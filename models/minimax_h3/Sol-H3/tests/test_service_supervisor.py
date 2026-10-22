"""A failing GPU pair must not tear down healthy sibling workers."""

import importlib.util
import io
from pathlib import Path
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location("h3_supervisor", Path(__file__).resolve().parents[1] / "serve_node.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def child(index, attempt=0, dead=False):
    process = Mock()
    process.poll.return_value = 1 if dead else None
    return {"index": index, "process": process, "attempt": attempt, "port": 18080 + index,
            "log": f"worker-{index}.log", "endpoint": f"http://localhost:{18080 + index}",
            "started": supervisor.time.monotonic(), "ready_seen": False}


def test_three_failures_disable_only_one_pair(tmp_path):
    healthy = child(1)
    children = [child(0, dead=True), healthy]
    opener = Mock()
    opener.open.side_effect = lambda *args, **kwargs: io.StringIO('{"status":"ready"}')
    start = Mock(side_effect=lambda index, attempt: child(index, attempt, dead=True))
    for _ in range(3):
        states = supervisor.refresh_workers(children, start, opener, tmp_path)
    assert [call.args for call in start.call_args_list] == [(0, 1), (0, 2)]
    assert children[0]["disabled"]
    assert children[1] is healthy
    healthy["process"].wait.assert_not_called()
    assert states[1]["health"]["status"] == "ready"
    assert supervisor.node_status(states) == "degraded"
    supervisor.refresh_workers(children, start, opener, tmp_path)
    assert start.call_count == 2


def test_all_exhausted_workers_release_node(tmp_path):
    children = [child(index, attempt=2, dead=True) for index in range(4)]
    start, opener = Mock(), Mock()
    states = supervisor.refresh_workers(children, start, opener, tmp_path)
    assert supervisor.node_status(states) == "failed"
    start.assert_not_called()
    opener.open.assert_not_called()


@pytest.mark.parametrize("was_ready", [False, True])
def test_startup_deadline_never_kills_previously_ready_worker(tmp_path, monkeypatch, was_ready):
    worker = child(0, attempt=2)
    worker.update(started=0, ready_seen=was_ready)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 2000)
    stop = Mock()
    monkeypatch.setattr(supervisor, "stop_child", stop)
    opener = Mock()
    opener.open.side_effect = OSError("transient health timeout")
    states = supervisor.refresh_workers([worker], Mock(), opener, tmp_path)
    assert stop.call_count == (0 if was_ready else 1)
    assert supervisor.node_status(states) == ("degraded" if was_ready else "failed")
