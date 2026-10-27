"""A failing GPU pair must not tear down healthy sibling workers."""

import importlib.util
import io
import errno
import json
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


def test_full_local_log_volume_disables_only_the_worker_being_restarted(tmp_path):
    healthy = child(1)
    children = [child(0, dead=True), healthy]
    opener = Mock()
    opener.open.side_effect = lambda *args, **kwargs: io.StringIO('{"status":"ready"}')
    start = Mock(side_effect=OSError(errno.ENOSPC, "cannot open new worker log"))
    for _ in range(2):
        states = supervisor.refresh_workers(children, start, opener, tmp_path)
        assert states[0]["disabled"] and states[1]["health"]["status"] == "ready"
        assert supervisor.node_status(states) == "degraded"
    assert start.call_count == 1
    healthy["process"].wait.assert_not_called()


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


def test_additional_allocation_preserves_original_registry_and_worker_ports(tmp_path):
    original, original_ports = supervisor.allocation_layout(tmp_path, "gpu-node", 2)
    extra, extra_ports = supervisor.allocation_layout(tmp_path, "gpu-node", 2, 18082)
    original.mkdir(parents=True)
    extra.mkdir(parents=True)
    supervisor.write_json(original / "node.json", {"job_id": "original", "ports": original_ports})
    saved = (original / "node.json").read_bytes()
    supervisor.write_json(extra / "node.json", {"job_id": "extra", "ports": extra_ports})
    assert original == tmp_path / "deployment/gpu-node"
    assert (original / "node.json").read_bytes() == saved
    assert set(original_ports).isdisjoint(extra_ports)
    assert len(list((tmp_path / "deployment").glob("*/node.json"))) == 2


@pytest.mark.parametrize("workers,base_port", [(0, 18080), (5, 18080), (2, 0), (2, 65535)])
def test_invalid_allocation_range_is_rejected(tmp_path, workers, base_port):
    with pytest.raises(ValueError):
        supervisor.allocation_layout(tmp_path, "gpu-node", workers, base_port)


@pytest.mark.parametrize("error_code", [errno.ENOSPC, errno.EIO])
@pytest.mark.parametrize("gpu_type,device_count,group_size", [("H200", 4, 2), ("H100", 8, 4)])
def test_shared_storage_failure_preserves_workers_and_recovers(tmp_path, monkeypatch, error_code,
                                                              gpu_type, device_count, group_size):
    root, local = tmp_path / "shared", tmp_path / "local"
    (root / "deployment").mkdir(parents=True)
    (root / "deployment/token").write_text("x" * 40)
    for key, value in {"H3_DEPLOY_ROOT": str(root), "H3_LOCAL_ROOT": str(local),
                       "SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(device_count))),
                       "H3_GPU_TYPE": gpu_type, "H3_BASE_PORT": "18082"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("H3_GPUS_PER_WORKER", raising=False)
    monkeypatch.delenv("H3_BIND_HOST", raising=False)
    monkeypatch.setattr(supervisor.socket, "gethostname", lambda: "gpu-node")
    monkeypatch.setattr(supervisor.socket, "gethostbyname", lambda _: "10.0.0.2")
    monkeypatch.setattr(supervisor.os, "sched_getaffinity", lambda _: set(range(64)))
    monkeypatch.setattr(supervisor.subprocess, "check_output", lambda *args, **kwargs: f"NVIDIA {gpu_type}\n" * device_count)
    popen = Mock(side_effect=lambda *args, **kwargs: Mock(pid=100, poll=Mock(return_value=None)))
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    stopped = Mock()
    monkeypatch.setattr(supervisor, "stop_child", stopped)
    handlers = {}
    monkeypatch.setattr(supervisor.signal, "signal", lambda signum, callback: handlers.update({signum: callback}))
    original_write = supervisor.write_json
    snapshots = []

    def write(path, value):
        snapshots.append(value)
        if len(snapshots) == 1:
            raise OSError(error_code, "injected shared filesystem failure")
        original_write(path, value)

    monkeypatch.setattr(supervisor, "write_json", write)

    def refresh(children, start, opener, runtime):
        assert runtime == local / "sol-h3-supervisor/gpu-node-p18082"
        assert len(children) == 2
        stopped.assert_not_called()
        assert popen.call_count == 2
        return [{"port": child["port"], "endpoint": child["endpoint"], "health": {"status": "ready"}}
                for child in children]

    monkeypatch.setattr(supervisor, "refresh_workers", refresh)

    def sleep(_):
        if len(snapshots) >= 2:
            path = root / "deployment/gpu-node-p18082/node.json"
            assert json.loads(path.read_text())["status"] == "ready"
            handlers[supervisor.signal.SIGTERM](None, None)

    monkeypatch.setattr(supervisor.time, "sleep", sleep)
    supervisor.main()
    assert [value["status"] for value in snapshots] == ["ready", "ready", "stopped"]
    assert stopped.call_count == 2
    assigned = []
    for call in popen.call_args_list:
        command = call.args[0]
        assert f"--nproc_per_node={group_size}" in command
        assert command[command.index("--expect-gpu") + 1] == gpu_type
        devices = call.kwargs["env"]["CUDA_VISIBLE_DEVICES"].split(",")
        assert len(devices) == group_size
        assigned.extend(devices)
        assert Path(command[command.index("--ready-file") + 1]).is_relative_to(local)
        assert Path(call.kwargs["stdout"].name).is_relative_to(local)
    assert sorted(assigned) == list(map(str, range(device_count)))


@pytest.mark.parametrize("devices,family,size", [(["0", "0"], "H200", 2), (["0", "1"], "H100", 4),
                                                   (["0", "1"], "A100", 2), (["0", "1"], "H200", 1)])
def test_invalid_gpu_groups_fail_before_launch(devices, family, size):
    with pytest.raises(ValueError):
        supervisor.gpu_layout(devices, family, size)
