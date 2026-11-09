"""The publication mode and code directory reach every worker start: serve_node.py, deploy_node.sh, and the
watchdog's run-worker.sh, with the same serve.py arguments and H3_ environment (launch parity)."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import Mock

import pytest

import serve

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_launch_supervisor", SOURCE / "serve_node.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)
spec = importlib.util.spec_from_file_location("h3_launch_watchdog", SOURCE / "watchdog/watchdog.py")
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)
RELEASE = "/home/jeff/sol-h3-h200/releases/20261110-async/Sol-H3"
REQUIRED = ["--model", "m", "--adapter", "a", "--output-dir", "o", "--token-file", "t", "--warmup-prompt-file", "w"]


def render(*arguments, env=None):
    """run-worker.sh's environment, CPU list, working directory, and serve.py arguments, as H3_RENDER_ONLY prints them."""
    result = subprocess.run(["bash", str(SOURCE / "watchdog/run-worker.sh"), *arguments], capture_output=True,
                            text=True, env={"H3_RENDER_ONLY": "1", "PATH": "/usr/bin:/bin", "SLURM_JOB_ID": "13",
                                            **(env or {})})
    if result.returncode:
        return result.returncode, result.stderr
    lines = result.stdout.splitlines()
    split = lines.index("--")
    return dict(line.split("=", 1) for line in lines[:split]), lines[split + 1:]


def without_ready(arguments):
    index = arguments.index("--ready-file")
    return arguments[:index] + arguments[index + 2:]


@pytest.mark.parametrize("setting,flag,ok", [("", False, True), ("", True, True), ("0", False, True),
                                             ("1", True, True), ("1", False, False), ("0", True, False),
                                             ("yes", True, False)])
def test_serve_flag_must_agree_with_the_deployment_setting(monkeypatch, setting, flag, ok):
    monkeypatch.setenv("H3_ASYNC_PUBLICATION", setting)
    arguments = REQUIRED + (["--async-publication"] if flag else [])
    if ok:
        assert serve.parse_args(arguments).async_publication is flag
    else:
        with pytest.raises(SystemExit):
            serve.parse_args(arguments)


def run_supervisor(tmp_path, monkeypatch, env):
    root, local = tmp_path / "shared", tmp_path / "local"
    (root / "deployment").mkdir(parents=True)
    (root / "deployment/token").write_text("x" * 40)
    (root / "Sol-H3").mkdir()
    (root / "Sol-H3/serve.py").write_text("")
    for key in ("H3_ASYNC_PUBLICATION", "H3_CODE_DIR", "H3_GPUS_PER_WORKER", "H3_BIND_HOST", "H3_BASE_PORT",
                "H3_MAX_NUM_FRAMES", "H3_ATTENTION_BACKEND", "H3_MIN_FREE_GIB"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {"H3_DEPLOY_ROOT": str(root), "H3_LOCAL_ROOT": str(local), "SLURM_JOB_ID": "13",
                       "CUDA_VISIBLE_DEVICES": "0,1,2,3", "H3_GPU_TYPE": "H200", **env}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(supervisor.socket, "gethostname", lambda: "gpu-node")
    monkeypatch.setattr(supervisor.socket, "gethostbyname", lambda _: "10.0.0.2")
    monkeypatch.setattr(supervisor.os, "sched_getaffinity", lambda _: set(range(64)))
    monkeypatch.setattr(supervisor.subprocess, "check_output", lambda *args, **kwargs: "".join(
        f"{index}, GPU-{index:08d}-test, NVIDIA H200\n" for index in range(4)))
    popen = Mock(side_effect=lambda *args, **kwargs: Mock(pid=100, poll=Mock(return_value=None)))
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor, "stop_child", Mock())
    handlers = {}
    monkeypatch.setattr(supervisor.signal, "signal", lambda signum, callback: handlers.update({signum: callback}))
    monkeypatch.setattr(supervisor, "refresh_workers", lambda children, *rest: [
        {"port": child["port"], "health": {"status": "ready"}} for child in children])
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: handlers[supervisor.signal.SIGTERM](None, None))
    supervisor.main()
    return popen, json.loads((root / "deployment/gpu-node/node.json").read_text()), root


def test_supervisor_starts_workers_in_the_allocation_mode_and_code_directory(tmp_path, monkeypatch):
    popen, registry, root = run_supervisor(tmp_path / "default", monkeypatch, {})
    assert registry["launch"] == {"code_dir": str(root / "Sol-H3"), "async_publication": False}
    for call in popen.call_args_list:
        assert call.kwargs["cwd"] == str(root / "Sol-H3") and "--async-publication" not in call.args[0]

    release = tmp_path / "release"
    release.mkdir()
    (release / "serve.py").write_text("")
    popen, registry, _ = run_supervisor(tmp_path / "async", monkeypatch,
                                        {"H3_ASYNC_PUBLICATION": "1", "H3_CODE_DIR": str(release)})
    assert registry["launch"] == {"code_dir": str(release), "async_publication": True}
    assert registry["status"] == "stopped"  # the final record keeps the launch settings too
    assert popen.call_count == 2
    for call in popen.call_args_list:
        assert call.kwargs["cwd"] == str(release) and call.args[0][-1] == "--async-publication"
        assert call.kwargs["env"]["H3_ASYNC_PUBLICATION"] == "1" and call.kwargs["env"]["H3_CODE_DIR"] == str(release)


@pytest.mark.parametrize("env", [{"H3_ASYNC_PUBLICATION": "yes"}, {"H3_CODE_DIR": "/nonexistent/release"},
                                 {"H3_CODE_DIR": "relative/Sol-H3"}])
def test_supervisor_refuses_invalid_launch_settings(tmp_path, env):
    root = tmp_path / "shared"
    (root / "Sol-H3").mkdir(parents=True)
    (root / "Sol-H3/serve.py").write_text("")
    with pytest.raises(ValueError):
        supervisor.launch_settings(env, root)


@pytest.mark.parametrize("settings", [(), ("0", RELEASE), ("1", RELEASE)])
def test_watchdog_restarts_with_the_supervisor_arguments(settings):
    environ = {} if not settings else {"H3_ASYNC_PUBLICATION": settings[0], "H3_CODE_DIR": settings[1]}
    local = Path("/var/tmp/jeff-sol-h3")
    expected = supervisor.serve_arguments(
        environ=environ, root=Path("/home/jeff/sol-h3-h200"), local=local,
        checkpoints=local / "sol-h3-checkpoints", token_file=Path("/home/jeff/sol-h3-h200/deployment/token"),
        address="10.2.201.3", port=18081, ready_file=Path("/ready.json"), gpu_type="H200")
    env, arguments = render("003", "18081", *settings)
    assert without_ready(arguments) == without_ready(expected)
    assert env["CWD"] == (settings[1] if settings else "/home/jeff/sol-h3-h200/Sol-H3")
    assert env["CPUS"] == "40-79" and env["H3_SERVING_MEMORY_FRACTION"] == "0.95"
    # Restarts of allocations without launch settings keep exactly the old environment.
    assert ({"H3_ASYNC_PUBLICATION", "H3_CODE_DIR"} <= set(env)) == bool(settings)


@pytest.mark.parametrize("arguments", [("003", "18081", "1"), ("003", "18081", "2", RELEASE),
                                       ("003", "18081", "1", "/tmp/elsewhere/Sol-H3"),
                                       ("003", "18081", "1", "/home/jeff/sol-h3-h200/../evil"),
                                       ("004", "18081"), ("003", "18084")])
def test_run_worker_refuses_invalid_arguments(arguments):
    assert render(*arguments)[0] == 2


def copy_launcher(code):
    code.mkdir(parents=True, exist_ok=True)
    for name in ("deploy_node.sh", "storage_preflight.py", "service_storage.py"):
        shutil.copy2(SOURCE / name, code / name)


def run_deploy_node(tmp_path, **env):
    """deploy_node.sh with prepared node-local files and a stand-in supervisor printing what it inherits."""
    root, local = tmp_path / "shared", tmp_path / "node-local"
    copy_launcher(root / "Sol-H3")
    venv = local / "prepared-env"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text('#!/bin/bash\necho "PWD=$PWD"\nenv | grep "^H3_" | sort\n')
    (venv / "bin/python").chmod(0o700)
    (local / "prepared-weights").mkdir()
    # What the fleet's launch.sh exports before running deploy_node.sh.
    base = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "H3_DEPLOY_ROOT": str(root),
            "H3_LOCAL_ROOT": str(local), "H3_VENV": str(venv), "H3_CHECKPOINT_ROOT": str(local / "prepared-weights"),
            "H3_GPU_TYPE": "H200", "H3_GPUS_PER_WORKER": "2", "H3_BASE_PORT": "18080", "H3_BIND_HOST": "10.2.201.3",
            "H3_MIN_FREE_GIB": "1", "SLURM_JOB_ID": "test-13"}
    result = subprocess.run(["bash", str(root / "Sol-H3/deploy_node.sh")], env={**base, **env}, capture_output=True,
                            text=True, timeout=30)
    log = local / "sol-h3-supervisor/supervisor-test-13.log"
    inherited = dict(line.split("=", 1) for line in log.read_text().splitlines()) if log.exists() else {}
    return result, inherited, root


def test_deploy_node_exports_the_launch_settings_run_worker_renders(tmp_path):
    result, inherited, root = run_deploy_node(tmp_path / "default")
    assert result.returncode == 0, result.stderr
    assert inherited["H3_ASYNC_PUBLICATION"] == "0" and inherited["PWD"] == str(root / "Sol-H3")
    assert inherited["H3_CODE_DIR"] == str(root / "Sol-H3")

    release = tmp_path / "release/Sol-H3"
    copy_launcher(release)
    result, inherited, _ = run_deploy_node(tmp_path / "async", H3_ASYNC_PUBLICATION="1", H3_CODE_DIR=str(release))
    assert result.returncode == 0, result.stderr
    assert (inherited["H3_ASYNC_PUBLICATION"], inherited["H3_CODE_DIR"], inherited["PWD"]) == ("1", str(release),
                                                                                                str(release))
    # Every H3_ variable a supervisor worker inherits is one the watchdog's restart sets (env_check.py's rule).
    # H3_MIN_FREE_GIB is set only for this test's small disk; the fleet's launch.sh leaves the default.
    env, _ = render("003", "18080", "1", RELEASE)
    supervised = {key for key in inherited if key.startswith("H3_")} - {"H3_MIN_FREE_GIB"}
    assert supervised == {key for key in env if key.startswith("H3_")}

    result, inherited, _ = run_deploy_node(tmp_path / "invalid", H3_ASYNC_PUBLICATION="true")
    assert result.returncode == 1 and "H3_ASYNC_PUBLICATION must be 0 or 1" in result.stderr and not inherited


def registry(job="13", launch=None, **values):
    state = {"updated_at": time.time(), "status": "degraded", "slurm_job_id": job,
             "workers": [{"port": 18081, "disabled": True}], **values}
    if launch is not None:
        state["launch"] = launch
    return state


@pytest.mark.parametrize("state,expected", [
    (registry(launch={"code_dir": RELEASE, "async_publication": True}), ["1", RELEASE]),
    (registry(launch={"code_dir": RELEASE, "async_publication": False}), ["0", RELEASE]),
    (registry(), []),  # allocations started before launch settings existed restart as before
    (registry(job="12", launch={"code_dir": RELEASE, "async_publication": True}), None),
    (registry(launch={"code_dir": RELEASE, "async_publication": "1"}), None),
])
def test_watchdog_uses_the_allocation_launch_settings(monkeypatch, state, expected):
    start = Mock()
    monkeypatch.setattr(watchdog, "start", start)
    monkeypatch.setattr(watchdog, "unit_state", lambda unit: {"ActiveState": "inactive", "failed_ago": None})
    monkeypatch.setattr(watchdog, "route", lambda url: "added to additional_workers")
    result = watchdog.decide("006", 18081, "http://10.2.201.6:18081", None, state, {"006": "13"}, set(), False)
    if expected is None:
        start.assert_not_called()
        assert result.startswith("down; node registry")
    else:
        start.assert_called_once_with("006", 18081, "13", expected)
        assert ("async publication" in result) == (expected[:1] == ["1"])
