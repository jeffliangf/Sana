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
from h3_runtime import adapters

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_launch_supervisor", SOURCE / "serve_node.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)
spec = importlib.util.spec_from_file_location("h3_launch_watchdog", SOURCE / "watchdog/watchdog.py")
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)
RELEASE = "/home/jeff/sol-h3-h200/releases/20261110-async/Sol-H3"
REQUIRED = ["--model", "m", "--adapter", "a", "--output-dir", "o", "--token-file", "t", "--warmup-prompt-file", "w"]
DEFAULT_PROFILE = "fasth3-dense-datafree"


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


def code_dir(path):
    """A stand-in code directory: serve.py plus the adapter profile registry the supervisor validates against."""
    (path / "h3_runtime").mkdir(parents=True, exist_ok=True)
    (path / "serve.py").write_text("")
    shutil.copy2(SOURCE / "adapter_profiles.json", path / "adapter_profiles.json")
    shutil.copy2(SOURCE / "h3_runtime/adapters.py", path / "h3_runtime/adapters.py")
    return path


def run_supervisor(tmp_path, monkeypatch, env):
    root, local = tmp_path / "shared", tmp_path / "local"
    (root / "deployment").mkdir(parents=True)
    (root / "deployment/token").write_text("x" * 40)
    code_dir(root / "Sol-H3")
    for key in ("H3_ASYNC_PUBLICATION", "H3_CODE_DIR", "H3_GPUS_PER_WORKER", "H3_BIND_HOST", "H3_BASE_PORT",
                "H3_MAX_NUM_FRAMES", "H3_ATTENTION_BACKEND", "H3_MIN_FREE_GIB", "H3_ADAPTER_PROFILE", "H3_GATE_FAILURE"):
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
    assert registry["launch"] == {"code_dir": str(root / "Sol-H3"), "async_publication": False,
                                  "adapter_profiles": [DEFAULT_PROFILE, DEFAULT_PROFILE], "gate_failure": "dense"}
    for call in popen.call_args_list:
        assert call.kwargs["cwd"] == str(root / "Sol-H3") and "--async-publication" not in call.args[0]
        assert profile_of(call.args[0]) == DEFAULT_PROFILE

    release = code_dir(tmp_path / "release")
    popen, registry, _ = run_supervisor(tmp_path / "async", monkeypatch,
                                        {"H3_ASYNC_PUBLICATION": "1", "H3_CODE_DIR": str(release)})
    assert registry["launch"] == {"code_dir": str(release), "async_publication": True,
                                  "adapter_profiles": [DEFAULT_PROFILE, DEFAULT_PROFILE], "gate_failure": "dense"}
    assert registry["status"] == "stopped"  # the final record keeps the launch settings too
    assert popen.call_count == 2
    for call in popen.call_args_list:
        assert call.kwargs["cwd"] == str(release) and call.args[0][-1] == "--async-publication"
        assert call.kwargs["env"]["H3_ASYNC_PUBLICATION"] == "1" and call.kwargs["env"]["H3_CODE_DIR"] == str(release)


def profile_of(command):
    return command[command.index("--adapter-profile") + 1]


@pytest.mark.parametrize("setting,expected", [
    ("lightx2v-fl2v-4step-v1.3", ["lightx2v-fl2v-4step-v1.3"] * 2),
    ("lightx2v-fl2v-4step-v1.2, lightx2v-fl2v-8step-v1.2", ["lightx2v-fl2v-4step-v1.2", "lightx2v-fl2v-8step-v1.2"]),
])
def test_supervisor_gives_each_worker_its_adapter_profile(tmp_path, monkeypatch, setting, expected):
    popen, registry, _ = run_supervisor(tmp_path, monkeypatch, {"H3_ADAPTER_PROFILE": setting})
    assert registry["launch"]["adapter_profiles"] == expected
    assert [profile_of(call.args[0]) for call in popen.call_args_list] == expected
    for call in popen.call_args_list:
        assert "--adapter" not in call.args[0] and "H3_ADAPTER_PROFILE" not in call.kwargs["env"]


@pytest.mark.parametrize("setting", ["no-such-profile", "lightx2v-ref2v-4step-v0.1",
                                     "fasth3-dense-datafree,fasth3-dense-datafree,fasth3-dense-datafree"])
def test_supervisor_refuses_unknown_or_miscounted_profiles(tmp_path, monkeypatch, setting):
    with pytest.raises(ValueError):
        run_supervisor(tmp_path, monkeypatch, {"H3_ADAPTER_PROFILE": setting})


def gate_of(command):
    return command[command.index("--gate-failure") + 1]


def test_supervisor_runs_songyangs_smd_e_setting(tmp_path, monkeypatch):
    """SMD-E with no numerical gate, as on 035/067: every worker, and the registry the watchdog restarts from."""
    popen, registry, _ = run_supervisor(tmp_path, monkeypatch,
                                        {"H3_ADAPTER_PROFILE": "smd-e-8step", "H3_GATE_FAILURE": "off"})
    assert registry["launch"]["adapter_profiles"] == ["smd-e-8step"] * 2
    assert registry["launch"]["gate_failure"] == "off"
    for call in popen.call_args_list:
        assert profile_of(call.args[0]) == "smd-e-8step" and gate_of(call.args[0]) == "off"
        assert "H3_GATE_FAILURE" not in call.kwargs["env"]


@pytest.mark.parametrize("setting", ["error", "none", "OFF"])
def test_supervisor_refuses_unknown_gate_settings(tmp_path, monkeypatch, setting):
    with pytest.raises(ValueError, match="H3_GATE_FAILURE"):
        run_supervisor(tmp_path, monkeypatch, {"H3_GATE_FAILURE": setting})


def test_run_worker_default_profile_is_the_registry_default():
    defaults, _ = adapters.load_profiles(SOURCE / "adapter_profiles.json")
    assert DEFAULT_PROFILE == defaults["t2v"]
    assert profile_of(render("003", "18081")[1]) == DEFAULT_PROFILE


@pytest.mark.parametrize("env", [{"H3_ASYNC_PUBLICATION": "yes"}, {"H3_CODE_DIR": "/nonexistent/release"},
                                 {"H3_CODE_DIR": "relative/Sol-H3"}])
def test_supervisor_refuses_invalid_launch_settings(tmp_path, env):
    root = tmp_path / "shared"
    (root / "Sol-H3").mkdir(parents=True)
    (root / "Sol-H3/serve.py").write_text("")
    with pytest.raises(ValueError):
        supervisor.launch_settings(env, root)


@pytest.mark.parametrize("settings", [(), ("0", RELEASE), ("1", RELEASE), ("1", RELEASE, "lightx2v-fl2v-8step-v1.2"),
                                      ("1", RELEASE, "smd-e-8step", "off"), ("1", RELEASE, "smd-e-8step", "dense")])
def test_watchdog_restarts_with_the_supervisor_arguments(settings):
    environ = {} if not settings else {"H3_ASYNC_PUBLICATION": settings[0], "H3_CODE_DIR": settings[1]}
    local = Path("/var/tmp/jeff-sol-h3")
    expected = supervisor.serve_arguments(
        environ=environ, root=Path("/home/jeff/sol-h3-h200"), local=local,
        checkpoints=local / "sol-h3-checkpoints", token_file=Path("/home/jeff/sol-h3-h200/deployment/token"),
        address="10.2.201.3", port=18081, ready_file=Path("/ready.json"), gpu_type="H200",
        adapter_profile=settings[2] if len(settings) > 2 else DEFAULT_PROFILE,
        gate_failure=settings[3] if len(settings) > 3 else "dense")
    env, arguments = render("003", "18081", *settings)
    assert without_ready(arguments) == without_ready(expected)
    assert env["CWD"] == (settings[1] if settings else "/home/jeff/sol-h3-h200/Sol-H3")
    assert env["CPUS"] == "40-79" and env["H3_SERVING_MEMORY_FRACTION"] == "0.95"
    # Restarts of allocations without launch settings keep exactly the old environment.
    assert ({"H3_ASYNC_PUBLICATION", "H3_CODE_DIR"} <= set(env)) == bool(settings)


@pytest.mark.parametrize("arguments", [("003", "18081", "1"), ("003", "18081", "2", RELEASE),
                                       ("003", "18081", "1", "/tmp/elsewhere/Sol-H3"),
                                       ("003", "18081", "1", "/home/jeff/sol-h3-h200/../evil"),
                                       ("004", "18081"), ("003", "18084"),
                                       ("003", "18081", "1", RELEASE, "Bad_Profile"),
                                       ("003", "18081", "1", RELEASE, "x; rm -rf /"),
                                       ("003", "18081", "", "", "fasth3-dense-datafree"),
                                       ("003", "18081", "1", RELEASE, "smd-e-8step", "error"),
                                       ("003", "18081", "1", RELEASE, "", "off"),
                                       ("003", "18081", "", "", "", "off")])
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
    # Port 18081 is the second worker, so it restarts with the second profile.
    (registry(launch={"code_dir": RELEASE, "async_publication": True,
                      "adapter_profiles": ["fasth3-dense-datafree", "lightx2v-fl2v-4step-v1.3"]}),
     ["1", RELEASE, "lightx2v-fl2v-4step-v1.3"]),
    (registry(launch={"code_dir": RELEASE, "async_publication": True, "adapter_profiles": ["fasth3-dense-datafree"]}),
     None),
    (registry(launch={"code_dir": RELEASE, "async_publication": True, "adapter_profiles": "fasth3-dense-datafree"}),
     None),
    # The gate setting follows the profile; a registry without one restarts with the dense-fallback gate.
    (registry(launch={"code_dir": RELEASE, "async_publication": True,
                      "adapter_profiles": ["smd-e-8step", "smd-e-8step"], "gate_failure": "off"}),
     ["1", RELEASE, "smd-e-8step", "off"]),
    (registry(launch={"code_dir": RELEASE, "async_publication": True,
                      "adapter_profiles": ["smd-e-8step", "smd-e-8step"], "gate_failure": "error"}), None),
    (registry(launch={"code_dir": RELEASE, "async_publication": True, "gate_failure": "off"}), None),
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
