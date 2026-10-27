#!/usr/bin/env python3
"""Supervise independent H200 or H100 HTTP workers in one Slurm node job."""

from __future__ import annotations

from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from service_storage import atomic_json, quiet_print

def write_json(path, value):
    atomic_json(path, value)


def publish_registry(path, value):
    """A shared-filesystem heartbeat failure must not stop healthy GPU workers."""
    try:
        write_json(path, value)
    except OSError as error:
        # Keep supervising and retry on the next iteration. Discovery may pause
        # while storage is unavailable, but resident workers retain their jobs.
        try:
            print(f"REGISTRY_WRITE_FAILED path={path} error={error!r}; will retry", flush=True)
        except OSError:
            pass
        return False
    return True


def stop_child(child):
    process = child["process"]
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def refresh_workers(children, start, opener, registry):
    """Quarantine an exhausted worker while leaving healthy GPU pairs running."""
    states = []
    for index, child in enumerate(children):
        if child["process"].poll() is not None:
            with suppress(OSError):
                (registry / f"{child['port']}.ready.json").unlink(missing_ok=True)
            if child["attempt"] >= 2:
                if not child.get("disabled"):
                    quiet_print(f"DISABLED worker={index} after three failures; see {child['log']}")
                child["disabled"] = True
            elif not child.get("disabled"):
                try:
                    children[index] = child = start(index, child["attempt"] + 1)
                except OSError as error:
                    child["disabled"] = True
                    child["restart_error"] = str(error)
                    quiet_print(f"RESTART_FAILED worker={index} error={error!r}; siblings remain running")
        if child.get("disabled"):
            health = {"status": "failed", "error": child.get("restart_error", "worker exhausted three attempts")}
        else:
            try:
                with opener.open(child["endpoint"] + "/healthz", timeout=2) as response:
                    health = json.load(response)
            except urllib.error.HTTPError as error:
                try:
                    health = json.load(error)
                except ValueError:
                    health = {"status": "unreachable"}
            except (OSError, ValueError):
                health = {"status": "unreachable"}
            if health.get("status") == "ready":
                child["ready_seen"] = True
            if not child["ready_seen"] and time.monotonic() - child["started"] > 1800:
                quiet_print(f"STARTUP_TIMEOUT worker={index}; see {child['log']}")
                stop_child(child)
                with suppress(OSError):
                    (registry / f"{child['port']}.ready.json").unlink(missing_ok=True)
                child["disabled"] = child["attempt"] >= 2
                health = {"status": "failed" if child["disabled"] else "restarting",
                          "error": "worker was not ready after 30 minutes"}
        states.append({key: value for key, value in child.items() if key not in {"process", "started"}} | {"health": health})
    return states


def node_status(states):
    if all(item["health"].get("status") == "ready" for item in states):
        return "ready"
    if all(item.get("disabled") for item in states):
        return "failed"
    if any(item.get("disabled") or item.get("ready_seen") for item in states):
        return "degraded"
    return "starting"


def allocation_layout(root, hostname, workers, base_port=18080):
    """Keep concurrent allocations on one host in separate port and registry ranges."""
    if not 1 <= workers <= 4 or not 1 <= base_port <= 65536 - workers:
        raise ValueError("Expected one to four workers with ports in 1..65535")
    name = hostname if base_port == 18080 else f"{hostname}-p{base_port}"
    return Path(root) / "deployment" / name, list(range(base_port, base_port + workers))


def gpu_layout(devices, gpu_type, gpus_per_worker):
    if gpu_type not in {"H100", "H200"}:
        raise ValueError("H3_GPU_TYPE must be H100 or H200")
    if gpus_per_worker not in {2, 4}:
        raise ValueError("H3_GPUS_PER_WORKER must be 2 or 4")
    if (not 2 <= len(devices) <= 8 or len(devices) % gpus_per_worker or
            any(not device.strip() for device in devices) or len(set(devices)) != len(devices)):
        raise ValueError("Assigned GPUs must be unique and divisible into complete worker groups")
    return [devices[i:i + gpus_per_worker] for i in range(0, len(devices), gpus_per_worker)]


def main():
    os.umask(0o077)
    root = Path(os.environ["H3_DEPLOY_ROOT"])
    local = Path(os.environ["H3_LOCAL_ROOT"])
    job_id = os.environ["SLURM_JOB_ID"]
    hostname = socket.gethostname()
    address = os.environ.get("H3_BIND_HOST") or socket.gethostbyname(hostname)
    token_file = root / "deployment" / "token"
    if len(token_file.read_text().strip()) < 32:
        raise ValueError("Missing deployment bearer token")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    gpu_type = os.environ.get("H3_GPU_TYPE", "H200").upper()
    group_size = int(os.environ.get("H3_GPUS_PER_WORKER", "4" if gpu_type == "H100" else "2"))
    groups = gpu_layout(devices, gpu_type, group_size)
    workers = len(groups)
    base_port = int(os.environ.get("H3_BASE_PORT", "18080"))
    registry, ports = allocation_layout(root, hostname, workers, base_port)
    registry.mkdir(parents=True, exist_ok=True)
    runtime = local / "sol-h3-supervisor" / registry.name
    runtime.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 32:
        raise ValueError(f"Expected at least 32 allocated logical CPUs, got {len(cpus)}")
    import fastapi
    import uvicorn
    names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
    if len(names) < len(devices) or any(gpu_type not in name for name in names):
        raise ValueError(f"The node allocation must contain {gpu_type} GPUs")
    quiet_print(json.dumps({"node": hostname, "address": address, "job_id": job_id, "cpus": cpus,
                      "gpu_type": gpu_type, "gpus_per_worker": group_size,
                      "devices": devices, "ports": ports, "registry": str(registry), "runtime": str(runtime),
                      "fastapi": fastapi.__version__, "uvicorn": uvicorn.__version__}))

    children = []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def start(index, attempt):
        port = ports[index]
        group = groups[index]
        assigned_cpus = cpus[index * len(cpus) // workers:(index + 1) * len(cpus) // workers]
        ready_file = runtime / f"{port}.ready.json"
        with suppress(OSError):
            ready_file.unlink(missing_ok=True)
        log_path = runtime / f"{job_id}-{port}-attempt{attempt}.log"
        log = log_path.open("a")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(group)}
        checkpoints = Path(os.environ.get("H3_CHECKPOINT_ROOT", str(local / "sol-h3-checkpoints")))
        command = ["taskset", "--cpu-list", ",".join(map(str, assigned_cpus)),
                   str(Path(sys.executable).with_name("torchrun")), "--standalone", "--nnodes=1", f"--nproc_per_node={group_size}",
                   "serve.py", "--model", str(checkpoints / "MiniMax-H3"),
                   "--adapter", str(checkpoints / "FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors"),
                   "--output-dir", str(local / f"sol-h3-service/{port}"), "--token-file", str(token_file),
                   "--host", address, "--port", str(port), "--attention-backend", "sol",
                   "--warmup-prompt-file", str(root / "demo_prompt.json"), "--ready-file", str(ready_file),
                   "--queue-capacity", "8", "--encoder-threads", "8", "--gate-failure", "dense",
                   "--expect-gpu", gpu_type, "--min-free-gib", os.environ.get("H3_MIN_FREE_GIB", "20")]
        try:
            process = subprocess.Popen(command, cwd=root / "Sol-H3", env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        finally:
            log.close()
        quiet_print(f"START worker={index} port={port} GPUs={group} pid={process.pid} attempt={attempt}")
        return {"process": process, "attempt": attempt, "started": time.monotonic(), "index": index,
                "port": port, "gpus": group, "endpoint": f"http://{address}:{port}", "log": str(log_path),
                "ready_seen": False}

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    states = []
    try:
        for index in range(workers):
            children.append(start(index, 0))
        while not stopping:
            states = refresh_workers(children, start, opener, runtime)
            status = node_status(states)
            publish_registry(registry / "node.json", {"hostname": hostname, "address": address, "slurm_job_id": job_id,
                                                "base_port": base_port,
                                                "updated_at": time.time(), "status": status,
                                                "workers": states})
            if status == "failed":
                raise RuntimeError("All workers exhausted their retries; releasing node allocation")
            for _ in range(10):
                if stopping:
                    break
                time.sleep(1)
    finally:
        for child in children:
            stop_child(child)
            with suppress(OSError):
                (runtime / f"{child['port']}.ready.json").unlink(missing_ok=True)
        publish_registry(registry / "node.json", {"hostname": hostname, "address": address, "slurm_job_id": job_id,
                                            "base_port": base_port,
                                            "updated_at": time.time(), "status": "stopped", "workers": states})


if __name__ == "__main__":
    main()
