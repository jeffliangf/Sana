#!/usr/bin/env python3
"""Supervise independent two-H200 HTTP workers in one Slurm node job."""

from __future__ import annotations

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


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


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
            (registry / f"{child['port']}.ready.json").unlink(missing_ok=True)
            if child["attempt"] >= 2:
                if not child.get("disabled"):
                    print(f"DISABLED worker={index} after three failures; see {child['log']}", flush=True)
                child["disabled"] = True
            else:
                children[index] = child = start(index, child["attempt"] + 1)
        if child.get("disabled"):
            health = {"status": "failed", "error": "worker exhausted three attempts"}
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
                print(f"STARTUP_TIMEOUT worker={index}; see {child['log']}", flush=True)
                stop_child(child)
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


def main():
    os.umask(0o077)
    root = Path(os.environ["H3_DEPLOY_ROOT"])
    local = Path(os.environ["H3_LOCAL_ROOT"])
    job_id = os.environ["SLURM_JOB_ID"]
    hostname = socket.gethostname()
    address = socket.gethostbyname(hostname)
    registry = root / "deployment" / hostname
    registry.mkdir(parents=True, exist_ok=True)
    token_file = root / "deployment" / "token"
    if len(token_file.read_text().strip()) < 32:
        raise ValueError("Missing deployment bearer token")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) not in {2, 4, 6, 8} or any(not item.strip() for item in devices):
        raise ValueError(f"Expected 2, 4, 6, or 8 Slurm-assigned GPUs, got {devices}")
    workers = len(devices) // 2
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 32:
        raise ValueError(f"Expected at least 32 allocated logical CPUs, got {len(cpus)}")
    import fastapi
    import uvicorn
    names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
    if len(names) < len(devices) or any("H200" not in name for name in names):
        raise ValueError("The node allocation must contain H200 GPUs")
    print(json.dumps({"node": hostname, "address": address, "job_id": job_id, "cpus": cpus,
                      "devices": devices, "fastapi": fastapi.__version__, "uvicorn": uvicorn.__version__}), flush=True)

    children = []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def start(index, attempt):
        port = 18080 + index
        pair = devices[2 * index:2 * index + 2]
        assigned_cpus = cpus[index * len(cpus) // workers:(index + 1) * len(cpus) // workers]
        ready_file = registry / f"{port}.ready.json"
        ready_file.unlink(missing_ok=True)
        log_path = registry / f"{job_id}-{port}-attempt{attempt}.log"
        log = log_path.open("a")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(pair)}
        command = ["taskset", "--cpu-list", ",".join(map(str, assigned_cpus)),
                   str(Path(sys.executable).with_name("torchrun")), "--standalone", "--nnodes=1", "--nproc_per_node=2",
                   "serve.py", "--model", str(local / "sol-h3-checkpoints/MiniMax-H3"),
                   "--adapter", str(local / "sol-h3-checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors"),
                   "--output-dir", str(local / f"sol-h3-service/{port}"), "--token-file", str(token_file),
                   "--host", address, "--port", str(port), "--attention-backend", "sol",
                   "--warmup-prompt-file", str(root / "demo_prompt.json"), "--ready-file", str(ready_file),
                   "--queue-capacity", "8", "--encoder-threads", "8", "--gate-failure", "dense"]
        process = subprocess.Popen(command, cwd=root / "Sol-H3", env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        log.close()
        print(f"START worker={index} port={port} GPUs={pair} pid={process.pid} attempt={attempt}", flush=True)
        return {"process": process, "attempt": attempt, "started": time.monotonic(), "index": index,
                "port": port, "gpus": pair, "endpoint": f"http://{address}:{port}", "log": str(log_path),
                "ready_seen": False}

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    states = []
    try:
        for index in range(workers):
            children.append(start(index, 0))
        while not stopping:
            states = refresh_workers(children, start, opener, registry)
            status = node_status(states)
            write_json(registry / "node.json", {"hostname": hostname, "address": address, "slurm_job_id": job_id,
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
            (registry / f"{child['port']}.ready.json").unlink(missing_ok=True)
        write_json(registry / "node.json", {"hostname": hostname, "address": address, "slurm_job_id": job_id,
                                            "updated_at": time.time(), "status": "stopped", "workers": states})


if __name__ == "__main__":
    main()
