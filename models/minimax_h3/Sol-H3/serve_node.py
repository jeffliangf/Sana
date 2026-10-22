#!/usr/bin/env python3
"""Supervise four independent two-H200 HTTP workers in one Slurm node job."""

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
    if len(devices) != 8 or any(not item.strip() for item in devices):
        raise ValueError(f"Expected eight Slurm-assigned GPUs, got {devices}")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 32:
        raise ValueError(f"Expected at least 32 allocated logical CPUs, got {len(cpus)}")
    import fastapi
    import uvicorn
    names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
    if len(names) != 8 or any("H200" not in name for name in names):
        raise ValueError("The node allocation must contain eight H200 GPUs")
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
        assigned_cpus = cpus[index * len(cpus) // 4:(index + 1) * len(cpus) // 4]
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
                   "--queue-capacity", "8", "--encoder-threads", "8"]
        process = subprocess.Popen(command, cwd=root / "Sol-H3", env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        log.close()
        print(f"START worker={index} port={port} GPUs={pair} pid={process.pid} attempt={attempt}", flush=True)
        return {"process": process, "attempt": attempt, "started": time.monotonic(), "index": index,
                "port": port, "gpus": pair, "endpoint": f"http://{address}:{port}", "log": str(log_path),
                "ready_seen": False}

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for index in range(4):
            children.append(start(index, 0))
        while not stopping:
            states = []
            for index, child in enumerate(children):
                if child["process"].poll() is not None:
                    if child["attempt"] >= 2:
                        raise RuntimeError(f"Worker {index} failed three times; see {child['log']}")
                    children[index] = child = start(index, child["attempt"] + 1)
                try:
                    with opener.open(child["endpoint"] + "/healthz", timeout=2) as response:
                        health = json.load(response)
                except urllib.error.HTTPError as error:
                    health = json.load(error)
                except (OSError, ValueError):
                    health = {"status": "unreachable"}
                if health.get("status") == "ready":
                    child["ready_seen"] = True
                # Shape compilation can briefly hold the GIL and delay HTTP.
                # A startup deadline must never become a lifetime deadline for
                # an already healthy worker after a single slow health probe.
                if not child["ready_seen"] and time.monotonic() - child["started"] > 1800:
                    raise RuntimeError(f"Worker {index} was not ready after 30 minutes; see {child['log']}")
                states.append({key: value for key, value in child.items() if key not in {"process", "started"}} | {"health": health})
            write_json(registry / "node.json", {"hostname": hostname, "address": address, "slurm_job_id": job_id,
                                                "updated_at": time.time(), "status": "ready" if all(x["health"].get("status") == "ready" for x in states) else "starting",
                                                "workers": states})
            for _ in range(10):
                if stopping:
                    break
                time.sleep(1)
    finally:
        for child in children:
            if child["process"].poll() is None:
                os.killpg(child["process"].pid, signal.SIGTERM)
        for child in children:
            try:
                child["process"].wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child["process"].pid, signal.SIGKILL)
                child["process"].wait()
            (registry / f"{child['port']}.ready.json").unlink(missing_ok=True)
        write_json(registry / "node.json", {"hostname": hostname, "address": address, "slurm_job_id": job_id,
                                            "updated_at": time.time(), "status": "stopped", "workers": []})


if __name__ == "__main__":
    main()
