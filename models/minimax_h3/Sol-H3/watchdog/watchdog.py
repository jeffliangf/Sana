#!/usr/bin/env python3
"""Hourly watchdog for the Sol-H3 H200 workers: restart a worker that is down when it is safe to.

Runs on the gateway host as the systemd --user unit h3-watchdog.service, started every hour by h3-watchdog.timer.
Manual run: python3 watchdog.py [--dry-run]. Log: watchdog.log here; last-run.json holds the latest per-worker view.

A worker whose process does not answer /healthz (3 tries, 10 s timeout, 10 s apart; a busy worker can take a few
seconds to answer) is restarted only when:
- its node's Slurm job h3-api-<node>-p18080 is RUNNING;
- it is not in pool.json excluded_workers (list a worker there to keep the watchdog away from it);
- serve_node.py no longer manages it: the node registry marks the port disabled, or is older than 5 minutes;
- its unit h3-worker-<node>-<port> is not running, starting or waiting to restart, and did not fail after its
  three allowed starts within the last 6 hours (a crash loop waits for a person or the cooldown).
The restart is the srun --overlap step used for the hand revivals of 2026-11-01/02, run as a systemd --user unit
(Restart=on-failure, RestartSec=60, at most 3 starts per 6 h). run-worker.sh refuses busy or unhealthy GPUs and a
taken port with exit 2, which the unit does not retry. The worker is added to pool.json additional_workers; the
gateway sends it jobs only once its /healthz reports ready, after a full warmup generation.

It never stops a worker whose process answers, never signals supervisors, and never submits or cancels Slurm jobs.
It runs only on the gateway host: ~/.config/systemd/user is on the shared /home, so every host where jeff has a
systemd --user manager also loads the timer; elsewhere it exits at once. The run lock is host-local (/run/user/<uid>)
because a lock on shared storage was held by another host's run from 2026-11-02 06:00 on and blocked every run here.
Since 2026-11-02 serve_node.py restarts crashed workers itself and disables a worker only after three failed
starts in a row, so this is a backstop for disabled workers. serve_node.py still releases its allocation when
every worker is disabled, which ends every step on the node, watchdog units included; each run logs a warning
for nodes with one enabled worker left.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

GATEWAY_HOST = "VM-187-fukushimadc"
ROOT = Path("/home/jeff/sol-h3-h200")
HERE = Path(__file__).resolve().parent
POOL = ROOT / "gateway/pool.json"
NODES = {"0002": "10.2.201.2", "0005": "10.2.201.5", "0006": "10.2.201.6", "0023": "10.2.201.23"}
PORTS = (18080, 18081, 18082, 18083)
CRASH_LOOP_COOLDOWN = 6 * 3600
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(message):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + message
    print(line, flush=True)
    with open(HERE / "watchdog.log", "a") as handle:
        handle.write(line + "\n")


def command(*args):
    return subprocess.run(args, capture_output=True, text=True, timeout=120, check=True).stdout


def health(url):
    """The worker's reported status (ready, warming, ...), or None when no process answers HTTP."""
    for attempt in range(3):
        try:
            with OPENER.open(url + "/healthz", timeout=10) as response:
                return json.load(response).get("status") or "unknown"
        except urllib.error.HTTPError as error:  # 503 while warming or busy: the process is alive
            try:
                return json.load(error).get("status") or f"http-{error.code}"
            except ValueError:
                return f"http-{error.code}"
        except (OSError, ValueError):
            if attempt < 2:
                time.sleep(10)
    return None


def registry(node):
    path = ROOT / f"deployment/fukushimadc-02-hgx-{node}/node.json"
    for _ in range(5):  # the MooseFS atomic replace can hide the file for a moment
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(1)
    return None


def owner(state, port):
    """'supervisor' while serve_node.py still manages the port, 'watchdog' once it does not, None if unknown."""
    if state is None:
        return None
    if time.time() - state.get("updated_at", 0) > 300 or state.get("status") in {"stopped", "failed"}:
        return "watchdog"
    disabled = {worker.get("port"): worker.get("disabled") for worker in state.get("workers", [])}
    return "watchdog" if disabled.get(port, True) else "supervisor"


def unit_state(unit):
    keys = ("ActiveState", "SubState", "Result", "ExecMainStatus", "InactiveEnterTimestampMonotonic")
    output = command("systemctl", "--user", "show", unit, *(f"--property={key}" for key in keys))
    state = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    stamp = int(state.get("InactiveEnterTimestampMonotonic") or 0)
    state["failed_ago"] = time.monotonic() - stamp / 1e6 if stamp else None
    return state


def start(node, port, job):
    index = port - 18080
    command("systemd-run", "--user", f"--unit=h3-worker-{node}-{port}",
            f"--description=H3 worker {node}:{port} on GPUs {2 * index},{2 * index + 1} "
            f"(extra step in Slurm job {job}, started by h3-watchdog)",
            "-p", "Restart=on-failure", "-p", "RestartSec=60", "-p", "StartLimitIntervalSec=6h",
            "-p", "StartLimitBurst=3", "-p", "RestartPreventExitStatus=2",
            "srun", f"--jobid={job}", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=160", "--mem=0",
            "--cpu-bind=none", f"--job-name=h3-{node}-{port}", f"--output={HERE}/srun-%j.%s.log",
            "bash", str(HERE / "run-worker.sh"), node, str(port))


def route(url):
    """Make sure the gateway can reach a worker the registries do not list."""
    pool = json.loads(POOL.read_text())
    if url in pool.get("additional_workers", []):
        return "routed"
    if "workers" in pool:
        return "not routed: pool.json has an explicit workers list"
    pool["additional_workers"] = sorted(set(pool.get("additional_workers", [])) | {url})
    temporary = POOL.with_name(".pool.json.watchdog.tmp")
    temporary.write_text(json.dumps(pool, indent=2) + "\n")
    os.replace(temporary, POOL)
    return "added to additional_workers"


def decide(node, port, url, status, state, jobs, excluded, dry_run):
    key, unit = f"{node}:{port}", f"h3-worker-{node}-{port}"
    who = owner(state, port)
    if url in excluded:
        return "excluded"
    if status is not None:
        if who == "watchdog" and not dry_run:
            routed = route(url)
            if routed != "routed":
                log(f"{key} answers but was not routed; {routed}")
        return f"up ({status})"
    if node not in jobs:
        return "down; no running allocation on the node"
    if who is None:
        return "down; node registry unreadable, retry next run"
    if who == "supervisor":
        return "down; serve_node.py manages it (restart in progress)"
    unit_now = unit_state(unit)
    active = unit_now.get("ActiveState", "")
    if active in {"active", "activating", "reloading", "deactivating"}:
        return f"down; {unit} is {active}/{unit_now.get('SubState')}, starting or restarting"
    failed_ago = unit_now["failed_ago"]
    if (active == "failed" and unit_now.get("ExecMainStatus") != "2" and failed_ago is not None
            and failed_ago < CRASH_LOOP_COOLDOWN):
        return (f"down; {unit} failed {failed_ago / 3600:.1f} h ago after its allowed restarts "
                f"({unit_now.get('Result')}); next try 6 h after that failure")
    if dry_run:
        return f"down; would start {unit} in job {jobs[node]}"
    if active == "failed":
        command("systemctl", "--user", "reset-failed", unit)
    start(node, port, jobs[node])
    return f"down; started {unit} in job {jobs[node]}, {route(url)}"


def main():
    parser = argparse.ArgumentParser(description="Restart Sol-H3 workers that are down, when it is safe to.")
    parser.add_argument("--dry-run", action="store_true", help="report what would be done; change nothing")
    dry_run = parser.parse_args().dry_run
    if socket.gethostname() != GATEWAY_HOST:
        print(f"h3-watchdog runs only on {GATEWAY_HOST}; this is {socket.gethostname()}; exiting", flush=True)
        return
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
    lock = open((runtime if runtime.is_dir() else Path("/tmp")) / f"h3-watchdog-{os.getuid()}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another watchdog run holds the lock; exiting")
        return
    prefix = "[dry run] " if dry_run else ""
    excluded = set(json.loads(POOL.read_text()).get("excluded_workers", []))
    jobs = {}
    for line in command("squeue", "-h", "-u", "jeff", "-t", "RUNNING", "-o", "%i %j %N").splitlines():
        fields = line.split()
        for node in NODES:
            if fields[1:3] == [f"h3-api-{node}-p18080", f"fukushimadc-02-hgx-{node}"]:
                jobs[node] = fields[0]
    states = {node: registry(node) for node in NODES}
    workers = [(node, port, f"http://{address}:{port}") for node, address in NODES.items() for port in PORTS]
    with concurrent.futures.ThreadPoolExecutor(len(workers)) as executor:
        statuses = list(executor.map(lambda worker: health(worker[2]), workers))
    report = {}
    for (node, port, url), status in zip(workers, statuses):
        try:
            report[f"{node}:{port}"] = decide(node, port, url, status, states[node], jobs, excluded, dry_run)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            report[f"{node}:{port}"] = f"error: {error!r}"
        if not report[f"{node}:{port}"].startswith(("up", "excluded")):
            log(f"{prefix}{node}:{port} {report[f'{node}:{port}']}")
    for node, state in states.items():  # serve_node.py releases the allocation once every worker is disabled
        alive = [w["port"] for w in (state or {}).get("workers", []) if not w.get("disabled")]
        fresh = state is not None and time.time() - state.get("updated_at", 0) < 300
        if node in jobs and fresh and len(alive) == 1:
            log(f"{prefix}warning: {node} allocation {jobs[node]} ends, with every worker on it, if its last enabled "
                f"serve_node.py worker ({alive[0]}) also fails three starts in a row")
    up = sum(value.startswith("up") for value in report.values())
    log(f"{prefix}summary: {up}/{len(report)} workers answering; allocations running: "
        f"{', '.join(f'{n}={j}' for n, j in sorted(jobs.items())) or 'none'}")
    if not dry_run:
        (HERE / "last-run.json").write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                        "jobs": jobs, "workers": report}, indent=1) + "\n")


if __name__ == "__main__":
    main()
