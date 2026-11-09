#!/usr/bin/env python3
"""Paired benchmark of synchronous and background MP4 publication on directly addressed workers.

Run once per publication mode against freshly started workers that are excluded from the gateway, with the
same manifest, seeds, and requests in flight. Every worker runs the whole manifest; each request gets a new
Idempotency-Key. Warm passes compile every prompt length first, so the measured pass is warm and its gate
fallbacks are comparable (the summary counts them; compare them across modes before trusting a difference).

  python3 publication_benchmark.py run --workers http://10.2.201.3:18080,http://10.2.201.3:18081 \\
      --manifest requests.jsonl --token-file /home/jeff/sol-h3-h200/deployment/token --output DIR \\
      [--inflight 2] [--warm-passes 1]
  python3 publication_benchmark.py compare SYNC_DIR ASYNC_DIR

DIR gets jobs.jsonl (client times plus the worker's final record), health.jsonl (/healthz every 2 s), and
summary.json. The overlap trace uses the worker's own timestamps (one clock per worker); client times are
used only for end-to-end latency. Use --inflight 1 to record the case with no next request waiting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LOCK = threading.Lock()


def call(url, token=None, body=None, key=None, timeout=60):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = None
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    if key:
        headers["Idempotency-Key"] = key
    with OPENER.open(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as response:
        return response.read()


def run_request(worker, token, body, phase):
    key = str(uuid.uuid4())
    row = {"worker": worker, "phase": phase, "key": key, "submitted": time.time()}
    while True:  # 429/503 and transport errors: repeat the same POST with the same key
        try:
            call(f"{worker}/v1/videos", token, body, key)
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 502, 503):
                return {**row, "error": f"HTTP {error.code} on submit"}
            time.sleep(float(error.headers.get("Retry-After") or 2))
        except OSError:
            time.sleep(2)
    while True:
        try:
            record = json.loads(call(f"{worker}/v1/jobs/{key}", token, timeout=30))
        except OSError:
            time.sleep(0.5)
            continue
        if record["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    row.update(seen=time.time(), record=record)
    if record["status"] == "succeeded":
        data = call(f"{worker}/v1/jobs/{key}/video", token, timeout=300)
        row.update(downloaded=time.time(), video_bytes=len(data), video_sha256=hashlib.sha256(data).hexdigest(),
                   valid=len(data) == record.get("bytes"))
    return row


def worker_pass(worker, token, manifest, inflight, phase, sink):
    queue = list(enumerate(manifest))

    def client():
        while True:
            with LOCK:
                if not queue:
                    return
                index, body = queue.pop(0)
            row = run_request(worker, token, body, phase)
            row["index"] = index
            with LOCK:
                sink.write(json.dumps(row) + "\n")
                sink.flush()
    threads = [threading.Thread(target=client) for _ in range(inflight)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def sample_health(workers, path, stop):
    with path.open("a") as output:
        while not stop.wait(2):
            for worker in workers:
                try:
                    health = json.loads(call(f"{worker}/healthz", timeout=5))
                except urllib.error.HTTPError as error:
                    health = json.loads(error.read() or b"{}")
                except OSError as error:
                    health = {"error": repr(error)}
                output.write(json.dumps({"at": time.time(), "worker": worker, "health": health}) + "\n")
            output.flush()


def percentiles(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {"n": len(values), "mean": statistics.fmean(values), "p50": pick(0.5), "p95": pick(0.95),
            "max": values[-1]}


def summarize_worker(rows, samples):
    done = [row for row in rows if row.get("valid")]
    records = sorted((row["record"] for row in done), key=lambda record: record["generation_started_at"])
    failures = {}
    for row in rows:
        if not row.get("valid"):
            reason = row.get("error") or row.get("record", {}).get("error_code") or "invalid download"
            failures[reason] = failures.get(reason, 0) + 1
    summary = {"completed": len(done), "failures": failures}
    if not records:
        return summary
    span = max(record["finished_at"] for record in records) - records[0]["generation_started_at"]
    starts = [record["generation_started_at"] for record in records]
    # Consecutive jobs on one worker: the next GPU start precedes this publication's end (overlap), and this
    # result became visible (finished_at) before the next generation ended.
    pairs = list(zip(records, records[1:]))
    overlapped = [(a["id"], b["id"]) for a, b in pairs if "published_at" in a and b["generation_started_at"] < a["published_at"]]
    visible_first = [(a["id"], b["id"]) for a, b in pairs
                     if b["generation_started_at"] < a.get("published_at", 0) and a["finished_at"] < b["generation_finished_at"]]
    summary.update({
        "publication_mode": records[0].get("publication_mode"),
        "videos_per_s": len(records) / span if span > 0 else None,
        "client_videos_per_s": len(done) / (max(r["downloaded"] for r in done) - min(r["submitted"] for r in done)),
        "generation_start_interval_s": percentiles(b - a for a, b in zip(starts, starts[1:])),
        "end_to_end_s": percentiles(row["downloaded"] - row["submitted"] for row in done),
        "processing_s": percentiles(record["processing_s"] for record in records),
        "pipeline_s": percentiles(record["pipeline_s"] for record in records),
        "gpu_slot_s": percentiles(record.get("gpu_slot_s") for record in records),
        "publication_slot_wait_s": percentiles(record.get("publication_slot_wait_s") for record in records),
        "publication_tail_s": percentiles(record.get("publication_tail_s") for record in records),
        "gpu_slot_occupancy": sum(record.get("gpu_slot_s") or record["processing_s"] for record in records) / span,
        "gate_fallback_jobs": sum(bool(record.get("used_gate_fallback")) for record in records),
        "overlap_pairs": len(overlapped), "visible_before_next_gpu_finished": len(visible_first),
        "overlap_examples": visible_first[:3],
        "peak_allocated_gib": max(m["peak_allocated_gib"] for record in records for m in record["memory"]),
        "pinned_peak_gib": max((record.get("host") or {}).get("pinned_peak_gib", 0) for record in records),
        "max_rss_gib": max(((s["health"].get("process") or {}).get("rss_gib", 0) for s in samples), default=None),
        "max_load_1m": max(((s["health"].get("process") or {}).get("load_1m", 0) for s in samples), default=None),
        "max_publishing_jobs": max((len(s["health"].get("publishing_jobs") or []) for s in samples), default=None),
        "release": next((s["health"].get("release") for s in samples if s["health"].get("release")), None),
    })
    return summary


def summarize(output):
    rows = [json.loads(line) for line in (output / "jobs.jsonl").read_text().splitlines()]
    samples = [json.loads(line) for line in (output / "health.jsonl").read_text().splitlines()]
    measured = [row for row in rows if row["phase"] == "measured"]
    workers = sorted({row["worker"] for row in measured})
    per_worker = {worker: summarize_worker([row for row in measured if row["worker"] == worker],
                                           [s for s in samples if s["worker"] == worker and "error" not in s["health"]])
                  for worker in workers}
    completed = sum(item["completed"] for item in per_worker.values())
    rates = [item["videos_per_s"] for item in per_worker.values() if item.get("videos_per_s")]
    return {"workers": per_worker, "completed": completed,
            "failed": sum(sum(item["failures"].values()) for item in per_worker.values()),
            "videos_per_s_total": sum(rates), "end_to_end_s": percentiles(
                row["downloaded"] - row["submitted"] for row in measured if row.get("valid"))}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    token = args.token_file.read_text().strip()
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    workers = [worker.rstrip("/") for worker in args.workers.split(",")]
    for worker in workers:  # warmed, ready, and idle before measuring
        health = json.loads(call(f"{worker}/healthz", timeout=10))
        assert health["status"] == "ready" and health.get("drained", True), f"{worker} is not ready and idle"
    stop = threading.Event()
    sampler = threading.Thread(target=sample_health, args=(workers, args.output / "health.jsonl", stop), daemon=True)
    sampler.start()
    with (args.output / "jobs.jsonl").open("a") as sink:
        # Each phase is one continuous stream, so repeating the manifest adds no idle gap between passes.
        for phase, requests in (("warm", manifest * args.warm_passes), ("measured", manifest * args.measured_passes)):
            threads = [threading.Thread(target=worker_pass, args=(worker, token, requests, args.inflight, phase, sink))
                       for worker in workers]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
    stop.set()
    sampler.join()
    summary = {"manifest": str(args.manifest), "manifest_requests": len(manifest), "inflight": args.inflight,
               "warm_passes": args.warm_passes, "measured_passes": args.measured_passes, **summarize(args.output)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({key: summary[key] for key in ("completed", "failed", "videos_per_s_total", "end_to_end_s")}))


def compare(args):
    sync, background = (json.loads((path / "summary.json").read_text()) for path in (args.sync, args.background))

    def p(summary, key, stat):
        return (summary.get(key) or {}).get(stat)
    result = {
        "throughput_ratio": background["videos_per_s_total"] / sync["videos_per_s_total"],
        "end_to_end_p50_s": [p(sync, "end_to_end_s", "p50"), p(background, "end_to_end_s", "p50")],
        "end_to_end_p95_s": [p(sync, "end_to_end_s", "p95"), p(background, "end_to_end_s", "p95")],
        "failed": [sync["failed"], background["failed"]],
        "gate_fallback_jobs": [sum(w.get("gate_fallback_jobs", 0) for w in s["workers"].values())
                               for s in (sync, background)],
        "per_worker_videos_per_s": {worker: [sync["workers"].get(worker, {}).get("videos_per_s"), item.get("videos_per_s")]
                                    for worker, item in background["workers"].items()},
    }
    print(json.dumps(result, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--workers", required=True, help="Comma-separated worker origins (excluded from the gateway)")
    runner.add_argument("--manifest", type=Path, required=True, help="JSONL request bodies, either request form")
    runner.add_argument("--token-file", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True, help="New directory")
    runner.add_argument("--inflight", type=int, default=2, help="Requests in flight per worker")
    runner.add_argument("--warm-passes", type=int, default=1, help="Unmeasured passes that compile every length")
    runner.add_argument("--measured-passes", type=int, default=1,
                        help="Measured passes over the manifest, run as one stream (64+ jobs per worker)")
    comparer = commands.add_parser("compare")
    comparer.add_argument("sync", type=Path)
    comparer.add_argument("background", type=Path)
    args = parser.parse_args()
    run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    main()
