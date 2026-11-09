"""Overlap of GPU generation with MP4 publication (serve.py --async-publication), without a model.

FakeEncode follows FastVideoEncodeJob's contract: save_async returns once the frames are staged, a writer
thread fills the hidden partial MP4, and wait() joins it. The frames live in a one-buffer "pinned pool", so
starting a second encoder while the first still owns them fails the test, as reusing the real zero-copy
frames would corrupt a video. Generation waits on per-job gates. Threads stand in for torch.distributed
ranks; they do not establish distributed CUDA correctness, which the live validation in DEPLOYMENT.md does.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import output_publication
import serve

TOKEN = "test-only-" + "x" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SOURCE = Path(__file__).resolve().parents[1]


def wait_for(predicate, timeout=10, message="condition"):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {message}"
        time.sleep(0.005)


class PinnedPool:
    """One frame buffer, which the caching host allocator would hand to the next video once freed."""

    def __init__(self):
        self.lock = threading.Lock()
        self.owner = None
        self.buffer = bytearray()
        self.live = self.peak = 0

    def acquire(self, owner, frames):
        with self.lock:
            assert self.owner is None, f"{owner}'s staging reused frames that {self.owner}'s encoder still owns"
            self.owner, self.buffer[:] = owner, frames
            self.live += 1
            self.peak = max(self.peak, self.live)

    def release(self, owner):
        with self.lock:
            assert self.owner == owner
            self.owner = None
            self.live -= 1


class FakeEncode:
    def __init__(self, harness, owner, frames, path):
        self.harness, self.owner, self.path = harness, owner, Path(path)
        harness.pool.acquire(owner, frames)
        harness.writers[self.path.name] = self
        self.error = self.elapsed_s = None
        self.staging_s, self.timings = 0.001, {"video_flush_s": 0.0, "close_s": 0.0}
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        started = time.perf_counter()
        try:
            with self.path.open("wb") as output:  # the container exists from the start, like av.open
                assert self.harness.encoder_gate(self.owner).wait(20), f"{self.owner}'s encoder was never released"
                failure = self.harness.encoder_failures.get(self.owner)
                if failure is not None:
                    raise failure
                output.write(bytes(self.harness.pool.buffer))  # zero-copy: reads the frames only now
        except BaseException as error:
            self.error = error
        self.elapsed_s = time.perf_counter() - started

    def wait(self):
        self.thread.join()
        self.harness.pool.release(self.owner)
        self.harness.events.append(("encoded", self.owner))
        if self.error is not None:
            self.path.unlink(missing_ok=True)
            raise RuntimeError(f"fast video encoder failed: {self.error}") from self.error


class FakeMedia:
    def __init__(self, harness, owner):
        self.harness, self.owner = harness, owner
        self.video, self.elapsed_s = f"frames-of-{owner}".encode(), 1.0

    def save_async(self, path, *, encoder_threads):
        failure = self.harness.staging_failures.get(self.owner)
        if failure is not None:
            raise failure
        self.harness.events.append(("stage", self.owner))
        encode = FakeEncode(self.harness, self.owner, self.video, path)
        self.video = None
        return encode

    def save(self, path, *, encoder_threads):  # publish_media's synchronous path
        self.save_async(path, encoder_threads=encoder_threads).wait()


class FakeRuntime:
    """Rank-local model stand-in: generation waits on its gate; metrics identify the job."""

    def __init__(self, harness, rank):
        self.harness, self.rank, self.current, self.count = harness, rank, None, 0

    def begin(self):
        self.count = 0

    def generate(self, job):
        name = self.current = job["prompt"]
        self.harness.events.append(("generate-start", name, self.rank))
        gate = self.harness.gpu_gates.get(name)
        assert gate is None or gate.wait(20), f"generation of {name} was never released"
        self.count = 4
        self.harness.generated.append((name, self.rank))
        self.harness.events.append(("generate-end", name, self.rank))
        return FakeMedia(self.harness, name) if self.rank == 0 else None

    def forwards(self):
        return self.count

    def metrics(self):
        mark = float(ord(self.current[0]))
        return {"memory": {"rank": self.rank, "peak_allocated_gib": mark},
                "attention": {"rank": self.rank, "sparse_calls": mark, "gate_fallback_calls": 0,
                              "gate": {"job": self.current}}}

    def host_memory(self):
        return {"rss_gib": 1.0, "pinned_gib": 0.5}


def inspect(path, duration, width, height):
    assert Path(path).name.endswith(".partial.mp4") and Path(path).stat().st_size > 0
    return {"width": width, "height": height, "fps": 24.0, "video_codec": "h264", "audio_codec": "aac",
            "sample_rate": 32000, "channels": 2, "duration_s": duration}


class Harness:
    """Rank zero's WorkerState, HTTP API, publisher and executor around the fake model."""

    def __init__(self, root, capacity=8, inspect_media=inspect):
        self.state = serve.WorkerState(root, capacity=capacity, min_free_gib=0,
                                       metadata={"publication_mode": "async", "publication_capacity": 1})
        self.state.phase = "ready"
        self.client = TestClient(serve.create_app(self.state, TOKEN))
        self.inspect_media = inspect_media
        self.pool = PinnedPool()
        self.gpu_gates, self.encoder_gates, self.encoder_failures, self.staging_failures = {}, {}, {}, {}
        self.events, self.generated, self.writers = [], [], {}
        self.publisher = output_publication.OutputPublisher(
            lambda task: serve.publish_staged(task, self.state, self.inspect_media))
        self.loop = None
        self.loop_errors = []

    def encoder_gate(self, owner):
        return self.encoder_gates.get(owner, _OPEN)

    def executor(self, rank=0, collectives=None):
        broadcast, gather, barrier = collectives or (lambda values: None, lambda value: [value], lambda: None)
        return serve.JobExecutor(rank=rank, runtime=FakeRuntime(self, rank), broadcast=broadcast, gather=gather,
                                 barrier=barrier, encoder_threads=8, inspect_media=self.inspect_media,
                                 state=self.state if rank == 0 else None,
                                 publisher=self.publisher if rank == 0 else None)

    def start(self, async_publication=True):
        """Run the real worker loop on a thread standing in for the main thread of a one-rank worker."""
        executor = self.executor()

        def run():
            try:
                serve.serve_jobs(executor, rank=0, broadcast=lambda values: None, state=self.state,
                                 output_dir=self.state.root, async_publication=async_publication, idle_seconds=0.005)
            except BaseException as error:
                self.loop_errors.append(error)

        self.loop = threading.Thread(target=run, daemon=True)
        self.loop.start()

    def stop(self):
        with self.state.lock:
            self.state.fatal_error = self.state.fatal_error or "test finished"
        self.loop.join(10)
        assert not self.loop.is_alive()

    def submit(self, prompt, key=None, seed=7):
        headers = {**AUTH, **({"Idempotency-Key": key} if key else {})}
        response = self.client.post("/v1/videos", json={"prompt": prompt, "duration": 5, "seed": seed}, headers=headers)
        assert response.status_code == 202, response.text
        return response.json()["id"]

    def job(self, job_id):
        return self.client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()

    def status(self, job_id):
        return self.state.get(job_id)["status"]

    def health(self):
        return self.client.get("/healthz").json()

    def index(self, event):
        return self.events.index(event)


_OPEN = threading.Event()
_OPEN.set()


def closed_gates(*names):
    return {name: threading.Event() for name in names}


def test_publication_becomes_downloadable_while_the_next_video_generates(tmp_path):
    h = Harness(tmp_path)
    h.gpu_gates = closed_gates("B")
    h.encoder_gates = closed_gates("A")
    h.start()
    a, b = h.submit("A"), h.submit("B")
    wait_for(lambda: ("generate-start", "B", 0) in h.events, message="B to enter generation")
    assert h.job(a)["stage"] == "publishing" and h.job(b)["stage"] == "generating"
    health = h.health()
    assert (health["gpu_active_job"], health["publishing_jobs"], health["active_job"]) == (b, [a], b)
    assert health["publication_mode"] == "async" and health["drained"] is False

    h.encoder_gates["A"].set()  # A finishes while B is still inside generation
    wait_for(lambda: h.status(a) == "succeeded", message="A to publish")
    assert ("generate-end", "B", 0) not in h.events
    done = h.job(a)
    assert done["status"] == "succeeded" and "stage" not in done and done["bytes"] == len(b"frames-of-A")
    video = h.client.get(f"/v1/jobs/{a}/video", headers=AUTH)
    assert video.status_code == 200 and video.content == b"frames-of-A"
    # Completing A cleared only A's publication; B keeps the GPU.
    health = h.health()
    assert (health["gpu_active_job"], health["publishing_jobs"], health["active_job"]) == (b, [], b)

    h.gpu_gates["B"].set()
    wait_for(lambda: h.status(b) == "succeeded", message="B to publish")
    h.stop()
    first, second = h.job(a), h.job(b)
    # The acceptance trace: B's GPU work starts before A's publication ends, and A is downloadable first.
    assert second["generation_started_at"] < first["published_at"] < second["generation_finished_at"]
    # Each record carries its own job's metrics, never its successor's.
    for record, name in ((first, "A"), (second, "B")):
        assert record["memory"] == [{"rank": 0, "peak_allocated_gib": float(ord(name))}]
        assert record["attention"][0]["gate"] == {"job": name} and record["transformer_forwards"] == 4
        assert record["publication_mode"] == "async" and record["host"]["pinned_gib"] == 0.5
        assert record["generation_started_at"] < record["generation_finished_at"] <= record["staged_at"]
        assert record["staged_at"] <= record["published_at"] <= record["finished_at"]
        assert 0 <= record["gpu_slot_s"] <= record["processing_s"]
        assert record["publication_tail_s"] <= record["processing_s"]
        assert record["encoder"]["staging_s"] == 0.001 and "close_s" in record["encoder"]
    assert h.state.completed == 2 and h.pool.peak == 1 and h.health()["drained"]


def test_slow_publication_holds_back_the_next_encoder_but_not_health(tmp_path):
    h = Harness(tmp_path)
    h.encoder_gates = closed_gates("A")
    h.start()
    a, b = h.submit("A"), h.submit("B")
    wait_for(lambda: h.state.active == b and h.state.gpu_stage == "staging",
             message="B to wait for the publication slot")
    assert [event for event in h.events if event[0] == "stage"] == [("stage", "A")]
    with ThreadPoolExecutor(1) as other_thread:  # a blocked lock would hang this request
        health = other_thread.submit(h.client.get, "/healthz").result(timeout=5).json()
    assert (health["gpu_active_job"], health["publishing_jobs"], health["active_job"]) == (b, [a], b)
    assert health["publication_capacity"] == 1 and h.job(b)["stage"] == "staging"
    time.sleep(0.2)
    h.encoder_gates["A"].set()
    wait_for(lambda: h.status(b) == "succeeded", message="B to publish")
    h.stop()
    # B's encoder started only after A's had finished with its frames; each video holds its own frames.
    assert h.index(("encoded", "A")) < h.index(("stage", "B")) and h.pool.peak == 1
    for job_id, name in ((a, "A"), (b, "B")):
        assert (tmp_path / f"{job_id}.mp4").read_bytes() == f"frames-of-{name}".encode()
    assert h.job(b)["publication_slot_wait_s"] >= 0.15 > h.job(a)["publication_slot_wait_s"]


def test_synchronous_mode_publishes_before_the_next_generation(tmp_path):
    h = Harness(tmp_path)
    h.start(async_publication=False)
    a, b = h.submit("A"), h.submit("B")
    wait_for(lambda: h.status(b) == "succeeded", message="both jobs")
    h.stop()
    assert h.index(("encoded", "A")) < h.index(("generate-start", "B", 0))
    record = h.job(a)
    assert record["publication_mode"] == "sync" and record["gpu_slot_s"] == record["processing_s"]
    assert "staged_at" not in record and h.state.completed == 2 and not h.state.publishing


def test_repeated_and_conflicting_keys_in_every_state_never_generate_twice(tmp_path):
    h = Harness(tmp_path)
    h.gpu_gates = closed_gates("A")
    h.encoder_gates = closed_gates("A")
    key = str(uuid4())

    def repeat():
        body = {"prompt": "A", "duration": 5, "seed": 7}
        same = h.client.post("/v1/videos", json=body, headers={**AUTH, "Idempotency-Key": key})
        other = h.client.post("/v1/videos", json={**body, "seed": 8}, headers={**AUTH, "Idempotency-Key": key})
        assert same.status_code == 202 and same.json()["id"] == key and other.status_code == 409
        return same.json()

    assert h.submit("A", key) == key
    assert repeat()["status"] == "queued"
    h.start()
    wait_for(lambda: h.state.active == key, message="A on the GPU")
    record = repeat()
    assert (record["status"], record["stage"]) == ("running", "generating")
    h.gpu_gates["A"].set()
    wait_for(lambda: key in h.state.publishing, message="A publishing")
    record = repeat()
    assert (record["status"], record["stage"]) == ("running", "publishing")
    h.encoder_gates["A"].set()
    wait_for(lambda: h.status(key) == "succeeded", message="A to publish")
    assert repeat()["status"] == "succeeded"
    h.stop()
    assert h.generated == [("A", 0)] and h.state.pending.empty() and h.state.completed == 1


def test_waiting_queue_capacity_is_separate_from_the_publication(tmp_path):
    h = Harness(tmp_path, capacity=2)
    h.gpu_gates = closed_gates("B")
    h.encoder_gates = closed_gates("A")
    h.start()
    a, b = h.submit("A"), h.submit("B")
    wait_for(lambda: ("generate-start", "B", 0) in h.events, message="B on the GPU")
    h.submit("C")
    h.submit("D")
    full = h.client.post("/v1/videos", json={"prompt": "E"}, headers=AUTH)
    assert full.status_code == 429
    health = h.health()
    assert (health["queued"], health["queue_capacity"], health["publishing_jobs"]) == (2, 2, [a])
    assert health["gpu_active_job"] == b
    h.encoder_gates["A"].set()
    h.gpu_gates["B"].set()
    wait_for(lambda: h.state.completed == 4, message="every job")
    h.stop()


def failing_probe(code):
    """serve.storage_probe replacement that fails while ``broken`` is set."""
    broken = threading.Event()
    broken.set()
    original = serve.storage_probe

    def probe(root):
        if broken.is_set():
            raise OSError(code, "injected storage probe failure")
        return original(root)
    return broken, probe


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS])
@pytest.mark.parametrize("point", ["encoder", "fsync", "rename"])
def test_output_storage_failure_pauses_admission_and_fails_the_paused_successor(tmp_path, monkeypatch, code, point):
    h = Harness(tmp_path)
    h.gpu_gates = closed_gates("B")
    h.encoder_gates = closed_gates("A")
    injected = OSError(code, "injected output failure")
    if point == "encoder":
        h.encoder_failures["A"] = injected
    elif point == "fsync":
        real_fsync = os.fsync

        def fsync(fd):
            if os.readlink(f"/proc/self/fd/{fd}").endswith(".partial.mp4"):
                raise injected
            return real_fsync(fd)
        monkeypatch.setattr(output_publication.os, "fsync", fsync)
    else:
        real_replace = Path.replace

        def replace(self, target):
            if self.name.endswith(".partial.mp4"):
                raise injected
            return real_replace(self, target)
        monkeypatch.setattr(Path, "replace", replace)
    real_unlink = Path.unlink

    def unlink(self, missing_ok=False):
        writer = h.writers.get(self.name)
        assert writer is None or not writer.thread.is_alive(), "a partial MP4 was removed before its writer ended"
        return real_unlink(self, missing_ok=missing_ok)
    monkeypatch.setattr(Path, "unlink", unlink)
    broken, probe = failing_probe(code)
    monkeypatch.setattr(serve, "storage_probe", probe)

    h.start()
    a, b, c = h.submit("A"), h.submit("B"), h.submit("C")
    wait_for(lambda: ("generate-start", "B", 0) in h.events, message="B on the GPU")
    h.encoder_gates["A"].set()
    wait_for(lambda: h.status(a) == "failed", message="A's storage failure")
    failed = h.job(a)
    assert failed["error_code"] == "storage_failure" and "injected output failure" in failed["storage_error"]
    assert not list(tmp_path.glob(".*.partial.mp4")) and not (tmp_path / f"{a}.mp4").exists()
    # Admission pauses at once, while B is still generating.
    assert h.client.post("/v1/videos", json={"prompt": "D"}, headers=AUTH).status_code == 507
    health = h.health()
    assert health["status"] == "storage_paused" and health["gpu_active_job"] == b and health["publishing_jobs"] == []
    # B reaches the staging boundary while storage is still failing: it fails coherently and is never staged.
    h.gpu_gates["B"].set()
    wait_for(lambda: h.status(b) == "failed", message="B to fail at the staging boundary")
    assert h.job(b)["error_code"] == "storage_failure" and ("stage", "B") not in h.events
    assert h.status(c) == "queued" and h.state.active is None  # C waits for recovery; no rerender of B
    monkeypatch.undo()  # the filesystem recovers
    wait_for(lambda: h.status(c) == "succeeded", message="C after recovery")
    h.stop()
    assert h.health()["storage"]["error"] is None and h.state.completed == 1
    assert [name for name, _ in h.generated] == ["A", "B", "C"]


def test_unwritable_terminal_record_stays_running_until_storage_recovers(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    original = serve.atomic_json
    broken = threading.Event()
    broken.set()

    def atomic_json(path, value):
        if broken.is_set() and value.get("status") == "succeeded":
            raise OSError(errno.EIO, "injected record failure")
        return original(path, value)
    monkeypatch.setattr(serve, "atomic_json", atomic_json)
    probe_broken, probe = failing_probe(errno.EIO)
    monkeypatch.setattr(serve, "storage_probe", probe)
    h.start()
    a = h.submit("A")
    wait_for(lambda: a in h.state.dirty, message="A's terminal record to fail")
    pending = h.job(a)
    assert pending["status"] == "running" and pending["persistence_pending"]
    assert h.client.get(f"/v1/jobs/{a}/video", headers=AUTH).status_code == 409
    health = h.health()
    assert health["status"] == "storage_paused" and health["storage"]["pending_writes"] == 1
    assert health["active_job"] is None and health["publishing_jobs"] == [] and health["drained"] is False
    assert h.client.post("/v1/videos", json={"prompt": "B"}, headers=AUTH).status_code == 507
    broken.clear()
    probe_broken.clear()
    wait_for(lambda: not h.state.dirty, message="the record to persist")
    h.stop()
    assert h.job(a)["status"] == "succeeded" and h.state.completed == 1 and h.health()["drained"]
    assert h.client.get(f"/v1/jobs/{a}/video", headers=AUTH).content == b"frames-of-A"
    assert serve.WorkerState(tmp_path, min_free_gib=0).get(a)["status"] == "succeeded"


@pytest.mark.parametrize("failure", ["encoder", "media"])
def test_unexpected_publication_error_stops_the_worker_at_the_next_boundary(tmp_path, failure):
    def malformed(*args):
        raise RuntimeError("Unexpected video format")
    h = Harness(tmp_path, inspect_media=malformed if failure == "media" else inspect)
    if failure == "encoder":
        h.encoder_failures["A"] = ValueError("x264 rejected a frame")  # an encoder error, not a disk error
    h.gpu_gates = closed_gates("B")
    h.encoder_gates = closed_gates("A")
    h.start()
    a, b, c = h.submit("A"), h.submit("B"), h.submit("C")
    wait_for(lambda: ("generate-start", "B", 0) in h.events, message="B on the GPU")
    h.encoder_gates["A"].set()
    wait_for(lambda: h.status(a) == "failed", message="A's fatal failure")
    assert h.job(a)["error_code"] == "inference_failed" and h.state.fatal_error
    assert h.health()["status"] == "failed"
    assert h.client.post("/v1/videos", json={"prompt": "D"}, headers=AUTH).status_code == 503
    h.gpu_gates["B"].set()  # B's generation finishes; the loop stops at its staging boundary
    h.loop.join(10)
    assert isinstance(h.loop_errors[0], RuntimeError) and ("stage", "B") not in h.events
    assert not h.publisher.busy
    serve.shut_down(h.state, h.publisher, timeout=1)  # main()'s handler
    assert [h.job(job)["error_code"] for job in (a, b, c)] == ["inference_failed"] * 3
    # A late or repeated completion changes nothing and is not counted.
    assert not h.state.publication_succeeded(a, {"bytes": 1}) and not h.state.publication_failed(a, storage="late")
    assert h.status(a) == "failed" and h.state.completed == 0


def test_staging_errors_keep_the_fatal_and_storage_policies(tmp_path):
    h = Harness(tmp_path)
    h.staging_failures["A"] = RuntimeError("could not stage")
    h.staging_failures["A"].__cause__ = OSError(errno.EROFS, "read-only output directory")
    h.staging_failures["B"] = RuntimeError("CUDA error: an illegal memory access was encountered")
    a, b = h.submit("A"), h.submit("B")  # both queued first: A's failure pauses admission at once
    h.start()
    wait_for(lambda: h.loop_errors, message="B's fatal staging error")
    assert h.job(a)["error_code"] == "storage_failure" and "read-only" in h.job(a)["storage_error"]
    assert "CUDA error" in str(h.loop_errors[0]) and h.publisher.wait_idle(0)  # the slot was returned
    serve.shut_down(h.state, h.publisher, timeout=1)
    assert h.job(b)["error_code"] == "inference_failed" and h.state.completed == 0


def run_job(h, executor=None):
    """One command on the test thread, as the main loop would run it."""
    command = h.state.next_command()
    path = h.state.root / f"{command['id']}.mp4"
    (executor or h.executor()).execute_async(command["id"], serve.job_from_record(command, h.state.root), path)
    return command["id"]


def test_shutdown_fails_the_other_jobs_at_once_and_lets_the_publication_finish(tmp_path):
    h = Harness(tmp_path)
    h.encoder_gates = closed_gates("A")
    h.submit("A")
    a = run_job(h)  # A is publishing; its encoder is held
    b, c = h.submit("B"), h.submit("C")
    assert h.state.next_command()["id"] == b  # B on the GPU, C queued
    shutdown = threading.Thread(target=serve.shut_down, args=(h.state, h.publisher), kwargs={"timeout": 10})
    shutdown.start()
    # A SIGTERM can end rank zero during the wait, so B and C are failed before it.
    wait_for(lambda: h.status(b) == h.status(c) == "failed", message="the unpublished jobs to fail")
    assert h.status(a) == "running" and shutdown.is_alive() and h.health()["status"] == "failed"
    h.encoder_gates["A"].set()
    shutdown.join(10)
    assert h.status(a) == "succeeded" and h.state.completed == 1
    assert [h.job(job)["error_code"] for job in (b, c)] == ["inference_failed"] * 2


def test_shutdown_is_bounded_and_a_late_publication_cannot_resurrect_its_job(tmp_path):
    h = Harness(tmp_path)
    h.encoder_gates = closed_gates("A")
    h.submit("A")
    a = run_job(h)
    started = time.monotonic()
    serve.shut_down(h.state, h.publisher, timeout=0.3)
    assert 0.25 < time.monotonic() - started < 3
    assert h.job(a)["error_code"] == "inference_failed" and h.health()["status"] == "failed"
    h.encoder_gates["A"].set()  # the publication finishes after its job was failed
    assert h.publisher.wait_idle(5)
    assert h.status(a) == "failed" and h.state.completed == 0
    assert h.client.get(f"/v1/jobs/{a}/video", headers=AUTH).status_code == 409


TORCHRUN_RANK = r"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import output_publication, serve
rank, root = int(os.environ["RANK"]), Path(sys.argv[2])
state = publisher = None
if rank == 0:
    state = serve.WorkerState(root, min_free_gib=0)
    state.phase = "ready"
    for prompt in ("published", "generating", "queued"):
        state.submit(serve.GenerateRequest(prompt=prompt))
    command = state.next_command()
    path = root / f"{command['id']}.mp4"

    class SlowEncode:  # this publication needs two more seconds
        staging_s = elapsed_s = 0.0
        timings = {}

        def wait(self):
            time.sleep(2)
            output_publication.partial_path(path).write_bytes(b"video")

    publisher = output_publication.OutputPublisher(lambda task: serve.publish_staged(task, state, lambda *a: {}))
    publisher.reserve()
    state.begin_publication(command["id"])
    publisher.start(output_publication.PublicationTask(
        job_id=command["id"], encode=SlowEncode(), path=path, result={}, started=time.perf_counter(),
        staged=time.perf_counter(), expected={"width": 1, "height": 1, "duration_seconds": 1}))
    state.next_command()  # the second job is on the GPU; the third stays queued
try:
    raise RuntimeError("the same generation error on every rank")
except BaseException:
    serve.stop_after_error(rank, state, publisher, async_publication=sys.argv[3] == "1", serving=True)
    raise
"""


def run_torchrun_ranks(tmp_path, async_publication="1"):
    script = tmp_path / "rank.py"
    script.write_text(TORCHRUN_RANK)
    root = tmp_path / "outputs"
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                             str(script), str(SOURCE), str(root), async_publication],
                            capture_output=True, text=True, timeout=120)
    records = {record["prompt"]: record for record in (json.loads(p.read_text()) for p in root.glob("*.json"))}
    return result, records


def test_an_error_on_every_rank_still_lets_rank_zero_finish_its_publication(tmp_path):
    """torchrun stops every rank once one exits; the other rank must not exit before rank zero's bounded wait."""
    pytest.importorskip("torch.distributed.run")
    started = time.monotonic()
    result, records = run_torchrun_ranks(tmp_path)
    assert result.returncode != 0 and time.monotonic() - started < 60
    assert records["published"]["status"] == "succeeded", result.stderr[-3000:]
    for prompt in ("generating", "queued"):
        assert (records[prompt]["status"], records[prompt]["error_code"]) == ("failed", "inference_failed")
        assert records[prompt]["finished_at"] < records["published"]["published_at"]


def test_drained_only_without_queued_generating_publishing_or_unpersisted_jobs(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    assert h.health()["drained"] is True
    h.encoder_gates = closed_gates("A")
    a = h.submit("A")
    assert h.health()["drained"] is False
    original = serve.atomic_json

    def atomic_json(path, value):
        if value.get("status") == "succeeded":
            raise OSError(errno.ENOSPC, "injected")
        return original(path, value)
    monkeypatch.setattr(serve, "atomic_json", atomic_json)
    run_job(h)
    health = h.health()
    assert (health["active_job"], health["gpu_active_job"], health["drained"]) == (a, None, False)
    h.encoder_gates["A"].set()
    wait_for(lambda: a in h.state.dirty, message="the unpersisted success")
    assert h.health()["active_job"] is None and h.health()["drained"] is False
    monkeypatch.undo()
    assert h.state.recover_storage() and h.health()["drained"] is True


class ControlGroup:
    """Threads standing in for the ranks' Gloo control group; records each rank's collective sequence."""

    def __init__(self, size):
        self.size = size
        self.sync = threading.Barrier(size, timeout=10)
        self.slots = [None] * size
        self.shared = None
        self.trace = [[] for _ in range(size)]

    def collectives(self, rank):
        def broadcast(values):
            self.trace[rank].append("broadcast")
            if rank == 0:
                self.shared = copy.deepcopy(values[0])
            self.sync.wait()
            values[0] = copy.deepcopy(self.shared)
            self.sync.wait()

        def gather(value):
            self.trace[rank].append("gather")
            self.slots[rank] = copy.deepcopy(value)
            self.sync.wait()
            values = copy.deepcopy(self.slots)
            self.sync.wait()
            return values

        def barrier():
            self.trace[rank].append("barrier")
            self.sync.wait()
        return broadcast, gather, barrier


def run_ranks(h, size, command):
    """Run one command's execute_async on every rank; returns each rank's exception or None."""
    group = ControlGroup(size)
    path = h.state.root / f"{command['id']}.mp4"
    job = serve.job_from_record(command, h.state.root)
    outcomes = [None] * size

    def rank_main(rank):
        try:
            h.executor(rank, group.collectives(rank)).execute_async(command["id"], job, path)
        except BaseException as error:
            outcomes[rank] = error
    threads = [threading.Thread(target=rank_main, args=(rank,)) for rank in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert len({tuple(trace) for trace in group.trace}) == 1, group.trace  # identical order on every rank
    return outcomes, group.trace[0]


@pytest.mark.parametrize("size", [2, 4])
def test_every_rank_takes_the_same_path_at_the_staging_boundary(tmp_path, monkeypatch, size):
    h = Harness(tmp_path)
    a, b, c, d = (h.submit(name) for name in "ABCD")
    outcomes, trace = run_ranks(h, size, h.state.next_command())
    assert outcomes == [None] * size and trace == ["gather", "broadcast", "barrier"]
    wait_for(lambda: h.status(a) == "succeeded", message="A to publish")
    assert [item["rank"] for item in h.job(a)["memory"]] == list(range(size))  # every rank's metrics, in order

    # B is generating when an earlier publication pauses storage, which is still failing at B's boundary:
    # every rank raises the same storage error, and B is never staged.
    command = h.state.next_command()
    monkeypatch.setattr(serve, "storage_probe", failing_probe(errno.ENOSPC)[1])
    h.state.storage_error = "[Errno 28] No space left on device"
    outcomes, trace = run_ranks(h, size, command)
    assert all(isinstance(error, serve.OutputStorageError) for error in outcomes) and trace == ["gather", "broadcast"]
    h.state.fail_output(command["id"], str(outcomes[0]))  # rank zero's handler in serve_jobs
    assert h.job(b)["error_code"] == "storage_failure" and h.status(c) == "queued"
    assert ("stage", "B") not in h.events and not h.publisher.busy

    # C is generating when a background publication fails fatally: every rank stops at C's boundary.
    monkeypatch.undo()
    command = h.state.next_command()
    assert command["id"] == c
    h.state.fatal_error = "RuntimeError: x264 failed"
    outcomes, trace = run_ranks(h, size, command)
    assert all(isinstance(error, RuntimeError) and "x264" in str(error) for error in outcomes)
    assert trace == ["gather", "broadcast"] and ("stage", "C") not in h.events and not h.publisher.busy
    assert h.status(d) == "queued"


def test_fatal_command_stops_every_rank_together(tmp_path):
    h = Harness(tmp_path)
    h.state.fatal_error = "RuntimeError: x264 failed"
    group = ControlGroup(2)
    errors = [None, None]

    def rank_main(rank):
        broadcast, _, _ = group.collectives(rank)
        try:
            serve.serve_jobs(h.executor(rank), rank=rank, broadcast=broadcast, state=h.state if rank == 0 else None,
                             output_dir=tmp_path, async_publication=True)
        except RuntimeError as error:
            errors[rank] = str(error)
    threads = [threading.Thread(target=rank_main, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert errors[0] == errors[1] and "x264" in errors[0]


CHILD = r"""
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import serve
state = serve.WorkerState(Path(sys.argv[2]), min_free_gib=0)
state.phase = "ready"
state.submit(serve.GenerateRequest(prompt="interrupted"))
command = state.next_command()
state.begin_publication(command["id"])
with (Path(sys.argv[2]) / f".{command['id']}.partial.mp4").open("wb") as output:  # an encoder mid-write
    output.write(b"partial")
    output.flush()
    print(command["id"], flush=True)
    time.sleep(60)
"""


def completed_job(root):
    state = serve.WorkerState(root, min_free_gib=0)
    state.phase = "ready"
    done = state.submit(serve.GenerateRequest(prompt="done"))["id"]
    state.next_command()
    (root / f"{done}.mp4").write_bytes(b"complete")
    state.finish(done, {"bytes": 8})
    return state, done


@pytest.mark.parametrize("moment", ["encoding", "after-rename", "persisting-record"])
def test_restart_after_a_forced_kill_fails_the_interrupted_job_and_keeps_completed_ones(tmp_path, moment):
    state, done = completed_job(tmp_path)
    if moment == "encoding":  # a real SIGKILL while an encoder writes the partial MP4
        child = subprocess.Popen([sys.executable, "-c", CHILD, str(SOURCE), str(tmp_path)],
                                 stdout=subprocess.PIPE, text=True)
        interrupted = child.stdout.readline().strip()
        child.send_signal(signal.SIGKILL)
        child.wait(10)
        leftover = tmp_path / f".{interrupted}.partial.mp4"
    else:
        interrupted = state.submit(serve.GenerateRequest(prompt="interrupted"))["id"]
        state.next_command()
        state.begin_publication(interrupted)
        if moment == "after-rename":  # the final MP4 exists, but its terminal record was never written
            leftover = tmp_path / f"{interrupted}.mp4"
            leftover.write_bytes(b"unrecorded")
        else:  # killed inside atomic_json: a torn temporary beside the previous version of the record
            leftover = tmp_path / f".{interrupted}.json.k2j3h4.tmp"
            leftover.write_text('{"status": "succ')
    assert leftover.exists()

    restarted = serve.WorkerState(tmp_path, min_free_gib=0)
    client = TestClient(serve.create_app(restarted, TOKEN))
    record = restarted.get(interrupted)
    assert (record["status"], record["error_code"]) == ("failed", "worker_restarted")  # never replayed
    assert client.get(f"/v1/jobs/{interrupted}/video", headers=AUTH).status_code == 409  # never served
    assert leftover.exists() == (moment == "after-rename")  # temporaries are removed, an orphan MP4 is inert
    assert client.get(f"/v1/jobs/{done}/video", headers=AUTH).content == b"complete"
    assert restarted.health()["drained"] and restarted.health()["active_job"] is None


def test_metadata_snapshot_is_immutable_and_publisher_is_bounded(tmp_path):
    live = {"gate": {"passed": True}, "memory": [{"rank": 0}]}
    frozen = output_publication.freeze(live)
    live["gate"]["passed"] = False
    live["memory"].append({"rank": 1})
    assert frozen["gate"] == {"passed": True} and frozen["memory"] == [{"rank": 0}]
    with pytest.raises(TypeError):
        frozen["gate"] = None
    publisher = output_publication.OutputPublisher(lambda task: None)
    with pytest.raises(RuntimeError):
        publisher.start(object())  # a publication needs the slot first
    assert publisher.reserve() < 1 and publisher.busy and not publisher.wait_idle(0.05)
    publisher.release()
    assert publisher.wait_idle(0) and output_publication.OutputPublisher.capacity == 1
    assert serve.job_from_record({"id": "x", "prompt": "p", "seed": 1, "duration": 10}, tmp_path)["output"] \
        == serve.legacy_output(10)


def test_storage_classification_matches_the_synchronous_path():
    def wrapped(cause):
        try:
            try:
                raise cause
            except OSError as error:
                raise RuntimeError("fast video encoder failed") from error
        except RuntimeError as error:
            return error
    for code in (errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS, errno.EACCES, errno.ENOENT):
        assert output_publication.storage_error_message(wrapped(OSError(code, "disk"))) is not None
    assert output_publication.storage_error_message(wrapped(OSError(errno.EINVAL, "bad argument"))) is None
    assert output_publication.storage_error_message(RuntimeError("CUDA error")) is None
    usage = serve.process_usage()
    assert usage["cpu_s"] >= 0 and usage["rss_gib"] > 0 and "load_1m" in usage


def test_source_digest_names_the_running_code(tmp_path):
    from service_storage import source_digest
    (tmp_path / "h3_runtime").mkdir()
    (tmp_path / "serve.py").write_text("a")
    (tmp_path / "h3_runtime/engine.py").write_text("b")
    (tmp_path / "notes.md").write_text("ignored")
    first = source_digest(tmp_path)
    (tmp_path / "notes.md").write_text("still ignored")
    assert source_digest(tmp_path) == first
    (tmp_path / "h3_runtime/engine.py").write_text("c")
    assert source_digest(tmp_path) != first
    assert json.loads(json.dumps({"digest": source_digest(SOURCE)}))["digest"]
