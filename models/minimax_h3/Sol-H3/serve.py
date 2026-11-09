#!/usr/bin/env python3
"""One resident, authenticated T2V and first-frame I2V worker per torchrun process group.

Requests use the original {prompt, duration, seed} form or the SGLang MiniMax-H3 form (see h3_request.py).

Each worker owns its bounded request queue and output directory. All CUDA work
and every collective run on the main thread; rank zero's HTTP thread only
validates and queues work. With --async-publication, rank zero's publisher
thread finishes the previous video's MP4 (CPU only, one at a time) while the
next video generates (output_publication.py).
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import queue
import secrets
import socket
import threading
import time
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from h3_request import (LEGACY_NUM_FRAMES, MAX_BODY_BYTES, MAX_NUM_FRAMES, VALID_NUM_FRAMES, GenerateRequest,
                        RequestError, capabilities, inspect_image, output_profile, parse_request,
                        resolve_canvas_size, same_request)
from output_publication import (OutputPublisher, PublicationTask, encoder_timings, finish_output, freeze,
                                partial_path, storage_error_message)
from service_storage import atomic_bytes, atomic_json, check_space, quiet_print, source_digest, storage_probe

__all__ = ["GenerateRequest", "WorkerState", "create_app", "caller_job_id", "atomic_json"]
LEGACY_CANVAS = resolve_canvas_size(16, 9)
TERMINAL = frozenset({"succeeded", "failed"})
INFERENCE_FAILURE = {"status": "failed", "error": "inference worker failed", "error_code": "inference_failed"}
ATTENTION_COUNTERS = ("sparse_calls", "dense_calls", "gate_fallback_calls", "gate_failures")
# After a fatal error, an in-flight background publication gets this long to finish before its job is failed;
# the other ranks delay their exit a little longer so torchrun does not stop rank zero first.
PUBLISHER_SHUTDOWN_SECONDS = 10
PEER_EXIT_DELAY_SECONDS = PUBLISHER_SHUTDOWN_SECONDS + 3


def legacy_output(duration):
    return output_profile(LEGACY_NUM_FRAMES[duration], LEGACY_CANVAS)


def read_request_body(payload):
    """Parse a decoded JSON body, mapping contract violations to HTTP errors."""
    try:
        return parse_request(payload)
    except RequestError as error:
        raise HTTPException(error.status_code, error.detail) from None


def caller_job_id(value):
    """An Idempotency-Key is used directly as the job ID, so it must be a canonical UUID."""
    if value is None:
        return None
    with suppress(ValueError):
        if str(UUID(value)) == value:
            return value
    raise HTTPException(400, "Idempotency-Key must be a lowercase canonical UUID")


class OutputStorageError(OSError):
    """Rank-zero output I/O failed after all ranks finished inference."""


def publish_media(media, path, expected, encoder_threads, inspect_media, *, rank, broadcast):
    """Tell every rank about output-storage failure before any GPU barrier.

    Encoder exceptions can wrap an OSError. Only known filesystem errors are
    recoverable here; CUDA, numerical, and malformed-media errors remain fatal.
    ``expected`` is the job's output profile: width, height, and duration_seconds.
    """
    outcome = [None]
    profile = None
    if rank == 0:
        temporary = partial_path(path)
        try:
            media.save(temporary, encoder_threads=encoder_threads)
            profile = inspect_media(temporary, expected["duration_seconds"], expected["width"], expected["height"])
            with temporary.open("rb") as source:
                os.fsync(source.fileno())
                size = os.fstat(source.fileno()).st_size
            temporary.replace(path)
            outcome[0] = {"bytes": size}
        except Exception as error:
            message = storage_error_message(error)
            if message is None:
                raise
            outcome[0] = {"error": message}
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
    broadcast(outcome)
    if "error" in outcome[0]:
        raise OutputStorageError(outcome[0]["error"])
    return {"media": profile, "bytes": outcome[0]["bytes"]}


def process_usage():
    """This process's resident memory and CPU time, and the node's load, for health and benchmarks."""
    usage = {"cpu_s": round(sum(os.times()[:2]), 1)}
    with suppress(OSError, ValueError, IndexError):
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        usage["rss_gib"] = round(pages * os.sysconf("SC_PAGE_SIZE") / 2**30, 3)
    with suppress(OSError):
        usage["load_1m"] = round(os.getloadavg()[0], 2)
    return usage


class WorkerState:
    """Rank zero's jobs: the admission queue, the job on the GPU (``active``), the job whose MP4 is
    published in the background (``publishing``, async mode only), and the durable records."""

    def __init__(self, output_dir, *, capacity=8, min_free_gib=20, metadata=None, max_num_frames=MAX_NUM_FRAMES):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending = queue.Queue(maxsize=capacity)
        self.lock = threading.RLock()
        self.phase = "loading"
        self.active = None  # the job on the GPU: generating, or staging its video
        self.gpu_stage = None
        self.publishing = {}  # job ID -> staging time, while the publisher finishes its MP4
        self.fatal_error = None  # a background publication failed fatally; the worker must stop
        self.completed = 0
        self.min_free_bytes = min_free_gib * 2**30
        self.max_num_frames = max_num_frames
        self.metadata = {**(metadata or {}), "api": capabilities(max_num_frames)}
        self.records = {}  # Accepted requests remain readable during an I/O outage.
        self.dirty = {}
        self.held = None
        self.storage_error = None
        self.storage = {}
        # No writer exists yet: remove interrupted MP4s and record/keyframe temporaries of a previous process.
        for pattern in (".*.partial.mp4", ".*.tmp"):
            for path in self.root.glob(pattern):
                with suppress(OSError):
                    path.unlink()
        # A killed distributed process cannot safely resume an in-flight job.
        # Preserve completed outputs, explicitly fail interrupted requests.
        for path in self.root.glob("*.json"):
            record = json.loads(path.read_text())
            if record.get("status") in {"queued", "running"}:
                record.update(status="failed", error="worker restarted", error_code="worker_restarted",
                              finished_at=time.time())
                self.persist(record)

    def persist(self, record):
        identifier = record["id"]
        try:
            atomic_json(self.record_path(identifier), record)
        except OSError as error:
            self.records[identifier] = self.dirty[identifier] = record
            self.storage_error = str(error)
            return False
        self.dirty.pop(identifier, None)
        if record["status"] in {"succeeded", "failed"}:
            self.records.pop(identifier, None)
        return True

    def recover_storage(self):
        """Pause inference/admission until accepted state can be persisted again."""
        with self.lock:
            for record in list(self.dirty.values()):
                if not self.persist(record):
                    return False
            try:
                self.storage = check_space(self.root, self.min_free_bytes)
                if self.storage_error:
                    storage_probe(self.root)
            except OSError as error:
                self.storage_error = str(error)
                return False
            self.storage_error = None
            return True

    def record_path(self, job_id):
        try:
            if str(UUID(job_id)) != job_id:
                raise ValueError
        except (ValueError, AttributeError):
            raise HTTPException(404, "Unknown job") from None
        return self.root / f"{job_id}.json"

    def get(self, job_id):
        with self.lock:
            path = self.record_path(job_id)
            if job_id in self.records:
                record = dict(self.records[job_id])
                if job_id in self.dirty:
                    record["persistence_pending"] = True
                    if record["status"] in TERMINAL:
                        record["status"] = "running"
                elif record["status"] == "running":
                    # Additive and never persisted: the status stays "running" until the MP4 is published.
                    stage = (self.gpu_stage if job_id == self.active else
                             "publishing" if job_id in self.publishing else None)
                    if stage:
                        record["stage"] = stage
                return record
            try:
                return json.loads(path.read_text())
            except FileNotFoundError:
                raise HTTPException(404, "Unknown job") from None
            except OSError as error:
                raise HTTPException(503, "Job storage is temporarily unavailable") from error

    def update(self, job_id, **values):
        with self.lock:
            record = dict(self.records[job_id]) if job_id in self.records else self.get(job_id)
            record.update(values)
            self.records[job_id] = record
            self.persist(record)
            return record

    def keyframe_path(self, job_id, index, kind):
        return self.root / f"{job_id}.keyframe{index}.{kind}"

    def submit(self, request, job_id=None):
        """Admit a validated request (a ParsedRequest, or a legacy GenerateRequest)."""
        parsed = request if hasattr(request, "identity") else read_request_body(request.model_dump())
        # Decode keyframes outside the lock: it is CPU work, and /healthz shares the lock.
        sizes = []
        for index, data in enumerate(parsed.images):
            try:
                info = inspect_image(data)
            except ValueError as error:
                raise HTTPException(422, [{"loc": ["body", "conditions", index, "uri"], "msg": str(error),
                                           "type": "value_error"}]) from None
            sizes.append((info["width"], info["height"]))
        try:
            output = parsed.resolve(sizes)
        except RequestError as error:
            raise HTTPException(error.status_code, error.detail) from None
        if output["num_frames"] > self.max_num_frames:
            raise HTTPException(422, f"This worker accepts at most {self.max_num_frames} frames")
        identity = parsed.identity
        with self.lock:
            if job_id is not None:
                # A repeated ID returns the original job in any state, so a lost
                # response can be retried without generating a second video.
                try:
                    record = self.get(job_id)
                except HTTPException as error:
                    if error.status_code != 404:
                        raise
                else:
                    if not same_request(record, identity):
                        raise HTTPException(409, "Job ID was already used with different inputs")
                    return record
            if self.phase != "ready":
                raise HTTPException(503, f"Worker is {self.phase}")
            if self.pending.full():
                raise HTTPException(429, "Worker queue is full", headers={"Retry-After": "10"})
            if self.dirty or self.storage_error:
                raise HTTPException(507, "Output storage is paused; retry after recovery")
            job_id = job_id or str(uuid4())
            record = {
                "id": job_id, "status": "queued", "created_at": time.time(),
                **identity, "task": parsed.task, "output": output,
                "status_url": f"/v1/jobs/{job_id}",
                "video_url": f"/v1/jobs/{job_id}/video",
            }
            if sizes:
                record["condition_sizes"] = [{"width": width, "height": height} for width, height in sizes]
            written = []
            try:
                self.storage = check_space(self.root, self.min_free_bytes,
                                           reserve_bytes=sum(len(data) for data in parsed.images))
                # The keyframe must exist before the record that refers to it.
                for index, (data, condition) in enumerate(zip(parsed.images, identity.get("conditions", ()))):
                    path = self.keyframe_path(job_id, index, condition["format"])
                    atomic_bytes(path, data)
                    written.append(path)
                atomic_json(self.record_path(job_id), record)
            except OSError as error:
                for path in written:
                    with suppress(OSError):
                        path.unlink(missing_ok=True)
                self.storage_error = str(error)
                raise HTTPException(507, "Output storage unavailable; request was not accepted") from error
            self.records[job_id] = record
            self.pending.put_nowait(record)
            return record

    def next_command(self):
        with self.lock:
            if self.fatal_error:
                return {"fatal": self.fatal_error}  # every rank stops at this boundary
            if not self.recover_storage():
                return {"idle": True}
            if self.held is None:
                try:
                    self.held = self.pending.get_nowait()
                except queue.Empty:
                    return {"idle": True}
                self.update(self.held["id"], status="running", started_at=time.time())
                if self.dirty:
                    return {"idle": True}
            command, self.held = self.held, None
            self.active, self.gpu_stage = command["id"], "generating"
            return command

    def health(self):
        usage = process_usage()
        with self.lock:
            publishing = list(self.publishing)
            # active_job stays set until a background publication has settled, for drain tooling and
            # the gateway's load estimate; gpu_active_job is only the job on the GPU.
            busy = self.active or next(iter(publishing), None)
            queued = self.pending.qsize() + int(self.held is not None)
            return {
                **self.metadata, "status": ("storage_paused" if self.phase == "ready" and
                                            (self.storage_error or self.dirty) else self.phase),
                "active_job": busy, "gpu_active_job": self.active, "publishing_jobs": publishing,
                "idempotency_key": True,
                "queued": queued, "queue_capacity": self.pending.maxsize,
                "completed_since_start": self.completed,
                # Safe to stop this worker: nothing queued, generating, publishing, or awaiting persistence.
                "drained": busy is None and not queued and not self.dirty,
                "fatal_error": self.fatal_error,
                "storage": {**self.storage, "error": self.storage_error, "pending_writes": len(self.dirty)},
                "process": usage,
            }

    def _settle(self, job_id, **values):
        """Apply a terminal transition once, with the lock held. A late or repeated one, such as a
        publication finishing after shutdown failed its job, changes nothing."""
        record = self.records.get(job_id)
        if record is None or record["status"] in TERMINAL:
            return False
        try:
            self.update(job_id, **{"finished_at": time.time(), **values})
        except BaseException:
            # An unwritable result (not an OSError, which leaves a dirty record) must not block the failure.
            self.records[job_id] = record
            raise
        return True

    def set_stage(self, job_id, stage):
        with self.lock:
            if self.active == job_id:
                self.gpu_stage = stage

    def storage_available(self):
        """Checked before staging a video: true unless storage is paused and still cannot recover."""
        with self.lock:
            return not (self.storage_error or self.dirty) or self.recover_storage()

    def finish(self, job_id, result):
        """Synchronous mode: record a published success once and free the GPU."""
        with self.lock:
            if self._settle(job_id, status="succeeded", **result):
                self.completed += 1
            if self.active == job_id:
                self.active = self.gpu_stage = None

    def fail_output(self, job_id, message):
        """The GPU job's output could not be written or staged (every rank agreed): pause storage, fail it."""
        with self.lock:
            self.storage_error = message
            self._settle(job_id, status="failed", error="output storage failure", error_code="storage_failure",
                         storage_error=message)
            if self.active == job_id:
                self.active = self.gpu_stage = None

    def begin_publication(self, job_id):
        """Async mode: the video is staged, so the GPU is free for the next job."""
        with self.lock:
            if self.active == job_id:
                self.active = self.gpu_stage = None
            self.publishing[job_id] = time.time()

    def publication_succeeded(self, job_id, result):
        """Publisher thread: the MP4 is final. Clears only this job's publication entry."""
        with self.lock:
            if job_id not in self.publishing:
                return False
            settled = self._settle(job_id, status="succeeded", **result)
            del self.publishing[job_id]
            if settled:
                self.completed += 1
            return settled

    def publication_failed(self, job_id, *, storage=None, fatal=None):
        """Publisher thread: pause storage (as for a synchronous output failure), or stop admission and
        make every rank stop at its next boundary (fatal: encoder, media, and other unexpected errors)."""
        with self.lock:
            if storage is not None:
                self.storage_error = storage
                failure = {"status": "failed", "error": "output storage failure", "error_code": "storage_failure",
                           "storage_error": storage}
            else:
                self.fatal_error = fatal or "background publication failed"
                self.phase = "failed"
                failure = INFERENCE_FAILURE
            if self.publishing.pop(job_id, None) is None:
                return False
            return self._settle(job_id, **failure)

    def fail_pending(self, *, keep_publishing=False):
        """Fail every unfinished job: held, on the GPU, queued, and publishing unless ``keep_publishing``.
        Settled jobs keep their result."""
        with self.lock:
            self.phase = "failed"
            unfinished = [self.held["id"]] if self.held else []
            unfinished += [self.active] if self.active else []
            self.held = self.active = self.gpu_stage = None
            if not keep_publishing:
                unfinished += list(self.publishing)
                self.publishing.clear()
            while True:
                try:
                    unfinished.append(self.pending.get_nowait()["id"])
                except queue.Empty:
                    break
            for job_id in unfinished:
                self._settle(job_id, **INFERENCE_FAILURE)


def publish_staged(task, state, inspect_media):
    """Publisher thread: finish a staged MP4 and settle its job. No collectives or CUDA work here."""
    try:
        output = finish_output(task, inspect_media)
        finished = time.perf_counter()
        state.publication_succeeded(task.job_id, {
            **task.result, **output, "processing_s": finished - task.started,
            "publication_tail_s": finished - task.staged, "published_at": time.time(),
            "encoder": encoder_timings(task.encode)})
    except BaseException as error:  # every outcome must reach the main loop
        message = storage_error_message(error)
        if message is None:
            quiet_print(f"PUBLICATION_FATAL job={task.job_id} error={error!r}")
            state.publication_failed(task.job_id, fatal=f"{type(error).__name__}: {error}")
        else:
            quiet_print(f"PUBLICATION_STORAGE_FAILURE job={task.job_id} error={message}")
            state.publication_failed(task.job_id, storage=message)


def shut_down(state, publisher, timeout=PUBLISHER_SHUTDOWN_SECONDS):
    """Rank zero after a fatal error: close admission and fail the held, GPU, and queued jobs at once (a
    SIGTERM from torchrun can end this process at any moment), then give an in-flight publication a
    bounded time and fail it if it is still running. A publication that finishes later changes nothing."""
    state.fail_pending(keep_publishing=True)
    if publisher is not None and not publisher.wait_idle(timeout):
        quiet_print(f"PUBLISHER_SHUTDOWN_TIMEOUT publication still running after {timeout} s; failing its job")
    state.fail_pending()


def stop_after_error(rank, state, publisher, *, async_publication, serving):
    """main()'s error path on every rank, before the exception ends the process. torchrun stops all ranks
    as soon as one exits, and SIGTERM ends a rank at once, so a rank that fails beside rank zero (the same
    error on every rank) stays up while rank zero gives its in-flight publication a bounded time. Once rank
    zero exits, torchrun stops this rank, so the wait lasts only as long as rank zero needs."""
    if state is not None:
        shut_down(state, publisher)
    elif rank != 0 and async_publication and serving:
        time.sleep(PEER_EXIT_DELAY_SECONDS)


def job_from_record(record, output_dir):
    """What a GPU job needs from a queued record; legacy records carry only a duration."""
    output = record.get("output") or legacy_output(record["duration"])
    image = None
    if record.get("conditions"):
        condition = record["conditions"][0]
        image = str(Path(output_dir) / f"{record['id']}.keyframe0.{condition['format']}")
    return {"prompt": record["prompt"], "seed": record["seed"], "output": output, "image": image}


class JobExecutor:
    """One job's GPU work on every rank's main thread, with the same collective order on each rank.

    ``runtime`` supplies the rank-local begin(), generate(job), forwards(), metrics(), and host_memory().
    broadcast/gather/barrier are the CPU control group's collectives. Rank zero also gets the
    WorkerState and, in async mode, the OutputPublisher. Tests drive this without a model.
    """

    def __init__(self, *, rank, runtime, broadcast, gather, barrier, encoder_threads, inspect_media,
                 state=None, publisher=None):
        self.rank, self.runtime = rank, runtime
        self.broadcast, self.gather, self.barrier = broadcast, gather, barrier
        self.encoder_threads, self.inspect_media = encoder_threads, inspect_media
        self.state, self.publisher = state, publisher

    def _generate(self, job):
        self.runtime.begin()
        started = time.perf_counter()
        timing = {"generation_started_at": time.time()}
        media = self.runtime.generate(job)
        forwards = self.runtime.forwards()
        if forwards != 4:
            raise RuntimeError(f"Expected four transformer forwards, got {forwards}")
        timing["generation_finished_at"] = time.time()
        timing["pipeline_s"] = media.elapsed_s if media is not None else None
        return media, started, timing

    @staticmethod
    def _summary(ranks):
        return {"transformer_forwards": 4, "memory": [item["memory"] for item in ranks],
                "attention": [item["attention"] for item in ranks],
                "used_gate_fallback": any(item["attention"] and item["attention"]["gate_fallback_calls"]
                                          for item in ranks)}

    def execute(self, job, path, job_id=None):
        """Generate and publish before returning (warmups and synchronous mode); rank zero gets the result."""
        media, started, timing = self._generate(job)
        if self.rank == 0 and job_id is not None and self.state is not None:
            self.state.set_stage(job_id, "publishing")
        output = publish_media(media, path, job["output"], self.encoder_threads, self.inspect_media,
                               rank=self.rank, broadcast=self.broadcast)
        self.barrier()
        elapsed = time.perf_counter() - started
        ranks = self.gather(self.runtime.metrics())
        if self.rank == 0:
            return {**timing, "processing_s": elapsed, **output, **self._summary(ranks),
                    "publication_mode": "sync", "gpu_slot_s": elapsed, "host": self.runtime.host_memory()}
        return None

    def execute_async(self, job_id, job, path):
        """Generate, stage, and hand the video to rank zero's publisher; the next job can start at once.

        Raises OutputStorageError on every rank if paused output storage prevents staging, and
        RuntimeError on every rank after a fatal background publication error.
        """
        media, started, timing = self._generate(job)
        # This job's per-rank metrics, copied before a later generation can change them.
        ranks = self.gather(copy.deepcopy(self.runtime.metrics()))
        outcome = [None]
        if self.rank == 0:
            outcome[0] = self._stage(job_id, media, path, job["output"], started,
                                     {**timing, **self._summary(ranks), "publication_mode": "async"})
        media = None  # a video that was not staged releases its GPU memory here
        self.broadcast(outcome)
        if "fatal" in outcome[0]:
            raise RuntimeError(f"A background publication failed: {outcome[0]['fatal']}")
        if "error" in outcome[0]:
            raise OutputStorageError(outcome[0]["error"])
        self.barrier()

    def _stage(self, job_id, media, path, expected, started, result):
        """Rank zero: wait for the publication slot, stage the video, and start its publication."""
        state, publisher = self.state, self.publisher
        state.set_stage(job_id, "staging")
        waited = publisher.reserve()
        handed_off = False
        try:
            if state.fatal_error:
                return {"fatal": state.fatal_error}
            if not state.storage_available():
                # An earlier publication paused storage: fail this job as a synchronous output failure would.
                return {"error": state.storage_error or "output storage is paused"}
            temporary = partial_path(path)
            try:
                encode = media.save_async(temporary, encoder_threads=self.encoder_threads)
            except Exception as error:
                message = storage_error_message(error)
                if message is None:
                    raise  # CUDA and other staging errors stay fatal, as in publish_media
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
                return {"error": message}
            staged = time.perf_counter()
            task = PublicationTask(
                job_id=job_id, encode=encode, path=path, expected=freeze(expected), started=started, staged=staged,
                result=freeze({**result, "gpu_slot_s": staged - started, "publication_slot_wait_s": waited,
                               "staged_at": time.time(), "host": self.runtime.host_memory()}))
            state.begin_publication(job_id)
            publisher.start(task)
            handed_off = True
            return {"staged": True}
        finally:
            if not handed_off:
                publisher.release()


def serve_jobs(executor, *, rank, broadcast, state, output_dir, async_publication, alive=lambda: True,
               idle_seconds=1):
    """The worker loop on every rank: rank zero picks the next command and every rank runs it."""
    output_dir = Path(output_dir)
    while True:
        command = None
        if rank == 0:
            if not alive():
                raise RuntimeError("HTTP server stopped")
            command = state.next_command()
        commands = [command]
        broadcast(commands)
        command = commands[0]
        if command.get("fatal"):
            raise RuntimeError(f"A background publication failed: {command['fatal']}")
        if command.get("idle"):
            time.sleep(idle_seconds)
            continue
        path = output_dir / f"{command['id']}.mp4"
        job = job_from_record(command, output_dir)
        try:
            if async_publication:
                executor.execute_async(command["id"], job, path)
                continue
            result = executor.execute(job, path, command["id"])
        except OutputStorageError as error:
            if rank == 0:
                state.fail_output(command["id"], str(error))
            continue
        if rank == 0:
            state.finish(command["id"], result)


def create_app(state, token):
    if len(token) < 32:
        raise ValueError("Use a random bearer token of at least 32 characters")
    app = FastAPI(title="Sol-H3 worker", docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
            raise HTTPException(401, "Bearer token required", headers={"WWW-Authenticate": "Bearer"})

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if request.method == "POST":
            try:
                length = int(request.headers.get("content-length", "-1"))
            except ValueError:
                return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
            if length < 0:
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
            if length > MAX_BODY_BYTES:
                return JSONResponse({"detail": "Request body too large"}, status_code=413)
        return await call_next(request)

    @app.get("/healthz")
    def health():
        status = state.health()
        return JSONResponse(status, status_code=200 if status["status"] == "ready" else 503)

    @app.post("/v1/videos", status_code=202, dependencies=[Depends(authorize)])
    async def submit(request: Request, idempotency_key: str | None = Header(default=None)):
        job_id = caller_job_id(idempotency_key)
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(422, [{"loc": ["body"], "msg": "the request body must be valid JSON",
                                       "type": "json_invalid"}]) from None
        # Decoding a base64 keyframe is CPU work; keep it off the event loop.
        parsed = await run_in_threadpool(read_request_body, payload)
        return await run_in_threadpool(state.submit, parsed, job_id)

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def status(job_id: str):
        return state.get(job_id)

    @app.get("/v1/jobs/{job_id}/video", dependencies=[Depends(authorize)])
    def video(job_id: str):
        record = state.get(job_id)
        if record["status"] != "succeeded":
            raise HTTPException(409, "Video is not ready")
        path = state.root / f"{job_id}.mp4"
        if not path.is_file():
            raise HTTPException(410, "Video was removed")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--attention-backend", choices=("sol", "sol_bsa", "dense"), default="sol")
    parser.add_argument("--gate-failure", choices=("error", "dense"), default="dense",
                        help="Use dense SDPA for shapes rejected by the numerical gate (service default)")
    parser.add_argument("--queue-capacity", type=int, default=8)
    parser.add_argument("--encoder-threads", type=int, default=8)
    parser.add_argument("--min-free-gib", type=int, default=20)
    parser.add_argument("--warmup-prompt-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--expect-gpu", default="H200")
    parser.add_argument("--max-num-frames", type=int, default=MAX_NUM_FRAMES,
                        help="Largest output frame count this worker admits (a 17n+5 value, 124-345)")
    parser.add_argument("--async-publication", action="store_true",
                        help="Finish and publish each MP4 on a background thread while the next video generates, "
                             "one publication at a time (DEPLOYMENT.md); default: publish before the next job")
    args = parser.parse_args(argv)
    if min(args.queue_capacity, args.encoder_threads, args.min_free_gib) < 1:
        parser.error("queue capacity, encoder threads, and free disk threshold must be positive")
    if args.max_num_frames not in VALID_NUM_FRAMES:
        parser.error(f"--max-num-frames must be one of {list(VALID_NUM_FRAMES)}")
    if not 1 <= args.port <= 65535:
        parser.error("invalid TCP port")
    # The launchers set both from H3_ASYNC_PUBLICATION; a mismatch means the supervisor and watchdog drifted.
    setting = os.environ.get("H3_ASYNC_PUBLICATION", "")
    if setting not in {"", "0", "1"}:
        parser.error("H3_ASYNC_PUBLICATION must be 0 or 1")
    if setting and (setting == "1") != args.async_publication:
        parser.error(f"--async-publication {'is missing' if setting == '1' else 'is set'} "
                     f"but H3_ASYNC_PUBLICATION={setting}")
    return args


def main():
    args = parse_args()
    os.umask(0o077)
    token = args.token_file.read_text().strip()
    if len(token) < 32:
        raise ValueError("Bearer token must contain at least 32 characters")
    if args.ready_file and int(os.environ.get("RANK", "0")) == 0:
        args.ready_file.unlink(missing_ok=True)

    import torch
    import torch.distributed as dist
    import uvicorn
    from benchmark import inspect_media, read_prompt
    from h3_runtime import MiniMaxH3Inference

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {1, 2, 4, 8} or int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))) != world_size:
        raise ValueError("Each worker must use 1, 2, 4, or 8 GPUs on a single node")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.expect_gpu.lower() not in torch.cuda.get_device_name(device).lower():
        raise ValueError(f"Expected {args.expect_gpu}, found {torch.cuda.get_device_name(device)}")
    # Optional cap on PyTorch's pool for the serving phase, as a fraction of GPU memory (see DEPLOYMENT.md). Loading
    # needs more memory than serving, so the cap is applied after warmup; it keeps room outside the pool for kernels
    # loaded for new prompt lengths.
    serving_fraction = float(os.environ.get("H3_SERVING_MEMORY_FRACTION") or 0)
    if not 0 <= serving_fraction < 1:
        raise ValueError("H3_SERVING_MEMORY_FRACTION must be at least 0 and below 1")
    # Split the text encoder and DiT blocks across the worker's GPUs; needed on 80 GB H100s (DEPLOYMENT.md).
    weight_sharding = os.environ.get("H3_WEIGHT_SHARDING", "0")
    if weight_sharding not in {"0", "1"}:
        raise ValueError("H3_WEIGHT_SHARDING must be '0' or '1'")
    weight_sharding = weight_sharding == "1"
    if args.attention_backend != "dense":
        os.environ["SOL_ATTN_STRICT"] = "1"
        os.environ["SOL_ATTN_CORRECTNESS_GATE"] = "1"
        os.environ["SOL_ATTN_GATE_FAILURE"] = args.gate_failure
        os.environ["SOL_ATTN_GATE_EACH_REQUEST"] = "1"
        if os.environ.get("H3_SOL_BACKEND", args.attention_backend) != args.attention_backend:
            raise ValueError("H3_SOL_BACKEND conflicts with selected backend")
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
        # Control commands stay on CPU, while model collectives use NCCL/NVLink.
        control = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    else:
        control = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    state = server = server_thread = publisher = None
    forwards = [0]
    serving = False
    try:
        if rank == 0:
            code_dir = Path(__file__).resolve().parent
            state = WorkerState(args.output_dir, capacity=args.queue_capacity, min_free_gib=args.min_free_gib,
                                metadata={"hostname": socket.gethostname(), "gpus": world_size,
                                          "gpu": torch.cuda.get_device_name(device), "attention": args.attention_backend,
                                          "gate_failure_policy": args.gate_failure, "gate_each_request": True,
                                          "duration_seconds": [5, 10], "width": 1344, "height": 768,
                                          "fps": 24, "transformer_forwards": 4,
                                          "encoder_threads": args.encoder_threads,
                                          "weight_sharding": weight_sharding,
                                          "publication_mode": "async" if args.async_publication else "sync",
                                          "publication_capacity": OutputPublisher.capacity if args.async_publication else 0,
                                          # The running code: its directory and a digest of its Python sources.
                                          "release": {"code_dir": str(code_dir), "source_sha256": source_digest(code_dir)},
                                          "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
                                max_num_frames=args.max_num_frames)
            server = uvicorn.Server(uvicorn.Config(create_app(state, token), host=args.host, port=args.port,
                                                  access_log=False, log_level="info", limit_concurrency=64))
            server_thread = threading.Thread(target=server.run, name="http", daemon=True)
            server_thread.start()
            deadline = time.monotonic() + 30
            while not server.started:
                if not server_thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("HTTP server failed to bind")
                time.sleep(0.1)

        engine = MiniMaxH3Inference(args.model, args.adapter, attention_backend=args.attention_backend,
                                    task="t2v", compute_quant="none", lora_mode="merged",
                                    weight_sharding=weight_sharding)

        def count_forward(*unused):
            forwards[0] += 1

        engine.transformer.register_forward_pre_hook(count_forward)

        def broadcast(values):
            if world_size > 1:
                dist.broadcast_object_list(values, src=0, group=control)

        def gather(value):
            if world_size == 1:
                return [value]
            values = [None] * world_size
            dist.all_gather_object(values, value, group=control)
            return values

        def barrier():
            if world_size > 1:
                dist.barrier()

        attention = engine.sparse_attention
        before = {}

        def begin():
            torch.cuda.reset_peak_memory_stats(device)
            forwards[0] = 0
            before.clear()
            before.update({key: getattr(attention, key) for key in ATTENTION_COUNTERS} if attention else {})

        def generate(job):
            expected = job["output"]
            return engine.generate(job["prompt"], num_frames=expected["num_frames"], height=expected["height"],
                                   width=expected["width"], seed=job["seed"], image=job["image"])

        def metrics():
            memory = {"rank": rank, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
            attention_stats = ({key: getattr(attention, key) - before[key] for key in ATTENTION_COUNTERS}
                               | {"gate": attention.gate_stats, "rejected_shapes": len(attention._rejected_shapes)}
                               if attention else None)
            return {"memory": memory, "attention": {"rank": rank, **attention_stats} if attention_stats else None}

        def host_memory():
            """Resident memory, CPU time, and PyTorch's pinned host pool, which holds rank zero's staged frames."""
            usage = process_usage()
            with suppress(Exception):  # the pinned-pool statistics are newer than some PyTorch releases
                stats = torch.cuda.memory.host_memory_stats()
                for name, key in (("pinned_gib", "allocated_bytes.current"), ("pinned_peak_gib", "allocated_bytes.peak")):
                    if key in stats:
                        usage[name] = round(stats[key] / 2**30, 3)
            return usage

        if rank == 0 and args.async_publication:
            publisher = OutputPublisher(lambda task: publish_staged(task, state, inspect_media))
        executor = JobExecutor(rank=rank, runtime=SimpleNamespace(begin=begin, generate=generate,
                                                                  forwards=lambda: forwards[0], metrics=metrics,
                                                                  host_memory=host_memory),
                               broadcast=broadcast, gather=gather, barrier=barrier,
                               encoder_threads=args.encoder_threads, inspect_media=inspect_media,
                               state=state, publisher=publisher)

        prompt = read_prompt(args.warmup_prompt_file)
        warmups = []
        if rank == 0:
            state.phase = "warming"
        for duration in (5, 10):
            for repeat in range(2):
                path = args.output_dir / f"warmup-{duration}s-{repeat}.mp4"
                while True:
                    available = [state.recover_storage() if rank == 0 else None]
                    broadcast(available)
                    if not available[0]:
                        time.sleep(5)
                        continue
                    try:
                        # Warmups always publish synchronously, so readiness follows complete outputs.
                        result = executor.execute({"prompt": prompt, "seed": 42, "output": legacy_output(duration),
                                                   "image": None}, path)
                        break
                    except OutputStorageError as error:
                        if rank == 0:
                            state.storage_error = str(error)
                        time.sleep(5)
                if rank == 0:
                    warmups.append({"duration": duration, "repeat": repeat, **result})
        if serving_fraction:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()  # return the load-time high-water mark
            torch.cuda.set_per_process_memory_fraction(serving_fraction, device)
            free, total = torch.cuda.mem_get_info(device)
            quiet_print(f"SERVING_MEMORY_CAP rank={rank} cap_gib={serving_fraction * total / 2**30:.1f} "
                        f"reserved_gib={torch.cuda.memory_reserved(device) / 2**30:.1f} free_gib={free / 2**30:.1f}")
        if rank == 0:
            state.phase = "ready"
            ready = {**state.health(), "endpoint": f"http://{args.host}:{args.port}", "warmups": warmups}
            if args.ready_file:
                try:
                    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(args.ready_file, ready)
                except OSError as error:
                    quiet_print(f"READY_FILE_WRITE_FAILED {error!r}; use /healthz")
            quiet_print("H3_WORKER_READY " + json.dumps(ready))

        serving = True
        serve_jobs(executor, rank=rank, broadcast=broadcast, state=state, output_dir=args.output_dir,
                   async_publication=args.async_publication,
                   alive=server_thread.is_alive if server_thread is not None else (lambda: True))
    except BaseException:
        stop_after_error(rank, state, publisher, async_publication=args.async_publication, serving=serving)
        raise
    finally:
        if rank == 0 and args.ready_file:
            with suppress(OSError):
                args.ready_file.unlink(missing_ok=True)
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=5)
        # Let torchrun kill peers on a rank failure; never continue after an OOM
        # or failed collective with potentially inconsistent distributed state.


if __name__ == "__main__":
    main()
