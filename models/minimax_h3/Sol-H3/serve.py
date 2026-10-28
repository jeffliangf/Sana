#!/usr/bin/env python3
"""One resident, authenticated T2V worker per torchrun process group.

Each worker owns its bounded request queue and output directory. All CUDA work
runs on the main thread; rank zero's HTTP thread only validates and queues work.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import timedelta
import errno
import json
import os
from pathlib import Path
import queue
import secrets
import socket
import threading
import time
from typing import Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from service_storage import atomic_json, check_space, quiet_print, storage_probe


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1, max_length=8000)
    duration: Literal[5, 10] = 5
    seed: int = Field(default=42, ge=0, le=2**63 - 1)

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value):
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return value.strip()


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


def publish_media(media, path, duration, encoder_threads, inspect_media, *, rank, broadcast):
    """Tell every rank about output-storage failure before any GPU barrier.

    Encoder exceptions can wrap an OSError. Only known filesystem errors are
    recoverable here; CUDA, numerical, and malformed-media errors remain fatal.
    """
    outcome = [None]
    profile = None
    if rank == 0:
        temporary = path.with_name(f".{path.stem}.partial.mp4")
        try:
            media.save(temporary, encoder_threads=encoder_threads)
            profile = inspect_media(temporary, duration)
            with temporary.open("rb") as source:
                os.fsync(source.fileno())
                size = os.fstat(source.fileno()).st_size
            temporary.replace(path)
            outcome[0] = {"bytes": size}
        except Exception as error:
            cause = error
            while cause.__cause__ is not None:
                cause = cause.__cause__
            if not isinstance(cause, OSError) or cause.errno not in {
                errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS, errno.EACCES,
                errno.ENOENT, errno.EMFILE, errno.ENFILE,
            }:
                raise
            outcome[0] = {"error": str(cause)}
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
    broadcast(outcome)
    if "error" in outcome[0]:
        raise OutputStorageError(outcome[0]["error"])
    return {"media": profile, "bytes": outcome[0]["bytes"]}


class WorkerState:
    def __init__(self, output_dir, *, capacity=8, min_free_gib=20, metadata=None):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending = queue.Queue(maxsize=capacity)
        self.lock = threading.RLock()
        self.phase = "loading"
        self.active = None
        self.completed = 0
        self.min_free_bytes = min_free_gib * 2**30
        self.metadata = metadata or {}
        self.records = {}  # Accepted requests remain readable during an I/O outage.
        self.dirty = {}
        self.held = None
        self.storage_error = None
        self.storage = {}
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
                    if record["status"] in {"succeeded", "failed"}:
                        record["status"] = "running"
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

    def submit(self, request, job_id=None):
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
                    if any(record.get(name) != value for name, value in request.model_dump().items()):
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
                **request.model_dump(),
                "status_url": f"/v1/jobs/{job_id}",
                "video_url": f"/v1/jobs/{job_id}/video",
            }
            try:
                self.storage = check_space(self.root, self.min_free_bytes)
                atomic_json(self.record_path(job_id), record)
            except OSError as error:
                self.storage_error = str(error)
                raise HTTPException(507, "Output storage unavailable; request was not accepted") from error
            self.records[job_id] = record
            self.pending.put_nowait(record)
            return record

    def next_command(self):
        with self.lock:
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
            self.active = command["id"]
            return command

    def health(self):
        with self.lock:
            return {
                **self.metadata, "status": ("storage_paused" if self.phase == "ready" and
                                            (self.storage_error or self.dirty) else self.phase),
                "active_job": self.active, "idempotency_key": True,
                "queued": self.pending.qsize() + int(self.held is not None), "queue_capacity": self.pending.maxsize,
                "completed_since_start": self.completed,
                "storage": {**self.storage, "error": self.storage_error, "pending_writes": len(self.dirty)},
            }

    def fail_pending(self):
        with self.lock:
            self.phase = "failed"
            failure = {"status": "failed", "error": "inference worker failed", "error_code": "inference_failed"}
            if self.held:
                self.update(self.held["id"], **failure, finished_at=time.time())
                self.held = None
            if self.active:
                self.update(self.active, **failure, finished_at=time.time())
                self.active = None
            while True:
                try:
                    record = self.pending.get_nowait()
                except queue.Empty:
                    break
                self.update(record["id"], **failure, finished_at=time.time())


def create_app(state, token):
    if len(token) < 32:
        raise ValueError("Use a random bearer token of at least 32 characters")
    app = FastAPI(title="Sol-H3 T2V worker", docs_url=None, redoc_url=None, openapi_url=None)

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
            if length > 32768:
                return JSONResponse({"detail": "Request body too large"}, status_code=413)
        return await call_next(request)

    @app.get("/healthz")
    def health():
        status = state.health()
        return JSONResponse(status, status_code=200 if status["status"] == "ready" else 503)

    @app.post("/v1/videos", status_code=202, dependencies=[Depends(authorize)])
    def submit(request: GenerateRequest, idempotency_key: str | None = Header(default=None)):
        return state.submit(request, caller_job_id(idempotency_key))

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
    args = parser.parse_args(argv)
    if min(args.queue_capacity, args.encoder_threads, args.min_free_gib) < 1:
        parser.error("queue capacity, encoder threads, and free disk threshold must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("invalid TCP port")
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
    state = server = server_thread = None
    forwards = [0]
    try:
        if rank == 0:
            state = WorkerState(args.output_dir, capacity=args.queue_capacity, min_free_gib=args.min_free_gib,
                                metadata={"hostname": socket.gethostname(), "gpus": world_size,
                                          "gpu": torch.cuda.get_device_name(device), "attention": args.attention_backend,
                                          "gate_failure_policy": args.gate_failure, "gate_each_request": True,
                                          "duration_seconds": [5, 10], "width": 1344, "height": 768,
                                          "fps": 24, "transformer_forwards": 4,
                                          "encoder_threads": args.encoder_threads,
                                          "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
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
                                    task="t2v", compute_quant="none", lora_mode="merged")

        def count_forward(*unused):
            forwards[0] += 1

        engine.transformer.register_forward_pre_hook(count_forward)

        def broadcast(values):
            if world_size > 1:
                dist.broadcast_object_list(values, src=0, group=control)

        def execute(prompt, duration, seed, path):
            torch.cuda.reset_peak_memory_stats(device)
            forwards[0] = 0
            attention = engine.sparse_attention
            counters = ("sparse_calls", "dense_calls", "gate_fallback_calls", "gate_failures")
            before = {key: getattr(attention, key) for key in counters} if attention else {}
            started = time.perf_counter()
            media = engine.generate(prompt, duration=duration, seed=seed)
            if forwards[0] != 4:
                raise RuntimeError(f"Expected four transformer forwards, got {forwards[0]}")
            pipeline_s = media.elapsed_s if media is not None else None
            output = publish_media(media, path, duration, args.encoder_threads, inspect_media,
                                   rank=rank, broadcast=broadcast)
            if world_size > 1:
                dist.barrier()
            elapsed = time.perf_counter() - started
            memory = {"rank": rank, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
            attention_stats = ({key: getattr(attention, key) - before[key] for key in counters}
                               | {"gate": attention.gate_stats, "rejected_shapes": len(attention._rejected_shapes)}
                               if attention else None)
            rank_result = {"memory": memory, "attention": {"rank": rank, **attention_stats} if attention_stats else None}
            ranks = [rank_result]
            if world_size > 1:
                ranks = [None] * world_size
                dist.all_gather_object(ranks, rank_result, group=control)
            if rank == 0:
                return {"pipeline_s": pipeline_s, "processing_s": elapsed, **output,
                        "transformer_forwards": forwards[0], "memory": [item["memory"] for item in ranks],
                        "attention": [item["attention"] for item in ranks],
                        "used_gate_fallback": any(item["attention"] and item["attention"]["gate_fallback_calls"] for item in ranks)}

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
                        result = execute(prompt, duration, 42, path)
                        break
                    except OutputStorageError as error:
                        if rank == 0:
                            state.storage_error = str(error)
                        time.sleep(5)
                if rank == 0:
                    warmups.append({"duration": duration, "repeat": repeat, **result})
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

        while True:
            command = None
            if rank == 0:
                if not server_thread.is_alive():
                    raise RuntimeError("HTTP server stopped")
                command = state.next_command()
            if world_size > 1:
                commands = [command]
                dist.broadcast_object_list(commands, src=0, group=control)
                command = commands[0]
            if command.get("idle"):
                time.sleep(1)
                continue
            path = args.output_dir / f"{command['id']}.mp4"
            try:
                result = execute(command["prompt"], command["duration"], command["seed"], path)
            except OutputStorageError as error:
                if rank == 0:
                    with state.lock:
                        state.storage_error = str(error)
                        state.update(command["id"], status="failed", error="output storage failure",
                                     error_code="storage_failure", finished_at=time.time(), storage_error=str(error))
                        state.active = None
                continue
            if rank == 0:
                with state.lock:
                    state.update(command["id"], status="succeeded", finished_at=time.time(), **result)
                    state.active = None
                    state.completed += 1
    except BaseException:
        if state is not None:
            state.fail_pending()
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
