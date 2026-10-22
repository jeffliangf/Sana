#!/usr/bin/env python3
"""One resident, authenticated T2V worker per torchrun process group.

Each worker owns its bounded request queue and output directory. All CUDA work
runs on the main thread; rank zero's HTTP thread only validates and queues work.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import socket
import threading
import time
from typing import Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator


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


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


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
        # A killed distributed process cannot safely resume an in-flight job.
        # Preserve completed outputs, explicitly fail interrupted requests.
        for path in self.root.glob("*.json"):
            record = json.loads(path.read_text())
            if record.get("status") in {"queued", "running"}:
                record.update(status="failed", error="worker restarted", finished_at=time.time())
                atomic_json(path, record)

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
            if not path.is_file():
                raise HTTPException(404, "Unknown job")
            return json.loads(path.read_text())

    def update(self, job_id, **values):
        with self.lock:
            record = self.get(job_id)
            record.update(values)
            atomic_json(self.record_path(job_id), record)
            return record

    def submit(self, request):
        with self.lock:
            if self.phase != "ready":
                raise HTTPException(503, f"Worker is {self.phase}")
            if self.pending.full():
                raise HTTPException(429, "Worker queue is full", headers={"Retry-After": "10"})
            if shutil.disk_usage(self.root).free < self.min_free_bytes:
                raise HTTPException(507, "Output filesystem has insufficient free space")
            job_id = str(uuid4())
            record = {
                "id": job_id, "status": "queued", "created_at": time.time(),
                **request.model_dump(),
                "status_url": f"/v1/jobs/{job_id}",
                "video_url": f"/v1/jobs/{job_id}/video",
            }
            atomic_json(self.record_path(job_id), record)
            self.pending.put_nowait(record)
            return record

    def health(self):
        with self.lock:
            return {
                **self.metadata, "status": self.phase, "active_job": self.active,
                "queued": self.pending.qsize(), "queue_capacity": self.pending.maxsize,
                "completed_since_start": self.completed,
            }

    def fail_pending(self):
        with self.lock:
            self.phase = "failed"
            if self.active:
                self.update(self.active, status="failed", error="inference worker failed", finished_at=time.time())
                self.active = None
            while True:
                try:
                    record = self.pending.get_nowait()
                except queue.Empty:
                    break
                self.update(record["id"], status="failed", error="inference worker failed", finished_at=time.time())


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
    def submit(request: GenerateRequest):
        return state.submit(request)

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

        def execute(prompt, duration, seed, path):
            torch.cuda.reset_peak_memory_stats(device)
            forwards[0] = 0
            started = time.perf_counter()
            media = engine.generate(prompt, duration=duration, seed=seed)
            if forwards[0] != 4:
                raise RuntimeError(f"Expected four transformer forwards, got {forwards[0]}")
            pipeline_s = media.elapsed_s if media is not None else None
            if rank == 0:
                temporary = path.with_name(f".{path.stem}.partial.mp4")
                media.save(temporary, encoder_threads=args.encoder_threads)
                # Validate both video and audio stream metadata before publishing.
                profile = inspect_media(temporary, duration)
                temporary.replace(path)
            if world_size > 1:
                dist.barrier()
            elapsed = time.perf_counter() - started
            memory = {"rank": rank, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
            memories = [memory]
            if world_size > 1:
                memories = [None] * world_size
                dist.all_gather_object(memories, memory, group=control)
            if rank == 0:
                return {"pipeline_s": pipeline_s, "processing_s": elapsed, "media": profile,
                        "transformer_forwards": forwards[0], "memory": memories,
                        "bytes": path.stat().st_size}

        prompt = read_prompt(args.warmup_prompt_file)
        warmups = []
        if rank == 0:
            state.phase = "warming"
        for duration in (5, 10):
            for repeat in range(2):
                path = args.output_dir / f"warmup-{duration}s-{repeat}.mp4"
                result = execute(prompt, duration, 42, path)
                if rank == 0:
                    warmups.append({"duration": duration, "repeat": repeat, **result})
        if rank == 0:
            state.phase = "ready"
            ready = {**state.health(), "endpoint": f"http://{args.host}:{args.port}", "warmups": warmups}
            if args.ready_file:
                args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(args.ready_file, ready)
            print("H3_WORKER_READY " + json.dumps(ready), flush=True)

        while True:
            command = None
            if rank == 0:
                if not server_thread.is_alive():
                    raise RuntimeError("HTTP server stopped")
                try:
                    command = state.pending.get(timeout=20)
                    with state.lock:
                        state.active = command["id"]
                        state.update(command["id"], status="running", started_at=time.time())
                except queue.Empty:
                    command = {"idle": True}
            if world_size > 1:
                commands = [command]
                dist.broadcast_object_list(commands, src=0, group=control)
                command = commands[0]
            if command.get("idle"):
                continue
            path = args.output_dir / f"{command['id']}.mp4"
            result = execute(command["prompt"], command["duration"], command["seed"], path)
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
            args.ready_file.unlink(missing_ok=True)
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=5)
        # Let torchrun kill peers on a rank failure; never continue after an OOM
        # or failed collective with potentially inconsistent distributed state.


if __name__ == "__main__":
    main()
