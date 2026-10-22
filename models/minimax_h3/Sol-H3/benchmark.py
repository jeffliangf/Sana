#!/usr/bin/env python3
"""Benchmark resident four-forward T2V inference, including MP4 completion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--prompt-file", type=Path, default=ROOT.parent / "demo_prompt.json")
    parser.add_argument("--durations", nargs="+", type=int, choices=(5, 10, 15), default=[5, 10])
    parser.add_argument("--attention-backend", choices=("dense", "sol", "sol_bsa"), default="dense")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expect-gpu", help="Require this substring in every GPU name, e.g. H200")
    args = parser.parse_args(argv)
    if args.warmups < 1 or args.repeats < 1:
        parser.error("--warmups and --repeats must be positive")
    if len(set(args.durations)) != len(args.durations):
        parser.error("--durations must not contain duplicates")
    if not args.adapter.is_file():
        parser.error("--adapter must be a local adapter_model.safetensors file")
    return args


def read_prompt(path):
    text = path.read_text(encoding="utf-8")
    prompt = json.loads(text).get("prompt") if path.suffix == ".json" else text
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt file must contain nonempty text or a JSON object with 'prompt'")
    return prompt.strip()


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_media(path, duration):
    import av

    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
            raise RuntimeError(f"Expected one video and one audio stream: {path}")
        video, audio = container.streams.video[0], container.streams.audio[0]
        profile = {
            "width": video.width, "height": video.height,
            "fps": float(video.average_rate), "video_codec": video.codec_context.name,
            "audio_codec": audio.codec_context.name, "sample_rate": audio.sample_rate,
            "channels": audio.codec_context.channels,
            "duration_s": container.duration / av.time_base if container.duration is not None else None,
        }
    if (profile["width"], profile["height"], profile["fps"]) != (1344, 768, 24.0):
        raise RuntimeError(f"Unexpected video format: {profile}")
    if (profile["sample_rate"], profile["channels"]) != (32000, 2):
        raise RuntimeError(f"Unexpected audio format: {profile}")
    if profile["duration_s"] is None or abs(profile["duration_s"] - duration) > 1:
        raise RuntimeError(f"Unexpected duration: {profile}")
    return profile


def summarize(records, world_size):
    summary = {}
    measured = [record for record in records if record["phase"] == "measured"]
    for duration in sorted({record["duration"] for record in measured}):
        group = [record for record in measured if record["duration"] == duration]
        pipeline = [record["pipeline_s"] for record in group]
        completion = [record["request_to_mp4_s"] for record in group]
        summary[str(duration)] = {
            "samples": len(group),
            "pipeline_median_s": statistics.median(pipeline),
            "request_to_mp4_median_s": statistics.median(completion),
            "request_to_mp4_min_s": min(completion),
            "request_to_mp4_max_s": max(completion),
            "serial_worker_clips_per_hour": 3600 * len(group) / sum(completion),
            "gpu_seconds_per_clip": world_size * statistics.mean(completion),
            "peak_allocated_gib_per_gpu": [
                max(record["ranks"][rank]["peak_allocated_bytes"] for record in group) / 2**30
                for rank in range(world_size)
            ],
            "peak_reserved_gib_per_gpu": [
                max(record["ranks"][rank]["peak_reserved_bytes"] for record in group) / 2**30
                for rank in range(world_size)
            ],
        }
    return summary


def main():
    args = parse_args()
    prompt = read_prompt(args.prompt_file)
    import torch
    import torch.distributed as dist

    from h3_runtime import MiniMaxH3Inference
    from h3_runtime.engine import DURATION_FRAMES, INFERENCE_STEPS

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; this benchmark cannot run on a login/CPU node")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (1, 2, 4, 8):
        raise ValueError("Use 1, 2, 4, or 8 GPUs within a single node")
    if int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))) != world_size:
        raise ValueError("This benchmark measures one node; run independent jobs on separate nodes")
    if args.attention_backend != "dense" and world_size == 1:
        raise ValueError("Sparse attention requires at least two GPU processes")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    if args.expect_gpu and args.expect_gpu.lower() not in properties.name.lower():
        raise RuntimeError(f"Expected {args.expect_gpu}, found {properties.name}")
    if args.attention_backend != "dense":
        # Exercise numerical checks during warmup and reject a silent dense fallback.
        os.environ["SOL_ATTN_STRICT"] = "1"
        os.environ["SOL_ATTN_CORRECTNESS_GATE"] = "1"
        if os.environ.get("H3_SOL_BACKEND", args.attention_backend) != args.attention_backend:
            raise ValueError("H3_SOL_BACKEND conflicts with --attention-backend")
    kernel_backend = "pytorch_sdpa" if args.attention_backend == "dense" else "cudnn_bsa"
    if args.attention_backend == "sol":
        from h3_runtime.sparse_attention import _import_kernel

        _import_kernel()
        from sol_attn import get_sol_attn_backend

        kernel_backend = get_sol_attn_backend(device)
        if torch.cuda.get_device_capability(device) == (9, 0) and kernel_backend != "cute_sm90":
            raise RuntimeError(f"Expected the Hopper CuTe kernel, got {kernel_backend}")
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank() if dist.is_initialized() else 0

    def barrier():
        torch.cuda.synchronize(device)
        if dist.is_initialized():
            dist.barrier()

    def gather(value):
        if not dist.is_initialized():
            return [value]
        values = [None] * world_size
        dist.all_gather_object(values, value)
        return values

    report = {"status": "initializing", "requests": []}
    created_output = False
    try:
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            created_output = True
        barrier()
        devices = gather({
            "rank": rank, "hostname": socket.gethostname(), "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "attention_kernel": kernel_backend,
        })
        if rank == 0:
            versions = {}
            for package in ("torch", "diffusers", "transformers", "triton", "nvidia-cutlass-dsl", "nvidia-cudnn-frontend", "av"):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
            source_file = ROOT.parent / "source.json"
            source = json.loads(source_file.read_text()) if source_file.is_file() else None
            report.update({
                "model": args.model, "adapter": str(args.adapter.resolve()),
                "adapter_sha256": sha256(args.adapter), "sana_revision": revision.stdout.strip(),
                "deployment_source": source,
                "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "devices": devices, "packages": versions, "cuda_version": torch.version.cuda,
                "attention_backend": args.attention_backend, "compute_quant": "none", "lora_mode": "merged",
                "scheduler_points": INFERENCE_STEPS, "expected_transformer_forwards": 4,
                "warmups_per_duration": args.warmups, "repeats_per_duration": args.repeats,
                "seed": args.seed, "durations": args.durations,
                "timing": "Warm requests include text encoding, four DiT forwards and AV decode; request_to_mp4_s also includes MP4 writing. Model load, warmup, validation and telemetry are excluded.",
                "environment": {key: value for key, value in os.environ.items()
                                if key.startswith(("H3_SOL_", "H3_ULYSSES_", "H3_VAE_", "SOL_ATTN_"))
                                or key in {"PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"}},
            })
            write_json(args.output_dir / "benchmark.json", report)
        barrier()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        engine = MiniMaxH3Inference(args.model, args.adapter, attention_backend=args.attention_backend,
                                    task="t2v", compute_quant="none", lora_mode="merged")
        barrier()
        initialization = gather({"rank": rank, "seconds": time.perf_counter() - started,
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        if rank == 0:
            report.update(status="running", initialization=initialization)
            write_json(args.output_dir / "benchmark.json", report)
        forwards = [0]

        def count_forward(_module, _inputs):
            forwards[0] += 1

        handle = engine.transformer.register_forward_pre_hook(count_forward)
        for duration in args.durations:
            for index in range(args.warmups + args.repeats):
                warmup = index < args.warmups
                phase = "warmup" if warmup else "measured"
                sample = index if warmup else index - args.warmups
                seed = args.seed + 10000 + sample if warmup else args.seed
                output = args.output_dir / f"{phase}-{duration}s-{sample:02d}.mp4"
                sparse_before = engine.sparse_attention.sparse_calls if engine.sparse_attention else 0
                forwards[0] = 0
                barrier()
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                media = engine.generate(prompt, duration=duration, seed=seed)
                pipeline_s = time.perf_counter() - started
                if rank == 0:
                    if media is None or media.audio is None:
                        raise RuntimeError("The pipeline did not return video with audio")
                    pipeline_s = media.elapsed_s
                    media.save(output)
                barrier()
                completion_s = time.perf_counter() - started
                local = {
                    "rank": rank, "pipeline_s": pipeline_s, "request_to_mp4_s": completion_s,
                    "transformer_forwards": forwards[0],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "sparse": engine.sparse_attention.stats() if engine.sparse_attention else None,
                    "sparse_calls": engine.sparse_attention.sparse_calls - sparse_before if engine.sparse_attention else 0,
                }
                ranks = gather(local)
                if any(item["transformer_forwards"] != 4 for item in ranks):
                    raise RuntimeError(f"Expected exactly four transformer forwards on each rank: {ranks}")
                if engine.sparse_attention and any(item["sparse_calls"] == 0 or not (item["sparse"]["gate"] or {}).get("passed") for item in ranks):
                    raise RuntimeError("Sparse attention did not execute with a passing numerical check on every rank")
                if rank == 0:
                    record = {
                        "phase": phase, "sample": sample, "duration": duration,
                        "frames": DURATION_FRAMES[duration], "seed": seed,
                        "pipeline_s": ranks[0]["pipeline_s"],
                        "request_to_mp4_s": max(item["request_to_mp4_s"] for item in ranks),
                        "ranks": ranks, "output": output.name,
                        "media": inspect_media(output, duration), "bytes": output.stat().st_size,
                        "sha256": sha256(output),
                    }
                    report["requests"].append(record)
                    report["summary"] = summarize(report["requests"], world_size)
                    write_json(args.output_dir / "benchmark.json", report)
                    print(json.dumps({key: record[key] for key in ("phase", "duration", "sample", "pipeline_s", "request_to_mp4_s")}), flush=True)
                del media
        handle.remove()
        if rank == 0:
            report["status"] = "complete"
            write_json(args.output_dir / "benchmark.json", report)
            print(json.dumps(report["summary"], indent=2), flush=True)
    except Exception as error:
        # A peer may fail before rank zero can write the report; status then stays
        # initializing/running, never complete. torchrun terminates the other ranks.
        if rank == 0 and created_output:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(args.output_dir / "benchmark.json", report)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
