#!/usr/bin/env python3
"""Download MiniMax-H3 and the adapter files of the selected profiles (adapter_profiles.json)."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download, snapshot_download

from h3_runtime.adapters import get_profile, load_profiles, verify_file


def main() -> int:
    defaults, profiles = load_profiles()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--task",
        choices=("t2v", "ref2va", "both"),
        default="t2v",
        help="Checkpoint partition to download (default: t2v)",
    )
    parser.add_argument(
        "--adapter-profile", action="append", choices=sorted(profiles), default=[],
        help="Profile whose adapter file to download; repeatable (default: the task's default profile)",
    )
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    model_patterns = [
        "model_index.json",
        "modular_model_index.json",
        "audio_scheduler/*",
        "audio_vae/*",
        "processor/*",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "vae/*",
    ]
    if args.task in {"t2v", "both"}:
        model_patterns.append("transformer/*")
    if args.task in {"ref2va", "both"}:
        model_patterns.append("transformer_ref/*")

    snapshot_download(
        "MiniMaxAI/MiniMax-H3",
        local_dir=root / "MiniMax-H3",
        allow_patterns=model_patterns,
    )
    names = args.adapter_profile or [defaults[task] for task in (("t2v", "ref2va") if args.task == "both"
                                                                  else (args.task,))]
    for name in dict.fromkeys(names):
        profile = profiles[name]
        source = profile.source
        target = profile.default_path(root / "MiniMax-H3")
        if "copy_from" in source:
            # Delivered outside the Hub (SMD-H3 E): copy the shared file unless it is already here.
            if not target.is_file() or target.stat().st_size != profile.bytes:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source["copy_from"], target)
            print(f"{name}: {target} sha256={verify_file(get_profile(name, task=profile.tasks[0]), target)}")
            continue
        # 'file' is <local repo directory>/<path in the repo>; hf_hub_download keeps the repo path below local_dir.
        local_dir = Path(profile.file).parts[0]
        if Path(profile.file) != Path(local_dir) / source["path"]:
            raise SystemExit(f"profile {name}: file {profile.file} is not <directory>/{source['path']}")
        # The pinned revision keeps a re-uploaded file from replacing the validated one; the digest check proves it.
        hf_hub_download(source["repo"], source["path"], revision=source.get("revision"), local_dir=root / local_dir)
        print(f"{name}: {target} sha256={verify_file(get_profile(name, task=profile.tasks[0]), target)}")
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
