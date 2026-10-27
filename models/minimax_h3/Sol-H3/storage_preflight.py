#!/usr/bin/env python3
"""Check capacity before staging a node; never delete existing data."""
import argparse
import json
from pathlib import Path

from service_storage import check_space, storage_probe


def required_staging_bytes(venv, checkpoints, stage):
    required = 0
    if not (venv / "bin/python").exists():
        # The staged tar is uncompressed. It approximates extraction size.
        required += (stage / "sol-h3-venv.tar").stat().st_size
    if not checkpoints.is_dir():
        manifest = json.loads((stage / "manifest.json").read_text())
        required += sum(item["bytes"] for name, item in manifest.items()
                        if name.startswith("sol-h3-checkpoints/"))
    return required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("local-root", "venv", "checkpoints", "stage"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--min-free-gib", type=int, default=20)
    args = parser.parse_args()
    if args.min_free_gib < 1:
        parser.error("minimum free space must be positive")
    required = required_staging_bytes(args.venv, args.checkpoints, args.stage)
    info = check_space(args.local_root, args.min_free_gib * 2**30, reserve_bytes=required)
    storage_probe(args.local_root)
    # Prepared environments/checkpoints may live on different mounts.
    for path in (args.venv, args.checkpoints):
        reserve = required if not path.exists() else 0
        while not path.exists():
            path = path.parent
        check_space(path, args.min_free_gib * 2**30, reserve_bytes=reserve)
    print(json.dumps({"local_root": str(args.local_root), "staging_bytes": required, **info}))


if __name__ == "__main__":
    main()
