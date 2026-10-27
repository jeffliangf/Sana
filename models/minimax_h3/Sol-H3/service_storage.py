"""Small filesystem primitives shared by the resident service and gateway."""
from contextlib import suppress
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile


def atomic_json(path, value):
    """Flush a complete record before replacing its previous version.

    Unique temporary files also allow independent writers to share a directory.
    This is not a transaction across files or a power-loss guarantee for a remote
    filesystem; the service still depends on that filesystem's durability.
    """
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def check_space(path, min_free_bytes, *, reserve_bytes=0):
    """Check bytes and inodes; a subsequent write can still fail (quota/races)."""
    free = shutil.disk_usage(path).free
    stats = os.statvfs(path)
    inodes = stats.f_favail if stats.f_files else None
    if free < min_free_bytes + max(0, reserve_bytes):
        raise OSError(errno.ENOSPC, "Filesystem free-space reserve would be exceeded", str(path))
    if inodes is not None and inodes < 128:
        raise OSError(errno.ENOSPC, "Filesystem has fewer than 128 available inodes", str(path))
    return {"free_bytes": free, "free_inodes": inodes, "min_free_bytes": min_free_bytes}


def storage_probe(root):
    # Keep this outside *.json job scans even if cleanup fails or we crash.
    path = Path(root) / ".storage-probe"
    atomic_json(path, {"probe": True})
    path.unlink()


def quiet_print(message):
    # A full log filesystem must not terminate the process it is diagnosing.
    with suppress(OSError):
        print(message, flush=True)
