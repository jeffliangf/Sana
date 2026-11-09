"""Bounded background publication of rank zero's MP4s (``serve.py --async-publication``).

The GPU loop stages each finished video into pinned host memory, hands the encoder job to
``OutputPublisher`` and starts the next generation. The publisher thread finishes the encode,
inspects, fsyncs and renames the MP4, then settles the job. One publication at a time: the next
video waits for the slot before its encoder starts, so at most one encoder and one set of pinned
frames exist besides the generation on the GPU. The publisher never issues collectives or CUDA work.
"""
from __future__ import annotations

from contextlib import suppress
import copy
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping

# Output errors that pause storage instead of terminating the worker. Every other failure,
# including a VideoEncodeError that does not wrap one of these, stays fatal.
STORAGE_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS, errno.EACCES,
                            errno.ENOENT, errno.EMFILE, errno.ENFILE})


def storage_error_message(error):
    """The root cause's message when ``error`` is a known filesystem failure, otherwise None."""
    cause = error
    while cause.__cause__ is not None:
        cause = cause.__cause__
    if isinstance(cause, OSError) and cause.errno in STORAGE_ERRNOS:
        return str(cause)
    return None


def partial_path(path):
    """Hidden name of an MP4 until it is complete; a restarted worker removes leftovers."""
    path = Path(path)
    return path.with_name(f".{path.stem}.partial.mp4")


def freeze(value):
    """A read-only deep copy, so later generations cannot change a staged job's metadata."""
    return MappingProxyType(copy.deepcopy(dict(value)))


@dataclass(frozen=True)
class PublicationTask:
    """One staged video with the metadata its terminal record needs, fixed at staging."""

    job_id: str
    encode: Any  # FastVideoEncodeJob; it owns the zero-copy pinned frames until wait() returns
    path: Path
    expected: Mapping[str, Any]  # output profile: width, height, duration_seconds
    result: Mapping[str, Any]  # per-rank metrics and GPU-side timings of this job only
    started: float  # time.perf_counter() at generation start
    staged: float  # time.perf_counter() when staging finished

    @property
    def temporary(self):
        return partial_path(self.path)


def encoder_timings(encode):
    """FastVideoEncodeJob phase timers. Its elapsed_s includes staging; phases overlap."""
    timings = {"staging_s": encode.staging_s, "elapsed_s": encode.elapsed_s, **(encode.timings or {})}
    return {name: None if value is None else round(value, 6) for name, value in timings.items()}


def finish_output(task, inspect_media):
    """Finish the encode, inspect, fsync, and atomically rename the MP4 (publisher thread)."""
    expected, temporary = task.expected, task.temporary
    try:
        # Joins the encoder thread: x264, the audio mux and the container close are done.
        task.encode.wait()
        profile = inspect_media(temporary, expected["duration_seconds"], expected["width"], expected["height"])
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
            size = os.fstat(source.fileno()).st_size
        temporary.replace(task.path)
    finally:
        # Only after the writer has ended, so it cannot recreate the file.
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return {"media": profile, "bytes": size}


class OutputPublisher:
    """The single publication slot, held from staging until the job is settled."""

    capacity = 1

    def __init__(self, publish):
        self._publish = publish  # publish(task): settles the job itself and never raises
        self._condition = threading.Condition()
        self._reserved = False

    @property
    def busy(self):
        with self._condition:
            return self._reserved

    def reserve(self):
        """Wait for the slot before starting an encoder; returns the seconds waited."""
        started = time.perf_counter()
        with self._condition:
            while self._reserved:
                self._condition.wait()
            self._reserved = True
        return time.perf_counter() - started

    def release(self):
        """Free the slot: after a publication, or when a reservation was not used."""
        with self._condition:
            self._reserved = False
            self._condition.notify_all()

    def start(self, task):
        """Publish ``task`` on a daemon thread, which inherits the reservation."""
        with self._condition:
            if not self._reserved:
                raise RuntimeError("reserve the publication slot before starting a publication")
        handoff = [task]

        def run():
            current = handoff.pop()
            try:
                self._publish(current)
            finally:
                # Drop the encoder job and its pinned frames before another encoder can start.
                del current
                self.release()

        threading.Thread(target=run, name=f"h3-publish-{task.job_id[:8]}", daemon=True).start()

    def wait_idle(self, timeout):
        """Wait at most ``timeout`` seconds for the slot; False if a publication is still running."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._reserved:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True
