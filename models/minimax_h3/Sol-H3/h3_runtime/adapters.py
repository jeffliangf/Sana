"""Adapter profiles: the few-step LoRA a worker merges and the sampling schedule it was distilled for.

A profile fixes everything that must match its adapter: the transformer forward count, the video and audio
flow shifts, the strength, and the length of the last step (SMD samples a last step half as long as the
others). The definitions live in ``adapter_profiles.json`` next to ``serve.py``, so adding an adapter is a
configuration change. A worker serves one profile for its lifetime: the AdaLN precompute bakes the step
schedule into its tables on the first request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

PROFILES_FILE = Path(__file__).resolve().parents[1] / "adapter_profiles.json"
TASKS = ("t2v", "i2v", "ref2va")
NAME = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AdapterProfile:
    name: str
    file: str
    tasks: tuple[str, ...]
    transformer_forwards: int
    video_shift: float
    audio_shift: float
    strength: float = 1.0
    final_step_size_scale: float = 1.0
    alpha: float | None = None
    sha256: str | None = None
    bytes: int | None = None
    description: str = ""
    reference: str = ""
    source: dict = field(default_factory=dict)

    @property
    def scheduler_points(self) -> int:
        """MiniMax-H3's scheduler counts sigma grid points including the terminal zero: N forwards need N + 1."""
        return self.transformer_forwards + 1

    def default_path(self, model_path: str | Path) -> Path:
        """The adapter file below the checkpoint root, the directory that holds the MiniMax-H3 model directory."""
        return Path(model_path).expanduser().resolve().parent / self.file

    def sigmas(self, shift: float) -> list[float] | None:
        """The shifted sigma grid this profile samples, or None for the scheduler's own grid. Only a shorter last
        step needs an explicit grid; with even steps the scheduler builds the same grid itself."""
        if self.final_step_size_scale == 1.0:
            return None
        return shifted_sigma_grid(raw_sigma_grid(self.transformer_forwards, self.final_step_size_scale), shift)

    def summary(self) -> dict:
        return {"profile": self.name, "transformer_forwards": self.transformer_forwards,
                "video_shift": self.video_shift, "audio_shift": self.audio_shift,
                "final_step_size_scale": self.final_step_size_scale, "strength": self.strength,
                "alpha": self.alpha, "sha256": self.sha256, "source": self.source}


def raw_sigma_grid(forwards: int, final_step_size_scale: float) -> list[float]:
    """SMD's sigma grid before the shift (``raw_sigma_grid`` in its release's tools/infer_minimax_h3.py): from 1 to
    0 in ``forwards`` equal steps, except that the last one is ``final_step_size_scale`` times as long."""
    segment = 1.0 / (forwards - 1 + final_step_size_scale)
    sigmas = [1.0]
    for step in range(forwards):
        sigmas.append(sigmas[-1] - segment * (final_step_size_scale if step == forwards - 1 else 1.0))
    sigmas[-1] = 0.0
    return sigmas


def shifted_sigma_grid(raw: list[float], shift: float) -> list[float]:
    """The flow shift MiniMaxH3Scheduler applies to its own grid: sigma' = s * sigma / (1 + (s - 1) * sigma)."""
    return [shift * sigma / (1.0 + (shift - 1.0) * sigma) for sigma in raw]


def install_explicit_schedule(scheduler, sigmas: list[float]) -> None:
    """Make ``scheduler`` sample ``sigmas`` whatever step count the pipeline asks for, as SMD's runner does.
    MiniMaxH3Scheduler.set_timesteps uses an explicit grid verbatim: no second shift, terminal 0 required."""
    original = scheduler.set_timesteps
    grid = [float(sigma) for sigma in sigmas]

    def set_timesteps(num_inference_steps=None, device=None, sigmas=None):
        return original(device=device, sigmas=grid if sigmas is None else sigmas)

    scheduler.set_timesteps = set_timesteps


def _profile(name: str, entry: dict) -> AdapterProfile:
    if not NAME.match(name):
        raise ValueError(f"invalid adapter profile name {name!r}")
    file = entry.get("file")
    if not isinstance(file, str) or not file or PurePosixPath(file).is_absolute() or ".." in PurePosixPath(file).parts:
        raise ValueError(f"adapter profile {name}: 'file' must be a relative path without '..'")
    tasks = tuple(entry.get("tasks") or ())
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError(f"adapter profile {name}: 'tasks' must be a non-empty subset of {TASKS}")
    forwards = entry.get("transformer_forwards")
    if not isinstance(forwards, int) or isinstance(forwards, bool) or not 1 <= forwards <= 64:
        raise ValueError(f"adapter profile {name}: 'transformer_forwards' must be an integer from 1 to 64")
    values = {}
    for key in ("video_shift", "audio_shift", "strength"):
        value = entry.get(key, 1.0 if key == "strength" else None)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 \
                or (key != "strength" and value <= 0):
            raise ValueError(f"adapter profile {name}: '{key}' must be a finite positive number")
        values[key] = float(value)
    scale = entry.get("final_step_size_scale", 1.0)
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) \
            or not 0 < scale <= 1:
        raise ValueError(f"adapter profile {name}: 'final_step_size_scale' must be a number in (0, 1]")
    values["final_step_size_scale"] = float(scale)
    alpha = entry.get("alpha")
    if alpha is not None and (not isinstance(alpha, (int, float)) or isinstance(alpha, bool)
                              or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError(f"adapter profile {name}: 'alpha' must be null or a positive number")
    digest = entry.get("sha256")
    if digest is not None and not SHA256.match(str(digest)):
        raise ValueError(f"adapter profile {name}: 'sha256' must be 64 lowercase hex characters")
    size = entry.get("bytes")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
        raise ValueError(f"adapter profile {name}: 'bytes' must be a positive integer")
    return AdapterProfile(name=name, file=file, tasks=tasks, transformer_forwards=forwards,
                          alpha=None if alpha is None else float(alpha), sha256=digest, bytes=size,
                          description=str(entry.get("description", "")), reference=str(entry.get("reference", "")),
                          source=dict(entry.get("source") or {}), **values)


def load_profiles(path: str | Path = PROFILES_FILE) -> tuple[dict[str, str], dict[str, AdapterProfile]]:
    """Return (task -> default profile name, name -> profile), validating the whole file."""
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version {data.get('schema_version')!r}")
    profiles = {name: _profile(name, entry) for name, entry in (data.get("profiles") or {}).items()}
    defaults = dict(data.get("defaults") or {})
    for task in TASKS:
        name = defaults.get(task)
        if name not in profiles or task not in profiles[name].tasks:
            raise ValueError(f"{path}: the default profile for {task} must exist and support {task}")
    return defaults, profiles


def get_profile(name: str | AdapterProfile | None = None, *, task: str = "t2v",
                path: str | Path = PROFILES_FILE) -> AdapterProfile:
    """The named profile (or the task's default) after checking that it supports ``task``."""
    if isinstance(name, AdapterProfile):
        profile = name
    else:
        defaults, profiles = load_profiles(path)
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        name = name or defaults[task]
        if name not in profiles:
            raise ValueError(f"unknown adapter profile {name!r}; known: {sorted(profiles)}")
        profile = profiles[name]
    if task not in profile.tasks:
        raise ValueError(f"adapter profile {profile.name} supports {list(profile.tasks)}, not {task}")
    return profile


def verify_file(profile: AdapterProfile, path: str | Path) -> str:
    """Check the adapter file against the profile's size and SHA-256; return its SHA-256."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adapter file for profile {profile.name} does not exist: {path}")
    if profile.bytes is not None and path.stat().st_size != profile.bytes:
        raise ValueError(f"{path} has {path.stat().st_size} bytes; profile {profile.name} expects {profile.bytes}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 2**20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if profile.sha256 is not None and value != profile.sha256:
        raise ValueError(f"{path} SHA-256 {value} does not match profile {profile.name} ({profile.sha256})")
    return value
