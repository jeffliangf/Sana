"""Request contract shared by the GPU worker and the gateway.

Two request forms are accepted on ``POST /v1/videos``:

* The original Sol-H3 form, ``{"prompt", "duration": 5|10, "seed"}``. It is unchanged.
* The SGLang MiniMax-H3 form, ``{"task", "prompt", "conditions", "target", "seed"}``, as
  served by ``sglang serve --model-variant fl2va``. This deployment serves ``t2va`` and
  first-frame ``fl2va``; it rejects the parts of that contract it cannot run (``ref2va``,
  last-frame keyframes, non-768 short edges, and per-request sampler settings).

Geometry follows MiniMax-H3 itself: a 768-pixel short edge with the area capped at
1344x768, both axes rounded to multiples of 32, and frame counts snapped up to the
``17 * n + 5`` lengths its video VAE decodes. ``target.aspect_ratio: "auto"`` takes the
canvas from the first keyframe, as the base pipeline does without an explicit size.

This module must stay importable without torch or Pillow: the gateway has neither.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import hashlib
import io
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

FPS = 24
SHORT_EDGE = 768
MAX_CANVAS_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32
MIN_ASPECT_RATIO, MAX_ASPECT_RATIO = 1 / 4, 4.0
FRAME_STEP = 17
MIN_NUM_FRAMES, MAX_NUM_FRAMES = 124, 345  # 5.17 s and 14.375 s
VALID_NUM_FRAMES = tuple(range(MIN_NUM_FRAMES, MAX_NUM_FRAMES + 1, FRAME_STEP))
MIN_DURATION_SECONDS, MAX_DURATION_SECONDS = 4.0, 15.0  # SGLang's accepted request range
LEGACY_NUM_FRAMES = {5: 124, 10: 243}
DEFAULT_ASPECT_RATIO = "16:9"  # 1344x768
ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
TASKS = ("t2va", "fl2va")
MAX_PROMPT_CHARS = 8000
DEFAULT_SEED = 42
MAX_SEED = 2**63 - 1
MAX_IMAGE_BYTES = 20 * 2**20
MAX_IMAGE_SIDE = 8192
MIN_IMAGE_SIDE = 32
MAX_IMAGE_PIXELS = 40_000_000
IMAGE_FORMATS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}
LEGACY_BODY_BYTES = 32 * 2**10
# A 20 MiB image is 26.7 MiB as base64; leave room for an 8,000-character prompt.
MAX_BODY_BYTES = 28 * 2**20
API_VERSION = 2
UNSUPPORTED_FIELDS = {
    "num_inference_steps": "this deployment runs a fixed four-step FastH3 profile",
    "flow_shift": "the video flow shift is fixed by the four-step FastH3 profile",
    "audio_flow_shift": "the audio flow shift is fixed by the four-step FastH3 profile",
    "guidance_scale": "MiniMax-H3 is guidance-distilled; there is no guidance scale",
    "guidance_scale_2": "MiniMax-H3 is guidance-distilled; there is no guidance scale",
    "true_cfg_scale": "MiniMax-H3 is guidance-distilled; there is no guidance scale",
    "negative_prompt": "MiniMax-H3 is guidance-distilled; there is no negative prompt",
}


def capabilities(max_num_frames=MAX_NUM_FRAMES):
    """What a worker built from this module accepts; published in /healthz as ``api``."""
    return {"version": API_VERSION, "tasks": list(TASKS), "aspect_ratios": ["auto", *ASPECT_RATIOS],
            "short_edges": [SHORT_EDGE], "keyframe_frame_indices": [[0]], "fps": FPS,
            "num_frames": {"min": MIN_NUM_FRAMES, "max": int(max_num_frames), "step": FRAME_STEP},
            "image_formats": sorted(IMAGE_FORMATS.values()), "max_image_bytes": MAX_IMAGE_BYTES}


class RequestError(Exception):
    """A request the service will not run; ``detail`` follows FastAPI's error shape."""

    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _error(loc, message, kind="value_error"):
    return {"loc": ["body", *loc], "msg": message, "type": kind}


def _invalid(*errors):
    return RequestError(422, list(errors))


def _from_validation_error(error, prefix=()):
    # Never echo request input back: a condition URI can be tens of megabytes.
    return [{"loc": ["body", *prefix, *item["loc"]], "msg": item["msg"], "type": item["type"]}
            for item in error.errors(include_url=False, include_context=False, include_input=False)]


def resolve_canvas_size(aspect_width, aspect_height):
    """MiniMax-H3's canvas for a display aspect ratio, as ``(height, width)``.

    Identical to ``diffusers.modular_pipelines.minimax_h3.packing.resolve_canvas_size``:
    the short edge starts at 768, the area is capped at 768x1344, and both axes are then
    rounded to the nearest multiple of 32.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height}.")
    if ratio >= 1.0:
        width, height = SHORT_EDGE * ratio, float(SHORT_EDGE)
    else:
        width, height = float(SHORT_EDGE), SHORT_EDGE / ratio
    area = width * height
    if area > MAX_CANVAS_PIXELS:
        scale = (MAX_CANVAS_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    multiple = CANVAS_MULTIPLE
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


def align_num_frames(num_frames):
    """Snap a frame count up to the next ``17 * n + 5`` the video VAE decodes."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    return num_frames + (5 - num_frames) % FRAME_STEP


def frames_for_duration(seconds):
    """SGLang's mapping: round to whole frames at 24 fps, then snap up to a VAE length."""
    return align_num_frames(int(round(float(seconds) * FPS)))


def output_profile(num_frames, canvas):
    height, width = canvas
    return {"width": int(width), "height": int(height), "num_frames": int(num_frames), "fps": FPS,
            "duration_seconds": round(num_frames / FPS, 6)}


def decode_image_uri(uri):
    """Decode a base64 keyframe URI to bytes and its image format.

    Accepts ``data:image/<type>;base64,<payload>`` and SGLang's ``base64://<payload>``,
    with the standard or URL-safe alphabet. Only PNG, JPEG, and WebP payloads are accepted.
    """
    if uri.startswith("data:"):
        header, separator, payload = uri.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("a data URI must be base64 encoded: data:image/png;base64,<payload>")
        if not header[len("data:"):].lower().startswith("image/"):
            raise ValueError("a data URI must carry an image/* media type")
    elif uri.startswith("base64://"):
        payload = uri[len("base64://"):]
    else:
        scheme = uri.split(":", 1)[0] if ":" in uri else "a bare path"
        raise ValueError(f"only base64 images are accepted (data:image/...;base64, or base64://); got {scheme}")
    if len(payload) > math.ceil(MAX_IMAGE_BYTES / 3) * 4 + 4:
        raise ValueError(f"the image exceeds {MAX_IMAGE_BYTES // 2**20} MiB")
    payload = payload.strip().replace("-", "+").replace("_", "/")
    try:
        data = base64.b64decode(payload + "=" * (-len(payload) % 4), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("the base64 image payload is invalid") from None
    if not data:
        raise ValueError("the base64 image payload is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"the image exceeds {MAX_IMAGE_BYTES // 2**20} MiB")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        kind = "png"
    elif data.startswith(b"\xff\xd8\xff"):
        kind = "jpg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        kind = "webp"
    else:
        raise ValueError("the image must be PNG, JPEG, or WebP")
    return data, kind


def inspect_image(data):
    """Fully decode an image and return its EXIF-upright size and format (requires Pillow)."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as image:
            kind = IMAGE_FORMATS.get(image.format)
            if kind is None:
                raise ValueError(f"the image must be PNG, JPEG, or WebP, got {image.format}")
            width, height = image.size
            if not (MIN_IMAGE_SIDE <= width <= MAX_IMAGE_SIDE and MIN_IMAGE_SIDE <= height <= MAX_IMAGE_SIDE):
                raise ValueError(f"image sides must be {MIN_IMAGE_SIDE}-{MAX_IMAGE_SIDE} pixels, got {width}x{height}")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"the image exceeds {MAX_IMAGE_PIXELS:,} pixels")
            image.load()  # Reject truncated or corrupt files at admission, not on the GPU.
            orientation = image.getexif().get(0x0112, 1)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError(f"the image could not be decoded: {error}") from None
    if orientation in (5, 6, 7, 8):  # The engine applies ImageOps.exif_transpose.
        width, height = height, width
    return {"format": kind, "width": width, "height": height}


class GenerateRequest(BaseModel):
    """The original Sol-H3 request: text to video at 1344x768."""

    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    duration: Literal[5, 10] = 5
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=MAX_SEED)

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value):
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return value.strip()


class KeyframeCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: str
    uri: str = Field(min_length=1)
    role: str
    frame_index: int


class VideoTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    short_edge: int = SHORT_EDGE
    aspect_ratio: str | None = None
    duration_seconds: float | None = None


class VideoRequest(BaseModel):
    """The SGLang MiniMax-H3 request, restricted to what this deployment runs."""

    model_config = ConfigDict(extra="forbid", strict=True)
    task: str
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    conditions: list[KeyframeCondition] = Field(default_factory=list)
    target: VideoTarget = Field(default_factory=VideoTarget)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    # SGLang transport fields: accepted; ``target.duration_seconds`` wins over ``seconds``.
    model: str | None = None
    seconds: float | None = None
    num_outputs_per_prompt: int | None = None
    n: int | None = None

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value):
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return value.strip()


@dataclass
class ParsedRequest:
    """A validated request.

    ``identity`` is what the job record stores and an Idempotency-Key repeat must match;
    ``payload`` is what the gateway forwards to a worker; ``images`` are decoded keyframes.
    ``canvas`` is ``(height, width)``, or None until the keyframe's size is known.
    """

    identity: dict
    payload: dict
    task: str
    num_frames: int
    canvas: tuple[int, int] | None
    images: list[bytes] = field(default_factory=list)
    legacy: bool = False

    def resolve(self, image_sizes=()):
        """The output profile once keyframe sizes (EXIF-upright ``(width, height)``) are known."""
        canvas = self.canvas
        if canvas is None:
            width, height = image_sizes[0]
            try:
                canvas = resolve_canvas_size(width, height)
            except ValueError as error:
                raise _invalid(_error(("conditions", 0, "uri"),
                                      f"{error} Use an explicit target.aspect_ratio to stretch it.")) from None
        return output_profile(self.num_frames, canvas)


def parse_request(payload):
    """Validate either request form. Raises RequestError with a FastAPI-style 422 detail."""
    if not isinstance(payload, dict):
        raise _invalid(_error((), "the request body must be a JSON object", "type_error"))
    if "task" not in payload:
        try:
            request = GenerateRequest.model_validate(payload)
        except ValidationError as error:
            raise _invalid(*_from_validation_error(error)) from None
        identity = request.model_dump()
        return ParsedRequest(identity=identity, payload=dict(identity), task="t2va",
                             num_frames=LEGACY_NUM_FRAMES[request.duration],
                             canvas=resolve_canvas_size(16, 9), legacy=True)

    unsupported = [_error((name,), f"{name} is not supported: {reason}", "unsupported_field")
                   for name, reason in UNSUPPORTED_FIELDS.items() if name in payload]
    if unsupported:
        raise _invalid(*unsupported)
    try:
        request = VideoRequest.model_validate(payload)
    except ValidationError as error:
        raise _invalid(*_from_validation_error(error)) from None

    errors = []
    task = request.task.strip().lower()
    if task == "ref2va":
        errors.append(_error(("task",), "ref2va needs the MiniMax-H3 Ref2VA checkpoint, which this deployment does not serve"))
    elif task not in TASKS:
        errors.append(_error(("task",), f"task must be one of {list(TASKS)}"))
    for name in ("num_outputs_per_prompt", "n"):
        if getattr(request, name) not in (None, 1):
            errors.append(_error((name,), "only one output per request is supported"))
    target = request.target
    if target.short_edge != SHORT_EDGE:
        errors.append(_error(("target", "short_edge"), f"only the released {SHORT_EDGE}-pixel short edge is supported"))
    aspect_ratio = target.aspect_ratio
    if aspect_ratio is None:
        aspect_ratio = "auto" if task == "fl2va" else DEFAULT_ASPECT_RATIO
    if aspect_ratio != "auto" and aspect_ratio not in ASPECT_RATIOS:
        errors.append(_error(("target", "aspect_ratio"), f"aspect_ratio must be 'auto' or one of {list(ASPECT_RATIOS)}"))
    duration = target.duration_seconds
    if duration is None:
        duration = request.seconds if request.seconds is not None else 5.0
    num_frames = None
    if not math.isfinite(duration) or not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        errors.append(_error(("target", "duration_seconds"),
                             f"duration_seconds must be between {MIN_DURATION_SECONDS:g} and {MAX_DURATION_SECONDS:g}"))
    else:
        num_frames = frames_for_duration(duration)
        if num_frames not in VALID_NUM_FRAMES:
            errors.append(_error(("target", "duration_seconds"),
                                 f"{duration:g} s resolves to {num_frames} frames; this deployment serves "
                                 f"{MIN_NUM_FRAMES}-{MAX_NUM_FRAMES} frames ({MIN_NUM_FRAMES / FPS:.3f}-"
                                 f"{MAX_NUM_FRAMES / FPS:.3f} s)"))

    conditions, images = [], []
    if task == "t2va" and request.conditions:
        errors.append(_error(("conditions",), "t2va takes no conditions; use fl2va for a first frame"))
    elif task == "fl2va":
        if len(request.conditions) != 1:
            errors.append(_error(("conditions",), "fl2va takes exactly one keyframe (the first frame) in this deployment"))
        for index, condition in enumerate(request.conditions[:1]):
            location = ("conditions", index)
            if condition.type != "image":
                errors.append(_error((*location, "type"), "a keyframe condition must have type 'image'"))
            if condition.role != "keyframe":
                errors.append(_error((*location, "role"), "fl2va conditions must have role 'keyframe'"))
            if condition.frame_index == -1:
                errors.append(_error((*location, "frame_index"),
                                     "last-frame keyframes (frame_index -1) are not supported yet; use 0"))
            elif condition.frame_index != 0:
                errors.append(_error((*location, "frame_index"), "frame_index must be 0 (the first frame)"))
            try:
                data, kind = decode_image_uri(condition.uri)
            except ValueError as error:
                errors.append(_error((*location, "uri"), str(error)))
                continue
            images.append(data)
            conditions.append({"type": "image", "role": "keyframe", "frame_index": 0, "format": kind,
                               "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if errors:
        raise _invalid(*errors)

    seed = DEFAULT_SEED if request.seed is None else request.seed
    resolved_target = {"short_edge": SHORT_EDGE, "aspect_ratio": aspect_ratio, "duration_seconds": float(duration)}
    identity = {"task": task, "prompt": request.prompt, "seed": seed, "target": resolved_target,
                "conditions": conditions}
    forwarded = {"task": task, "prompt": request.prompt, "seed": seed, "target": resolved_target,
                 "conditions": [{"type": c.type, "uri": c.uri, "role": c.role, "frame_index": c.frame_index}
                                for c in request.conditions]}
    canvas = None if aspect_ratio == "auto" and task == "fl2va" else resolve_canvas_size(
        *map(int, (aspect_ratio if aspect_ratio != "auto" else DEFAULT_ASPECT_RATIO).split(":")))
    return ParsedRequest(identity=identity, payload=forwarded, task=task, num_frames=num_frames,
                         canvas=canvas, images=images)


def accepts(api, parsed):
    """Whether a worker's published capabilities cover a request."""
    if parsed.legacy:
        return True
    if not isinstance(api, dict) or api.get("version", 0) < API_VERSION:
        return False
    frames = api.get("num_frames") or {}
    return parsed.task in api.get("tasks", ()) and frames.get("min", 0) <= parsed.num_frames <= frames.get("max", 0)


def sglang_status(status):
    """SGLang's job status names: ``completed`` instead of ``succeeded``."""
    return "completed" if status == "succeeded" else status


def same_request(record, identity):
    """Whether a stored job was created from the same request identity."""
    return all(record.get(name) == value for name, value in identity.items())


def as_json_detail(detail: Any):
    return {"detail": detail}
