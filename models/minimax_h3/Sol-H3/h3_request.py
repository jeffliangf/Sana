"""Request contract shared by the GPU worker and the gateway.

Two request forms are accepted on ``POST /v1/videos``:

* The original Sol-H3 form, ``{"prompt", "duration": 5|10, "seed"}``. It is unchanged.
* The SGLang MiniMax-H3 form, ``{"task", "prompt", "conditions", "target", "seed"}``, as
  served by ``sglang serve --model-variant fl2va`` or ``ref2va``. A T2V worker serves ``t2va``
  and first-frame ``fl2va``; a Ref2VA worker serves ``ref2va`` with MiniMax-H3's ordered
  omni-references (``role: "reference"``): up to nine images, three videos and three audio
  clips, twelve in all. Both reject the parts of that contract they cannot run (last-frame
  keyframes, keyframes in ``ref2va``, ``start_time_seconds``, non-768 short edges, and
  per-request sampler settings). Each worker publishes its tasks in /healthz ``api.tasks``.

Geometry follows MiniMax-H3 itself: a 768-pixel short edge with the area capped at
1344x768, both axes rounded to multiples of 32, and frame counts snapped up to the
``17 * n + 5`` lengths its video VAE decodes. ``target.aspect_ratio: "auto"`` takes the
canvas from the first keyframe, as the base pipeline does without an explicit size; for
``ref2va`` it is 16:9, the Ref2VA pipeline's (and SGLang's) own default (references are not frames).

This module must stay importable without torch, Pillow, or PyAV: the gateway may have none of them.
The gateway checks what it can from the bytes; the worker decodes every reference at admission.
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
# t2va and fl2va: 5.17 s to 14.375 s, what a two-GPU H200 worker holds at every canvas.
MIN_NUM_FRAMES, MAX_NUM_FRAMES = 124, 345
VALID_NUM_FRAMES = tuple(range(MIN_NUM_FRAMES, MAX_NUM_FRAMES + 1, FRAME_STEP))
MIN_DURATION_SECONDS, MAX_DURATION_SECONDS = 4.0, 15.0  # SGLang's accepted request range
# ref2va: MiniMax-H3's whole published 4-15 s range as SGLang resolves it, 4 s to 107 frames (4.458 s) and
# 15 s to 362 (15.083 s). Ref2VA workers are sharded four-GPU workers with room for the longest.
REF2VA_MIN_NUM_FRAMES, REF2VA_MAX_NUM_FRAMES = 107, 362
REF2VA_VALID_NUM_FRAMES = tuple(range(REF2VA_MIN_NUM_FRAMES, REF2VA_MAX_NUM_FRAMES + 1, FRAME_STEP))
LEGACY_NUM_FRAMES = {5: 124, 10: 243}
DEFAULT_ASPECT_RATIO = "16:9"  # 1344x768
ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
TASKS = ("t2va", "fl2va", "ref2va")
# What a worker serves, by its engine task (the adapter profile's): T2V workers also run first-frame I2V.
WORKER_TASKS = {"t2v": ("t2va", "fl2va"), "ref2va": ("ref2va",)}
# MiniMax-H3 Ref2VA's published input limits (its README, "Model Variants and Input Specifications").
REFERENCE_TYPES = ("image", "video", "video_audio", "audio")  # SGLang's names; video_audio requires sound
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCES = 12
MIN_CLIP_SECONDS = 2.0
# "Each clip 2-15 seconds, at most 15 seconds in total", read on H3's own 24 fps grid: H3 renders 15 s as 362
# frames (15.083 s), and a 15 s clip from a 24 fps generator is often 361 frames (15.042 s), one past 15.000.
MAX_CLIP_SECONDS = REF2VA_MAX_NUM_FRAMES / FPS
MAX_AUDIO_CHANNELS = 2  # The pipeline takes mono or stereo; it upmixes mono
MAX_MEDIA_BYTES = 64 * 2**20  # one video or audio reference
# Containers a video or audio reference may use: SGLang's list, by FFmpeg format name.
MEDIA_CONTAINERS = frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "wav", "mp3", "flac",
                              "ogg"})
# About 3,500 Qwen tokens of English or 12,300 of Chinese. The model has no limit of its own: the text encoder
# reads the whole prompt and each token becomes one DiT row, next to about 104,000 rows for a 345-frame clip.
MAX_PROMPT_CHARS = 16000
DEFAULT_SEED = 42
MAX_SEED = 2**63 - 1
MAX_IMAGE_BYTES = 20 * 2**20
MAX_IMAGE_SIDE = 8192
MIN_IMAGE_SIDE = 32
MAX_IMAGE_PIXELS = 40_000_000
IMAGE_FORMATS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}
LEGACY_BODY_BYTES = 32 * 2**10
# A 20 MiB image is 26.7 MiB as base64; leave room for a 16,000-character prompt.
MAX_BODY_BYTES = 28 * 2**20
# Ref2VA bodies carry up to twelve references: 96 MiB of media as base64, plus the prompt.
REF2VA_MAX_BODY_BYTES = 128 * 2**20
API_VERSION = 2
UNSUPPORTED_FIELDS = {
    "num_inference_steps": "each worker runs a fixed adapter profile (see /healthz adapter)",
    "flow_shift": "the video flow shift is fixed by the worker's adapter profile",
    "audio_flow_shift": "the audio flow shift is fixed by the worker's adapter profile",
    "guidance_scale": "MiniMax-H3 is guidance-distilled; there is no guidance scale",
    "guidance_scale_2": "MiniMax-H3 is guidance-distilled; there is no guidance scale",
    "true_cfg_scale": "MiniMax-H3 is guidance-distilled; there is no guidance scale",
    "negative_prompt": "MiniMax-H3 is guidance-distilled; there is no negative prompt",
}


def frame_range(task):
    """The ``(min, max)`` output frame counts of a task."""
    if task == "ref2va":
        return REF2VA_MIN_NUM_FRAMES, REF2VA_MAX_NUM_FRAMES
    return MIN_NUM_FRAMES, MAX_NUM_FRAMES


def max_body_bytes(tasks):
    """The largest request body a worker serving ``tasks`` takes."""
    return REF2VA_MAX_BODY_BYTES if "ref2va" in tasks else MAX_BODY_BYTES


def capabilities(max_num_frames=None, tasks=WORKER_TASKS["t2v"]):
    """What a worker serving ``tasks`` accepts; published in /healthz as ``api``."""
    low, high = frame_range(tasks[0])
    api = {"version": API_VERSION, "tasks": list(tasks), "aspect_ratios": ["auto", *ASPECT_RATIOS],
           "short_edges": [SHORT_EDGE]}
    if "fl2va" in tasks:
        api["keyframe_frame_indices"] = [[0]]
    api.update({"fps": FPS, "num_frames": {"min": low, "max": int(max_num_frames or high), "step": FRAME_STEP},
                "image_formats": sorted(IMAGE_FORMATS.values()), "max_image_bytes": MAX_IMAGE_BYTES})
    if "ref2va" in tasks:
        api.update(reference_types=list(REFERENCE_TYPES), max_references=MAX_REFERENCES,
                   max_references_by_type={"image": MAX_REFERENCE_IMAGES, "video": MAX_REFERENCE_VIDEOS,
                                           "audio": MAX_REFERENCE_AUDIOS},
                   reference_clip_seconds={"min": MIN_CLIP_SECONDS, "max": round(MAX_CLIP_SECONDS, 6),
                                           "total_per_type": round(MAX_CLIP_SECONDS, 6)},
                   media_containers=sorted(MEDIA_CONTAINERS), max_media_bytes=MAX_MEDIA_BYTES,
                   max_body_bytes=REF2VA_MAX_BODY_BYTES)
    return api


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


def sniff_container(data):
    """The container of a video or audio file from its first bytes, as a file extension; None if unknown."""
    if data[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"):  # ISO BMFF: MP4, MOV, M4A, 3GP
        brand = data[8:12] if data[4:8] == b"ftyp" else b""
        return "mov" if brand == b"qt  " else "m4a" if brand in (b"M4A ", b"M4B ") else "mp4"
    if data[:4] == b"\x1aE\xdf\xa3":  # EBML: Matroska or WebM
        return "webm" if b"webm" in data[:64] else "mkv"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE6 == 0xE2):  # MPEG layer III
        return "mp3"
    return None


AUDIO_ONLY_CONTAINERS = frozenset({"wav", "flac", "ogg", "mp3"})


def decode_media_uri(uri, condition_type):
    """Decode a base64 video or audio reference URI to bytes and its container (a file extension).

    Accepts ``data:<type>;base64,<payload>`` (any media type: the bytes decide) and ``base64://<payload>``.
    The container must be one SGLang takes: MP4/MOV/M4A/3GP, Matroska/WebM, or for audio also WAV, MP3,
    FLAC and Ogg. Streams and lengths are checked when the worker decodes the file.
    """
    noun = "audio" if condition_type == "audio" else "video"
    if uri.startswith("data:"):
        header, separator, payload = uri.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError(f"a data URI must be base64 encoded: data:{noun}/...;base64,<payload>")
    elif uri.startswith("base64://"):
        payload = uri[len("base64://"):]
    else:
        scheme = uri.split(":", 1)[0] if ":" in uri else "a bare path"
        raise ValueError(f"only base64 {noun} is accepted (data:...;base64, or base64://); got {scheme}")
    limit = f"the {noun} exceeds {MAX_MEDIA_BYTES // 2**20} MiB"
    if len(payload) > math.ceil(MAX_MEDIA_BYTES / 3) * 4 + 4:
        raise ValueError(limit)
    payload = payload.strip().replace("-", "+").replace("_", "/")
    try:
        data = base64.b64decode(payload + "=" * (-len(payload) % 4), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"the base64 {noun} payload is invalid") from None
    if not data:
        raise ValueError(f"the base64 {noun} payload is empty")
    if len(data) > MAX_MEDIA_BYTES:
        raise ValueError(limit)
    kind = sniff_container(data)
    if kind is None:
        raise ValueError(f"the {noun} must be in an MP4, MOV, Matroska or WebM container"
                         + (", or WAV, MP3, FLAC or Ogg" if noun == "audio" else ""))
    if noun == "video" and kind in AUDIO_ONLY_CONTAINERS:
        raise ValueError(f"a video reference cannot be a {kind.upper()} file")
    return data, kind


def _soundtrack_facts(container, stream):
    """Channels, sample rate and decoded length of an audio stream: the samples the pipeline will read."""
    rate = stream.codec_context.sample_rate
    if not rate:
        raise ValueError("the audio has no sample rate")
    container.seek(0)
    samples = sum(frame.samples for frame in container.decode(stream))
    return {"channels": stream.codec_context.channels, "sample_rate": rate, "seconds": round(samples / rate, 6)}


def inspect_media(data, condition_type):
    """Fully decode a video or audio reference as the worker's pipeline will (requires PyAV).

    Returns the container, and for a video its display ``width`` and ``height`` (rotation applied), frame
    rate, frame count and length in ``seconds``; ``audio`` is its soundtrack (channels, sample rate, decoded
    length) or None. For an audio reference, ``seconds`` is the decoded length.
    """
    import av

    try:
        with av.open(io.BytesIO(data)) as container:
            names = set(container.format.name.split(","))
            if not names & MEDIA_CONTAINERS:
                raise ValueError(f"the container {container.format.name} is not supported")
            video = container.streams.video[0] if container.streams.video else None
            audio = container.streams.audio[0] if container.streams.audio else None
            facts = {"container": container.format.name.split(",")[0]}
            if condition_type == "audio":
                if audio is None:
                    raise ValueError("the audio reference has no audio stream")
                facts.update(_soundtrack_facts(container, audio))
                return facts
            if video is None:
                raise ValueError("the video reference has no video stream")
            if condition_type == "video_audio" and audio is None:
                raise ValueError("a video_audio reference needs a soundtrack; use type 'video' for a silent video")
            video.thread_type = "AUTO"
            frames, rotation = 0, 0
            for frame in container.decode(video):
                frames += 1
                rotation = getattr(frame, "rotation", 0) or 0
            rate = float(video.average_rate or video.guessed_rate or 0)
            if not frames or rate <= 0:
                raise ValueError("the video has no decodable frames")
            width, height = video.codec_context.width, video.codec_context.height
            if round(rotation / 90) % 2:  # The pipeline turns the frames upright, as ffmpeg displays them.
                width, height = height, width
            facts.update(width=width, height=height, fps=round(rate, 6), frames=frames,
                         seconds=round(frames / rate, 6))
            facts["audio"] = None if audio is None else _soundtrack_facts(container, audio)
            return facts
    except (av.error.FFmpegError, UnicodeDecodeError, IndexError, ZeroDivisionError) as error:
        raise ValueError(f"the {'audio' if condition_type == 'audio' else 'video'} could not be decoded: "
                         f"{error}") from None


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
    """A keyframe (``fl2va``: role ``keyframe`` with a ``frame_index``) or a reference (``ref2va``: role
    ``reference``, no ``frame_index``)."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: str
    uri: str = Field(min_length=1)
    role: str
    frame_index: int | None = None
    start_time_seconds: float | None = None  # SGLang's; parsed only to reject it by name


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
    ``payload`` is what the gateway forwards to a worker; ``media`` are the decoded condition files, in
    condition order. ``canvas`` is ``(height, width)``, or None until the keyframe's size is known.
    """

    identity: dict
    payload: dict
    task: str
    num_frames: int
    canvas: tuple[int, int] | None
    media: list[bytes] = field(default_factory=list)
    legacy: bool = False

    @property
    def condition_types(self):
        return [condition["type"] for condition in self.identity.get("conditions", ())]

    def resolve(self, facts=()):
        """The output profile once every condition is decoded.

        ``facts`` has one entry per condition: ``inspect_image``'s EXIF-upright ``width`` and ``height`` (or a
        ``(width, height)`` pair) for an image, ``inspect_media``'s result for a video or audio reference.
        """
        facts = [{"width": item[0], "height": item[1]} if isinstance(item, (tuple, list)) else item
                 for item in facts]
        canvas = self.canvas
        if self.task == "ref2va":
            errors, totals = [], {"video": 0.0, "audio": 0.0}
            for index, (kind, item) in enumerate(zip(self.condition_types, facts)):
                location = ("conditions", index, "uri")
                noun = "audio" if kind == "audio" else kind.split("_")[0]
                if kind != "audio":
                    width, height = item["width"], item["height"]
                    if width > 4 * height or height > 4 * width:
                        errors.append(_error(location, f"a reference {noun} must be within 1:4 and 4:1, "
                                                       f"got {width}x{height}"))
                if kind == "image":
                    continue
                sound = item if kind == "audio" else item.get("audio")
                if sound and not 1 <= sound["channels"] <= MAX_AUDIO_CHANNELS:
                    errors.append(_error(location, f"reference audio must be mono or stereo, got "
                                                   f"{sound['channels']} channels"))
                seconds = item["seconds"]
                if not MIN_CLIP_SECONDS <= seconds <= MAX_CLIP_SECONDS + 1e-6:
                    errors.append(_error(location, f"a {noun} reference must be {MIN_CLIP_SECONDS:g}-15 s long "
                                                   f"(at most {MAX_CLIP_SECONDS:.3f} s, H3's 15 s), got {seconds:.3f} s"))
                totals[noun] += seconds
            for noun, total in totals.items():
                if total > MAX_CLIP_SECONDS + 1e-6:
                    errors.append(_error(("conditions",), f"{noun} references add up to {total:.3f} s; MiniMax-H3 "
                                                          f"takes at most 15 s ({MAX_CLIP_SECONDS:.3f} s) in total"))
            if errors:
                raise _invalid(*errors)
        if canvas is None:
            width, height = facts[0]["width"], facts[0]["height"]
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
    if task not in TASKS:
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
    low, high = frame_range(task)
    if not math.isfinite(duration) or not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        errors.append(_error(("target", "duration_seconds"),
                             f"duration_seconds must be between {MIN_DURATION_SECONDS:g} and {MAX_DURATION_SECONDS:g}"))
    else:
        num_frames = frames_for_duration(duration)
        if not (low <= num_frames <= high):
            errors.append(_error(("target", "duration_seconds"),
                                 f"{duration:g} s resolves to {num_frames} frames; this deployment serves "
                                 f"{low}-{high} frames ({low / FPS:.3f}-{high / FPS:.3f} s) for {task}"))

    conditions, media = [], []
    for index, condition in enumerate(request.conditions):
        if condition.start_time_seconds is not None:
            errors.append(_error(("conditions", index, "start_time_seconds"),
                                 "start_time_seconds is not supported; send the segment itself"))
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
            media.append(data)
            conditions.append({"type": "image", "role": "keyframe", "frame_index": 0, "format": kind,
                               "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    elif task == "ref2va":
        counts = {"image": 0, "video": 0, "audio": 0}
        for condition in request.conditions:
            kind = condition.type.split("_")[0] if condition.type in REFERENCE_TYPES else None
            if kind:
                counts[kind] += 1
        if not request.conditions:
            errors.append(_error(("conditions",), "ref2va needs at least one reference"))
        elif len(request.conditions) > MAX_REFERENCES:
            errors.append(_error(("conditions",), f"ref2va takes at most {MAX_REFERENCES} references"))
        for kind, limit in (("image", MAX_REFERENCE_IMAGES), ("video", MAX_REFERENCE_VIDEOS),
                            ("audio", MAX_REFERENCE_AUDIOS)):
            if counts[kind] > limit:
                errors.append(_error(("conditions",), f"ref2va takes at most {limit} {kind} references, "
                                                      f"got {counts[kind]}"))
        if counts["audio"] and not (counts["image"] or counts["video"]):
            errors.append(_error(("conditions",), "an audio reference must be paired with at least one image or "
                                                  "video reference"))
        # Order matters: the prompt names references per type in this order, <Picture 1>, <Video 1>, <Audio 1>.
        for index, condition in enumerate(request.conditions[:MAX_REFERENCES]):
            location = ("conditions", index)
            if condition.role != "reference":
                errors.append(_error((*location, "role"), "ref2va conditions must have role 'reference'"))
            if condition.frame_index is not None:
                errors.append(_error((*location, "frame_index"), "a reference is not a keyframe; omit frame_index"))
            if condition.type not in REFERENCE_TYPES:
                errors.append(_error((*location, "type"), f"a reference type must be one of {list(REFERENCE_TYPES)}"))
                continue
            try:
                if condition.type == "image":
                    data, kind = decode_image_uri(condition.uri)
                else:
                    data, kind = decode_media_uri(condition.uri, condition.type)
            except ValueError as error:
                errors.append(_error((*location, "uri"), str(error)))
                continue
            media.append(data)
            conditions.append({"type": condition.type, "role": "reference", "format": kind,
                               "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if errors:
        raise _invalid(*errors)

    seed = DEFAULT_SEED if request.seed is None else request.seed
    resolved_target = {"short_edge": SHORT_EDGE, "aspect_ratio": aspect_ratio, "duration_seconds": float(duration)}
    identity = {"task": task, "prompt": request.prompt, "seed": seed, "target": resolved_target,
                "conditions": conditions}
    forwarded = {"task": task, "prompt": request.prompt, "seed": seed, "target": resolved_target,
                 "conditions": [{"type": c.type, "uri": c.uri, "role": c.role,
                                 **({"frame_index": c.frame_index} if c.frame_index is not None else {})}
                                for c in request.conditions]}
    canvas = None if aspect_ratio == "auto" and task == "fl2va" else resolve_canvas_size(
        *map(int, (aspect_ratio if aspect_ratio != "auto" else DEFAULT_ASPECT_RATIO).split(":")))
    return ParsedRequest(identity=identity, payload=forwarded, task=task, num_frames=num_frames,
                         canvas=canvas, media=media)


def accepts(api, parsed):
    """Whether a worker's published capabilities cover a request."""
    upgraded = isinstance(api, dict) and api.get("version", 0) >= API_VERSION
    if parsed.legacy:  # text to video; every worker before API v2 serves it
        return not upgraded or "t2va" in api.get("tasks", ())
    if not upgraded:
        return False
    frames = api.get("num_frames") or {}
    if parsed.task == "ref2va" and not set(parsed.condition_types) <= set(api.get("reference_types", ("image",))):
        return False  # a Ref2VA worker from before video and audio references
    return parsed.task in api.get("tasks", ()) and frames.get("min", 0) <= parsed.num_frames <= frames.get("max", 0)


def sglang_status(status):
    """SGLang's job status names: ``completed`` instead of ``succeeded``."""
    return "completed" if status == "succeeded" else status


def same_request(record, identity):
    """Whether a stored job was created from the same request identity."""
    return all(record.get(name) == value for name, value in identity.items())


def as_json_detail(detail: Any):
    return {"detail": detail}
