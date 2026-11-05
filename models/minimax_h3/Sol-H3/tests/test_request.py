"""The SGLang-form request contract: tasks, targets, keyframes, and their limits."""
import base64
import io

import pytest

import h3_request
from h3_request import RequestError, parse_request

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def image_bytes(width, height, kind="PNG", **save):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(buffer, kind, **save)
    return buffer.getvalue()


def data_uri(data, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def fl2va(uri, **target):
    body = {"task": "fl2va", "prompt": "The frame comes alive.",
            "conditions": [{"type": "image", "uri": uri, "role": "keyframe", "frame_index": 0}]}
    if target:
        body["target"] = target
    return body


def rejected(body):
    with pytest.raises(RequestError) as error:
        parse_request(body)
    assert error.value.status_code == 422
    return error.value.detail


def test_legacy_requests_are_unchanged():
    parsed = parse_request({"prompt": " A fox. ", "duration": 10, "seed": 3})
    assert parsed.legacy and parsed.identity == {"prompt": "A fox.", "duration": 10, "seed": 3}
    assert parsed.num_frames == 243 and parsed.canvas == (768, 1344)
    for invalid in ({"duration": 15}, {"duration": "5"}, {"seed": True}, {"prompt": " "}, {"steps": 4}):
        rejected({"prompt": "x", **invalid})


def test_t2va_defaults_to_16_9_and_five_seconds():
    parsed = parse_request({"task": "t2va", "prompt": "A fox."})
    assert parsed.resolve() == {"width": 1344, "height": 768, "num_frames": 124, "fps": 24,
                                "duration_seconds": round(124 / 24, 6)}
    assert parsed.identity == {"task": "t2va", "prompt": "A fox.", "seed": 42, "conditions": [],
                               "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0}}


@pytest.mark.parametrize("ratio,canvas", [("21:9", (672, 1536)), ("16:9", (768, 1344)), ("4:3", (768, 1024)),
                                          ("1:1", (768, 768)), ("3:4", (1024, 768)), ("9:16", (1344, 768)),
                                          ("auto", (768, 1344))])
def test_aspect_ratios_use_minimax_geometry(ratio, canvas):
    parsed = parse_request({"task": "t2va", "prompt": "x", "target": {"aspect_ratio": ratio}})
    assert parsed.canvas == canvas


def test_every_vae_length_from_124_to_345_frames_is_reachable():
    for frames in h3_request.VALID_NUM_FRAMES:
        body = {"task": "t2va", "prompt": "x", "target": {"duration_seconds": frames / 24}}
        assert parse_request(body).num_frames == frames
    assert parse_request({"task": "t2va", "prompt": "x", "target": {"duration_seconds": 8}}).num_frames == 192
    assert parse_request({"task": "t2va", "prompt": "x", "target": {"duration_seconds": 4.5}}).num_frames == 124
    for seconds in (3.9, 4.4, 14.4, 15.0, 15.1, float("nan")):
        rejected({"task": "t2va", "prompt": "x", "target": {"duration_seconds": seconds}})


def test_sglang_transport_fields_and_unsupported_sampler_settings():
    parsed = parse_request({"task": "t2va", "prompt": "x", "model": "MiniMaxAI/MiniMax-H3", "seconds": 10,
                            "num_outputs_per_prompt": 1, "seed": None})
    assert parsed.num_frames == 243 and parsed.identity["seed"] == 42
    target_wins = {"task": "t2va", "prompt": "x", "seconds": 5, "target": {"duration_seconds": 14.375}}
    assert parse_request(target_wins).num_frames == 345
    for field, value in (("num_inference_steps", 50), ("flow_shift", 12.0), ("audio_flow_shift", 3.0),
                         ("guidance_scale", 1.0), ("negative_prompt", "blur")):
        detail = rejected({"task": "t2va", "prompt": "x", field: value})
        assert detail[0]["loc"] == ["body", field] and "not supported" in detail[0]["msg"]
    for invalid in ({"task": "ref2va"}, {"task": "i2v"}, {"n": 2}, {"target": {"short_edge": 480}},
                    {"target": {"aspect_ratio": ""}}, {"target": {"aspect_ratio": "2:1"}},
                    {"target": {"width": 1344}}, {"unexpected": True}, {"seed": -1}, {"seed": 1.5}):
        rejected({"task": "t2va", "prompt": "x", **invalid})


def test_fl2va_first_frame_follows_the_image_and_rejects_other_keyframes():
    portrait = image_bytes(1024, 1536)
    parsed = parse_request(fl2va(data_uri(portrait)))
    assert parsed.canvas is None and parsed.images == [portrait]
    assert parsed.resolve([(1024, 1536)])["width"] == 768 and parsed.resolve([(1024, 1536)])["height"] == 1152
    condition = parsed.identity["conditions"][0]
    assert condition["format"] == "png" and condition["bytes"] == len(portrait) and len(condition["sha256"]) == 64
    assert "uri" not in condition and parsed.payload["conditions"][0]["uri"].startswith("data:image/png")
    explicit = parse_request(fl2va(data_uri(portrait), aspect_ratio="16:9"))
    assert explicit.canvas == (768, 1344)  # Stretched onto 16:9, as SGLang and Diffusers do.
    assert parse_request(fl2va("base64://" + base64.urlsafe_b64encode(portrait).decode().rstrip("="))).images == [portrait]
    with pytest.raises(RequestError):
        parse_request(fl2va(data_uri(image_bytes(1000, 200)))).resolve([(1000, 200)])
    uri = data_uri(portrait)
    for body in (fl2va(uri) | {"conditions": []},
                 fl2va(uri) | {"conditions": [fl2va(uri)["conditions"][0]] * 2},
                 {**fl2va(uri), "task": "t2va"}):
        rejected(body)
    for change, message in ((("frame_index", -1), "last-frame"), (("frame_index", 5), "must be 0"),
                            (("role", "reference"), "keyframe"), (("type", "video"), "image")):
        body = fl2va(uri)
        body["conditions"][0][change[0]] = change[1]
        assert message in rejected(body)[0]["msg"]


@pytest.mark.parametrize("uri", ["https://example.com/a.png", "file:///tmp/a.png", "/tmp/a.png",
                                 "data:image/png,notbase64", "data:text/plain;base64,aGVsbG8=",
                                 "data:image/png;base64,***", "data:image/png;base64," + base64.b64encode(b"GIF89a").decode()])
def test_only_base64_png_jpeg_webp_keyframes_are_accepted(uri):
    detail = rejected(fl2va(uri))
    assert detail[0]["loc"] == ["body", "conditions", 0, "uri"]
    assert uri not in str(detail)


def test_image_limits_and_exif_orientation():
    big = b"\x89PNG\r\n\x1a\n" + bytes(h3_request.MAX_IMAGE_BYTES)
    rejected(fl2va(data_uri(big)))
    for kind, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
        info = h3_request.inspect_image(image_bytes(640, 360, kind))
        assert (info["width"], info["height"]) == (640, 360)
    exif = Image.Exif()
    exif[0x0112] = 6  # Rotated 90 degrees: the upright image is portrait.
    rotated = image_bytes(640, 360, "JPEG", exif=exif.tobytes())
    assert (h3_request.inspect_image(rotated)["width"], h3_request.inspect_image(rotated)["height"]) == (360, 640)
    with pytest.raises(ValueError):
        h3_request.inspect_image(image_bytes(640, 360)[:200])
    with pytest.raises(ValueError):
        h3_request.inspect_image(image_bytes(16, 16))


def test_canvas_matches_the_diffusers_pipeline():
    packing = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.packing")
    for width in range(100, 4000, 37):
        for height in (100, 333, 768, 1080, 1500, 2600):
            if 0.25 <= width / height <= 4:
                assert h3_request.resolve_canvas_size(width, height) == packing.resolve_canvas_size(width, height)
    for frames in range(100, 400):
        assert h3_request.align_num_frames(frames) == packing.align_num_frames(frames)
