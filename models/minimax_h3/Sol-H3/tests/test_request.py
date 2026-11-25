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


def test_prompts_up_to_16000_characters_are_accepted_in_both_forms():
    longest = "猫" * 8000 + "x" * 8000  # Characters, not bytes or tokens, are what the limit counts.
    assert parse_request({"prompt": longest}).identity["prompt"] == longest
    assert parse_request({"task": "t2va", "prompt": longest}).identity["prompt"] == longest
    for body in ({"prompt": longest + "x"}, {"task": "t2va", "prompt": longest + "x"}):
        assert rejected(body)[0]["type"] == "string_too_long"


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
    assert parsed.canvas is None and parsed.media == [portrait]
    assert parsed.resolve([(1024, 1536)])["width"] == 768 and parsed.resolve([(1024, 1536)])["height"] == 1152
    condition = parsed.identity["conditions"][0]
    assert condition["format"] == "png" and condition["bytes"] == len(portrait) and len(condition["sha256"]) == 64
    assert "uri" not in condition and parsed.payload["conditions"][0]["uri"].startswith("data:image/png")
    explicit = parse_request(fl2va(data_uri(portrait), aspect_ratio="16:9"))
    assert explicit.canvas == (768, 1344)  # Stretched onto 16:9, as SGLang and Diffusers do.
    assert parse_request(fl2va("base64://" + base64.urlsafe_b64encode(portrait).decode().rstrip("="))).media == [portrait]
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


def ref2va(*uris, **target):
    body = {"task": "ref2va", "prompt": "<Subject 1> from <Picture 1> meets <Subject 2> from <Picture 2>.",
            "conditions": [{"type": "image", "uri": uri, "role": "reference"} for uri in uris]}
    if target:
        body["target"] = target
    return body


def test_ref2va_takes_ordered_image_references():
    person, toy = image_bytes(1024, 1024), image_bytes(640, 960, "JPEG")
    parsed = parse_request(ref2va(data_uri(person), data_uri(toy, "image/jpeg")))
    assert parsed.task == "ref2va" and parsed.media == [person, toy]  # request order is the <Picture n> order
    assert parsed.canvas == (768, 1344) and parsed.num_frames == 124  # 16:9 and 5 s, as for t2va
    assert [c["role"] for c in parsed.identity["conditions"]] == ["reference", "reference"]
    assert [c["format"] for c in parsed.identity["conditions"]] == ["png", "jpg"]
    assert all("uri" not in c and "frame_index" not in c for c in parsed.identity["conditions"])
    assert [set(c) for c in parsed.payload["conditions"]] == [{"type", "uri", "role"}] * 2
    assert parse_request(parsed.payload).identity == parsed.identity  # what the worker re-parses
    assert parsed.resolve([(1024, 1024), (640, 960)])["width"] == 1344
    # References are not frames: 'auto' is the Ref2VA pipeline's own 16:9 default.
    assert parse_request(ref2va(data_uri(person), aspect_ratio="auto")).canvas == (768, 1344)
    assert parse_request(ref2va(data_uri(person), aspect_ratio="9:16", duration_seconds=10)).canvas == (1344, 768)
    with pytest.raises(RequestError) as wide:
        parse_request(ref2va(data_uri(person), data_uri(toy))).resolve([(1024, 1024), (1000, 200)])
    assert wide.value.detail[0]["loc"] == ["body", "conditions", 1, "uri"]
    uri = data_uri(person)
    assert "at least one" in rejected(ref2va())[0]["msg"]
    assert "at most 9 image" in rejected(ref2va(*[uri] * 10))[0]["msg"]
    for change, message in ((("role", "keyframe"), "role 'reference'"), (("frame_index", 0), "omit frame_index"),
                            (("type", "text"), "must be one of"), (("type", "video"), "container"),
                            (("start_time_seconds", 1.0), "not supported"),
                            (("uri", "https://example.com/a.png"), "base64")):
        body = ref2va(uri)
        body["conditions"][0][change[0]] = change[1]
        assert message in rejected(body)[0]["msg"]
    # A first-frame request is still the fl2va form, and t2va still takes no conditions.
    assert parse_request(fl2va(uri)).payload["conditions"][0]["frame_index"] == 0
    rejected({**ref2va(uri), "task": "t2va"})


def reference(kind, data, mime):
    return {"type": kind, "role": "reference", "uri": data_uri(data, mime)}


def test_ref2va_takes_video_and_audio_references_within_minimax_limits():
    import media_samples

    pytest.importorskip("av")
    picture, clip, voice = image_bytes(512, 512), media_samples.mp4(3), media_samples.wav(4)
    body = {"task": "ref2va", "prompt": "<Picture 1> speaks like <Audio 2> and moves like <Video 1>.",
            "conditions": [reference("image", picture, "image/png"), reference("video", clip, "video/mp4"),
                           reference("audio", voice, "audio/wav")], "target": {"duration_seconds": 15}}
    parsed = parse_request(body)
    assert parsed.media == [picture, clip, voice] and parsed.condition_types == ["image", "video", "audio"]
    assert [c["format"] for c in parsed.identity["conditions"]] == ["png", "mp4", "wav"]
    assert parse_request(parsed.payload).identity == parsed.identity  # what the worker re-parses
    # MiniMax-H3's 4-15 s: 4 s is 107 frames and 15 s is 362 for ref2va; T2V keeps 124-345.
    assert parsed.num_frames == 362
    assert parse_request({**body, "target": {"duration_seconds": 4}}).num_frames == 107
    for seconds in (4, 15):
        assert "124-345" in rejected({"task": "t2va", "prompt": "x", "target": {"duration_seconds": seconds}})[0]["msg"]
    facts = [{"width": 512, "height": 512}] + [h3_request.inspect_media(data, kind)
                                               for data, kind in ((clip, "video"), (voice, "audio"))]
    assert facts[1]["audio"]["channels"] == 2 and facts[2]["seconds"] == 4.0
    assert parsed.resolve(facts)["num_frames"] == 362
    # Counts: 9 images, 3 videos and 3 audio clips, 12 in all, and audio never alone.
    image, video, audio = body["conditions"]
    for conditions, message in (([image] * 2 + [video] * 4, "at most 3 video"), ([image] + [audio] * 4, "at most 3 audio"),
                                ([image] * 9 + [video] * 3 + [audio], "at most 12 references"),
                                ([audio], "paired with at least one image or video")):
        assert message in rejected({**body, "conditions": conditions})[0]["msg"]
    assert parse_request({**body, "conditions": [{**video, "type": "video_audio"}, audio]}).condition_types == [
        "video_audio", "audio"]
    for kind, data, message in (("video", voice, "cannot be a WAV"), ("audio", b"not a media file", "container"),
                                ("video", b"x" * (h3_request.MAX_MEDIA_BYTES + 1), "64 MiB")):
        assert message in rejected({**body, "conditions": [image, reference(kind, data, "video/mp4")]})[0]["msg"]
    # Lengths, shapes and channels come from the worker's decode: 2-15 s each (15.083 s on H3's grid), 15 s per type.
    def problems(*items):
        kinds = ["image"] + [kind for kind, _ in items]
        request = parse_request({**body, "conditions": [image] + [video if kind == "video" else audio
                                                                    for kind in kinds[1:]]})
        with pytest.raises(RequestError) as error:
            request.resolve([{"width": 512, "height": 512}, *[item for _, item in items]])
        return " | ".join(entry["msg"] for entry in error.value.detail)

    clip_facts = {"width": 1280, "height": 720, "seconds": 5.0, "audio": {"channels": 2}}
    assert "2-15 s" in problems(("video", {**clip_facts, "seconds": 1.5}))
    assert "2-15 s" in problems(("video", {**clip_facts, "seconds": 15.2}))
    parse_request({**body, "conditions": [image, video]}).resolve([facts[0], {**clip_facts, "seconds": 15.042}])
    assert "add up to 16.000 s" in problems(("video", {**clip_facts, "seconds": 8.0}), ("video", {**clip_facts, "seconds": 8.0}))
    assert "within 1:4 and 4:1" in problems(("video", {**clip_facts, "width": 2000, "height": 400}))
    assert "mono or stereo" in problems(("video", {**clip_facts, "audio": {"channels": 6}}))
    assert "mono or stereo" in problems(("audio", {"seconds": 5.0, "channels": 6}))
    assert "add up to 16.000 s" in problems(("audio", {"seconds": 8.0, "channels": 1}), ("audio", {"seconds": 8.0, "channels": 1}))


def test_inspect_media_reads_what_the_pipeline_decodes():
    import media_samples

    pytest.importorskip("av")
    clip = h3_request.inspect_media(media_samples.mp4(2.5, 48, 208, audio_channels=1), "video")
    assert (clip["width"], clip["height"], clip["frames"], clip["seconds"]) == (48, 208, 60, 2.5)
    assert clip["audio"]["channels"] == 1 and clip["audio"]["seconds"] == pytest.approx(2.5, abs=0.1)  # AAC pads to whole frames
    silent = media_samples.mp4(3, audio_channels=0)
    assert h3_request.inspect_media(silent, "video")["audio"] is None
    assert h3_request.inspect_media(media_samples.mp4(3), "audio")["channels"] == 2  # a video's soundtrack
    assert h3_request.inspect_media(media_samples.wav(3, channels=6), "audio")["channels"] == 6
    for data, kind, message in ((silent, "video_audio", "needs a soundtrack"), (media_samples.wav(3), "video",
                                                                               "no video stream"),
                                (silent, "audio", "no audio stream"), (b"\0\0\0\x20ftypisom" + b"\0" * 64, "video",
                                                                       "could not be decoded")):
        with pytest.raises(ValueError, match=message):
            h3_request.inspect_media(data, kind)
    assert [h3_request.sniff_container(data) for data in (silent, media_samples.wav(1), b"ID3\x04", b"fLaC",
                                                          b"OggS", b"\x1aE\xdf\xa3....webm", b"GIF89a")] == [
        "mp4", "wav", "mp3", "flac", "ogg", "webm", None]


def test_capabilities_follow_the_worker_tasks():
    t2v = h3_request.capabilities()
    assert t2v["tasks"] == ["t2va", "fl2va"] and t2v["keyframe_frame_indices"] == [[0]]
    assert list(t2v) == ["version", "tasks", "aspect_ratios", "short_edges", "keyframe_frame_indices", "fps",
                         "num_frames", "image_formats", "max_image_bytes"]  # unchanged for T2V workers
    assert t2v["num_frames"] == {"min": 124, "max": 345, "step": 17}
    ref = h3_request.capabilities(tasks=h3_request.WORKER_TASKS["ref2va"])
    assert ref["tasks"] == ["ref2va"] and "keyframe_frame_indices" not in ref
    assert ref["num_frames"] == {"min": 107, "max": 362, "step": 17}
    assert ref["reference_types"] == ["image", "video", "video_audio", "audio"] and ref["max_references"] == 12
    assert ref["max_references_by_type"] == {"image": 9, "video": 3, "audio": 3}
    assert ref["reference_clip_seconds"] == {"min": 2.0, "max": 15.083333, "total_per_type": 15.083333}
    assert ref["max_body_bytes"] == 128 * 2**20
    parsed_ref = parse_request(ref2va(data_uri(image_bytes(64, 64))))
    parsed_t2v = parse_request({"task": "t2va", "prompt": "x"})
    legacy = parse_request({"prompt": "x"})
    assert h3_request.accepts(ref, parsed_ref) and not h3_request.accepts(t2v, parsed_ref)
    assert h3_request.accepts(t2v, parsed_t2v) and not h3_request.accepts(ref, parsed_t2v)
    assert h3_request.accepts(t2v, legacy) and h3_request.accepts(None, legacy)  # pre-v2 workers serve it
    assert not h3_request.accepts(ref, legacy)
    # A Ref2VA worker from before video and audio references (images, 124-345 frames) gets only what it runs.
    older = {**ref, "reference_types": ["image"], "num_frames": {"min": 124, "max": 345, "step": 17}}
    assert h3_request.accepts(older, parsed_ref)
    audio_request = parse_request({**ref2va(data_uri(image_bytes(64, 64))), "conditions": [
        ref2va(data_uri(image_bytes(64, 64)))["conditions"][0],
        {"type": "audio", "role": "reference", "uri": data_uri(b"RIFF\0\0\0\0WAVEfmt ", "audio/wav")}]})
    assert h3_request.accepts(ref, audio_request) and not h3_request.accepts(older, audio_request)
    fifteen = parse_request(ref2va(data_uri(image_bytes(64, 64)), duration_seconds=15))
    assert h3_request.accepts(ref, fifteen) and not h3_request.accepts(older, fifteen)


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
