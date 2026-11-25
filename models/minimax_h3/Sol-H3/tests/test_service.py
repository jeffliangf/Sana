"""Exercise HTTP admission, durable jobs, and restart behavior without GPUs."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
from uuid import uuid4

from fastapi.testclient import TestClient


spec = importlib.util.spec_from_file_location("h3_serve", Path(__file__).resolve().parents[1] / "serve.py")
service = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = service
spec.loader.exec_module(service)
TOKEN = "test-only-" + "x" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make_client(tmp_path, capacity=2):
    state = service.WorkerState(tmp_path, capacity=capacity, min_free_gib=0)
    return state, TestClient(service.create_app(state, TOKEN))


def test_auth_readiness_validation_and_queue_limit(tmp_path):
    state, client = make_client(tmp_path, capacity=1)
    payload = {"prompt": "A bird sings.", "duration": 5, "seed": 0}
    assert client.get("/healthz").status_code == 503
    assert client.post("/v1/videos", json=payload).status_code == 401
    assert client.post("/v1/videos", json=payload, headers=AUTH).status_code == 503
    state.phase = "ready"
    for invalid in ({"duration": 15}, {"duration": "5"}, {"seed": True}, {"seed": -1},
                    {"prompt": " "}, {"prompt": "x" * 16001}, {"unexpected": "field"}):
        assert client.post("/v1/videos", json={**payload, **invalid}, headers=AUTH).status_code == 422
    first = client.post("/v1/videos", json=payload, headers=AUTH)
    assert first.status_code == 202
    second = client.post("/v1/videos", json=payload, headers=AUTH)
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "10"
    job = first.json()
    assert client.get(job["status_url"]).status_code == 401
    assert client.get(job["status_url"], headers=AUTH).json()["status"] == "queued"
    assert client.get(job["video_url"], headers=AUTH).status_code == 409
    assert client.get("/v1/jobs/not-a-uuid", headers=AUTH).status_code == 404


def test_job_id_header_returns_the_saved_job_across_full_queues_and_restarts(tmp_path):
    state, client = make_client(tmp_path, capacity=1)
    state.phase = "ready"
    payload = {"prompt": "A bird sings.", "duration": 5, "seed": 0}
    keyed = {**AUTH, "Idempotency-Key": str(uuid4())}
    assert client.get("/healthz").json()["idempotency_key"] is True
    first = client.post("/v1/videos", json=payload, headers=keyed)
    repeat = client.post("/v1/videos", json=payload, headers=keyed)  # Queue is now full.
    assert first.status_code == repeat.status_code == 202
    assert first.json()["id"] == repeat.json()["id"] == keyed["Idempotency-Key"]
    assert state.pending.qsize() == 1
    assert client.post("/v1/videos", json={**payload, "seed": 1}, headers=keyed).status_code == 409
    assert client.post("/v1/videos", json=payload, headers={**AUTH, "Idempotency-Key": "bad"}).status_code == 400
    _, restarted = make_client(tmp_path)  # Still loading, so only an existing job can be returned.
    again = restarted.post("/v1/videos", json=payload, headers=keyed)
    assert again.status_code == 202
    assert again.json()["status"] == "failed" and again.json()["error_code"] == "worker_restarted"


def test_completed_media_and_interrupted_jobs_survive_restart(tmp_path):
    state, client = make_client(tmp_path)
    state.phase = "ready"
    done = state.submit(service.GenerateRequest(prompt="done"))
    pending = state.submit(service.GenerateRequest(prompt="pending"))
    state.update(done["id"], status="succeeded")
    path = tmp_path / f"{done['id']}.mp4"
    path.write_bytes(b"test-mp4")
    state2, client2 = make_client(tmp_path)
    assert state2.get(pending["id"])["status"] == "failed"
    assert state2.get(done["id"])["status"] == "succeeded"
    response = client2.get(done["video_url"], headers=AUTH)
    assert response.content == b"test-mp4"
    assert response.headers["content-type"] == "video/mp4"
    path.unlink()
    assert client2.get(done["video_url"], headers=AUTH).status_code == 410


def test_concurrent_admission_never_exceeds_capacity(tmp_path):
    state, client = make_client(tmp_path, capacity=3)
    state.phase = "ready"
    def submit(index):
        return client.post("/v1/videos", json={"prompt": f"request {index}"}, headers=AUTH).status_code
    with ThreadPoolExecutor(max_workers=12) as pool:
        codes = list(pool.map(submit, range(12)))
    assert codes.count(202) == 3
    assert codes.count(429) == 9
    assert len(list(tmp_path.glob("*.json"))) == 3
    state.fail_pending()
    assert state.pending.empty()
    assert all(state.get(p.stem)["status"] == "failed" for p in tmp_path.glob("*.json"))


def test_disk_pressure_and_body_limit(tmp_path):
    state, client = make_client(tmp_path)
    state.phase = "ready"
    state.min_free_bytes = 2**70
    assert client.post("/v1/videos", json={"prompt": "test"}, headers=AUTH).status_code == 507
    assert state.max_body_bytes == 28 * 2**20  # a T2V worker: one 20 MiB first frame
    assert client.post("/v1/videos", content=b"x" * (state.max_body_bytes + 1), headers=AUTH).status_code == 413
    assert client.post("/v1/videos", content=b"not json", headers=AUTH).status_code == 422
    assert not list(tmp_path.glob("*.json"))


def keyframe_request(width=1024, height=1536, seed=0, **target):
    from PIL import Image
    import base64
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (seed % 255, 90, 160)).save(buffer, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    body = {"task": "fl2va", "prompt": "The still frame starts to move.", "seed": 5,
            "conditions": [{"type": "image", "uri": uri, "role": "keyframe", "frame_index": 0}]}
    return {**body, "target": target} if target else body, buffer.getvalue()


def test_sglang_form_t2va_and_first_frame_jobs(tmp_path):
    import pytest
    pytest.importorskip("PIL")
    state, client = make_client(tmp_path, capacity=4)
    state.phase = "ready"
    health = client.get("/healthz").json()
    assert health["api"]["tasks"] == ["t2va", "fl2va"] and health["api"]["num_frames"]["max"] == 345
    t2va = client.post("/v1/videos", json={"task": "t2va", "prompt": "A fox.",
                                           "target": {"aspect_ratio": "9:16", "duration_seconds": 14.375}},
                       headers=AUTH)
    assert t2va.status_code == 202
    assert t2va.json()["output"] == {"width": 768, "height": 1344, "num_frames": 345, "fps": 24,
                                     "duration_seconds": 14.375}
    body, image = keyframe_request()
    key = str(uuid4())
    first = client.post("/v1/videos", json=body, headers={**AUTH, "Idempotency-Key": key})
    assert first.status_code == 202, first.text
    job = first.json()
    assert job["task"] == "fl2va" and job["output"]["width"] == 768 and job["output"]["height"] == 1152
    assert job["condition_sizes"] == [{"width": 1024, "height": 1536}]
    assert "uri" not in json_text(job)
    assert (tmp_path / f"{key}.keyframe0.png").read_bytes() == image
    repeat = client.post("/v1/videos", json=body, headers={**AUTH, "Idempotency-Key": key})
    assert repeat.status_code == 202 and repeat.json()["id"] == key and state.pending.qsize() == 2
    other, _ = keyframe_request(seed=9)
    assert client.post("/v1/videos", json=other, headers={**AUTH, "Idempotency-Key": key}).status_code == 409
    for invalid in ({**body, "num_inference_steps": 50}, {**body, "task": "ref2va"},
                    keyframe_request(width=1600, height=200)[0]):
        response = client.post("/v1/videos", json=invalid, headers=AUTH)
        assert response.status_code == 422, response.text
    capped = service.WorkerState(tmp_path / "capped", capacity=2, min_free_gib=0, max_num_frames=243)
    capped.phase = "ready"
    with TestClient(service.create_app(capped, TOKEN)) as small:
        long = {"task": "t2va", "prompt": "x", "target": {"duration_seconds": 12}}
        assert small.post("/v1/videos", json=long, headers=AUTH).status_code == 422
        assert small.get("/healthz").json()["api"]["num_frames"]["max"] == 243


def json_text(value):
    import json
    return json.dumps(value)


def test_ref2va_worker_admits_only_ordered_image_references(tmp_path):
    import base64
    import io

    import pytest
    pytest.importorskip("PIL")
    from PIL import Image

    def png(width, height, color):
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), color).save(buffer, "PNG")
        return buffer.getvalue()

    images = [png(512, 512, (200, 40, 40)), png(400, 600, (40, 200, 40))]
    body = {"task": "ref2va", "prompt": "<Picture 1> and <Picture 2>.", "seed": 7,
            "conditions": [{"type": "image", "role": "reference",
                            "uri": "data:image/png;base64," + base64.b64encode(data).decode()} for data in images]}
    state = service.WorkerState(tmp_path, capacity=4, min_free_gib=0, tasks=("ref2va",))
    state.phase = "ready"
    with TestClient(service.create_app(state, TOKEN)) as client:
        api = client.get("/healthz").json()["api"]
        assert api["tasks"] == ["ref2va"] and api["reference_types"] == ["image", "video", "video_audio", "audio"]
        assert api["num_frames"]["min"] == 107 and api["num_frames"]["max"] == 362
        assert state.max_body_bytes == 128 * 2**20
        for other in ({"prompt": "A bird sings.", "duration": 5}, {"task": "t2va", "prompt": "x"}):
            response = client.post("/v1/videos", json=other, headers=AUTH)
            assert response.status_code == 422 and "ref2va" in response.text
        key = str(uuid4())
        response = client.post("/v1/videos", json=body, headers={**AUTH, "Idempotency-Key": key})
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["task"] == "ref2va" and job["output"]["width"] == 1344 and job["output"]["num_frames"] == 124
        assert job["condition_sizes"] == [{"width": 512, "height": 512}, {"width": 400, "height": 600}]
        assert "base64" not in json_text(job)
        reordered = {**body, "conditions": body["conditions"][::-1]}
        assert client.post("/v1/videos", json=reordered, headers={**AUTH, "Idempotency-Key": key}).status_code == 409
    gpu_job = service.job_from_record(state.next_command(), tmp_path)
    assert gpu_job["image"] is None and [(kind, Path(p).name) for kind, p in gpu_job["references"]] == [
        ("image", f"{key}.keyframe0.png"), ("image", f"{key}.keyframe1.png")]
    assert [Path(p).read_bytes() for _, p in gpu_job["references"]] == images
    reference = service.warmup_reference()
    assert reference.size == (512, 512) and reference.mode == "RGB"
    # A T2V worker keeps its contract and turns Ref2VA away.
    t2v, t2v_client = make_client(tmp_path / "t2v")
    t2v.phase = "ready"
    assert t2v_client.post("/v1/videos", json=body, headers=AUTH).status_code == 422
    assert service.job_from_record({"id": "x", "prompt": "p", "seed": 1, "duration": 5}, tmp_path)["references"] is None


def test_ref2va_worker_decodes_video_and_audio_references_at_admission(tmp_path):
    import base64

    import pytest
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    import media_samples
    from PIL import Image
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), (90, 90, 200)).save(buffer, "PNG")
    picture, clip, voice = buffer.getvalue(), media_samples.mp4(3), media_samples.wav(4)

    def ref(kind, data):  # images keep their image/* rule; video and audio are read from their bytes
        mime = "image/png" if kind == "image" else "application/octet-stream"
        return {"type": kind, "role": "reference", "uri": f"data:{mime};base64," + base64.b64encode(data).decode()}

    body = {"task": "ref2va", "prompt": "<Picture 1> talks like <Audio 2> in the room of <Video 1>.", "seed": 3,
            "target": {"duration_seconds": 15, "aspect_ratio": "9:16"},
            "conditions": [ref("image", picture), ref("video_audio", clip), ref("audio", voice)]}
    state = service.WorkerState(tmp_path, capacity=4, min_free_gib=0, tasks=("ref2va",))
    state.phase = "ready"
    with TestClient(service.create_app(state, TOKEN)) as client:
        key = str(uuid4())
        response = client.post("/v1/videos", json=body, headers={**AUTH, "Idempotency-Key": key})
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["output"] == {"width": 768, "height": 1344, "num_frames": 362, "fps": 24, "duration_seconds": 15.083333}
        sizes = job["condition_sizes"]
        assert sizes[0] == {"width": 512, "height": 512}
        assert (sizes[1]["width"], sizes[1]["height"], sizes[1]["frames"], sizes[1]["seconds"]) == (64, 48, 72, 3.0)
        assert sizes[1]["audio"]["channels"] == 2 and sizes[2]["channels"] == 1 and sizes[2]["seconds"] == 4.0
        # Only a decode finds these: too short, too many channels, a silent video_audio, a broken file.
        for index, data, kind, message in ((1, media_samples.mp4(1.5), "video", "2-15 s"),
                                           (2, media_samples.wav(4, channels=6), "audio", "mono or stereo"),
                                           (1, media_samples.mp4(3, audio_channels=0), "video_audio", "soundtrack"),
                                           (1, b"\0\0\0\x20ftypisom" + b"\0" * 64, "video", "could not be decoded")):
            conditions = list(body["conditions"])
            conditions[index] = ref(kind, data)
            response = client.post("/v1/videos", json={**body, "conditions": conditions}, headers=AUTH)
            assert response.status_code == 422 and message in response.text, response.text
            assert response.json()["detail"][0]["loc"][:3] == ["body", "conditions", index]
    gpu_job = service.job_from_record(state.next_command(), tmp_path)
    assert [(kind, Path(path).name) for kind, path in gpu_job["references"]] == [
        ("image", f"{key}.keyframe0.png"), ("video_audio", f"{key}.keyframe1.mp4"), ("audio", f"{key}.keyframe2.wav")]
    assert [Path(path).read_bytes() for _, path in gpu_job["references"]] == [picture, clip, voice]
    assert gpu_job["output"]["num_frames"] == 362
