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
                    {"prompt": " "}, {"prompt": "x" * 8001}, {"unexpected": "field"}):
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
    assert client.post("/v1/videos", content=b"x" * (service.MAX_BODY_BYTES + 1), headers=AUTH).status_code == 413
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
