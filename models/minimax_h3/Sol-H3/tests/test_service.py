"""Exercise HTTP admission, durable jobs, and restart behavior without GPUs."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys

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
    assert client.post("/v1/videos", content=b"x" * 32769, headers=AUTH).status_code == 413
    assert not list(tmp_path.glob("*.json"))
