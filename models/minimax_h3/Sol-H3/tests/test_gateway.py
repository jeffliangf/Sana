"""Verify stable jobs and downloads while the worker pool changes."""
import asyncio
import collections
import importlib.util
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("h3_gateway_test", Path(__file__).resolve().parents[1] / "gateway.py")
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)
TOKEN = "gateway-test-" + "x" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
URLS = ["http://worker-a:18080", "http://worker-b:18080", "http://worker-c:18080"]


class Fleet:
    def __init__(self):
        self.calls = []
        self.jobs = {}
        self.errors = {}
        self.media = b"example-mp4"
        self.expected_bytes = len(self.media)

    async def handle(self, request):
        endpoint = str(request.url).split("/v1/")[0].removesuffix("/healthz")
        self.calls.append((request.method, endpoint, request.url.path))
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ready", "gpus": 2, "queued": 0,
                                             "queue_capacity": 8, "active_job": None})
        assert request.headers["Authorization"] == AUTH["Authorization"]
        if request.method == "POST":
            error = self.errors.get(endpoint)
            if isinstance(error, Exception):
                raise error
            if error:
                return httpx.Response(error)
            await asyncio.sleep(0.005)
            identifier = str(uuid4())
            value = {"id": identifier, "status": "queued", **json.loads(request.content)}
            self.jobs[identifier] = {"endpoint": endpoint, "value": value}
            return httpx.Response(202, json=value)
        identifier = request.url.path.split("/")[3]
        job = self.jobs[identifier]
        assert job["endpoint"] == endpoint, "Job must stay on the accepting worker"
        if request.url.path.endswith("/video"):
            return httpx.Response(200, content=self.media)
        return httpx.Response(200, json={**job["value"], "status": "succeeded", "bytes": self.expected_bytes})


def make_gateway(tmp_path, fleet):
    registry = tmp_path / "registry/node"
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "node.json").write_text(json.dumps({"updated_at": time.time(), "status": "ready",
        "workers": [{"endpoint": url, "disabled": False} for url in URLS]}))
    client = httpx.AsyncClient(transport=httpx.MockTransport(fleet.handle))
    return gateway.Gateway(registry.parent, tmp_path / "state", TOKEN, client=client,
                           min_free_gib=0, refresh_seconds=0.01)


def test_concurrent_requests_spread_across_workers(tmp_path):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        values = await asyncio.gather(*(router.submit(gateway.GenerateRequest(prompt=f"request {i}")) for i in range(6)))
        assert len({v["id"] for v in values}) == 6
        assert collections.Counter(x["endpoint"] for x in fleet.jobs.values()) == {url: 2 for url in URLS}
        assert all(router.load(v["id"])["worker"] == fleet.jobs[v["id"]]["endpoint"] for v in values)
        await router.client.aclose()
    asyncio.run(run())


def test_restart_and_pool_changes_preserve_inflight_jobs_and_cached_downloads(tmp_path):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        value = await router.submit(gateway.GenerateRequest(prompt="original"))
        identifier = value["id"]
        original_worker = router.load(identifier)["worker"]
        await router.client.aclose()
        restarted = make_gateway(tmp_path, fleet)
        restarted.pool_file.write_text(json.dumps({"excluded_workers": [original_worker]}))
        await restarted.refresh_pool()
        assert original_worker not in restarted.ready
        await restarted.refresh_job(identifier)
        assert restarted.public(restarted.load(identifier))["status"] == "succeeded"
        assert (restarted.root / "videos" / f"{identifier}.mp4").read_bytes() == fleet.media
        assert len([c for c in fleet.calls if c[0] == "POST"]) == 1
        restarted.pool_file.write_text('{"workers": []}')
        await restarted.refresh_pool()
        assert restarted.health()["status"] == "unavailable"
        assert restarted.public(restarted.load(identifier))["status"] == "succeeded"
        await restarted.client.aclose()
    asyncio.run(run())


def test_pool_discovery_tracks_added_removed_disabled_and_stale_workers(tmp_path):
    async def run():
        router = make_gateway(tmp_path, Fleet())
        await router.refresh_pool()
        assert router.health()["workers_ready"] == 3
        path = router.registry / "node/node.json"
        value = json.loads(path.read_text())
        value["workers"][0]["disabled"] = True
        value["workers"].pop()
        path.write_text(json.dumps(value))
        await router.refresh_pool()
        assert list(router.ready) == [URLS[1]]
        value["updated_at"] = 0
        path.write_text(json.dumps(value))
        await router.refresh_pool()
        assert not router.ready
        await router.client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("status", [429, 503, 507])
def test_explicit_rejection_can_use_another_worker(tmp_path, status):
    async def run():
        fleet = Fleet()
        fleet.errors[URLS[0]] = status
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        job = await router.submit(gateway.GenerateRequest(prompt="test"))
        assert router.load(job["id"])["worker"] == URLS[1]
        assert len(fleet.jobs) == 1
        assert not list((router.root / "uncertain").glob("*.json"))
        await router.client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("status", [429, 503, 507])
def test_all_workers_reject_without_uncertain_submissions(tmp_path, status):
    async def run():
        fleet = Fleet()
        fleet.errors = {url: status for url in URLS}
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        with pytest.raises(HTTPException) as error:
            await router.submit(gateway.GenerateRequest(prompt="test"))
        assert error.value.status_code == status
        assert len([call for call in fleet.calls if call[0] == "POST"]) == len(URLS)
        assert not fleet.jobs
        assert not router.pending
        assert not list((router.root / "uncertain").glob("*.json"))
        await router.client.aclose()
    asyncio.run(run())


def test_ambiguous_post_is_never_retried_on_another_worker(tmp_path):
    async def run():
        fleet = Fleet()
        fleet.errors[URLS[0]] = httpx.ReadTimeout("response lost")
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        with pytest.raises(HTTPException) as error:
            await router.submit(gateway.GenerateRequest(prompt="test"))
        assert error.value.status_code == 502
        assert len([x for x in fleet.calls if x[0] == "POST"]) == 1
        assert len(list((router.root / "uncertain").glob("*.json"))) == 1
        assert not router.pending
        await router.client.aclose()
    asyncio.run(run())


def test_success_is_not_published_until_video_is_cached_completely(tmp_path):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        identifier = (await router.submit(gateway.GenerateRequest(prompt="test")))["id"]
        fleet.expected_bytes += 1
        await router.refresh_job(identifier)
        assert router.public(router.load(identifier))["status"] == "running"
        assert not (router.root / "videos" / f"{identifier}.mp4").exists()
        fleet.media += b"x"
        await router.refresh_job(identifier)
        assert router.public(router.load(identifier))["status"] == "succeeded"
        assert not router.pending
        await router.client.aclose()
    asyncio.run(run())


def test_http_contract_auth_and_persistence_without_workers(tmp_path):
    fleet = Fleet()
    router = make_gateway(tmp_path, fleet)
    with TestClient(gateway.create_app(router)) as client:
        assert client.get("/healthz").json()["gpus"] == 6
        assert client.post("/v1/videos", json={"prompt": "test"}).status_code == 401
        assert client.post("/v1/videos", json={"prompt": ""}, headers=AUTH).status_code == 422
        assert client.post("/v1/videos", content=b"x" * 32769, headers=AUTH).status_code == 413
        response = client.post("/v1/videos", json={"prompt": "test"}, headers=AUTH)
        assert response.status_code == 202
        job = response.json()
        deadline = time.monotonic() + 5
        while client.get(job["status_url"], headers=AUTH).json()["status"] != "succeeded":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert client.get(job["video_url"]).status_code == 401
        assert client.get(job["video_url"], headers=AUTH).content == fleet.media
        assert client.get("/v1/jobs/not-a-uuid", headers=AUTH).status_code == 404
    restarted = make_gateway(tmp_path, fleet)
    restarted.pool_file.write_text('{"workers": []}')
    with TestClient(gateway.create_app(restarted)) as client:
        assert client.get("/healthz").status_code == 503
        assert client.post("/v1/videos", json={"prompt": "new"}, headers=AUTH).status_code == 503
        assert client.get(job["video_url"], headers=AUTH).content == fleet.media
        assert client.get(job["status_url"], headers=AUTH).json()["status"] == "succeeded"


def test_disk_pressure_prevents_new_backend_submission(tmp_path):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        router.min_free_bytes = 2**70
        with pytest.raises(HTTPException) as error:
            await router.submit(gateway.GenerateRequest(prompt="test"))
        assert error.value.status_code == 507
        assert not fleet.jobs
        await router.client.aclose()
    asyncio.run(run())
