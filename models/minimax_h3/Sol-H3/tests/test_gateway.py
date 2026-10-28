"""Verify stable jobs and downloads while the worker pool changes."""
import asyncio
import collections
import errno
import hashlib
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
        self.lost = set()  # Accept the next POST, then lose its response.
        self.down = set()  # Refuse every connection.
        self.legacy = set()  # Predates job ID support.
        self.media = b"example-mp4"
        self.expected_bytes = len(self.media)

    async def handle(self, request):
        endpoint = str(request.url).split("/v1/")[0].removesuffix("/healthz")
        self.calls.append((request.method, endpoint, request.url.path))
        if endpoint in self.down:
            raise httpx.ConnectError("connection refused")
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ready", "gpus": 2, "queued": 0, "queue_capacity": 8,
                                             "active_job": None, "idempotency_key": endpoint not in self.legacy})
        assert request.headers["Authorization"] == AUTH["Authorization"]
        if request.method == "POST":
            error = self.errors.get(endpoint)
            if isinstance(error, Exception):
                raise error
            if error:
                return httpx.Response(error)
            await asyncio.sleep(0.005)
            identifier = request.headers["Idempotency-Key"]
            value = {"id": identifier, "status": "queued", **json.loads(request.content)}
            job = self.jobs.setdefault(identifier, {"endpoint": endpoint, "value": value})
            assert job["endpoint"] == endpoint, "A job ID must never be generated on a second worker"
            if endpoint in self.lost:
                self.lost.discard(endpoint)
                raise httpx.ReadTimeout("response lost after acceptance")
            return httpx.Response(202, json=job["value"])
        identifier = request.url.path.split("/")[3]
        job = self.jobs.get(identifier)
        if job is None:
            return httpx.Response(404, json={"detail": "Unknown job"})
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


@pytest.mark.parametrize("status", [429, 503, 507, httpx.ConnectError("connection refused")])
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


def test_idempotency_key_returns_one_job_for_repeats_and_conflicts(tmp_path):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        key = str(uuid4())
        request = gateway.GenerateRequest(prompt="same attempt", seed=7)
        values = await asyncio.gather(*(router.submit(request, key) for _ in range(3)))
        assert {value["id"] for value in values} == {key}
        assert (await router.submit(request, key))["id"] == key
        with pytest.raises(HTTPException) as conflict:
            await router.submit(gateway.GenerateRequest(prompt="same attempt", seed=8), key)
        assert conflict.value.status_code == 409
        assert len([call for call in fleet.calls if call[0] == "POST"]) == 1
        await router.client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("restart", [False, True])
def test_lost_acceptance_is_recovered_from_its_worker_without_a_second_post(tmp_path, restart):
    async def run():
        fleet = Fleet()
        fleet.lost.add(URLS[0])
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        key = str(uuid4())
        request = gateway.GenerateRequest(prompt="response lost")
        with pytest.raises(HTTPException) as error:
            await router.submit(request, key)
        assert error.value.status_code == 502
        if restart:
            await router.client.aclose()
            router = make_gateway(tmp_path, fleet)
            await router.refresh_pool()
        assert (await router.submit(request, key))["id"] == key
        assert router.load(key)["worker"] == URLS[0]
        assert len([call for call in fleet.calls if call[0] == "POST"]) == 1
        assert not list((router.root / "uncertain").glob("*.json"))
        await router.refresh_job(key)
        job = router.public(router.load(key))
        assert job["status"] == "succeeded"
        assert job["video_sha256"] == hashlib.sha256(fleet.media).hexdigest()
        assert job["video_bytes"] == len(fleet.media)
        await router.client.aclose()
    asyncio.run(run())


def test_unaccepted_ambiguous_submission_is_dispatched_again(tmp_path):
    async def run():
        fleet = Fleet()
        fleet.errors[URLS[0]] = httpx.ReadTimeout("request lost before acceptance")
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        key = str(uuid4())
        request = gateway.GenerateRequest(prompt="never accepted")
        with pytest.raises(HTTPException):
            await router.submit(request, key)
        del fleet.errors[URLS[0]]
        assert (await router.submit(request, key))["id"] == key  # Its worker reports 404.
        assert len(fleet.jobs) == 1
        assert len([call for call in fleet.calls if call[0] == "POST"]) == 2
        await router.client.aclose()
    asyncio.run(run())


def test_unreachable_worker_turns_unfinished_jobs_into_worker_lost(tmp_path):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        router.worker_lost_seconds = 0
        await router.refresh_pool()
        identifier = (await router.submit(gateway.GenerateRequest(prompt="worker vanishes")))["id"]
        fleet.down.add(router.load(identifier)["worker"])
        await router.refresh_job(identifier)
        assert identifier in router.pending  # The first failure only starts the clock.
        await asyncio.sleep(0.01)
        await router.refresh_job(identifier)
        job = router.public(router.load(identifier))
        assert job["status"] == "failed" and job["error_code"] == "worker_lost"
        assert identifier not in router.pending
        await router.client.aclose()
    asyncio.run(run())


def test_ambiguous_submission_on_a_lost_worker_resolves_to_worker_lost(tmp_path):
    async def run():
        fleet = Fleet()
        fleet.lost.add(URLS[0])
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        key = str(uuid4())
        request = gateway.GenerateRequest(prompt="worker vanished after accepting")
        with pytest.raises(HTTPException):
            await router.submit(request, key)
        fleet.down.add(URLS[0])
        with pytest.raises(HTTPException) as unknown:
            await router.submit(request, key)
        assert unknown.value.status_code == 502  # Too early to declare the worker lost.
        router.worker_lost_seconds = 0
        job = await router.submit(request, key)
        assert job["id"] == key and job["status"] == "failed" and job["error_code"] == "worker_lost"
        assert len([call for call in fleet.calls if call[0] == "POST"]) == 1
        await router.client.aclose()
    asyncio.run(run())


def test_workers_without_job_id_support_receive_no_jobs(tmp_path):
    async def run():
        fleet = Fleet()
        fleet.legacy.add(URLS[0])
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        assert set(router.ready) == set(URLS[1:])
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


def test_http_idempotency_key_contract(tmp_path):
    router = make_gateway(tmp_path, Fleet())
    key = str(uuid4())
    keyed = {**AUTH, "Idempotency-Key": key}
    with TestClient(gateway.create_app(router)) as client:
        assert client.get("/healthz").json()["idempotency_key"] is True
        for bad in ("not-a-uuid", key.upper()):
            response = client.post("/v1/videos", json={"prompt": "x"}, headers={**AUTH, "Idempotency-Key": bad})
            assert response.status_code == 400
        first = client.post("/v1/videos", json={"prompt": "x"}, headers=keyed)
        repeat = client.post("/v1/videos", json={"prompt": "x"}, headers=keyed)
        assert first.status_code == repeat.status_code == 202
        assert first.json()["id"] == repeat.json()["id"] == key
        assert client.post("/v1/videos", json={"prompt": "y"}, headers=keyed).status_code == 409


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


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EIO])
def test_journal_failure_is_rejected_before_any_backend_post(tmp_path, monkeypatch, code):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs):
                raise OSError(code, "injected journal failure")
            patch.setattr(gateway, "atomic_json", fail)
            with pytest.raises(HTTPException) as error:
                await router.submit(gateway.GenerateRequest(prompt="not accepted"))
        assert error.value.status_code == 507
        assert not fleet.jobs and not router.pending
        assert not any(call[0] == "POST" for call in fleet.calls)
        assert all(count == 0 for count in router.inflight.values())
        assert router.health()["status"] == "storage_paused"
        assert router.storage_available()
        assert (await router.submit(gateway.GenerateRequest(prompt="recovered")))["id"]
        await router.client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("restart", [False, True])
def test_accepted_job_record_failure_is_recovered_without_resubmission(tmp_path, monkeypatch, restart):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        await router.refresh_pool()
        original = gateway.atomic_json
        def fail_route(path, value):
            if path.parent.name == "jobs":
                raise OSError(errno.ENOSPC, "accepted but route filesystem full")
            original(path, value)
        with monkeypatch.context() as patch:
            patch.setattr(gateway, "atomic_json", fail_route)
            with pytest.raises(HTTPException) as error:
                await router.submit(gateway.GenerateRequest(prompt="accepted exactly once"))
            assert error.value.status_code == 502
            identifier = error.value.detail["id"]
            assert error.value.detail["status_url"] == f"/v1/jobs/{identifier}"
            assert router.public(router.load(identifier))["persistence_pending"]
            await router.refresh_job(identifier)  # Repeated write failure must not escape maintenance.
            assert router.completed == 0 and identifier in router.pending
            with pytest.raises(HTTPException) as paused:
                await router.submit(gateway.GenerateRequest(prompt="new request"))
            assert paused.value.status_code == 507
        if restart:
            await router.client.aclose()
            router = make_gateway(tmp_path, fleet)
            assert identifier in router.pending
        await router.refresh_job(identifier)
        assert router.public(router.load(identifier))["status"] == "succeeded"
        assert router.completed == 1 and not router.pending and not router.unpersisted
        assert len([call for call in fleet.calls if call[0] == "POST"]) == 1
        assert not list((router.root / "uncertain").glob("*.json"))
        await router.client.aclose()
    asyncio.run(run())


def test_archive_metadata_failure_keeps_video_and_retries_only_metadata(tmp_path, monkeypatch):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        identifier = (await router.submit(gateway.GenerateRequest(prompt="archive")))["id"]
        with monkeypatch.context() as patch:
            def fail(*args):
                raise OSError(errno.EIO, "cannot persist archived status")
            patch.setattr(router, "save", fail)
            await router.refresh_job(identifier)
            assert router.pending[identifier]["video_cached"]
            assert router.public(router.load(identifier))["status"] == "running"
            assert router.completed == 0
            await router.refresh_job(identifier)
            assert router.completed == 0
        await router.refresh_job(identifier)
        await router.refresh_job(identifier)
        assert router.public(router.load(identifier))["status"] == "succeeded"
        assert router.completed == 1
        assert len([call for call in fleet.calls if call[2].endswith("/video")]) == 1
        await router.client.aclose()
    asyncio.run(run())


def test_failed_job_metadata_can_recover_after_worker_is_gone(tmp_path, monkeypatch):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        identifier = (await router.submit(gateway.GenerateRequest(prompt="failed inference")))["id"]
        router.pending[identifier]["job"].update(status="failed", error="inference worker failed")
        before = list(fleet.calls)
        with monkeypatch.context() as patch:
            def fail(*args):
                raise OSError(errno.ENOSPC, "terminal metadata could not be written")
            patch.setattr(router, "save", fail)
            await router.refresh_job(identifier)
            assert identifier in router.unpersisted
        await router.refresh_job(identifier)
        assert not router.pending and not router.unpersisted
        assert router.load(identifier)["job"]["status"] == "failed"
        assert fleet.calls == before  # Terminal metadata needs no live worker.
        await router.client.aclose()
    asyncio.run(run())


def test_full_archive_volume_keeps_job_on_original_worker_until_recovery(tmp_path):
    async def run():
        fleet = Fleet()
        router = make_gateway(tmp_path, fleet)
        identifier = (await router.submit(gateway.GenerateRequest(prompt="archive")))["id"]
        worker = router.pending[identifier]["worker"]
        router.min_free_bytes = 2**70
        await router.refresh_job(identifier)
        assert router.public(router.load(identifier))["status"] == "running"
        assert router.health()["status"] == "storage_paused"
        router.pool_file.write_text('{"workers": []}')
        await router.refresh_pool()
        router.min_free_bytes = 0
        await router.refresh_job(identifier)
        assert router.load(identifier)["worker"] == worker
        assert router.public(router.load(identifier))["status"] == "succeeded"
        assert len([call for call in fleet.calls if call[0] == "POST"]) == 1
        await router.client.aclose()
    asyncio.run(run())


def test_bad_registry_is_isolated_grace_is_bounded_and_stop_wins(tmp_path, monkeypatch):
    async def run():
        router = make_gateway(tmp_path, Fleet())
        now = time.time()
        await router.refresh_pool()
        known = router.registry / "node/node.json"
        bad = router.registry / "bad/node.json"
        bad.parent.mkdir()
        bad.write_text("{")
        known.write_text("{")
        monkeypatch.setattr(gateway.time, "time", lambda: now + 180)
        await router.refresh_pool()
        assert set(router.ready) == set(URLS)  # Last valid heartbeat plus fresh health probes.
        assert len(router.registry_errors) == 2
        router.pool_file.write_text(json.dumps({"excluded_workers": [URLS[0]]}))
        await router.refresh_pool()
        assert URLS[0] not in router.ready
        monkeypatch.setattr(gateway.time, "time", lambda: now + 301)
        await router.refresh_pool()
        assert not router.ready
        known.write_text(json.dumps({"updated_at": now + 301, "status": "stopped",
                                    "workers": [{"endpoint": url} for url in URLS]}))
        await router.refresh_pool()
        assert not router.ready
        known.write_text("{")
        await router.refresh_pool()
        assert not router.ready  # Cached stop must not resurrect the preceding live record.
        await router.client.aclose()
    asyncio.run(run())


def test_manual_pool_config_errors_fail_closed_and_tunnels_can_supplement_discovery(tmp_path):
    async def run():
        router = make_gateway(tmp_path, Fleet())
        tunnel = "http://127.0.0.1:28084"
        router.pool_file.write_text(json.dumps({"additional_workers": [tunnel], "excluded_workers": [URLS[0]]}))
        await router.refresh_pool()
        assert set(router.ready) == {URLS[1], URLS[2], tunnel}
        router.pool_file.write_text("{")
        await router.refresh_pool()
        assert not router.ready and router.discovery_error
        router.pool_file.write_text('{"workers": [], "additional_workers": ["http://127.0.0.1:28084"]}')
        await router.refresh_pool()
        assert not router.ready  # Explicit global pause overrides supplemental workers.
        await router.client.aclose()
    asyncio.run(run())


def test_leftover_storage_probe_does_not_break_restart_job_scan(tmp_path, monkeypatch):
    from service_storage import storage_probe
    fleet = Fleet()
    router = make_gateway(tmp_path, fleet)
    original_unlink = Path.unlink
    def fail_probe_cleanup(path, *args, **kwargs):
        if path.name == ".storage-probe":
            raise OSError(errno.EIO, "probe cleanup failed")
        return original_unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_probe_cleanup)
        with pytest.raises(OSError):
            storage_probe(router.root / "jobs")
    assert (router.root / "jobs/.storage-probe").exists()
    restarted = make_gateway(tmp_path, fleet)
    assert not restarted.pending
    asyncio.run(router.client.aclose())
    asyncio.run(restarted.client.aclose())


@pytest.mark.parametrize("bad", [{"workers": [None]}, {"workers": [{"endpoint": 12}]}, {"updated_at": float("nan")}])
def test_malformed_registry_cannot_remove_healthy_siblings(tmp_path, bad):
    async def run():
        router = make_gateway(tmp_path, Fleet())
        path = router.registry / "invalid/node.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"updated_at": time.time(), "status": "ready", **bad}))
        await router.refresh_pool()
        assert set(router.ready) == set(URLS)
        assert str(path) in router.registry_errors
        await router.client.aclose()
    asyncio.run(run())
