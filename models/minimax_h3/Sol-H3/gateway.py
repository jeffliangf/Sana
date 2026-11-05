#!/usr/bin/env python3
"""Stable H3 API: discover workers, pin jobs, and retain completed videos."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import secrets
import time
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from h3_request import (API_VERSION, MAX_BODY_BYTES, GenerateRequest, accepts, capabilities, same_request,
                        sglang_status)
from serve import atomic_json, caller_job_id, read_request_body
from service_storage import check_space, storage_probe

__all__ = ["Gateway", "GenerateRequest", "create_app"]

LOG = logging.getLogger("h3.gateway")


def origin(value):
    if not isinstance(value, str):
        raise ValueError("Worker origin must be a string")
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError(f"Invalid worker origin: {value}")
    _ = parsed.port
    return value.rstrip("/")


def job_id(value):
    try:
        if str(UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise HTTPException(404, "Unknown job") from None
    return value


class Gateway:
    def __init__(self, registry, state_dir, token, *, pool_file=None, client=None,
                 min_free_gib=20, refresh_seconds=2, worker_lost_seconds=600):
        if len(token) < 32:
            raise ValueError("Bearer token must contain at least 32 characters")
        self.registry, self.root = Path(registry), Path(state_dir)
        self.pool_file = Path(pool_file) if pool_file else self.root / "pool.json"
        for directory in ("jobs", "videos", "uncertain"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self.token = token
        self.owns_client = client is None
        self.client = client or httpx.AsyncClient(trust_env=False, follow_redirects=False,
                                                 limits=httpx.Limits(max_connections=64))
        self.min_free_bytes = min_free_gib * 2**30
        self.refresh_seconds = refresh_seconds
        self.worker_lost_seconds = worker_lost_seconds
        self.claims = {}
        self.ready = {}
        self.discovered = []
        self.updated_at = 0.0
        self.discovery_error = None
        self.registry_errors = {}
        self.registry_cache = {}
        self.storage_error = None
        self.storage = {}
        self.unpersisted = set()
        self.pending = {}
        self.inflight = {}
        self.last_selected = {}
        self.sequence = 0
        self.completed = 0
        self.selection = asyncio.Lock()
        self.io_limit = asyncio.Semaphore(16)
        self.maintenance = []
        # Only unfinished jobs need memory or polling; completed records stay on disk.
        for path in (self.root / "jobs").glob("*.json"):
            record = json.loads(path.read_text())
            if record["job"]["status"] != "failed" and not record.get("video_cached"):
                self.pending[record["job"]["id"]] = record
        # A known acceptance may have reached its journal but not jobs/<id>.json.
        # Unknown outcomes remain for reconciliation; never replay their POSTs.
        for path in (self.root / "uncertain").glob("*.json"):
            attempt = json.loads(path.read_text())
            if attempt.get("status") == "accepted":
                record = attempt["record"]
                identifier = job_id(record["job"]["id"])
                if not (self.root / "jobs" / f"{identifier}.json").exists():
                    record["persistence_pending"] = True
                    self.pending[identifier] = record
                    self.unpersisted.add(identifier)

    def storage_available(self):
        try:
            for name in ("jobs", "uncertain", "videos"):
                path = self.root / name
                self.storage[name] = check_space(path, self.min_free_bytes)
                if self.storage_error:
                    storage_probe(path)
        except OSError as error:
            self.storage_error = str(error)
            return False
        if self.unpersisted:
            return False
        self.storage_error = None
        return True

    def candidates(self):
        try:
            config = json.loads(self.pool_file.read_text())
        except FileNotFoundError:
            config = {}
        if not isinstance(config, dict):
            raise ValueError("Pool configuration must be an object")
        for key in ("workers", "additional_workers", "excluded_workers"):
            if key in config and (not isinstance(config[key], list) or
                                  any(not isinstance(url, str) for url in config[key])):
                raise ValueError(f"Pool {key} must be an array of origins")
        self.registry_errors = {}
        if "workers" in config:
            endpoints = config["workers"]
        else:
            endpoints = list(config.get("additional_workers", []))
            paths = list(self.registry.glob("*/node.json"))
            self.registry_cache = {p: s for p, s in self.registry_cache.items() if p in paths}
            for path in paths:
                try:
                    state = json.loads(path.read_text())
                    if (not isinstance(state, dict) or not isinstance(state.get("updated_at"), (int, float)) or
                            not math.isfinite(state["updated_at"]) or state["updated_at"] > time.time() + 30):
                        raise ValueError("Invalid registry timestamp")
                    workers = state.get("workers", [])
                    if not isinstance(workers, list) or any(not isinstance(w, dict) for w in workers):
                        raise ValueError("Invalid registry workers")
                    urls = [origin(w["endpoint"]) for w in workers if not w.get("disabled")]
                    state = {**state, "origins": urls}
                    self.registry_cache[path] = state
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.registry_errors[str(path)] = str(error)
                    state = self.registry_cache.get(path, {})
                # Bounded grace for failed heartbeats; every candidate still needs
                # a fresh direct health probe. Explicit stop/disable wins at once.
                age = time.time() - state.get("updated_at", 0)
                if age > 300 or state.get("status") in {"stopped", "failed"}:
                    continue
                if age > 120:
                    self.registry_errors[str(path)] = "Stale heartbeat; using health probes for at most 300 seconds"
                endpoints.extend(state.get("origins", []))
        excluded = {origin(url) for url in config.get("excluded_workers", [])}
        return sorted({origin(url) for url in endpoints} - excluded)

    async def refresh_pool(self):
        try:
            candidates = self.candidates()
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.discovery_error = str(error)
            self.ready = {}
            self.updated_at = time.time()
            return

        async def inspect(endpoint):
            try:
                response = await self.client.get(endpoint + "/healthz", timeout=2)
                response.raise_for_status()
                health = response.json()
                # Older workers ignore the job ID header, so they would break deduplication.
                if health.get("status") == "ready" and health["queue_capacity"] > 0 and health.get("idempotency_key"):
                    return endpoint, health
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                pass
            return None

        results = await asyncio.gather(*(inspect(endpoint) for endpoint in candidates))
        self.discovered = candidates
        self.ready = dict(result for result in results if result)
        self.updated_at = time.time()
        self.discovery_error = None

    def health(self):
        ready = self.ready if time.time() - self.updated_at < 15 else {}
        storage_ready = not self.storage_error and not self.unpersisted
        # Advertise the SGLang-form contract only while upgraded workers can serve it.
        upgraded = [h["api"] for h in ready.values()
                    if isinstance(h.get("api"), dict) and h["api"].get("version", 0) >= API_VERSION]
        api = ({**capabilities(max(a["num_frames"]["max"] for a in upgraded)), "workers": len(upgraded)}
               if upgraded else {"version": 1, "tasks": ["t2va"], "workers": 0})
        return {"status": "storage_paused" if not storage_ready else ("ready" if ready else "unavailable"),
                "service": "Sol-H3 gateway", "idempotency_key": True, "api": api,
                "workers_ready": len(ready), "workers_discovered": len(self.discovered),
                "gpus": sum(value.get("gpus", 0) for value in ready.values()),
                "queued": sum(value.get("queued", 0) for value in ready.values()),
                "queue_capacity": sum(value["queue_capacity"] for value in ready.values()),
                "active_job": None, "jobs_pending": len(self.pending),
                "completed_since_start": self.completed, "pool_updated_at": self.updated_at,
                "discovery_error": self.discovery_error, "registry_errors": self.registry_errors,
                "storage": {**self.storage, "error": self.storage_error, "pending_writes": len(self.unpersisted)}}

    def load(self, identifier):
        path = self.root / "jobs" / f"{job_id(identifier)}.json"
        if identifier in self.pending:
            return self.pending[identifier]
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            raise HTTPException(404, "Unknown job") from None
        except OSError as error:
            raise HTTPException(503, "Job storage is temporarily unavailable") from error

    def find(self, identifier):
        try:
            return self.load(identifier)
        except HTTPException as error:
            if error.status_code != 404:
                raise
        return None

    def save(self, record):
        value = {k: v for k, v in record.items() if k not in {"persistence_pending", "unreachable_since"}}
        atomic_json(self.root / "jobs" / f"{record['job']['id']}.json", value)

    def persist(self, record):
        identifier = record["job"]["id"]
        try:
            self.save(record)
        except OSError as error:
            record["persistence_pending"] = True
            self.unpersisted.add(identifier)
            self.storage_error = str(error)
            return False
        record.pop("persistence_pending", None)
        self.unpersisted.discard(identifier)
        if record.get("attempt_id"):
            with suppress(OSError):
                (self.root / "uncertain" / f"{record['attempt_id']}.json").unlink(missing_ok=True)
        return True

    @staticmethod
    def public(record):
        value = dict(record["job"])
        identifier = value["id"]
        value.update(status_url=f"/v1/jobs/{identifier}", video_url=f"/v1/jobs/{identifier}/video")
        if record.get("persistence_pending"):
            value["persistence_pending"] = True
            if value["status"] in {"succeeded", "failed"}:
                value["status"] = "running"
        if value["status"] == "succeeded" and not record.get("video_cached"):
            value["status"] = "running"  # Publish success only after the video is durable here.
        if value["status"] == "succeeded":
            value.update(video_sha256=record.get("video_sha256"), video_bytes=record.get("video_bytes"))
        return value

    async def submit(self, body, key=None):
        """Accept a request, or return the job already created for this key.

        The Idempotency-Key is the job ID on both the gateway and the worker, so
        a repeated POST finds the original job without a separate lookup table.
        Requests without a key get a random ID through the same path.
        ``body`` is a parsed request (or a legacy GenerateRequest).
        """
        parsed = body if hasattr(body, "identity") else read_request_body(body.model_dump())
        identifier = key or str(uuid4())
        while identifier in self.claims:  # Concurrent repeats share one dispatch.
            await self.claims[identifier].wait()
        self.claims[identifier] = asyncio.Event()
        try:
            request = parsed.identity
            record = self.find(identifier) or await self.recover(identifier, request)
            if record is None:
                return await self.dispatch(identifier, parsed)
            if not same_request(record["job"], request):
                raise HTTPException(409, "Idempotency-Key was already used with different inputs")
            return self.public(record)
        finally:
            self.claims.pop(identifier).set()

    async def recover(self, identifier, request):
        """Resolve an earlier ambiguous submission on the only worker that could hold it."""
        journal = self.root / "uncertain" / f"{identifier}.json"
        try:
            attempt = json.loads(journal.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            raise HTTPException(503, "Submission journal is temporarily unavailable") from error
        if attempt.get("status") == "accepted":  # Normally loaded at startup; never dispatch it again.
            self.pending[identifier] = record = {**attempt["record"], "persistence_pending": True}
            self.unpersisted.add(identifier)
            return record
        if attempt["request"] != request:
            raise HTTPException(409, "Idempotency-Key was already used with different inputs")
        try:
            response = await self.client.get(attempt["worker"] + f"/v1/jobs/{identifier}",
                                             headers={"Authorization": f"Bearer {self.token}"}, timeout=10)
            if response.status_code == 404:
                with suppress(OSError):
                    journal.unlink(missing_ok=True)
                return None  # That worker never accepted it, so a new dispatch is safe.
            response.raise_for_status()
            value = response.json()
            if value["id"] != identifier or value["status"] not in {"queued", "running", "succeeded", "failed"}:
                raise ValueError("Unexpected worker status response")
        except (httpx.HTTPError, ValueError, KeyError) as error:
            if time.time() - attempt["submitted_at"] < self.worker_lost_seconds:
                raise HTTPException(502, "Submission outcome is still unknown; repeat with the same Idempotency-Key later") from error
            value = {"id": identifier, "status": "failed", **request, "error": "worker lost",
                     "error_code": "worker_lost", "finished_at": time.time()}
        record = {"worker": attempt["worker"], "job": value, "video_cached": False,
                  "gateway_submitted_at": attempt["submitted_at"], "attempt_id": identifier}
        self.pending[identifier] = record
        self.persist(record)  # On failure, maintenance keeps retrying the write.
        return record

    async def dispatch(self, identifier, parsed):
        if not self.storage_available():
            raise HTTPException(507, "Gateway storage unavailable; request was not accepted")
        if time.time() - self.updated_at > 5:
            await self.refresh_pool()
        tried = set()
        rejections = set()
        attempt_path = self.root / "uncertain" / f"{identifier}.json"
        # The journal keeps the request identity (keyframes by SHA-256), never the base64 payload.
        request = parsed.identity
        while True:
            async with self.selection:
                capable = {url: h for url, h in self.ready.items() if accepts(h.get("api"), parsed)}
                candidates = [url for url, h in capable.items() if url not in tried
                              and h.get("queued", 0) < h["queue_capacity"]]
                if not candidates:
                    if self.ready and not capable:
                        raise HTTPException(503, f"No ready worker accepts {parsed.task} with {parsed.num_frames} "
                                                 "frames yet", headers={"Retry-After": "5"})
                    code = next(iter(rejections)) if len(rejections) == 1 else (429 if capable else 503)
                    raise HTTPException(code, "No worker currently available", headers={"Retry-After": "5"})

                def load(url):
                    health = self.ready[url]
                    observed = health.get("queued", 0) + int(health.get("active_job") is not None)
                    assigned = sum(r["worker"] == url for r in self.pending.values())
                    return max(observed, assigned) + self.inflight.get(url, 0), self.last_selected.get(url, 0), url

                endpoint = min(candidates, key=load)
                self.inflight[endpoint] = self.inflight.get(endpoint, 0) + 1
                self.sequence += 1
                self.last_selected[endpoint] = self.sequence
            tried.add(endpoint)
            try:
                try:
                    atomic_json(attempt_path, {"worker": endpoint, "request": request,
                                               "submitted_at": time.time(), "status": "submitting"})
                except OSError as error:
                    self.storage_error = str(error)
                    raise HTTPException(507, "Gateway could not journal request; no worker accepted it") from error
                try:
                    response = await self.client.post(endpoint + "/v1/videos", json=parsed.payload, timeout=60,
                                                      headers={"Authorization": f"Bearer {self.token}",
                                                               "Idempotency-Key": identifier})
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    response = None  # The request never reached this worker.
                if response is None or response.status_code in {429, 503, 507}:
                    rejections.add(503 if response is None else response.status_code)
                    with suppress(OSError):
                        attempt_path.unlink(missing_ok=True)
                    continue  # No job was accepted, so another worker is safe.
                if 400 <= response.status_code < 500:
                    with suppress(OSError):
                        attempt_path.unlink(missing_ok=True)
                    try:
                        detail = response.json().get("detail") or "Worker rejected request"
                    except (ValueError, AttributeError):
                        detail = "Worker rejected request"
                    raise HTTPException(response.status_code, detail)
                response.raise_for_status()
                value = response.json()
                if value["id"] != identifier or value.get("status") not in {"queued", "running", "succeeded", "failed"}:
                    raise ValueError("Unexpected worker admission response")
                record = {"worker": endpoint, "job": value, "video_cached": False,
                          "gateway_submitted_at": time.time(), "attempt_id": identifier}
                # Keep tracking a known acceptance even if shared storage fails
                # between the worker acknowledgement and writing its route.
                self.pending[identifier] = record
                if not self.persist(record):
                    with suppress(OSError):
                        atomic_json(attempt_path, {"status": "accepted", "record": record})
                    raise HTTPException(502, {"message": "Worker accepted the job but gateway persistence failed; poll this ID, do not resubmit",
                                              "id": identifier, "status_url": f"/v1/jobs/{identifier}"})
                return self.public(record)
            except (httpx.HTTPError, ValueError, KeyError) as error:
                # A timeout can occur after acceptance. Only a repeat of this job ID is safe.
                raise HTTPException(502, "Worker submission outcome is uncertain; repeat only with the same Idempotency-Key") from error
            finally:
                self.inflight[endpoint] -= 1

    async def refresh_job(self, identifier):
        async with self.io_limit:
            record = self.pending.get(identifier)
            if record is None:
                return
            try:
                if identifier in self.unpersisted and not self.persist(record):
                    return
                if record["job"]["status"] not in {"succeeded", "failed"}:
                    response = await self.client.get(record["worker"] + f"/v1/jobs/{identifier}",
                                                     headers={"Authorization": f"Bearer {self.token}"}, timeout=10)
                    response.raise_for_status()
                    value = response.json()
                    if value["id"] != identifier or value["status"] not in {"queued", "running", "succeeded", "failed"}:
                        raise ValueError("Unexpected worker status response")
                    record["job"] = value
                if record["job"]["status"] == "succeeded" and not record.get("video_cached"):
                    await self.archive(record)
                record.pop("last_error", None)
                record.pop("unreachable_since", None)
                record["updated_at"] = time.time()
                if not self.persist(record):
                    return
                if record.get("video_cached") or record["job"]["status"] == "failed":
                    self.pending.pop(identifier, None)
                    self.completed += int(record.get("video_cached", False))
            except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
                now = time.time()
                record.update(last_error=str(error), updated_at=now)
                if isinstance(error, OSError):
                    self.storage_error = str(error)
                elif now - record.setdefault("unreachable_since", now) > self.worker_lost_seconds:
                    # Otherwise a vanished worker is polled forever. The caller can
                    # regenerate under a new job ID; a late result is never archived.
                    record["job"].update(status="failed", error="worker lost", error_code="worker_lost", finished_at=now)
                if self.persist(record) and record["job"]["status"] == "failed":
                    self.pending.pop(identifier, None)
                LOG.warning("Job %s remains pending: %s", identifier, error)

    async def archive(self, record):
        identifier = record["job"]["id"]
        target = self.root / "videos" / f"{identifier}.mp4"
        check_space(target.parent, self.min_free_bytes, reserve_bytes=record["job"]["bytes"])
        temporary = target.with_suffix(".partial")
        digest, size = hashlib.sha256(), 0
        try:
            async with self.client.stream("GET", record["worker"] + f"/v1/jobs/{identifier}/video",
                                          headers={"Authorization": f"Bearer {self.token}"}, timeout=120) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    async for block in response.aiter_bytes(1024 * 1024):
                        check_space(target.parent, self.min_free_bytes, reserve_bytes=len(block))
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                    output.flush()
                    os.fsync(output.fileno())
            if size != record["job"]["bytes"] or size == 0:
                raise ValueError("Incomplete worker video download")
            temporary.replace(target)
            record.update(video_cached=True, video_sha256=digest.hexdigest(), video_bytes=size)
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    async def maintain_pool(self):
        while True:
            try:
                await self.refresh_pool()
            except Exception:
                LOG.exception("Gateway pool refresh failed; retrying")
            await asyncio.sleep(self.refresh_seconds)

    async def maintain_jobs(self):
        while True:
            try:
                await asyncio.gather(*(self.refresh_job(identifier) for identifier in list(self.pending)))
                self.storage_available()
            except Exception:
                LOG.exception("Gateway maintenance failed; retrying")
            await asyncio.sleep(self.refresh_seconds)

    async def start(self):
        self.storage_available()
        await self.refresh_pool()
        self.maintenance = [asyncio.create_task(self.maintain_pool()), asyncio.create_task(self.maintain_jobs())]

    async def close(self):
        for task in self.maintenance:
            task.cancel()
        for task in self.maintenance:
            with suppress(asyncio.CancelledError):
                await task
        if self.owns_client:
            await self.client.aclose()


def create_app(gateway):
    @asynccontextmanager
    async def lifespan(app):
        await gateway.start()
        try:
            yield
        finally:
            await gateway.close()

    app = FastAPI(title="Sol-H3 gateway", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization.encode(), f"Bearer {gateway.token}".encode()):
            raise HTTPException(401, "Bearer token required", headers={"WWW-Authenticate": "Bearer"})

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if request.method == "POST":
            try:
                length = int(request.headers.get("content-length", "-1"))
            except ValueError:
                return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
            if length < 0:
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
            if length > MAX_BODY_BYTES:
                return JSONResponse({"detail": "Request body too large"}, status_code=413)
        return await call_next(request)

    @app.get("/healthz")
    async def health():
        value = gateway.health()
        return JSONResponse(value, status_code=200 if value["status"] == "ready" else 503)

    @app.post("/v1/videos", status_code=202, dependencies=[Depends(authorize)])
    async def submit(request: Request, idempotency_key: str | None = Header(default=None)):
        key = caller_job_id(idempotency_key)
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(422, [{"loc": ["body"], "msg": "the request body must be valid JSON",
                                       "type": "json_invalid"}]) from None
        # Base64 keyframes are decoded and hashed here; keep that off the event loop.
        parsed = await run_in_threadpool(read_request_body, payload)
        return await gateway.submit(parsed, key)

    @app.get("/v1/jobs/{identifier}", dependencies=[Depends(authorize)])
    async def status(identifier: str):
        return gateway.public(gateway.load(identifier))

    def video_file(identifier):
        record = gateway.load(identifier)
        if not record.get("video_cached") or record.get("persistence_pending"):
            raise HTTPException(409, "Video is not ready")
        path = gateway.root / "videos" / f"{identifier}.mp4"
        if not path.is_file():
            raise HTTPException(410, "Video was removed")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/v1/jobs/{identifier}/video", dependencies=[Depends(authorize)])
    async def video(identifier: str):
        return video_file(identifier)

    # SGLang-compatible routes: the same job, with SGLang's "completed" for a finished video.
    @app.get("/v1/videos/{identifier}", dependencies=[Depends(authorize)])
    async def sglang_status_route(identifier: str):
        value = gateway.public(gateway.load(identifier))
        return {**value, "status": sglang_status(value["status"])}

    @app.get("/v1/videos/{identifier}/content", dependencies=[Depends(authorize)])
    async def sglang_content(identifier: str):
        return video_file(identifier)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--pool-file", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--fd", type=int, help="Inherited systemd listening socket; preserves the port on restart")
    parser.add_argument("--min-free-gib", type=int, default=20)
    args = parser.parse_args()
    os.umask(0o077)
    import uvicorn

    gateway = Gateway(args.registry, args.state_dir, args.token_file.read_text().strip(),
                      pool_file=args.pool_file, min_free_gib=args.min_free_gib)
    # Exactly one process owns dispatch and durable job records.
    uvicorn.run(create_app(gateway), host=args.host, port=args.port, fd=args.fd, workers=1, access_log=False,
                limit_concurrency=256, timeout_graceful_shutdown=35)


if __name__ == "__main__":
    main()
