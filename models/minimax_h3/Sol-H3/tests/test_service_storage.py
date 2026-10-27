"""Inject storage failures at admission, generation, and persistence boundaries."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

import service_storage as storage
import serve
from storage_preflight import required_staging_bytes


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EIO, errno.EDQUOT, errno.EROFS])
def test_atomic_flush_failure_preserves_previous_record(tmp_path, monkeypatch, code):
    path = tmp_path / "job.json"
    storage.atomic_json(path, {"status": "running"})
    monkeypatch.setattr(storage.os, "fsync", Mock(side_effect=OSError(code, "injected")))
    with pytest.raises(OSError):
        storage.atomic_json(path, {"status": "succeeded"})
    assert json.loads(path.read_text()) == {"status": "running"}
    assert list(tmp_path.iterdir()) == [path]


def test_inode_exhaustion_and_archive_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    monkeypatch.setattr(storage.os, "statvfs", lambda _: SimpleNamespace(f_files=1000, f_favail=127))
    with pytest.raises(OSError, match="inodes"):
        storage.check_space(tmp_path, 0)
    monkeypatch.setattr(storage.os, "statvfs", lambda _: SimpleNamespace(f_files=0))
    with pytest.raises(OSError, match="reserve"):
        storage.check_space(tmp_path, 20, reserve_bytes=81)
    assert storage.check_space(tmp_path, 20, reserve_bytes=80)["free_bytes"] == 100


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EIO])
def test_admission_write_failure_does_not_accept_or_queue(tmp_path, monkeypatch, code):
    state = serve.WorkerState(tmp_path, min_free_gib=0)
    state.phase = "ready"
    with monkeypatch.context() as patch:
        patch.setattr(serve, "atomic_json", Mock(side_effect=OSError(code, "injected")))
        with pytest.raises(HTTPException) as error:
            state.submit(serve.GenerateRequest(prompt="never accepted"))
        assert error.value.status_code == 507
    assert state.pending.empty() and not state.records and not state.dirty
    assert state.health()["status"] == "storage_paused"
    assert state.recover_storage()
    assert state.health()["status"] == "ready"
    assert state.submit(serve.GenerateRequest(prompt="after recovery"))["status"] == "queued"


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EIO])
def test_accepted_jobs_pause_and_resume_without_losing_completion(tmp_path, monkeypatch, code):
    state = serve.WorkerState(tmp_path, min_free_gib=0)
    state.phase = "ready"
    record = state.submit(serve.GenerateRequest(prompt="accepted"))
    identifier = record["id"]
    with monkeypatch.context() as patch:
        patch.setattr(serve, "atomic_json", Mock(side_effect=OSError(code, "injected")))
        assert state.next_command() == {"idle": True}
        assert state.held["id"] == identifier and state.active is None
        with pytest.raises(HTTPException) as error:
            state.submit(serve.GenerateRequest(prompt="rejected"))
        assert error.value.status_code == 507
        assert state.get(identifier)["persistence_pending"]
    assert state.next_command()["id"] == identifier
    with monkeypatch.context() as patch:
        patch.setattr(serve, "atomic_json", Mock(side_effect=OSError(code, "injected")))
        state.update(identifier, status="succeeded", bytes=123)
        assert state.get(identifier)["status"] == "running"
        assert state.next_command() == {"idle": True}
        assert state.health()["status"] == "storage_paused"
    assert state.recover_storage()
    assert state.get(identifier)["status"] == "succeeded"
    assert not state.dirty and not state.records
    restarted = serve.WorkerState(tmp_path, min_free_gib=0)
    assert restarted.get(identifier)["status"] == "succeeded"


def test_failed_job_updates_do_not_mask_original_worker_failure(tmp_path, monkeypatch):
    state = serve.WorkerState(tmp_path, min_free_gib=0)
    state.phase = "ready"
    ids = [state.submit(serve.GenerateRequest(prompt="pending"))["id"] for _ in range(2)]
    state.next_command()
    monkeypatch.setattr(serve, "atomic_json", Mock(side_effect=OSError(errno.EIO, "injected")))
    state.fail_pending()
    assert state.phase == "failed" and state.pending.empty() and state.active is None
    assert set(state.dirty) == set(ids)
    assert all(record["status"] == "failed" for record in state.dirty.values())


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EIO])
def test_output_failure_is_broadcast_to_all_ranks_and_next_output_works(tmp_path, code):
    # CPU stand-in for the Gloo object broadcast. All ranks must reach it before
    # returning/raising; a rank-zero-only storage exception would strand peers.
    barrier = threading.Barrier(4)
    shared = {}
    def run(rank, failing):
        def broadcast(values):
            if rank == 0:
                shared["value"] = values[0]
            barrier.wait(timeout=5)
            values[0] = shared["value"]
            barrier.wait(timeout=5)
        def save(path, **kwargs):
            if failing:
                try:
                    raise OSError(code, "disk failed")
                except OSError as error:
                    raise RuntimeError("encoder wrapped filesystem error") from error
            path.write_bytes(b"mp4")
        media = SimpleNamespace(save=save) if rank == 0 else None
        call = lambda: serve.publish_media(media, tmp_path / "video.mp4", 5, 8, lambda *_: {"audio": True},
                                           rank=rank, broadcast=broadcast)
        if failing:
            with pytest.raises(serve.OutputStorageError):
                call()
        else:
            assert call()["bytes"] == 3
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda rank: run(rank, True), range(4)))
        assert not list(tmp_path.iterdir())
        list(pool.map(lambda rank: run(rank, False), range(4)))
    assert (tmp_path / "video.mp4").read_bytes() == b"mp4"


def test_nonstorage_encoder_failure_remains_fatal(tmp_path):
    media = SimpleNamespace(save=Mock(side_effect=RuntimeError("CUDA failure")))
    broadcast = Mock()
    with pytest.raises(RuntimeError, match="CUDA"):
        serve.publish_media(media, tmp_path / "video.mp4", 5, 8, Mock(), rank=0, broadcast=broadcast)
    broadcast.assert_not_called()


def test_staging_capacity_includes_only_missing_artifacts(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "sol-h3-venv.tar").write_bytes(b"x" * 50)
    (stage / "manifest.json").write_text(json.dumps({"sol-h3-venv.tar": {"bytes": 50},
                                                   "sol-h3-checkpoints/model": {"bytes": 100}}))
    venv, checkpoints = tmp_path / "venv", tmp_path / "checkpoints"
    assert required_staging_bytes(venv, checkpoints, stage) == 150
    checkpoints.mkdir()
    assert required_staging_bytes(venv, checkpoints, stage) == 50
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_bytes(b"python")
    assert required_staging_bytes(venv, checkpoints, stage) == 0


def test_legacy_slurm_entry_launches_prepared_h100_paths_without_staging(tmp_path):
    # Slurm copies the submitted script to its spool. The compatibility entry
    # must find the deployment through H3_DEPLOY_ROOT, not its own directory.
    source = Path(__file__).resolve().parents[1]
    root, local = tmp_path / "shared", tmp_path / "node-local"
    code = root / "Sol-H3"
    code.mkdir(parents=True)
    for name in ("deploy_node.sh", "storage_preflight.py", "service_storage.py"):
        shutil.copy2(source / name, code / name)
    spool = tmp_path / "slurm_script"
    shutil.copy2(source / "deploy_h200_node.sh", spool)
    venv = local / "prepared-env"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.write_text('#!/bin/bash\nprintf "FAKE_SUPERVISOR %s %s %s %s\\n" "$*" "$H3_GPU_TYPE" "$H3_GPUS_PER_WORKER" "$CUDA_VISIBLE_DEVICES"\n')
    python.chmod(0o700)
    checkpoints = local / "prepared-weights"
    checkpoints.mkdir()
    env = {**os.environ, "H3_DEPLOY_ROOT": str(root), "H3_LOCAL_ROOT": str(local),
           "H3_VENV": str(venv), "H3_CHECKPOINT_ROOT": str(checkpoints), "H3_GPU_TYPE": "H100",
           "H3_GPUS_PER_WORKER": "4", "H3_MIN_FREE_GIB": "1", "SLURM_JOB_ID": "test-987",
           "CUDA_VISIBLE_DEVICES": "4,5,6,7"}
    result = subprocess.run(["bash", str(spool)], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '"staging_bytes": 0' in result.stdout
    log = local / "sol-h3-supervisor/supervisor-test-987.log"
    assert log.read_text().strip() == "FAKE_SUPERVISOR serve_node.py H100 4 4,5,6,7"
    assert not (root / "deployment/staging").exists()
