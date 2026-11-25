"""Adapter profiles: the checked-in configuration, its validation, and file verification (CPU only)."""
import hashlib
import json

import pytest

import serve
from h3_runtime import adapters

REQUIRED = ["--model", "m", "--output-dir", "o", "--token-file", "t", "--warmup-prompt-file", "w"]


def test_checked_in_profiles_carry_the_recommended_settings():
    defaults, profiles = adapters.load_profiles()
    assert defaults == {"t2v": "fasth3-dense-datafree", "i2v": "fasth3-dense-datafree",
                        "ref2va": "lightx2v-ref2v-4step-v0.1"}
    expected = {  # name: (forwards, video shift, audio shift, final step scale, strength, explicit alpha)
        "smd-e-8step": (8, 12.0, 3.0, 0.5, 1.0, None),
        "smd-ref2va-8step": (8, 12.0, 3.0, 0.5, 1.0, None),
        "fasth3-dense-datafree": (4, 12.0, 3.0, 1.0, 1.0, None),
        "lightx2v-fl2v-4step-v1.2": (4, 6.0, 3.0, 1.0, 1.0, None),
        "lightx2v-fl2v-4step-v1.3": (4, 6.0, 3.0, 1.0, 1.0, None),
        "lightx2v-fl2v-8step-v1.2": (8, 12.0, 3.0, 1.0, 1.0, None),
        "lightx2v-ref2v-4step-v0.1": (4, 12.0, 3.0, 1.0, 1.0, None),
    }
    assert {name: (p.transformer_forwards, p.video_shift, p.audio_shift, p.final_step_size_scale, p.strength,
                   p.alpha) for name, p in profiles.items()} == expected
    for profile in profiles.values():
        assert profile.scheduler_points == profile.transformer_forwards + 1
        assert profile.sha256 and profile.bytes
        if "copy_from" in profile.source:
            # A file delivered outside the Hub: download_checkpoints.py copies it to the same name.
            assert profile.source["copy_from"].startswith("/")
            assert profile.source["copy_from"].rsplit("/", 1)[1] == profile.file.rsplit("/", 1)[1]
        else:
            assert profile.source["revision"]
            # download_checkpoints.py's layout rule: <local repo directory>/<path in the repo>.
            assert profile.file.split("/", 1)[1] == profile.source["path"]


# The SMD release's generation receipt (examples/generation.json): 8 steps, final_step_size_scale 0.5.
SMD_RAW_SIGMAS = [1.0, 0.8666666666666667, 0.7333333333333334, 0.6000000000000001, 0.4666666666666668,
                  0.3333333333333335, 0.20000000000000015, 0.06666666666666682, 0.0]


def test_smd_profile_samples_the_release_schedule():
    assert adapters.raw_sigma_grid(8, 0.5) == SMD_RAW_SIGMAS
    profile = adapters.get_profile("smd-e-8step", task="i2v")
    for shift in (12.0, 3.0):
        sigmas = profile.sigmas(shift)
        assert sigmas == [shift * s / (1 + (shift - 1) * s) for s in SMD_RAW_SIGMAS]
        assert sigmas[0] == 1.0 and sigmas[-1] == 0.0 and all(a > b for a, b in zip(sigmas, sigmas[1:]))
    assert len(profile.sigmas(12.0)) == profile.scheduler_points
    assert profile.summary()["final_step_size_scale"] == 0.5
    # Even steps keep the scheduler's own grid, so the existing profiles sample exactly as before.
    assert adapters.get_profile("fasth3-dense-datafree").sigmas(12.0) is None
    assert adapters.raw_sigma_grid(4, 1.0) == [1.0, 0.75, 0.5, 0.25, 0.0]


def test_smd_ref2va_profile_matches_its_release():
    profile = adapters.get_profile("smd-ref2va-8step", task="ref2va")
    assert profile.tasks == ("ref2va",) and profile.worker_task == "ref2va"
    assert profile.reference_image_short_edge == 512  # the release's infer.py: --reference-image-short-edge 512
    assert profile.sigmas(12.0) == [12.0 * s / (1 + 11.0 * s) for s in SMD_RAW_SIGMAS]
    assert profile.sha256 == "f8aabc9258d443a5c04833a637246ecb3551cb53847e5aba1ef4a00b444e462c"
    assert profile.source["repo"] == "zjwfufu/SMD-MiniMax-H3-Ref2VA-8step"
    assert profile.summary()["reference_image_short_edge"] == 512
    # Profiles without the field report exactly what they did before.
    assert "reference_image_short_edge" not in adapters.get_profile("smd-e-8step", task="t2v").summary()
    assert adapters.get_profile("smd-e-8step", task="t2v").worker_task == "t2v"
    assert adapters.get_profile("lightx2v-ref2v-4step-v0.1", task="ref2va").reference_image_short_edge is None


def test_reference_short_edge_is_a_ref2va_multiple_of_32(tmp_path):
    path = write_profiles(tmp_path)
    data = json.loads(path.read_text())
    data["profiles"]["r"]["reference_image_short_edge"] = 512
    path.write_text(json.dumps(data))
    assert adapters.get_profile("r", task="ref2va", path=path).reference_image_short_edge == 512
    for value, profile in ((500, "r"), (0, "r"), (True, "r"), (512.0, "r"), (512, "x")):
        data = json.loads(write_profiles(tmp_path).read_text())
        data["profiles"][profile]["reference_image_short_edge"] = value
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="reference_image_short_edge"):
            adapters.load_profiles(path)


def test_explicit_schedule_replaces_the_pipeline_step_count():
    calls = []

    class Scheduler:
        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None):
            calls.append((num_inference_steps, device, sigmas))

    scheduler = Scheduler()
    adapters.install_explicit_schedule(scheduler, [1.0, 0.5, 0.0])
    scheduler.set_timesteps(9, device="cuda:0")  # what the pipeline's set-timesteps step passes
    scheduler.set_timesteps(sigmas=[1.0, 0.0])   # an explicit caller still wins
    assert calls == [(None, "cuda:0", [1.0, 0.5, 0.0]), (None, None, [1.0, 0.0])]


def test_explicit_schedule_drives_the_minimax_h3_scheduler():
    scheduling = pytest.importorskip("diffusers.schedulers.scheduling_minimax_h3")
    torch = pytest.importorskip("torch")
    profile = adapters.get_profile("smd-e-8step")
    scheduler = scheduling.MiniMaxH3Scheduler()
    scheduler.set_shift(profile.video_shift)
    adapters.install_explicit_schedule(scheduler, profile.sigmas(profile.video_shift))
    scheduler.set_timesteps(profile.scheduler_points, device="cpu")
    expected = torch.tensor(profile.sigmas(profile.video_shift), dtype=torch.float32)
    assert torch.equal(scheduler.sigmas, expected)  # verbatim: the shift is not applied twice
    assert scheduler.num_inference_steps == profile.transformer_forwards


def test_get_profile_checks_names_and_tasks():
    assert adapters.get_profile(task="t2v").name == "fasth3-dense-datafree"
    assert adapters.get_profile(task="ref2va").name == "lightx2v-ref2v-4step-v0.1"
    assert adapters.get_profile("lightx2v-fl2v-8step-v1.2", task="i2v").transformer_forwards == 8
    with pytest.raises(ValueError, match="unknown adapter profile"):
        adapters.get_profile("lightx2v-fl2v-4step-v9")
    with pytest.raises(ValueError, match="supports"):
        adapters.get_profile("lightx2v-ref2v-4step-v0.1", task="t2v")
    with pytest.raises(ValueError, match="supports"):
        adapters.get_profile("fasth3-dense-datafree", task="ref2va")


def write_profiles(tmp_path, **overrides):
    entry = {"tasks": ["t2v", "i2v"], "file": "repo/adapter.safetensors", "transformer_forwards": 4,
             "video_shift": 6.0, "audio_shift": 3.0, **overrides}
    ref = {**entry, "tasks": ["ref2va"]}
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"schema_version": 1, "defaults": {"t2v": "x", "i2v": "x", "ref2va": "r"},
                                "profiles": {"x": entry, "r": ref}}))
    return path


@pytest.mark.parametrize("overrides", [{"file": "../escape.safetensors"}, {"file": "/abs.safetensors"},
                                       {"transformer_forwards": 0}, {"transformer_forwards": 4.0},
                                       {"video_shift": 0}, {"audio_shift": float("nan")}, {"strength": -1},
                                       {"alpha": 0}, {"alpha": True}, {"tasks": ["v2v"]}, {"sha256": "ABC"},
                                       {"bytes": 0}, {"final_step_size_scale": 0}, {"final_step_size_scale": 1.5},
                                       {"final_step_size_scale": True}, {"final_step_size_scale": float("nan")}])
def test_invalid_profiles_are_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        adapters.load_profiles(write_profiles(tmp_path, **overrides))


def test_defaults_must_support_their_task(tmp_path):
    path = write_profiles(tmp_path)
    data = json.loads(path.read_text())
    data["defaults"]["t2v"] = "r"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="default profile for t2v"):
        adapters.load_profiles(path)


def test_verify_file_checks_size_and_digest(tmp_path):
    payload = b"adapter bytes"
    path = tmp_path / "repo/adapter.safetensors"
    path.parent.mkdir()
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    profile = adapters.get_profile("x", path=write_profiles(tmp_path, sha256=digest, bytes=len(payload)))
    assert profile.default_path(tmp_path / "MiniMax-H3") == path.resolve()
    assert adapters.verify_file(profile, path) == digest
    path.write_bytes(payload[:-1] + b"X")
    with pytest.raises(ValueError, match="SHA-256"):
        adapters.verify_file(profile, path)
    path.write_bytes(payload + b"!")
    with pytest.raises(ValueError, match="bytes"):
        adapters.verify_file(profile, path)
    with pytest.raises(FileNotFoundError):
        adapters.verify_file(profile, tmp_path / "missing.safetensors")


def test_serve_resolves_the_profile_before_loading_anything():
    assert serve.parse_args(REQUIRED).adapter_profile == "fasth3-dense-datafree"
    args = serve.parse_args(REQUIRED + ["--adapter-profile", "lightx2v-fl2v-4step-v1.3"])
    assert args.adapter_profile == "lightx2v-fl2v-4step-v1.3" and args.adapter is None
    assert args.task == "t2v"
    with pytest.raises(SystemExit):
        serve.parse_args(REQUIRED + ["--adapter-profile", "no-such-profile"])
    # A Ref2VA-only profile makes a Ref2VA worker, which cannot use direct Sol attention.
    with pytest.raises(SystemExit):
        serve.parse_args(REQUIRED + ["--adapter-profile", "smd-ref2va-8step"])
    for backend in ("dense", "sol_bsa"):
        args = serve.parse_args(REQUIRED + ["--adapter-profile", "smd-ref2va-8step", "--attention-backend", backend])
        assert (args.adapter_profile, args.task) == ("smd-ref2va-8step", "ref2va")
