"""How the loader scales each layer: alpha from per-layer tensors, file metadata, or the profile (never a guess)."""
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from h3_runtime import lora


def model():
    m = torch.nn.Module()
    m.attn = torch.nn.Module()
    m.attn.to_q = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    m.attn.to_v = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    m.adaln_proj = torch.nn.Module()
    m.adaln_proj.linear = torch.nn.Linear(8, 16, dtype=torch.bfloat16)
    return m.eval()


def checkpoint(tmp_path, ranks, *, metadata=None, alphas=None, hybrid=False):
    """``ranks``: module name -> rank; ``alphas``: module name -> per-layer alpha tensor value."""
    torch.manual_seed(7)
    shapes = {"attn.to_q": (8, 8), "attn.to_v": (8, 8), "adaln_proj.linear": (16, 8)}
    tensors = {}
    for name, rank in ranks.items():
        out_features, in_features = shapes[name]
        suffix = "" if hybrid else ".default"
        tensors[f"{name}.lora_A{suffix}.weight"] = torch.randn(rank, in_features).bfloat16()
        tensors[f"{name}.lora_B{suffix}.weight"] = torch.randn(out_features, rank).bfloat16()
    for name, value in (alphas or {}).items():
        tensors[f"{name}.alpha"] = torch.tensor(float(value), dtype=torch.bfloat16)
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, path, metadata=metadata or {})
    return path


def plan(path, *, alpha=None, modulation=True, transformer=None):
    with safe_open(path, framework="pt", device="cpu") as handle:
        return lora._plan(transformer or model(), handle, path, alpha=alpha, modulation=modulation)


def layers(result):
    return {name: (rank, alpha) for name, _, _, rank, alpha in result.pairs}


def test_metadata_alpha_applies_to_every_layer(tmp_path):
    # LightX2V 4-step v1.2/v1.3: rank 128 with metadata alpha 8, so each update is scaled by 1/16.
    result = plan(checkpoint(tmp_path, {"attn.to_q": 4, "attn.to_v": 4}, metadata={"format": "pt", "alpha": "8"}))
    assert result.alpha_source == "file metadata" and not result.hybrid
    assert layers(result) == {"attn.to_q": (4, 8.0), "attn.to_v": (4, 8.0)}


def test_per_layer_alpha_wins_over_metadata_and_allows_mixed_ranks(tmp_path):
    # LightX2V 8-step v1.2: metadata says 128, but the rank-16 AdaLN layers carry alpha 16 per layer.
    path = checkpoint(tmp_path, {"attn.to_q": 4, "adaln_proj.linear": 2}, metadata={"alpha": "128"},
                      alphas={"attn.to_q": 4, "adaln_proj.linear": 2})
    result = plan(path)
    assert result.alpha_source == "per-layer tensors"
    assert layers(result) == {"attn.to_q": (4, 4.0), "adaln_proj.linear": (2, 2.0)}


def test_missing_alpha_is_an_error_unless_the_profile_sets_one(tmp_path):
    path = checkpoint(tmp_path, {"attn.to_q": 4})
    with pytest.raises(ValueError, match="embeds no LoRA alpha"):
        plan(path)
    result = plan(path, alpha=8)
    assert result.alpha_source == "explicit" and layers(result) == {"attn.to_q": (4, 8.0)}


def test_an_explicit_alpha_cannot_contradict_the_file(tmp_path):
    path = checkpoint(tmp_path, {"attn.to_q": 4}, metadata={"alpha": "8"})
    assert layers(plan(path, alpha=8)) == {"attn.to_q": (4, 8.0)}
    with pytest.raises(ValueError, match="contradicts"):
        plan(path, alpha=64)  # the old hard-coded T2V value would have scaled this file 8x


def test_fastvideo_hybrid_files_use_alpha_equal_to_rank(tmp_path):
    metadata = {"format": "fastvideo-lora-v2", "rank": "4", "low_rank_tensors": "4", "diff_tensors": "0"}
    path = checkpoint(tmp_path, {"attn.to_q": 4, "attn.to_v": 4}, metadata=metadata, hybrid=True)
    result = plan(path)
    assert result.hybrid and result.alpha_source == "rank (FastVideo hybrid)"
    assert set(layers(result).values()) == {(4, 4.0)}
    assert plan(path, alpha=4).alpha_source == "rank (FastVideo hybrid)"
    with pytest.raises(ValueError, match="alpha = rank"):
        plan(path, alpha=64)


def test_per_layer_alphas_must_cover_exactly_the_lora_layers(tmp_path):
    path = checkpoint(tmp_path, {"attn.to_q": 4, "attn.to_v": 4}, alphas={"attn.to_q": 4})
    with pytest.raises(ValueError, match="coverage"):
        plan(path)


def test_modulation_targets_need_merged_mode(tmp_path):
    path = checkpoint(tmp_path, {"adaln_proj.linear": 2}, metadata={"alpha": "2"})
    assert layers(plan(path, modulation=True)) == {"adaln_proj.linear": (2, 2.0)}
    with pytest.raises(ValueError, match="Unsupported LoRA target module"):
        plan(path, modulation=False)
    with pytest.raises(ValueError, match="Unsupported LoRA target module"):
        lora.load_lora_branches(model(), path)


def test_separate_branches_reject_mixed_ranks(tmp_path):
    path = checkpoint(tmp_path, {"attn.to_q": 4, "attn.to_v": 2}, metadata={"alpha": "4"})
    with pytest.raises(ValueError, match="Mixed LoRA ranks"):
        lora.load_lora_branches(model(), path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fuse_lora merges on CUDA only")
def test_fuse_lora_scales_each_layer_by_its_alpha_over_rank(tmp_path):
    path = checkpoint(tmp_path, {"attn.to_q": 4, "adaln_proj.linear": 2}, alphas={"attn.to_q": 8,
                                                                                    "adaln_proj.linear": 2})
    m = model().cuda()
    before = {name: m.get_submodule(name).weight.detach().clone() for name in ("attn.to_q", "adaln_proj.linear")}
    report = lora.fuse_lora(m, path, scale=0.5)
    assert report.ranks == (2, 4) and report.alpha_source == "per-layer tensors"
    assert report.effective_scales == (0.5, 1.0)  # 0.5 * 2/2 and 0.5 * 8/4
    with safe_open(path, framework="pt", device="cuda") as handle:
        for name, multiplier in (("attn.to_q", 1.0), ("adaln_proj.linear", 0.5)):
            a = handle.get_tensor(f"{name}.lora_A.default.weight")
            b = handle.get_tensor(f"{name}.lora_B.default.weight")
            expected = before[name].clone().addmm_(b, a, beta=1.0, alpha=multiplier)
            assert torch.equal(m.get_submodule(name).weight, expected)
