"""Fuse MiniMax-H3 inference adapters into the base transformer.

The loader applies the adapter directly to the resident transformer, avoiding
a PEFT runtime dependency after startup.

FastVideo's ``fastvideo-lora-v2`` checkpoints are hybrid adapters: low-rank
updates use ``W += B @ A`` and selected parameters additionally carry tiny
``.diff``/``.diff_b`` corrections. Those corrections must be accumulated in
float32 or BF16 rounding can erase them. Replacement-weight checkpoints
(``.set_weight``, FastVideo's VSA variants) are rejected.

The LoRA alpha decides the update's scale (``alpha / rank``), so it is never
guessed. FastVideo hybrid files use alpha = rank. Other PEFT files use, in
order: per-layer ``<module>.alpha`` tensors (LightX2V 8-step v1.2: alpha = rank
at ranks 128 and 16), the file's ``alpha`` metadata (LightX2V v1.0/v1.1: 128;
v1.2/v1.3 and Ref2VA v0.1: 8, all at rank 128), and only then an explicit
alpha (an adapter profile's ``alpha``). A file with neither is rejected, and an
explicit alpha that contradicts the file's is an error.

Checkpoint conventions follow:
https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/inference_minimax_h3.py
https://github.com/ModelTC/LightX2V/blob/main/scripts/minimax_h3/README.md (LoRA alpha order)
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open


LORA_SUFFIXES = {
    ".lora_A.default.weight": "A",
    ".lora_B.default.weight": "B",
    ".lora_A.weight": "A",
    ".lora_B.weight": "B",
}
ALPHA_SUFFIX = ".alpha"
LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
# Modulation projections. LightX2V 8-step v1.2 adds rank-16 updates to them. Merged mode applies them before the
# AdaLN precompute first evaluates adaln_proj, so its cached tables include the update.
MODULATION_TARGET_MODULES = ("adaln_proj.linear", "norm_out.linear")


@dataclass(frozen=True)
class FuseReport:
    path: str
    format: str
    pairs: int
    diffs: int
    replacements: int
    ranks: tuple[int, ...]
    alpha_source: str
    alphas: tuple[float, ...]
    scale: float
    effective_scales: tuple[float, ...]
    device: str
    fuse_s: float

    def summary(self) -> dict:
        return {"format": self.format, "pairs": self.pairs, "diffs": self.diffs, "ranks": list(self.ranks),
                "alpha_source": self.alpha_source, "alphas": list(self.alphas), "strength": self.scale,
                "effective_scales": list(self.effective_scales), "fuse_s": round(self.fuse_s, 3)}


@dataclass(frozen=True)
class _Plan:
    format: str
    hybrid: bool
    pairs: tuple[tuple[str, str, str, int, float], ...]  # module, A key, B key, rank, alpha
    diffs: tuple[tuple[str, str], ...]  # parameter, diff key
    alpha_source: str


def _payload(
    keys: list[str], path: Path, *, hybrid: bool
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]], dict[str, str]]:
    a_keys, b_keys, diffs, alphas, replacements, unsupported = {}, {}, {}, {}, [], []
    for key in keys:
        matched = False
        for suffix, side in LORA_SUFFIXES.items():
            if key.endswith(suffix):
                module_name = key[: -len(suffix)].removeprefix("diffusion_model.")
                (a_keys if side == "A" else b_keys)[module_name] = key
                matched = True
                break
        if matched:
            continue
        if hybrid and key.endswith(".diff_b"):
            param_name = key[:-len(".diff_b")].removeprefix("diffusion_model.") + ".bias"
            diffs[param_name] = (key, "bias")
        elif hybrid and key.endswith(".diff"):
            param_name = key[:-len(".diff")].removeprefix("diffusion_model.") + ".weight"
            diffs[param_name] = (key, "weight")
        elif not hybrid and key.endswith(ALPHA_SUFFIX):
            alphas[key[: -len(ALPHA_SUFFIX)].removeprefix("diffusion_model.")] = key
        elif key.endswith(".set_weight"):
            replacements.append(key)
        else:
            unsupported.append(key)
    if replacements:
        raise ValueError(
            f"{path} contains {len(replacements)} replacement tensors; "
            "this runtime accepts low-rank and diff adapters only"
        )
    if unsupported:
        raise ValueError(
            f"{path} contains unsupported adapter keys: "
            f"{unsupported[:3]}"
        )
    if not a_keys:
        raise ValueError(f"No LoRA A tensors found in {path}")
    missing_a = sorted(b_keys.keys() - a_keys.keys())
    missing_b = sorted(a_keys.keys() - b_keys.keys())
    if missing_a or missing_b:
        raise ValueError(
            f"Unpaired LoRA tensors: missing A={missing_a[:3]}, "
            f"missing B={missing_b[:3]}"
        )
    pairs = {name: (a_keys[name], b_keys[name]) for name in sorted(a_keys)}
    return pairs, {name: diffs[name] for name in sorted(diffs)}, alphas


def _positive(value: float, what: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be finite and positive, got {value}")
    return value


def _plan(transformer, checkpoint, path: Path, *, alpha: float | None, modulation: bool) -> _Plan:
    """Validate the whole checkpoint against the transformer and resolve every layer's rank and alpha."""
    metadata = checkpoint.metadata() or {}
    adapter_format = metadata.get("format", "peft-lora")
    hybrid = adapter_format == "fastvideo-lora-v2"
    pairs, diffs, alpha_keys = _payload(list(checkpoint.keys()), path, hybrid=hybrid)
    if hybrid:
        expected = (int(metadata.get("low_rank_tensors", len(pairs) * 2)),
                    int(metadata.get("diff_tensors", len(diffs))),
                    int(metadata.get("set_weight_tensors", 0)))
        if expected != (len(pairs) * 2, len(diffs), 0):
            raise ValueError(
                "FastVideo adapter metadata/payload mismatch: "
                f"low_rank={expected[0]}/{len(pairs) * 2}, diffs={expected[1]}/{len(diffs)}, "
                f"replacements={expected[2]}"
            )
    allowed = LORA_TARGET_MODULES + (MODULATION_TARGET_MODULES if modulation else ())
    ranks = {}
    for module_name, (a_key, b_key) in pairs.items():
        if not hybrid and not module_name.endswith(allowed):
            raise ValueError(f"Unsupported LoRA target module: {module_name}")
        try:
            module = transformer.get_submodule(module_name)
        except AttributeError as error:
            raise ValueError(f"LoRA target is absent from transformer: {module_name}") from error
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"LoRA target {module_name} is {type(module).__name__}, expected Linear")
        a_shape = checkpoint.get_slice(a_key).get_shape()
        b_shape = checkpoint.get_slice(b_key).get_shape()
        if (len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] < 1 or a_shape[0] != b_shape[1]
                or tuple(module.weight.shape) != (b_shape[0], a_shape[1])):
            raise ValueError(
                f"LoRA/base shape mismatch for {module_name}: base{tuple(module.weight.shape)}, "
                f"A{a_shape}, B{b_shape}"
            )
        ranks[module_name] = a_shape[0]
    validated_diffs = []
    for param_name, (diff_key, _) in diffs.items():
        try:
            parameter = transformer.get_parameter(param_name)
        except AttributeError as error:
            raise ValueError(f"Adapter diff target is absent: {param_name}") from error
        diff_shape = checkpoint.get_slice(diff_key).get_shape()
        if tuple(parameter.shape) != tuple(diff_shape):
            raise ValueError(
                f"Adapter diff/base mismatch for {param_name}: "
                f"base{tuple(parameter.shape)}, diff{diff_shape}"
            )
        validated_diffs.append((param_name, diff_key))

    if alpha is not None:
        alpha = _positive(alpha, "LoRA alpha")
    if hybrid:
        distinct = set(ranks.values())
        if len(distinct) != 1:
            raise ValueError(f"Mixed LoRA ranks are unsupported in FastVideo adapters: {sorted(distinct)}")
        rank = distinct.pop()
        if int(metadata.get("rank", rank)) != rank:
            raise ValueError(f"FastVideo metadata rank {metadata.get('rank')} does not match tensor rank {rank}")
        if alpha is not None and alpha != rank:
            raise ValueError(f"FastVideo adapters use alpha = rank ({rank}); an explicit alpha {alpha} contradicts it")
        alphas, source = {name: float(rank) for name in pairs}, "rank (FastVideo hybrid)"
    else:
        if alpha_keys:
            stray = sorted(alpha_keys.keys() - pairs.keys())
            missing = sorted(pairs.keys() - alpha_keys.keys())
            if stray or missing:
                raise ValueError(f"Per-layer LoRA alpha coverage mismatch: missing={missing[:3]}, stray={stray[:3]}")
            embedded = {name: _positive(checkpoint.get_tensor(alpha_keys[name]).float().item(),
                                        f"alpha of {name}") for name in pairs}
            source = "per-layer tensors"
        elif "alpha" in metadata:
            value = _positive(metadata["alpha"], f"metadata alpha of {path}")
            embedded, source = {name: value for name in pairs}, "file metadata"
        else:
            embedded = None
        if embedded is None:
            if alpha is None:
                raise ValueError(f"{path} embeds no LoRA alpha; set the adapter profile's 'alpha'")
            alphas, source = {name: alpha for name in pairs}, "explicit"
        else:
            if alpha is not None and any(value != alpha for value in embedded.values()):
                raise ValueError(f"explicit alpha {alpha} contradicts the alpha embedded in {path} ({source})")
            alphas = embedded
    return _Plan(format=adapter_format, hybrid=hybrid, alpha_source=source, diffs=tuple(validated_diffs),
                 pairs=tuple((name, *pairs[name], ranks[name], alphas[name]) for name in pairs))


def _check_arguments(path: str | Path, alpha: float | None, scale: float) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {path}")
    if alpha is not None:
        _positive(alpha, "LoRA alpha")
    if not math.isfinite(float(scale)) or float(scale) < 0:
        raise ValueError("LoRA scale must be finite and non-negative")
    return path


@torch.no_grad()
def fuse_lora(
    transformer: torch.nn.Module,
    path: str | Path,
    *,
    alpha: float | None = None,
    scale: float = 1.0,
) -> FuseReport:
    """Validate and fuse a PEFT LoRA or FastVideo v2 hybrid adapter in-place."""
    started = time.perf_counter()
    path = _check_arguments(path, alpha, scale)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        # Validate the complete checkpoint before modifying any base weight.
        plan = _plan(transformer, checkpoint, path, alpha=alpha, modulation=True)
        modules = [transformer.get_submodule(name) for name, *_ in plan.pairs]
        parameters = [transformer.get_parameter(name) for name, _ in plan.diffs]
        devices = {module.weight.device for module in modules} | {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise RuntimeError(f"LoRA targets span multiple devices: {sorted(map(str, devices))}")
        device = devices.pop()
        if device.type != "cuda":
            raise RuntimeError(
                "Refusing to fuse the adapter on CPU: the low-rank matrix products "
                "would make service startup impractically slow. Move the "
                "transformer to its rank-local CUDA device before calling fuse_lora()."
            )
        for module, (_, a_key, b_key, rank, layer_alpha) in zip(modules, plan.pairs):
            dtype = module.weight.dtype
            a = checkpoint.get_tensor(a_key).to(device=device, dtype=dtype)
            b = checkpoint.get_tensor(b_key).to(device=device, dtype=dtype)
            module.weight.addmm_(b, a, beta=1.0, alpha=float(scale) * layer_alpha / rank)
            del a, b
        for parameter, (_, diff_key) in zip(parameters, plan.diffs):
            delta = checkpoint.get_tensor(diff_key).to(device=parameter.device)
            merged = parameter.float().add_(delta.float(), alpha=float(scale))
            parameter.copy_(merged.to(dtype=parameter.dtype))
            del delta, merged

    gc.collect()
    torch.cuda.empty_cache()
    transformer.requires_grad_(False)
    transformer.eval()
    return FuseReport(
        path=str(path),
        format=plan.format,
        pairs=len(plan.pairs),
        diffs=len(plan.diffs),
        replacements=0,
        ranks=tuple(sorted({rank for *_, rank, _ in plan.pairs})),
        alpha_source=plan.alpha_source,
        alphas=tuple(sorted({layer_alpha for *_, layer_alpha in plan.pairs})),
        scale=float(scale),
        effective_scales=tuple(sorted({float(scale) * layer_alpha / rank for *_, rank, layer_alpha in plan.pairs})),
        device=str(device),
        fuse_s=time.perf_counter() - started,
    )


@torch.no_grad()
def load_lora_branches(transformer, path: str | Path, *, alpha: float | None = None, scale: float = 1.0):
    """Load the public adapter formats into native PEFT Linear branches.

    Validate every tensor before installing wrappers. Hybrid ``.diff`` updates
    apply to the base parameters in FP32, as in ``fuse_lora``; A/B remain separate.
    Branches need one rank and one alpha and cover attention/FFN projections only;
    other adapters (LightX2V 8-step v1.2) load in merged mode.
    """
    from peft import LoraConfig
    from peft.tuners.lora import Linear

    path = _check_arguments(path, alpha, scale)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        plan = _plan(transformer, checkpoint, path, alpha=alpha, modulation=False)
        ranks = {rank for *_, rank, _ in plan.pairs}
        alphas = {layer_alpha for *_, layer_alpha in plan.pairs}
        if len(ranks) != 1 or len(alphas) != 1:
            raise ValueError(f"Mixed LoRA ranks are unsupported for separate branches (ranks {sorted(ranks)}, "
                             f"alphas {sorted(alphas)}); use merged mode")
        rank, applied_alpha = ranks.pop(), alphas.pop()
        validated = []
        for name, a_key, b_key, _, _ in plan.pairs:
            module = transformer.get_submodule(name)
            if type(module) is not torch.nn.Linear or module.weight.dtype != torch.bfloat16:
                raise ValueError(f"Separate LoRA requires an unwrapped BF16 Linear: {name}")
            validated.append((name, module, a_key, b_key))
        config = LoraConfig(r=rank, lora_alpha=applied_alpha, lora_dropout=0)
        replacements = []
        for name, module, a_key, b_key in validated:
            wrapper = Linear(module, adapter_name="default", config=config,
                             r=rank, lora_alpha=applied_alpha, lora_dropout=0)
            wrapper = wrapper.to(device=module.weight.device, dtype=module.weight.dtype)
            wrapper.lora_A["default"].weight.copy_(checkpoint.get_tensor(a_key))
            wrapper.lora_B["default"].weight.copy_(checkpoint.get_tensor(b_key))
            wrapper.scaling["default"] *= float(scale)
            replacements.append((name, wrapper.requires_grad_(False).eval()))
        for name, key in plan.diffs:
            parameter = transformer.get_parameter(name)
            delta = checkpoint.get_tensor(key).to(device=parameter.device, dtype=torch.float32)
            parameter.copy_(parameter.float().add(delta, alpha=float(scale)).to(parameter.dtype))
        for name, wrapper in replacements:
            parent, _, leaf = name.rpartition(".")
            setattr(transformer.get_submodule(parent), leaf, wrapper)
    transformer.requires_grad_(False).eval()
    return {"pairs": len(replacements), "diffs": len(plan.diffs), "rank": rank, "alpha_source": plan.alpha_source,
            "effective_scale": float(scale) * applied_alpha / rank}
