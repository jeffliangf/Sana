"""Numerical rejection must never publish a rejected kernel output."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch


# Load only the attention module; these CPU tests do not need model weights.
root = Path(__file__).resolve().parents[1] / "h3_runtime"
package = types.ModuleType("h3_gate_test_runtime")
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(package.__name__ + ".sparse_attention", root / "sparse_attention.py")
attention = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = attention
spec.loader.exec_module(attention)


@pytest.fixture
def slot(monkeypatch):
    for name in ("SOL_ATTN_GATE_MAX_ABS", "SOL_ATTN_GATE_MEAN_ABS", "SOL_ATTN_GATE_REL_L2"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SOL_ATTN_GATE_FAILURE", "dense")
    monkeypatch.setenv("H3_SOL_COMPILE_BUCKET_SIZE", "0")
    monkeypatch.setenv("SOL_ATTN_STRICT", "1")
    return attention.H3SparseAttention(gate=True)


def test_outlier_rejection_keeps_limits_and_caches_dense(slot):
    want = torch.full((1, 32768, 1, 1), 32, dtype=torch.bfloat16)
    got = want.clone()
    got[0, 0, 0, 0] += 0.25
    key = slot._gate_key(want)
    assert slot._finish_gate(key, got, want) is False
    assert slot.gate_stats["max_abs"] == 0.25
    assert slot.gate_stats["limits"]["max_abs"] == 0.15
    assert slot.gate_stats["mean_abs"] < 0.002
    assert slot.gate_stats["rel_l2"] < 0.005
    assert slot._cached_gate(key) is False
    assert key not in slot._gated_shapes


def test_default_benchmark_policy_still_raises(monkeypatch):
    monkeypatch.delenv("SOL_ATTN_GATE_FAILURE", raising=False)
    slot = attention.H3SparseAttention(gate=True)
    want = torch.ones(1, 4, 1, 1)
    with pytest.raises(RuntimeError, match="correctness gate failed"):
        slot._finish_gate(slot._gate_key(want), want + 1, want)
    assert not slot._rejected_shapes


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_reference_cannot_fall_back(slot, bad):
    want = torch.full((1, 4, 1, 1), bad)
    with pytest.raises(RuntimeError, match="nonfinite dense reference"):
        slot._finish_gate(slot._gate_key(want), torch.ones_like(want), want)
    assert not slot._rejected_shapes


@pytest.mark.parametrize("peer_rejected", [False, True])
def test_rejected_call_returns_dense_and_never_retries_kernel(slot, monkeypatch, peer_rejected):
    monkeypatch.setattr(slot, "_declined", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    calls = []

    def kernel(q, k, v, **kwargs):
        calls.append(kwargs["tau"])
        result = attention._dense(q[0], k[0], v[0]).unsqueeze(0)
        return result if peer_rejected else result + 0.25

    monkeypatch.setattr(attention, "_import_kernel", lambda: kernel)
    if peer_rejected:
        verdicts = iter([True, False])  # finite reference; another rank rejects
        monkeypatch.setattr(attention, "_agree", lambda passed: next(verdicts))
    q, k, v = (torch.randn(16, 1, 128).bfloat16() for _ in range(3))
    want = attention._dense(q, k, v)
    for _ in range(2):
        torch.testing.assert_close(slot(q, k, v), want, rtol=0, atol=0)
    assert calls == [-1000.0]  # no production call and no retry of the rejected shape
    assert slot.sparse_calls == 0
    assert slot.gate_fallback_calls == slot.dense_calls == 2
    assert slot.gate_stats["local_passed"] == peer_rejected
    slot.request, slot.step = 0, 3
    slot._close_request()  # an explicitly reported dense request is allowed
    assert slot.last_request_gate_fallback_calls == 2
    with pytest.raises(RuntimeError, match="without one sparse call"):
        slot._close_request()  # an accidental all-dense request still fails


def test_bsa_rejection_returns_dense(slot, monkeypatch):
    slot.backend = "sol_bsa"
    monkeypatch.setattr(slot, "_declined", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))

    def bsa(q, k, v, *args, **kwargs):
        return {"o_tensor": attention._dense(q[0], k[0], v[0]).unsqueeze(0) + 0.25}

    monkeypatch.setattr(attention, "_import_bsa", lambda: types.SimpleNamespace(block_sparse_attention_forward=bsa))
    q, k, v = (torch.randn(17, 1, 128).bfloat16() for _ in range(3))
    torch.testing.assert_close(slot(q, k, v), attention._dense(q, k, v), rtol=0, atol=0)
    assert slot.gate_fallback_calls == 1 and slot.sparse_calls == 0


def test_service_rechecks_passed_shapes_between_requests(slot):
    slot.gate_each_request = True
    slot._gated_shapes.add(("sol", 16, 1, 128))
    slot._rejected_shapes[("sol", 17, 1, 128)] = {"passed": False}
    slot.observe(torch.arange(2, 16), torch.arange(2), None, torch.zeros(16, 3), torch.tensor([0.0]))
    assert not slot._gated_shapes
    assert slot._cached_gate(("sol", 17, 1, 128)) is False


def test_passing_gate_is_cached(slot):
    want = torch.ones(1, 4, 1, 1)
    key = slot._gate_key(want)
    assert slot._finish_gate(key, want, want)
    assert slot._cached_gate(key)
    assert slot.gate_failures == 0


def test_nonfinite_kernel_diagnostics_can_be_saved(slot):
    want = torch.ones(1, 4, 1, 1)
    assert not slot._finish_gate(slot._gate_key(want), torch.full_like(want, float("nan")), want)
    json.dumps(slot.gate_stats, allow_nan=False)
    assert slot.gate_stats["max_abs"] is None


def test_kernel_exception_is_not_suppressed(slot, monkeypatch):
    monkeypatch.setattr(slot, "_declined", lambda *args: None)

    def broken(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(attention, "_import_kernel", lambda: broken)
    q = torch.ones(16, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        slot(q, q, q)
    assert slot.gate_fallback_calls == 0
