"""Weight sharding must reproduce replicated outputs bit for bit (GPU) and keep hidden_states[50] (CPU)."""

from __future__ import annotations

import os
import socket

import pytest
import torch
from torch import nn


class _Block(nn.Module):
    """Odd sizes exercise the alignment padding; `gate` stands in for an excluded `adaln_proj`."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.up = nn.Linear(width, 3 * width + 5)
        self.down = nn.Linear(3 * width + 5, width, bias=False)
        self.gate = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate(x).sigmoid() * self.down(torch.nn.functional.silu(self.up(self.norm(x))))


def _models(seed: int):
    torch.manual_seed(seed)
    first = nn.ModuleList(_Block(96) for _ in range(5))  # odd length, like the 51 text layers
    second = nn.ModuleList(_Block(64) for _ in range(4))
    return first, second


def _run(first, second, x):
    y = x
    for block in first:
        y = block(y)
    z = y[..., :64].contiguous()
    for block in second:
        z = block(z)
    return y, z


def _worker(rank: int, world: int, port: int, failures) -> None:
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
        from h3_runtime.weight_sharding import BlockSharding

        reference_first, reference_second = _models(0)
        reference_first.to(device, torch.bfloat16)
        reference_second.to(device, torch.bfloat16)
        first, second = _models(0)
        first.to(torch.bfloat16)  # left on the CPU, like the text encoder
        second.to(device, torch.bfloat16)  # already on the GPU, like the DiT
        gate = first[0].gate.weight
        sharding = BlockSharding(device)
        stats = sharding.shard([first, second], exclude={0: ("gate",)})
        assert stats["blocks"] == 9
        assert first[0].gate.weight is gate and gate.device.type == "cpu"  # excluded: untouched
        for block in first:
            block.gate.to(device)
        assert first[1].up.weight.data_ptr() != first[0].up.weight.data_ptr()  # parity slots
        assert first[2].up.weight.data_ptr() == first[0].up.weight.data_ptr()
        torch.manual_seed(1)
        x = torch.randn(2, 33, 96, device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = _run(reference_first, reference_second, x)
            for _ in range(3):  # later passes re-gather blocks whose slot was reused
                actual = _run(first, second, x)
                for want, got in zip(expected, actual):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
        torch.distributed.destroy_process_group()
    except BaseException as error:  # surface the rank's failure in the parent
        failures.put(f"rank {rank}: {error!r}")
        raise


@pytest.mark.parametrize("world", [2, 4])
def test_sharded_blocks_match_replicated_bitwise(world):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    import torch.multiprocessing as mp

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    context = mp.get_context("spawn")
    failures = context.SimpleQueue()
    mp.start_processes(_worker, args=(world, port, failures), nprocs=world, start_method="spawn")
    assert failures.empty()


def test_truncated_text_encoder_keeps_hidden_state_50():
    transformers = pytest.importorskip("transformers")
    from h3_runtime.weight_sharding import TEXT_ENCODER_LAYERS_KEPT, truncate_text_encoder

    config = transformers.Qwen3VLConfig(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=64, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, vocab_size=320,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[0]),
    )
    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(config).eval()
    ids = torch.randint(0, 300, (1, 23))
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=True)
    with torch.no_grad():
        full = model.model(**kwargs).hidden_states
        truncate_text_encoder(model)
        truncated = model.model(**kwargs).hidden_states
    assert len(model.model.language_model.layers) == TEXT_ENCODER_LAYERS_KEPT
    assert len(full) == 65 and len(truncated) == TEXT_ENCODER_LAYERS_KEPT + 1
    torch.testing.assert_close(truncated[50], full[50], rtol=0, atol=0)
    assert not torch.equal(full[64], full[63])  # the last entry really is the normed one
