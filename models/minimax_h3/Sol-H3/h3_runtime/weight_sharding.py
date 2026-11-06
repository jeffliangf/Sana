"""Split the repeated blocks' weights across a worker's GPUs and gather each block as it runs.

One rank's pipeline does not fit on an 80 GB H100: the Qwen3-VL text encoder is 62 GiB in BF16
and the DiT 62 GiB (36 GiB of blocks once the AdaLN table replaces `adaln_proj`), and Ulysses
splits activations, not weights. Here every rank keeps 1/world of each text-encoder decoder
layer and each DiT block as one flat shard. A forward pre-hook all-gathers the block into one of
two full-size buffers, which the block's parameters view permanently, and starts gathering the
next block on a side stream while this one runs. Values are copied, never transformed, so each
block computes exactly what it computes with replicated weights.

Between gathers a block's parameters show whichever block last used its buffer, so nothing may
read a sharded parameter outside that block's own forward. Modules that are read from outside
(`adaln_proj`, which the AdaLN precompute evaluates once and then frees) are excluded and stay
replicated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import nn

# MiniMax-H3 conditions on `hidden_states[50]` of its Qwen3-VL text encoder (diffusers
# `minimax_h3/encoders.py`). transformers records the input of the first decoder layer and the
# output of every layer, then replaces the last entry with the final-norm output. With 51 layers,
# entry 50 is still the raw output of layer 49, exactly as with all 64; layers 51-63 and the LM
# head are never read. Keep 51 rather than 50, or entry 50 would become the normed output.
TEXT_ENCODER_LAYERS_KEPT = 51
_ALIGN = 128  # elements; keeps every parameter view and all-gather chunk 256-byte aligned


def truncate_text_encoder(text_encoder, keep: int = TEXT_ENCODER_LAYERS_KEPT) -> int:
    """Drop the decoder layers after `keep` and the unused LM head. Returns the bytes released."""
    layers = text_encoder.model.language_model.layers
    if len(layers) < keep:
        raise ValueError(f"text encoder has {len(layers)} decoder layers, fewer than {keep}")
    dropped = [*layers[keep:], text_encoder.lm_head]
    released = sum(p.numel() * p.element_size() for module in dropped for p in module.parameters())
    del layers[keep:]
    # The pipeline calls `text_encoder.model(...)` and never the head; Identity keeps the attribute valid.
    text_encoder.lm_head = nn.Identity()
    return released


@dataclass(eq=False)
class _Block:
    module: nn.Module
    params: list[tuple[nn.Parameter, int]]  # (parameter, element offset in the flat block)
    padded: int  # flat length, a multiple of world * _ALIGN
    slot: int
    shard: torch.Tensor | None = None
    next: "_Block | None" = None


@dataclass(eq=False)
class _Slot:
    buffer: torch.Tensor
    ready: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    free: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    owner: _Block | None = None


class BlockSharding:
    """Shard sequences of blocks over `group`; each sequence must run its blocks in order."""

    def __init__(self, device: torch.device, group=None) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("weight sharding needs an initialized process group")
        # A separate communicator lets the gathers overlap the Ulysses all-to-alls on the default one.
        self.group = group if group is not None else dist.new_group(backend="nccl")
        self.world = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        if self.world < 2:
            raise ValueError("weight sharding needs at least two ranks")
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(self.device)
        self.blocks: list[_Block] = []
        self.slots: list[_Slot] = []
        self.shard_bytes = 0
        self.released_bytes = 0

    def shard(self, sequences, exclude: dict[int, tuple[str, ...]] | None = None) -> dict:
        """Shard every block of each sequence (lists of modules). `exclude[i]` names submodules of
        sequence i's blocks to leave replicated. Parameters may be on the CPU or this GPU."""
        exclude = exclude or {}
        plans, seen, dtype = [], set(), None
        for index, blocks in enumerate(sequences):
            skip = tuple(f"{name}." for name in exclude.get(index, ()))
            previous = None
            for position, module in enumerate(blocks):
                params, offset = [], 0
                for name, param in module.named_parameters():
                    if name.startswith(skip):
                        continue
                    if id(param) in seen:
                        raise ValueError(f"parameter {name} is shared between blocks")
                    seen.add(id(param))
                    if dtype is None:
                        dtype = param.dtype
                    if param.dtype != dtype:
                        raise TypeError(f"{name} is {param.dtype}; sharded blocks must all be {dtype}")
                    params.append((param, offset))
                    offset += -(-param.numel() // _ALIGN) * _ALIGN
                if not params:
                    raise ValueError(f"block {position} of sequence {index} has no parameters to shard")
                unit = self.world * _ALIGN
                block = _Block(module, params, -(-offset // unit) * unit, slot=position % 2)
                if previous is not None:
                    previous.next = block
                previous = block
                plans.append(block)
        if not plans:
            raise ValueError("nothing to shard")
        # Two buffers: the running block's and the one being gathered next.
        largest = max(block.padded for block in plans)
        self.slots = [_Slot(torch.empty(largest, dtype=dtype, device=self.device)) for _ in range(2)]
        for block in plans:
            self._shard_block(block)
        self.blocks = plans
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        return {
            "blocks": len(plans),
            "world": self.world,
            "shard_gib": self.shard_bytes / 2**30,
            "buffers_gib": sum(s.buffer.numel() * s.buffer.element_size() for s in self.slots) / 2**30,
            "released_gib": self.released_bytes / 2**30,
        }

    @torch.no_grad()
    def _shard_block(self, block: _Block) -> None:
        size = block.padded // self.world
        start, stop = self.rank * size, (self.rank + 1) * size
        shard = torch.zeros(size, dtype=self.slots[0].buffer.dtype, device=self.device)
        buffer = self.slots[block.slot].buffer
        for param, offset in block.params:
            numel = param.numel()
            low, high = max(start, offset), min(stop, offset + numel)
            if low < high:
                shard[low - start:high - start].copy_(param.data.reshape(-1)[low - offset:high - offset])
            self.released_bytes += numel * param.element_size()
            # The parameter now views its slot; the full tensor it held is released.
            param.requires_grad_(False)
            param.data = buffer[offset:offset + numel].view(param.shape)
        block.shard = shard
        self.shard_bytes += shard.numel() * shard.element_size()
        block.module.register_forward_pre_hook(lambda module, args, block=block: self._before(block))
        block.module.register_forward_hook(lambda module, args, output, block=block: self._after(block))

    def _gather(self, block: _Block) -> None:
        slot = self.slots[block.slot]
        with torch.cuda.stream(self.stream):
            # Wait until the compute stream has finished reading the slot's previous block.
            self.stream.wait_event(slot.free)
            dist.all_gather_into_tensor(slot.buffer[:block.padded], block.shard, group=self.group)
            slot.ready.record(self.stream)
        slot.owner = block

    def _before(self, block: _Block) -> None:
        slot = self.slots[block.slot]
        if slot.owner is not block:
            self._gather(block)
        torch.cuda.current_stream(self.device).wait_event(slot.ready)
        following = block.next
        if following is not None and self.slots[following.slot].owner is not following:
            self._gather(following)

    def _after(self, block: _Block) -> None:
        self.slots[block.slot].free.record(torch.cuda.current_stream(self.device))
