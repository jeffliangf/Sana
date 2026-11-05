"""Memory-lean replacement for the MiniMax-H3 video decode step's post-processing.

Diffusers' ``MiniMaxH3VideoDecodeStep`` turns the float16 decode into float32 with
``(video.float() * std + mean).clamp(0, 1)`` and then ``postprocess_video(..., "pt")``, which
for this processor (``do_normalize=False``) only permutes each video and ``torch.stack``s it.
That holds the float16 decode plus up to two float32 copies at once, and the stack makes a
third: about 4.3 GB each for a 345-frame 1344x768 clip on every rank.

This version applies the same float32 operations in place and returns the permuted view, so
the values are bit-identical while at most one float16 and one float32 copy coexist. Other
output types still run the original step.
"""

from __future__ import annotations

import torch


def install():
    """Patch ``MiniMaxH3VideoDecodeStep.__call__``; returns an uninstall callable."""
    from diffusers.modular_pipelines.minimax_h3 import decoders
    from diffusers.modular_pipelines.minimax_h3.packing import (
        MINIMAX_H3_PIXEL_MEAN,
        MINIMAX_H3_PIXEL_STD,
        unpatchify_video_tokens,
    )

    step = decoders.MiniMaxH3VideoDecodeStep
    if getattr(step, "_h3_lean_decode", False):
        return lambda: None
    original_call = step.__call__

    @torch.no_grad()
    def lean_call(self, components, state):
        block_state = self.get_block_state(state)
        if block_state.output_type != "pt":
            return original_call(self, components, state)
        device = components._execution_device
        latents = unpatchify_video_tokens(
            block_state.latents[block_state.num_condition_video_rows :],
            block_state.num_latent_frames,
            block_state.latent_height,
            block_state.latent_width,
            components.vae_latent_channels,
            components.patch_size,
        )
        latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latents = latents * latents_std + latents_mean
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            video = components.vae.decode(latents, return_dict=False)[0]
        del latents
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        frames = video.float()
        del video
        frames.mul_(pixel_std).add_(pixel_mean).clamp_(0, 1)
        # postprocess_video(frames, "pt") == stack of frames[b].permute(1, 0, 2, 3): the same view, uncopied.
        block_state.videos = frames.permute(0, 2, 1, 3, 4)
        self.set_block_state(state, block_state)
        return components, state

    step.__call__ = lean_call
    step._h3_lean_decode = True

    def uninstall():
        step.__call__ = original_call
        step._h3_lean_decode = False

    return uninstall
