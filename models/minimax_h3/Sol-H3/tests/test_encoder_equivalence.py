"""Background publication encodes exactly what synchronous publication encodes. Needs CUDA, PyAV, and the
service's Diffusers, so run it in an allocation on a GPU no worker is using (about 1 GiB).

The same fixed source tensors go through GeneratedMedia.save (the synchronous path) and through save_async
plus finish_output on another thread (the background path). The MP4 container bytes need not match; the
decoded video frames and audio samples must.
"""
import math
import threading

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocation required")
FRAMES, SIZE, RATE = 124, 768, 32000  # 5.17 s on the 1:1 canvas


def decode(path):
    import av
    import numpy as np
    with av.open(str(path)) as container:
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    with av.open(str(path)) as container:
        samples = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=-1)
    return frames, samples


def test_background_publication_decodes_to_the_synchronous_frames_and_audio(tmp_path):
    import numpy as np
    from benchmark import inspect_media
    from h3_runtime.engine import GeneratedMedia
    from output_publication import PublicationTask, finish_output, partial_path

    # Smooth moving gradients and stereo tones: deterministic, and quick for x264 and AAC.
    t = torch.arange(FRAMES, device="cuda").view(-1, 1, 1, 1) / FRAMES
    y = torch.linspace(0, 1, SIZE, device="cuda").view(1, 1, -1, 1)
    x = torch.linspace(0, 1, SIZE, device="cuda").view(1, 1, 1, -1)
    channel = torch.tensor([0.0, 0.33, 0.66], device="cuda").view(1, 3, 1, 1)
    video = torch.remainder(x + 0.5 * y + t + channel, 1.0)
    seconds = torch.arange(round(FRAMES / 24 * RATE)) / RATE
    audio = torch.stack([torch.sin(2 * math.pi * 440 * seconds), torch.sin(2 * math.pi * 660 * seconds)]) * 0.5

    def media():  # save_async only reads the source tensors
        return GeneratedMedia(video=video, audio=audio, audio_sample_rate=RATE, elapsed_s=0.0,
                              duration=FRAMES / 24, seed=0)

    synchronous = tmp_path / "synchronous.mp4"
    media().save(synchronous, encoder_threads=8)
    background = tmp_path / "background.mp4"
    task = PublicationTask(job_id="equivalence", encode=media().save_async(partial_path(background), encoder_threads=8),
                           path=background, expected={"width": SIZE, "height": SIZE, "duration_seconds": FRAMES / 24},
                           result={}, started=0.0, staged=0.0)
    outcome = {}
    publisher = threading.Thread(target=lambda: outcome.update(finish_output(task, inspect_media)))
    publisher.start()
    publisher.join(600)
    assert outcome["bytes"] == background.stat().st_size and not partial_path(background).exists()
    assert (outcome["media"]["width"], outcome["media"]["sample_rate"], outcome["media"]["channels"]) == (SIZE, RATE, 2)
    expected_frames, expected_samples = decode(synchronous)
    frames, samples = decode(background)
    assert frames.shape == (FRAMES, SIZE, SIZE, 3) and np.array_equal(frames, expected_frames)
    assert samples.shape == expected_samples.shape and np.array_equal(samples, expected_samples)
