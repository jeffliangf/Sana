"""A Ref2VA video reference reaches the model upright, as ffmpeg (MiniMax-H3's reference decode) shows it."""
import numpy as np
import pytest


def test_quarter_turn_reference_videos_come_out_upright():
    pytest.importorskip("torch")
    from h3_runtime.engine import upright_reference_video

    coded = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)  # frames, height, width, RGB
    for rotation in (-90, 90, 180, -180, 0):
        turns = round(rotation / 90) % 4

        def diffusers_decode(media, turns=turns):  # Diffusers turns the frames by k=-turns
            return np.ascontiguousarray(np.rot90(coded, k=-turns, axes=(1, 2))), 30.0, None

        decode = upright_reference_video(diffusers_decode, rotation_of=lambda media, value=rotation: value)
        frames, rate, soundtrack = decode("clip.mov")
        # ffmpeg rotates the decoded frame counterclockwise by the display matrix angle.
        assert np.array_equal(frames, np.rot90(coded, k=turns, axes=(1, 2))), rotation
        assert frames.flags["C_CONTIGUOUS"] and rate == 30.0 and soundtrack is None
