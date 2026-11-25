"""Small real media files for the Ref2VA reference tests: PyAV writes the MP4s, the wave module the WAVs."""
import io
import wave

import numpy as np


def wav(seconds, channels=1, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * channels * round(seconds * rate))
    return buffer.getvalue()


def mp4(seconds, width=64, height=48, fps=24, audio_channels=2, rate=16000):
    """An MPEG-4 Part 2 video, with an AAC soundtrack of ``audio_channels`` (0 for a silent clip)."""
    import av

    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        video = container.add_stream("mpeg4", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        layout = {1: "mono", 2: "stereo"}.get(audio_channels)
        audio = container.add_stream("aac", rate=rate, layout=layout) if audio_channels else None
        for index in range(round(seconds * fps)):
            pixels = np.full((height, width, 3), (index * 7) % 256, np.uint8)
            for packet in video.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio is not None:
            total, start = round(seconds * rate), 0
            while start < total:
                count = min(1024, total - start)
                frame = av.AudioFrame.from_ndarray(np.zeros((audio_channels, count), np.float32), format="fltp",
                                                   layout=layout)
                frame.sample_rate, frame.pts = rate, start
                for packet in audio.encode(frame):
                    container.mux(packet)
                start += count
            for packet in audio.encode():
                container.mux(packet)
    return buffer.getvalue()
