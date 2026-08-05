from __future__ import annotations

import subprocess
from pathlib import Path

from backend import media


def test_probe_video_falls_back_to_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"fixture")
    diagnostic = """
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'sample.mp4':
  Duration: 00:01:02.50, start: 0.000000, bitrate: 900 kb/s
  Stream #0:0[0x1](und): Video: h264 (High), yuv420p, 1920x1080, 23.98 fps
  Stream #0:1[0x2](jpn): Audio: aac (LC), 48000 Hz, stereo
  Stream #0:2[0x3](eng): Subtitle: mov_text (tx3g / 0x67337874)
"""
    monkeypatch.setattr(media, "find_binary", lambda name: None)
    monkeypatch.setattr(
        media,
        "_run_media_command",
        lambda command: subprocess.CompletedProcess(command, 1, "", diagnostic),
    )

    result = media.probe_video(source, ffmpeg=Path("ffmpeg.exe"))

    assert result.duration_ms == 62_500
    assert (result.width, result.height) == (1920, 1080)
    assert result.frame_rate == 23.98
    assert result.video_codec == "h264"
    assert result.audio_codec == "aac"
    assert result.subtitle_streams == (2,)
    assert result.has_subtitle_stream is True
