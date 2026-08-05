from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from backend.models import Cue
from backend.timing import refine_cue_timing


def _cue(cue_id: str, start_ms: int, end_ms: int) -> Cue:
    return Cue(
        cue_id=cue_id,
        start_ms=start_ms,
        end_ms=end_ms,
        source_kind="speech",
        source_text="fixture speech",
    )


def _write_voice_fixture(path: Path, intervals: list[tuple[float, float]], duration: float = 2.2) -> None:
    sample_rate = 16_000
    samples = np.zeros(round(duration * sample_rate), dtype=np.float32)
    for start, end in intervals:
        first = round(start * sample_rate)
        last = round(end * sample_rate)
        time = np.arange(last - first, dtype=np.float32) / sample_rate
        samples[first:last] = 0.42 * np.sin(2 * math.pi * 220 * time)
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_refinement_snaps_rough_asr_edges_to_voice_activity(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    _write_voice_fixture(audio, [(0.50, 1.50)])

    refined, stats = refine_cue_timing(audio, [_cue("ASR1", 350, 1680)])

    assert stats.method == "asr_timestamp+energy_vad"
    assert stats.changed_cues == 1
    assert 460 <= refined[0].start_ms <= 520
    assert 1480 <= refined[0].end_ms <= 1540


def test_refinement_does_not_cross_non_overlapping_neighbor(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    _write_voice_fixture(audio, [(0.40, 0.95), (1.10, 1.70)])
    cues = [_cue("ASR1", 300, 1000), _cue("ASR2", 1050, 1800)]

    refined, _stats = refine_cue_timing(audio, cues)

    assert refined[0].end_ms <= cues[1].start_ms
    assert refined[1].start_ms >= cues[0].end_ms


def test_refinement_falls_back_when_audio_is_unavailable(tmp_path: Path) -> None:
    cue = _cue("ASR1", 300, 1000)

    refined, stats = refine_cue_timing(tmp_path / "missing.wav", [cue])

    assert refined == [cue]
    assert stats.unavailable is True
    assert stats.method == "asr_timestamp"
