from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .models import Cue


@dataclass(frozen=True)
class TimingRefinementStats:
    method: str
    changed_cues: int
    total_cues: int
    unavailable: bool = False


def _read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read the worker's PCM WAV without importing an audio backend."""
    with wave.open(str(path), "rb") as handle:
        channels = max(1, handle.getnchannels())
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2 or not frames:
        raise ValueError("timing refinement expects a 16-bit PCM WAV")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio[: len(audio) - (len(audio) % channels)].reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def _frame_db(audio: np.ndarray, sample_rate: int, frame_ms: int = 20, hop_ms: int = 10) -> tuple[np.ndarray, float]:
    frame_size = max(1, round(sample_rate * frame_ms / 1000))
    hop_size = max(1, round(sample_rate * hop_ms / 1000))
    if audio.size == 0:
        return np.empty(0, dtype=np.float32), hop_ms / 1000.0
    frame_count = max(1, 1 + math.ceil(max(0, audio.size - frame_size) / hop_size))
    padded_size = (frame_count - 1) * hop_size + frame_size
    padded = np.pad(audio, (0, max(0, padded_size - audio.size)))
    starts = np.arange(frame_count, dtype=np.int64) * hop_size
    cumulative = np.concatenate(([0.0], np.cumsum(padded * padded, dtype=np.float64)))
    energy = (cumulative[starts + frame_size] - cumulative[starts]) / frame_size
    return (10.0 * np.log10(np.maximum(energy, 1e-10))).astype(np.float32), hop_ms / 1000.0


def _active_runs(active: np.ndarray, *, min_frames: int = 4, gap_frames: int = 8) -> list[tuple[int, int]]:
    if active.size == 0:
        return []
    values = active.astype(bool, copy=True)
    index = 0
    while index < len(values):
        if values[index]:
            index += 1
            continue
        end = index
        while end < len(values) and not values[end]:
            end += 1
        if index > 0 and end < len(values) and end - index <= gap_frames:
            values[index:end] = True
        index = end
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(values):
        if not values[index]:
            index += 1
            continue
        end = index
        while end < len(values) and values[end]:
            end += 1
        if end - index >= min_frames:
            runs.append((index, end))
        index = end
    return runs


def refine_cue_timing(
    audio_path: Path,
    cues: Sequence[Cue],
    *,
    search_before_ms: int = 450,
    search_after_ms: int = 500,
    max_adjust_ms: int = 450,
) -> tuple[list[Cue], TimingRefinementStats]:
    """Refine ASR boundaries against local voice activity.

    ASR supplies the semantic interval; this pass only moves each edge to the
    nearest stable voiced run in a bounded neighborhood. That keeps pauses and
    neighboring/overlapping cues intact while removing decoder padding.
    """
    original = list(cues)
    if not original:
        return original, TimingRefinementStats("asr_timestamp+energy_vad", 0, 0)
    try:
        audio, sample_rate = _read_pcm16_mono(audio_path)
        db, hop_seconds = _frame_db(audio, sample_rate)
    except (OSError, ValueError, wave.Error):
        return original, TimingRefinementStats("asr_timestamp", 0, len(original), True)
    if db.size == 0 or not np.isfinite(db).any():
        return original, TimingRefinementStats("asr_timestamp", 0, len(original), True)

    finite = db[np.isfinite(db)]
    noise_floor = float(np.percentile(finite, 20))
    high_level = float(np.percentile(finite, 90))
    global_threshold = max(noise_floor + 6.0, min(noise_floor + 12.0, high_level - 8.0))
    duration_seconds = len(audio) / max(1, sample_rate)
    refined: list[Cue] = []
    changed = 0

    for index, cue in enumerate(original):
        start = cue.start_ms / 1000.0
        end = cue.end_ms / 1000.0
        search_start = max(0.0, start - search_before_ms / 1000.0)
        search_end = min(duration_seconds, end + search_after_ms / 1000.0)
        first = max(0, int(search_start / hop_seconds))
        last = min(len(db), max(first + 1, int(math.ceil(search_end / hop_seconds))))
        local_db = db[first:last]
        if local_db.size == 0:
            refined.append(cue)
            continue
        local_finite = local_db[np.isfinite(local_db)]
        local_noise = float(np.percentile(local_finite, 20)) if local_finite.size else noise_floor
        local_peak = float(np.percentile(local_finite, 95)) if local_finite.size else high_level
        threshold = max(
            global_threshold - 3.0,
            local_noise + 6.0,
            min(local_noise + 12.0, local_peak - 8.0),
        )
        runs = _active_runs(local_db >= threshold)
        if not runs:
            refined.append(cue)
            continue
        absolute_runs = [
            (first + run_start, first + run_end)
            for run_start, run_end in runs
        ]
        overlapping = [
            run for run in absolute_runs
            if run[1] * hop_seconds > start and run[0] * hop_seconds < end
        ]
        if overlapping:
            selected = overlapping
        else:
            midpoint = (start + end) / 2.0
            selected = [
                min(absolute_runs, key=lambda run: abs((run[0] + run[1]) * hop_seconds / 2 - midpoint))
            ]
            center = (selected[0][0] + selected[0][1]) * hop_seconds / 2
            if abs(center - midpoint) > 0.35:
                refined.append(cue)
                continue
        candidate_start = max(0.0, selected[0][0] * hop_seconds - 0.02)
        candidate_end = min(duration_seconds, selected[-1][1] * hop_seconds + 0.02)
        candidate_start = max(start - max_adjust_ms / 1000.0, candidate_start)
        candidate_end = min(end + max_adjust_ms / 1000.0, candidate_end)

        # Do not let a non-overlapping cue consume its neighbor's interval.
        if index and original[index - 1].end_ms <= cue.start_ms:
            candidate_start = max(candidate_start, original[index - 1].end_ms / 1000.0)
        if index + 1 < len(original) and cue.end_ms <= original[index + 1].start_ms:
            candidate_end = min(candidate_end, original[index + 1].start_ms / 1000.0)
        new_start = max(0, round(candidate_start * 1000))
        new_end = max(new_start + 1, round(candidate_end * 1000))
        if new_end - new_start < 80:
            refined.append(cue)
            continue
        if new_start != cue.start_ms or new_end != cue.end_ms:
            changed += 1
            refined.append(cue.model_copy(update={"start_ms": new_start, "end_ms": new_end}))
        else:
            refined.append(cue)
    return refined, TimingRefinementStats("asr_timestamp+energy_vad", changed, len(original))
