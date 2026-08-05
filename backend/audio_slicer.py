from __future__ import annotations

import json
import math
import os
import wave
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio_pipeline import AudioPipelineError


# RMS and silence-boundary decisions are adapted from GPT-SoVITS tools/slicer2.py
# (MIT, Copyright (c) 2024 RVC-Boss). WAV I/O and manifest handling are local.
ProgressCallback = Callable[[float, str], None]
DEFAULT_MANIFEST_FILENAME = "slices.json"


@dataclass(frozen=True)
class SlicerSettings:
    threshold_db: float = -34.0
    min_length_ms: int = 4_000
    min_interval_ms: int = 200
    hop_size_ms: int = 10
    max_sil_kept_ms: int = 500
    max_length_ms: int = 30_000

    def validate(self) -> None:
        values = (
            self.threshold_db,
            self.min_length_ms,
            self.min_interval_ms,
            self.hop_size_ms,
            self.max_sil_kept_ms,
            self.max_length_ms,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("slicer settings must be finite")
        if self.min_length_ms < self.min_interval_ms:
            raise ValueError("min_length_ms must be at least min_interval_ms")
        if self.min_interval_ms < self.hop_size_ms:
            raise ValueError("min_interval_ms must be at least hop_size_ms")
        if self.max_sil_kept_ms < self.hop_size_ms:
            raise ValueError("max_sil_kept_ms must be at least hop_size_ms")
        if self.max_length_ms < self.min_length_ms:
            raise ValueError("max_length_ms must be at least min_length_ms")
        if self.hop_size_ms <= 0:
            raise ValueError("hop_size_ms must be positive")


@dataclass(frozen=True)
class AudioSlice:
    index: int
    path: Path
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "path": str(self.path.resolve()),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class AudioSliceManifest:
    manifest_path: Path
    source_path: Path
    sample_rate: int
    channels: int
    duration_ms: int
    settings: SlicerSettings
    slices: tuple[AudioSlice, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "source_path": str(self.source_path.resolve()),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_ms": self.duration_ms,
            "parameters": asdict(self.settings),
            "slices": [item.to_dict() for item in self.slices],
        }


def get_rms(
    samples: np.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Return centered frame RMS without importing librosa."""
    if samples.ndim != 1:
        raise ValueError("get_rms expects mono samples")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    source = samples.astype(np.float64, copy=False)
    padded = np.pad(source, (frame_length // 2, frame_length // 2))
    frame_count = max(1, 1 + (len(padded) - frame_length) // hop_length)
    starts = np.arange(frame_count, dtype=np.int64) * hop_length
    squared = padded * padded
    cumulative = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))
    energy = (cumulative[starts + frame_length] - cumulative[starts]) / frame_length
    return np.sqrt(np.maximum(energy, 0.0))


class Slicer:
    def __init__(self, sample_rate: int, settings: SlicerSettings | None = None):
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.settings = settings or SlicerSettings()
        self.settings.validate()
        self.threshold = 10 ** (self.settings.threshold_db / 20.0)
        self.hop_samples = max(
            1, round(self.sample_rate * self.settings.hop_size_ms / 1000)
        )
        min_interval_samples = round(
            self.sample_rate * self.settings.min_interval_ms / 1000
        )
        self.window_samples = max(
            1, min(min_interval_samples, 4 * self.hop_samples)
        )
        self.min_length_frames = max(
            1,
            round(
                self.sample_rate
                * self.settings.min_length_ms
                / 1000
                / self.hop_samples
            ),
        )
        self.min_interval_frames = max(
            1, round(min_interval_samples / self.hop_samples)
        )
        self.max_sil_kept_frames = max(
            1,
            round(
                self.sample_rate
                * self.settings.max_sil_kept_ms
                / 1000
                / self.hop_samples
            ),
        )

    def slice_ranges(self, waveform: np.ndarray) -> list[tuple[int, int]]:
        if waveform.ndim == 1:
            mono = waveform
            sample_count = waveform.shape[0]
        elif waveform.ndim == 2:
            mono = waveform.mean(axis=0)
            sample_count = waveform.shape[1]
        else:
            raise ValueError("waveform must have shape [samples] or [channels, samples]")
        if sample_count == 0:
            return []

        rms = get_rms(
            mono,
            frame_length=self.window_samples,
            hop_length=self.hop_samples,
        )
        if len(rms) <= self.min_length_frames:
            return [(0, sample_count)]

        silence_tags: list[tuple[int, int]] = []
        silence_start: int | None = None
        clip_start = 0
        for index, value in enumerate(rms):
            if value < self.threshold:
                if silence_start is None:
                    silence_start = index
                continue
            if silence_start is None:
                continue

            leading = silence_start == 0 and index > self.max_sil_kept_frames
            middle = (
                index - silence_start >= self.min_interval_frames
                and index - clip_start >= self.min_length_frames
            )
            if not leading and not middle:
                silence_start = None
                continue

            silence_length = index - silence_start
            if silence_length <= self.max_sil_kept_frames:
                position = int(np.argmin(rms[silence_start : index + 1])) + silence_start
                silence_tags.append((0, position) if silence_start == 0 else (position, position))
                clip_start = position
            elif silence_length <= self.max_sil_kept_frames * 2:
                center = (
                    int(
                        np.argmin(
                            rms[
                                index
                                - self.max_sil_kept_frames : silence_start
                                + self.max_sil_kept_frames
                                + 1
                            ]
                        )
                    )
                    + index
                    - self.max_sil_kept_frames
                )
                left = (
                    int(
                        np.argmin(
                            rms[
                                silence_start : silence_start
                                + self.max_sil_kept_frames
                                + 1
                            ]
                        )
                    )
                    + silence_start
                )
                right = (
                    int(
                        np.argmin(
                            rms[index - self.max_sil_kept_frames : index + 1]
                        )
                    )
                    + index
                    - self.max_sil_kept_frames
                )
                if silence_start == 0:
                    silence_tags.append((0, right))
                    clip_start = right
                else:
                    silence_tags.append((min(left, center), max(right, center)))
                    clip_start = max(right, center)
            else:
                left = (
                    int(
                        np.argmin(
                            rms[
                                silence_start : silence_start
                                + self.max_sil_kept_frames
                                + 1
                            ]
                        )
                    )
                    + silence_start
                )
                right = (
                    int(
                        np.argmin(
                            rms[index - self.max_sil_kept_frames : index + 1]
                        )
                    )
                    + index
                    - self.max_sil_kept_frames
                )
                silence_tags.append((0, right) if silence_start == 0 else (left, right))
                clip_start = right
            silence_start = None

        total_frames = len(rms)
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval_frames
        ):
            silence_end = min(
                total_frames, silence_start + self.max_sil_kept_frames
            )
            position = (
                int(np.argmin(rms[silence_start : silence_end + 1]))
                + silence_start
            )
            silence_tags.append((position, total_frames + 1))

        raw_ranges: list[tuple[int, int]] = []
        if not silence_tags:
            raw_ranges.append((0, sample_count))
        else:
            if silence_tags[0][0] > 0:
                raw_ranges.append((0, silence_tags[0][0] * self.hop_samples))
            for current, following in zip(silence_tags, silence_tags[1:]):
                raw_ranges.append(
                    (current[1] * self.hop_samples, following[0] * self.hop_samples)
                )
            if silence_tags[-1][1] < total_frames:
                raw_ranges.append(
                    (silence_tags[-1][1] * self.hop_samples, sample_count)
                )

        bounded = [
            (max(0, start), min(sample_count, end))
            for start, end in raw_ranges
            if end > start
        ]
        return self._enforce_max_length(bounded, rms)

    def _enforce_max_length(
        self, ranges: Sequence[tuple[int, int]], rms: np.ndarray
    ) -> list[tuple[int, int]]:
        maximum = round(self.sample_rate * self.settings.max_length_ms / 1000)
        minimum = round(self.sample_rate * self.settings.min_length_ms / 1000)
        search = round(self.sample_rate * self.settings.max_sil_kept_ms / 1000)
        result: list[tuple[int, int]] = []
        for range_start, range_end in ranges:
            cursor = range_start
            while range_end - cursor > maximum:
                target = cursor + maximum
                if range_end - target < minimum:
                    target = max(cursor + minimum, range_end - minimum)
                lower = max(cursor + minimum, target - search)
                upper = max(lower, target)
                first_frame = max(0, lower // self.hop_samples)
                last_frame = min(len(rms), upper // self.hop_samples + 1)
                if last_frame > first_frame:
                    boundary = (
                        first_frame
                        + int(np.argmin(rms[first_frame:last_frame]))
                    ) * self.hop_samples
                    boundary = min(target, max(cursor + minimum, boundary))
                else:
                    boundary = target
                result.append((cursor, boundary))
                cursor = boundary
            if range_end > cursor:
                result.append((cursor, range_end))
        return result


def _read_pcm16_wav(path: Path) -> tuple[np.ndarray, int, int]:
    source = path.resolve()
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise AudioPipelineError(f"audio slicing could not read WAV: {exc}") from exc
    if sample_width != 2:
        raise AudioPipelineError("audio slicing requires a 16-bit PCM WAV")
    if channels <= 0 or sample_rate <= 0 or not raw:
        raise AudioPipelineError("audio slicing received an empty WAV")
    values = np.frombuffer(raw, dtype="<i2")
    usable = len(values) - (len(values) % channels)
    if usable <= 0:
        raise AudioPipelineError("audio slicing received an invalid PCM payload")
    waveform = values[:usable].reshape(-1, channels).T.astype(np.float32) / 32768.0
    return waveform, sample_rate, channels


def _write_pcm16_wav(
    path: Path, waveform: np.ndarray, sample_rate: int, channels: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(waveform, -1.0, 1.0)
    interleaved = np.rint(clipped.T.reshape(-1) * 32767.0).astype("<i2")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(interleaved.tobytes())
    temporary.replace(path)


def _write_manifest(manifest: AudioSliceManifest) -> None:
    path = manifest.manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def slice_audio(
    input_path: Path,
    output_dir: Path,
    *,
    threshold_db: float = -34.0,
    min_length_ms: int = 4_000,
    min_interval_ms: int = 200,
    hop_size_ms: int = 10,
    max_sil_kept_ms: int = 500,
    max_length_ms: int = 30_000,
    manifest_path: Path | None = None,
    progress: ProgressCallback | None = None,
) -> AudioSliceManifest:
    source = input_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    settings = SlicerSettings(
        threshold_db=threshold_db,
        min_length_ms=min_length_ms,
        min_interval_ms=min_interval_ms,
        hop_size_ms=hop_size_ms,
        max_sil_kept_ms=max_sil_kept_ms,
        max_length_ms=max_length_ms,
    )
    settings.validate()
    if progress:
        progress(0.05, "Reading speech audio for silence slicing")
    waveform, sample_rate, channels = _read_pcm16_wav(source)
    if progress:
        progress(0.2, "Detecting stable silence boundaries")
    ranges = Slicer(sample_rate, settings).slice_ranges(waveform)
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    slices: list[AudioSlice] = []
    for index, (start_sample, end_sample) in enumerate(ranges, start=1):
        start_ms = round(start_sample * 1000 / sample_rate)
        end_ms = max(start_ms + 1, round(end_sample * 1000 / sample_rate))
        path = destination / (
            f"slice-{index:06d}-{start_ms:09d}-{end_ms:09d}.wav"
        )
        _write_pcm16_wav(
            path,
            waveform[:, start_sample:end_sample],
            sample_rate,
            channels,
        )
        slices.append(
            AudioSlice(
                index=index,
                path=path.resolve(),
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
        if progress:
            progress(
                0.25 + 0.7 * index / max(1, len(ranges)),
                f"Writing speech slice {index}/{len(ranges)}",
            )

    resolved_manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else destination / DEFAULT_MANIFEST_FILENAME
    )
    manifest = AudioSliceManifest(
        manifest_path=resolved_manifest,
        source_path=source,
        sample_rate=sample_rate,
        channels=channels,
        duration_ms=round(waveform.shape[1] * 1000 / sample_rate),
        settings=settings,
        slices=tuple(slices),
    )
    _write_manifest(manifest)
    if progress:
        progress(1.0, f"Created {len(slices)} speech slices")
    return manifest


def load_slice_manifest(path: Path) -> AudioSliceManifest:
    manifest_path = path.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AudioPipelineError(f"invalid audio slice manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise AudioPipelineError("unsupported audio slice manifest version")
    try:
        parameters = payload.get("parameters") or {}
        settings = SlicerSettings(**parameters)
        settings.validate()
        source = Path(str(payload["source_path"])).resolve()
        sample_rate = int(payload["sample_rate"])
        channels = int(payload["channels"])
        duration_ms = int(payload["duration_ms"])
        rows = payload["slices"]
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioPipelineError(f"invalid audio slice manifest fields: {exc}") from exc
    if not isinstance(rows, list):
        raise AudioPipelineError("audio slice manifest slices must be a list")
    slices: list[AudioSlice] = []
    for expected_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise AudioPipelineError("audio slice manifest contains an invalid row")
        try:
            slice_path = Path(str(row["path"]))
            if not slice_path.is_absolute():
                slice_path = manifest_path.parent / slice_path
            item = AudioSlice(
                index=int(row.get("index", expected_index)),
                path=slice_path.resolve(),
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AudioPipelineError(f"invalid audio slice row: {exc}") from exc
        if item.index != expected_index or item.start_ms < 0 or item.end_ms <= item.start_ms:
            raise AudioPipelineError("audio slice manifest has invalid ordering or timing")
        if item.end_ms > duration_ms + settings.hop_size_ms:
            raise AudioPipelineError("audio slice exceeds source duration")
        if not item.path.is_file() or item.path.stat().st_size <= 44:
            raise AudioPipelineError(f"audio slice file is missing or invalid: {item.path}")
        slices.append(item)
    return AudioSliceManifest(
        manifest_path=manifest_path,
        source_path=source,
        sample_rate=sample_rate,
        channels=channels,
        duration_ms=duration_ms,
        settings=settings,
        slices=tuple(slices),
    )
