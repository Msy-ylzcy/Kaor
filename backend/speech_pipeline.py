from __future__ import annotations

import math
import re
import shutil
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .audio_slicer import AudioSlice, AudioSliceManifest, load_slice_manifest
from .audio_pipeline import (
    AudioPipelineError,
    AsrModelSpec,
    UVR_CONFIG_FILENAME,
    UVR_MODEL_FILENAME,
    ensure_uvr_checkpoint,
    resolve_uvr_assets,
)
from .diarization import SPEAKER_COLORS, diarize_cues
from .models import Cue
from .timing import refine_cue_timing
from .task_state import atomic_write_json, read_json


ProgressCallback = Callable[[float, str], None]
SliceCheckpoint = Callable[[int, list[Cue]], None]


def _resolve_device(requested: str) -> str:
    value = (requested or "auto").strip().lower()
    try:
        import torch
    except ImportError as exc:
        raise AudioPipelineError("PyTorch is required for local audio recognition") from exc
    if value in {"auto", "cuda", "gpu"}:
        if torch.cuda.is_available():
            return "cuda:0"
        if value != "auto":
            raise AudioPipelineError("CUDA was requested but PyTorch cannot access a GPU")
        return "cpu"
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise AudioPipelineError("CUDA was requested but PyTorch cannot access a GPU")
        return value.replace("cuda", "cuda:", 1) if value[4:].isdigit() else value
    if value == "cpu":
        return value
    raise AudioPipelineError(f"unsupported audio device: {requested}")


def separate_vocals(
    input_wav: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    progress: ProgressCallback | None = None,
) -> Path:
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise AudioPipelineError(
            "audio-separator is required for local BS-Roformer vocal separation"
        ) from exc

    ensure_uvr_checkpoint(
        progress=(
            (lambda value, message: progress(0.01 + value * 0.34, message))
            if progress
            else None
        )
    )
    model_path, config_path, error = resolve_uvr_assets()
    if error or model_path is None or config_path is None:
        raise AudioPipelineError(error or "UVR model assets are missing")
    selected_device = _resolve_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    class LocalUvrSeparator(Separator):
        def download_model_files(self, model_filename: str):  # type: ignore[no-untyped-def]
            if Path(model_filename).name not in {
                UVR_MODEL_FILENAME,
                UVR_CONFIG_FILENAME,
            }:
                raise AudioPipelineError(f"unexpected UVR model request: {model_filename}")
            return (
                UVR_MODEL_FILENAME,
                "MDXC",
                "BS-Roformer-Viperx-1297",
                str(model_path),
                str(config_path),
            )

    if progress:
        progress(0.36, f"Loading local UVR5 BS-Roformer: {model_path}")
    try:
        separator = LocalUvrSeparator(
            output_dir=str(output_dir),
            model_file_dir=str(output_dir / "model-cache"),
            output_format="WAV",
            output_single_stem="Vocals",
            use_autocast=selected_device.startswith("cuda"),
            mdxc_params={
                "segment_size": 256,
                "override_model_segment_size": False,
                "batch_size": 1,
                "overlap": 8,
                "pitch_shift": 0,
            },
        )
        separator.load_model(model_filename=UVR_MODEL_FILENAME)
        if progress:
            progress(0.42, "Separating vocals from accompaniment")
        outputs = separator.separate(
            str(input_wav.resolve()), custom_output_names={"Vocals": "vocals"}
        )
    except AudioPipelineError:
        raise
    except Exception as exc:
        raise AudioPipelineError(f"vocal separation failed: {exc}") from exc

    candidates = [Path(value) for value in outputs if isinstance(value, str)]
    candidates.extend(output_dir.glob("*.wav"))
    vocals = next(
        (
            path
            for path in candidates
            if path.name.lower() == "vocals.wav" or "vocal" in path.stem.lower()
        ),
        None,
    )
    if vocals is None:
        raise AudioPipelineError("vocal separation did not return a vocals WAV file")
    if not vocals.is_absolute():
        vocals = output_dir / vocals
    vocals = vocals.resolve()
    if not vocals.is_file() or vocals.stat().st_size <= 44:
        raise AudioPipelineError("vocal separation produced an invalid WAV file")
    canonical = (output_dir / "vocals.wav").resolve()
    if vocals != canonical:
        shutil.move(str(vocals), str(canonical))
    if progress:
        progress(1.0, "Vocal separation completed")
    return canonical


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _confidence(value: Any) -> float | None:
    result = _number(value)
    if result is None or not 0.0 <= result <= 1.0:
        return None
    return round(result, 4)


def _mean_confidence(values: Any) -> float | None:
    if not isinstance(values, (list, tuple)):
        return None
    valid = [confidence for value in values if (confidence := _confidence(value)) is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def _cue(
    index: int,
    *,
    start_seconds: float,
    end_seconds: float,
    text: str,
    confidence: float | None = None,
    speaker: str = "",
) -> Cue:
    start_ms = max(0, round(start_seconds * 1000))
    end_ms = max(start_ms + 1, round(end_seconds * 1000))
    normalized_speaker = str(speaker).strip()
    return Cue(
        cue_id=f"ASR{index:06d}",
        start_ms=start_ms,
        end_ms=end_ms,
        track_id="speech",
        speaker_id=(f"SPK_{normalized_speaker}" if normalized_speaker else ""),
        speaker_name=(f"Speaker {normalized_speaker}" if normalized_speaker else ""),
        speaker_color="#FFFFFF",
        source_kind="speech",
        source_text=text.strip(),
        ocr_confidence=confidence,
        review_status="ocr_ok" if confidence is not None and confidence >= 0.85 else "needs_review",
    )


def _timestamp_value(row: dict[str, Any], name: str) -> float | None:
    direct = _number(row.get(name))
    if direct is not None:
        return direct
    milliseconds = _number(row.get(f"{name}_ms"))
    return milliseconds / 1000 if milliseconds is not None else None


def _segments_from_words(words: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    current: list[str] = []
    start = end = 0.0
    for row in words:
        word = str(row.get("word") or row.get("char") or "").strip()
        word_start = _timestamp_value(row, "start")
        word_end = _timestamp_value(row, "end")
        if not word or word_start is None or word_end is None:
            continue
        should_flush = bool(
            current
            and (
                word_start - end >= 0.65
                or word_end - start >= 7.5
                or re.search(r"[.!?。！？]$", current[-1])
            )
        )
        if should_flush:
            segments.append((start, end, " ".join(current)))
            current = []
        if not current:
            start = word_start
        current.append(word)
        end = max(word_end, word_start + 0.04)
    if current:
        segments.append((start, end, " ".join(current)))
    return segments


def _nemo_segments(hypothesis: Any) -> list[tuple[float, float, str]]:
    timestamps = getattr(hypothesis, "timestamp", None) or getattr(
        hypothesis, "timestep", None
    )
    if isinstance(timestamps, dict):
        raw_segments = timestamps.get("segment")
        if isinstance(raw_segments, list):
            segments: list[tuple[float, float, str]] = []
            for row in raw_segments:
                if not isinstance(row, dict):
                    continue
                start = _timestamp_value(row, "start")
                end = _timestamp_value(row, "end")
                text = str(row.get("segment") or row.get("text") or "").strip()
                if start is not None and end is not None and text:
                    segments.append((start, end, text))
            if segments:
                return segments
        raw_words = timestamps.get("word")
        if isinstance(raw_words, list):
            segments = _segments_from_words(
                [row for row in raw_words if isinstance(row, dict)]
            )
            if segments:
                return segments
    text = str(getattr(hypothesis, "text", hypothesis) or "").strip()
    return [(0.0, 0.001, text)] if text else []


def _nemo_segment_confidences(
    hypothesis: Any, segments: list[tuple[float, float, str]]
) -> list[float | None]:
    if not segments:
        return []
    word_confidences = getattr(hypothesis, "word_confidence", None)
    timestamps = getattr(hypothesis, "timestamp", None) or getattr(
        hypothesis, "timestep", None
    )
    timed_words: list[tuple[float, float, float]] = []
    if isinstance(timestamps, dict):
        raw_words = timestamps.get("word")
        if isinstance(raw_words, list):
            for index, row in enumerate(raw_words):
                if not isinstance(row, dict):
                    continue
                start = _timestamp_value(row, "start")
                end = _timestamp_value(row, "end")
                direct_confidence = _confidence(
                    row.get("confidence", row.get("score"))
                )
                if direct_confidence is None and isinstance(word_confidences, (list, tuple)):
                    if index < len(word_confidences):
                        direct_confidence = _confidence(word_confidences[index])
                if start is not None and end is not None and direct_confidence is not None:
                    timed_words.append((start, end, direct_confidence))

    if timed_words:
        per_segment: list[float | None] = []
        for segment_start, segment_end, _text in segments:
            overlapping = [
                confidence
                for word_start, word_end, confidence in timed_words
                if word_end > segment_start and word_start < segment_end
            ]
            per_segment.append(_mean_confidence(overlapping))
        if any(value is not None for value in per_segment):
            return per_segment

    if len(segments) == 1:
        confidence = _mean_confidence(word_confidences)
        if confidence is None:
            confidence = _mean_confidence(getattr(hypothesis, "token_confidence", None))
        return [confidence]
    return [None] * len(segments)


def _enable_nemo_confidence(model: Any) -> bool:
    try:
        from omegaconf import OmegaConf, open_dict

        current = model.cfg.decoding
        decoding_cfg = OmegaConf.create(
            OmegaConf.to_container(current, resolve=False)
        )
        with open_dict(decoding_cfg):
            decoding_cfg.compute_timestamps = True
            decoding_cfg.confidence_cfg = {
                "preserve_frame_confidence": True,
                "preserve_token_confidence": True,
                "preserve_word_confidence": True,
                "exclude_blank": True,
                "aggregation": "mean",
                "tdt_include_duration": False,
                "method_cfg": {
                    "name": "max_prob",
                    "alpha": 1.0,
                    "entropy_type": "gibbs",
                    "entropy_norm": "lin",
                },
            }
        model.change_decoding_strategy(decoding_cfg, verbose=False)
    except Exception:
        return False
    return True


def _wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path.resolve()), "rb") as handle:
            return max(1, round(handle.getnframes() * 1000 / handle.getframerate()))
    except (OSError, ValueError, wave.Error, ZeroDivisionError):
        return 1


def _normalize_nemo_hypotheses(outputs: Any, expected: int) -> list[Any]:
    candidates: list[Any]
    if isinstance(outputs, tuple):
        candidates = [candidate for candidate in outputs if isinstance(candidate, list)]
        if not candidates:
            raise AudioPipelineError("NeMo returned an unsupported transcription tuple")
        matching = [candidate for candidate in candidates if len(candidate) == expected]
        outputs = matching[0] if matching else candidates[0]
    if not isinstance(outputs, list):
        raise AudioPipelineError("NeMo returned an unsupported transcription response")
    if len(outputs) == 1 and isinstance(outputs[0], list) and expected != 1:
        outputs = outputs[0]
    if len(outputs) != expected:
        raise AudioPipelineError(
            f"NeMo returned {len(outputs)} hypotheses for {expected} speech slices"
        )
    return outputs


def _nemo_slice_cues(
    hypothesis: Any,
    audio_slice: AudioSlice,
    *,
    first_index: int,
) -> list[Cue]:
    segments = _nemo_segments(hypothesis)
    confidences = _nemo_segment_confidences(hypothesis, segments)
    offset_seconds = audio_slice.start_ms / 1000.0
    duration_seconds = audio_slice.duration_ms / 1000.0
    timestamp_payload = getattr(hypothesis, "timestamp", None) or getattr(
        hypothesis, "timestep", None
    )
    cues: list[Cue] = []
    for position, (start, end, text) in enumerate(segments):
        local_start = max(0.0, start)
        local_end = max(local_start + 0.001, end)
        if not isinstance(timestamp_payload, dict) and local_end <= 0.001:
            local_end = duration_seconds
        local_start = min(duration_seconds, local_start)
        local_end = min(duration_seconds, local_end)
        if not text.strip() or local_end <= local_start:
            continue
        cues.append(
            _cue(
                first_index + len(cues),
                start_seconds=offset_seconds + local_start,
                end_seconds=offset_seconds + local_end,
                text=text,
                confidence=confidences[position] if position < len(confidences) else None,
            )
        )
    return cues


def _recognize_nemo_slices(
    slices: Sequence[AudioSlice],
    model_dir: Path,
    device: str,
    progress: ProgressCallback | None,
    *,
    batch_size: int = 4,
    start_slice: int = 0,
    initial_cues: Sequence[Cue] = (),
    checkpoint: SliceCheckpoint | None = None,
) -> list[Cue]:
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError as exc:
        raise AudioPipelineError("NVIDIA NeMo ASR runtime is not installed") from exc
    checkpoints = sorted(
        model_dir.rglob("*.nemo"), key=lambda path: path.stat().st_size, reverse=True
    )
    if not checkpoints:
        raise AudioPipelineError(
            f"downloaded NeMo model has no .nemo checkpoint: {model_dir}"
        )
    if not slices:
        raise AudioPipelineError("audio slice manifest contains no speech slices")
    selected_batch_size = max(1, min(int(batch_size), 64))
    if progress:
        progress(0.05, "Loading language-specific NeMo ASR model")
    try:
        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(checkpoints[0]), map_location=torch.device(device)
        )
        model.to(device)
        model.eval()
        confidence_enabled = _enable_nemo_confidence(model)
        cues: list[Cue] = [cue.model_copy(deep=True) for cue in initial_cues]
        total = len(slices)
        start_slice = max(0, min(int(start_slice), total))
        with torch.inference_mode():
            for batch_start in range(start_slice, total, selected_batch_size):
                batch = list(slices[batch_start : batch_start + selected_batch_size])
                batch_end = batch_start + len(batch)
                if progress:
                    detail = " with native confidence" if confidence_enabled else ""
                    progress(
                        0.18 + 0.8 * batch_start / total,
                        f"Transcribing speech slices {batch_start + 1}-{batch_end}/{total}{detail}",
                    )
                outputs = model.transcribe(
                    audio=[str(item.path.resolve()) for item in batch],
                    batch_size=min(selected_batch_size, len(batch)),
                    timestamps=True,
                )
                hypotheses = _normalize_nemo_hypotheses(outputs, len(batch))
                for audio_slice, hypothesis in zip(batch, hypotheses):
                    cues.extend(
                        _nemo_slice_cues(
                            hypothesis,
                            audio_slice,
                            first_index=len(cues) + 1,
                        )
                    )
                if checkpoint:
                    checkpoint(batch_end, [cue.model_copy(deep=True) for cue in cues])
                if progress:
                    progress(
                        0.18 + 0.8 * batch_end / total,
                        f"Transcribed speech slices {batch_end}/{total}",
                    )
    except AudioPipelineError:
        raise
    except Exception as exc:
        raise AudioPipelineError(f"NeMo speech recognition failed: {exc}") from exc
    if not cues:
        raise AudioPipelineError(
            f"NeMo recognized 0 speech cues from {len(slices)} sliced audio files"
        )
    if progress:
        progress(1.0, f"Recognized {len(cues)} speech cues")
    return cues


def _recognize_nemo(
    audio_path: Path,
    model_dir: Path,
    device: str,
    progress: ProgressCallback | None,
) -> list[Cue]:
    source = audio_path.resolve()
    return _recognize_nemo_slices(
        [
            AudioSlice(
                index=1,
                path=source,
                start_ms=0,
                end_ms=_wav_duration_ms(source),
            )
        ],
        model_dir,
        device,
        progress,
        batch_size=1,
    )


def _funasr_result_cues(
    result: dict[str, Any],
    audio_slice: AudioSlice,
    *,
    first_index: int,
) -> list[Cue]:
    offset_seconds = audio_slice.start_ms / 1000.0
    sentence_info = result.get("sentence_info")
    cues: list[Cue] = []
    if isinstance(sentence_info, list):
        for row in sentence_info:
            if not isinstance(row, dict):
                continue
            start = _number(row.get("start"))
            end = _number(row.get("end"))
            text = str(row.get("text") or "").strip()
            if start is None or end is None or not text:
                continue
            confidence_value = (
                row["confidence"] if "confidence" in row else row.get("score")
            )
            cues.append(
                _cue(
                    first_index + len(cues),
                    start_seconds=offset_seconds + start / 1000,
                    end_seconds=offset_seconds + end / 1000,
                    text=text,
                    confidence=_confidence(confidence_value),
                    speaker=str(row.get("spk") or ""),
                )
            )
    if cues:
        return cues

    text = str(result.get("text") or "").strip()
    timestamps = result.get("timestamp")
    if text and isinstance(timestamps, list) and timestamps:
        starts = [
            _number(row[0])
            for row in timestamps
            if isinstance(row, (list, tuple)) and len(row) >= 2
        ]
        ends = [
            _number(row[1])
            for row in timestamps
            if isinstance(row, (list, tuple)) and len(row) >= 2
        ]
        valid_starts = [value for value in starts if value is not None]
        valid_ends = [value for value in ends if value is not None]
        if valid_starts and valid_ends:
            confidence_value = (
                result["confidence"]
                if "confidence" in result
                else result.get("score")
            )
            return [
                _cue(
                    first_index,
                    start_seconds=offset_seconds + min(valid_starts) / 1000,
                    end_seconds=offset_seconds + max(valid_ends) / 1000,
                    text=text,
                    confidence=_confidence(confidence_value),
                )
            ]
    return []


def _recognize_funasr_slices(
    slices: Sequence[AudioSlice],
    model_dir: Path,
    device: str,
    progress: ProgressCallback | None,
    *,
    start_slice: int = 0,
    initial_cues: Sequence[Cue] = (),
    checkpoint: SliceCheckpoint | None = None,
) -> list[Cue]:
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise AudioPipelineError("FunASR runtime is not installed") from exc
    if not slices:
        raise AudioPipelineError("audio slice manifest contains no speech slices")
    if progress:
        progress(0.05, "Loading language-specific FunASR model")
    try:
        model = AutoModel(
            model=str(model_dir),
            device=device,
            disable_update=True,
            trust_remote_code=True,
        )
        cues: list[Cue] = [cue.model_copy(deep=True) for cue in initial_cues]
        total = len(slices)
        start_slice = max(0, min(int(start_slice), total))
        for index, audio_slice in enumerate(
            slices[start_slice:], start=start_slice + 1
        ):
            if progress:
                progress(
                    0.15 + 0.83 * (index - 1) / total,
                    f"Transcribing speech slice {index}/{total}",
                )
            outputs = model.generate(
                input=str(audio_slice.path.resolve()),
                batch_size_s=30,
                merge_vad=True,
                merge_length_s=15,
            )
            if not isinstance(outputs, list):
                continue
            result = next((row for row in outputs if isinstance(row, dict)), None)
            if result is not None:
                cues.extend(
                    _funasr_result_cues(
                        result,
                        audio_slice,
                        first_index=len(cues) + 1,
                    )
                )
            if checkpoint:
                checkpoint(index, [cue.model_copy(deep=True) for cue in cues])
            if progress:
                progress(
                    0.15 + 0.83 * index / total,
                    f"Transcribed speech slices {index}/{total}",
                )
    except Exception as exc:
        raise AudioPipelineError(f"FunASR speech recognition failed: {exc}") from exc
    if not cues:
        raise AudioPipelineError(
            f"FunASR recognized 0 speech cues from {len(slices)} sliced audio files"
        )
    if progress:
        progress(1.0, f"Recognized {len(cues)} speech cues")
    return cues


def _recognize_funasr(
    audio_path: Path,
    model_dir: Path,
    device: str,
    progress: ProgressCallback | None,
) -> list[Cue]:
    source = audio_path.resolve()
    return _recognize_funasr_slices(
        [
            AudioSlice(
                index=1,
                path=source,
                start_ms=0,
                end_ms=_wav_duration_ms(source),
            )
        ],
        model_dir,
        device,
        progress,
    )


def recognize_speech(
    audio_path: Path,
    model: AsrModelSpec,
    model_dir: Path,
    *,
    device: str = "auto",
    slice_manifest: Path | AudioSliceManifest | None = None,
    batch_size: int = 4,
    forced_alignment: bool = True,
    diarization: bool = True,
    diarization_output_dir: Path | None = None,
    progress: ProgressCallback | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_signature: str = "",
) -> list[Cue]:
    selected_device = _resolve_device(device)
    if batch_size < 1 or batch_size > 64:
        raise AudioPipelineError("ASR batch_size must be between 1 and 64")
    manifest = (
        load_slice_manifest(slice_manifest)
        if isinstance(slice_manifest, Path)
        else slice_manifest
    )
    if manifest is not None and (
        manifest.sample_rate != 16_000 or manifest.channels != 1
    ):
        raise AudioPipelineError(
            "ASR slice manifest must contain 16 kHz mono PCM audio"
        )
    checkpoint_payload = read_json(checkpoint_path) if checkpoint_path else None
    checkpoint_stage = ""
    next_slice = 0
    checkpoint_cues: list[Cue] = []
    if (
        manifest is not None
        and isinstance(checkpoint_payload, dict)
        and checkpoint_payload.get("signature") == checkpoint_signature
    ):
        try:
            if int(checkpoint_payload.get("slice_count", -1)) != len(manifest.slices):
                raise ValueError("ASR checkpoint slice count changed")
            checkpoint_stage = str(checkpoint_payload.get("stage") or "running")
            next_slice = max(
                0,
                min(int(checkpoint_payload.get("next_slice", 0)), len(manifest.slices)),
            )
            checkpoint_cues = [
                Cue.model_validate(row) for row in checkpoint_payload.get("cues", [])
            ]
        except (TypeError, ValueError):
            checkpoint_stage = ""
            next_slice = 0
            checkpoint_cues = []
    if checkpoint_stage == "complete" and checkpoint_cues:
        if progress:
            progress(1.0, f"Resumed {len(checkpoint_cues)} completed speech cues")
        return checkpoint_cues

    def save_checkpoint(stage: str, completed_slices: int, rows: list[Cue]) -> None:
        if checkpoint_path is None or manifest is None:
            return
        atomic_write_json(
            checkpoint_path,
            {
                "signature": checkpoint_signature,
                "stage": stage,
                "slice_count": len(manifest.slices),
                "next_slice": completed_slices,
                "cues": [cue.model_dump(mode="json") for cue in rows],
            },
        )

    asr_progress = (
        (lambda value, message: progress(value * 0.7, message))
        if progress is not None
        else None
    )
    raw_ready = bool(
        manifest is not None
        and checkpoint_cues
        and next_slice >= len(manifest.slices)
    )
    if checkpoint_stage == "transcribed" or raw_ready:
        cues = checkpoint_cues
        if progress:
            progress(0.7, f"Resumed {len(cues)} transcribed speech cues")
    elif model.engine == "nemo":
        cues = (
            _recognize_nemo_slices(
                manifest.slices,
                model_dir,
                selected_device,
                asr_progress,
                batch_size=batch_size,
                start_slice=next_slice,
                initial_cues=checkpoint_cues,
                checkpoint=lambda index, rows: save_checkpoint(
                    "running", index, rows
                ),
            )
            if manifest is not None
            else _recognize_nemo(audio_path, model_dir, selected_device, asr_progress)
        )
    elif model.engine == "funasr":
        cues = (
            _recognize_funasr_slices(
                manifest.slices,
                model_dir,
                selected_device,
                asr_progress,
                start_slice=next_slice,
                initial_cues=checkpoint_cues,
                checkpoint=lambda index, rows: save_checkpoint(
                    "running", index, rows
                ),
            )
            if manifest is not None
            else _recognize_funasr(audio_path, model_dir, selected_device, asr_progress)
        )
    else:
        raise AudioPipelineError(f"unsupported ASR engine: {model.engine}")
    if not cues:
        raise AudioPipelineError("speech recognition completed with 0 cues")
    save_checkpoint(
        "transcribed",
        len(manifest.slices) if manifest is not None else 1,
        cues,
    )

    if forced_alignment and cues:
        if progress:
            progress(0.72, "Refining speech boundaries with local voice activity")
        cues, refinement = refine_cue_timing(audio_path, cues)
        if progress:
            if refinement.unavailable:
                progress(0.8, "Using native ASR timestamps (audio refinement unavailable)")
            else:
                progress(
                    0.8,
                    f"Speech timing refined ({refinement.changed_cues}/{refinement.total_cues} cues)",
                )

    if diarization and cues:
        if all(cue.speaker_id.strip() or cue.speaker_name.strip() for cue in cues):
            ordered_speakers: dict[str, int] = {}
            colored: list[Cue] = []
            for cue in cues:
                identity = cue.speaker_id.strip() or cue.speaker_name.strip()
                speaker_number = ordered_speakers.setdefault(identity, len(ordered_speakers) + 1)
                color = SPEAKER_COLORS[(speaker_number - 1) % len(SPEAKER_COLORS)]
                colored.append(
                    cue.model_copy(update={"speaker_color": color})
                    if cue.speaker_color == "#FFFFFF"
                    else cue
                )
            cues = colored
            if progress:
                progress(0.98, f"Preserved native labels for {len(ordered_speakers)} speakers")
        else:
            output_dir = diarization_output_dir or audio_path.parent / "diarization"
            result = diarize_cues(
                audio_path,
                cues,
                output_dir,
                device=selected_device,
                progress=(
                    (lambda value, message: progress(0.81 + value * 0.18, message))
                    if progress is not None
                    else None
                ),
            )
            cues = list(result.cues)
            if progress:
                if result.success and result.stats is not None:
                    progress(
                        0.99,
                        f"Assigned {result.stats.assigned_cues} cues across "
                        f"{result.stats.speaker_count} speakers",
                    )
                else:
                    progress(0.99, f"Speaker diarization needs review: {result.error}")

    if progress:
        progress(1.0, f"Recognized {len(cues)} speech cues")
    save_checkpoint(
        "complete",
        len(manifest.slices) if manifest is not None else 1,
        cues,
    )
    return cues
