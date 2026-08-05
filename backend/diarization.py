from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .media import application_root
from .models import Cue


SPEAKER_COLORS = (
    "#F4D35E",
    "#76C7C0",
    "#FF7F6E",
    "#73A9FF",
    "#C792EA",
    "#8BD17C",
    "#FF9F43",
    "#E879B9",
)

VAD_MODEL_FILENAME = "vad_multilingual_marblenet.nemo"
SPEAKER_MODEL_FILENAME = "titanet_large.nemo"
VAD_MODEL_NAME = "vad_multilingual_marblenet"
SPEAKER_MODEL_NAME = "titanet_large"


def default_model_dir() -> Path:
    return application_root() / "models" / "diarization"


def resolve_model_assets(
    model_dir: str | Path | None = None,
) -> tuple[Path, Path, list[str]]:
    models = Path(model_dir).resolve() if model_dir is not None else default_model_dir()
    vad_model = models / VAD_MODEL_FILENAME
    speaker_model = models / SPEAKER_MODEL_FILENAME
    missing = [
        path.name
        for path in (vad_model, speaker_model)
        if not path.is_file() or path.stat().st_size < 10_000
    ]
    return vad_model, speaker_model, missing


def ensure_model_assets(
    model_dir: str | Path | None = None,
    *,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[Path, Path]:
    """Download NeMo's universal VAD and speaker encoders when absent."""

    vad_model, speaker_model, missing = resolve_model_assets(model_dir)
    if not missing:
        return vad_model, speaker_model

    try:
        from nemo.collections.asr.models import (
            EncDecClassificationModel,
            EncDecSpeakerLabelModel,
        )
    except ImportError as exc:
        raise RuntimeError("NVIDIA NeMo diarization runtime is not installed") from exc

    vad_model.parent.mkdir(parents=True, exist_ok=True)
    downloads = [
        (
            speaker_model,
            SPEAKER_MODEL_NAME,
            EncDecSpeakerLabelModel,
            "speaker encoder",
        ),
        (vad_model, VAD_MODEL_NAME, EncDecClassificationModel, "voice activity model"),
    ]
    missing_set = set(missing)
    for index, (destination, model_name, model_class, label) in enumerate(downloads):
        if destination.name not in missing_set:
            continue
        if progress is not None:
            progress(index / len(downloads), f"Downloading local NeMo {label}")
        temporary = destination.with_suffix(destination.suffix + ".download")
        try:
            model = model_class.from_pretrained(model_name=model_name, map_location="cpu")
            model.save_to(str(temporary))
            if not temporary.is_file() or temporary.stat().st_size < 10_000:
                raise RuntimeError(f"downloaded {label} archive is invalid")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            if "model" in locals():
                del model

    vad_model, speaker_model, missing = resolve_model_assets(model_dir)
    if missing:
        raise RuntimeError("local diarization model missing after download: " + ", ".join(missing))
    if progress is not None:
        progress(1.0, "Local speaker diarization models are ready")
    return vad_model, speaker_model


@dataclass(frozen=True, slots=True)
class DiarizationSegment:
    recording_id: str
    start_ms: int
    end_ms: int
    speaker_label: str


@dataclass(frozen=True, slots=True)
class SpeakerAssignmentStats:
    assigned_cues: int
    preserved_cues: int
    unmatched_cues: int
    speaker_count: int


@dataclass(frozen=True, slots=True)
class DiarizationRunResult:
    success: bool
    segments: tuple[DiarizationSegment, ...] = ()
    rttm_path: Path | None = None
    error: str = ""


@dataclass(frozen=True, slots=True)
class DiarizationCueResult:
    success: bool
    cues: tuple[Cue, ...]
    segments: tuple[DiarizationSegment, ...] = ()
    rttm_path: Path | None = None
    stats: SpeakerAssignmentStats | None = None
    error: str = ""


def parse_rttm(path: str | Path) -> list[DiarizationSegment]:
    """Parse SPEAKER records from an RTTM file into millisecond intervals."""

    rttm_path = Path(path)
    segments: list[DiarizationSegment] = []
    for line_number, raw_line in enumerate(
        rttm_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or fields[0].upper() != "SPEAKER":
            continue
        try:
            start_seconds = float(fields[3])
            duration_seconds = float(fields[4])
        except ValueError as exc:
            raise ValueError(
                f"invalid RTTM timing at {rttm_path}:{line_number}"
            ) from exc
        if start_seconds < 0 or duration_seconds <= 0:
            raise ValueError(f"invalid RTTM interval at {rttm_path}:{line_number}")
        start_ms = round(start_seconds * 1000)
        end_ms = max(start_ms + 1, round((start_seconds + duration_seconds) * 1000))
        speaker_label = fields[7].strip()
        if not speaker_label or speaker_label == "<NA>":
            raise ValueError(f"missing RTTM speaker label at {rttm_path}:{line_number}")
        segments.append(
            DiarizationSegment(
                recording_id=fields[1],
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_label=speaker_label,
            )
        )
    return sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.speaker_label))


def _speaker_order(segments: Iterable[DiarizationSegment]) -> dict[str, int]:
    first_seen: dict[str, tuple[int, int, str]] = {}
    for index, segment in enumerate(segments):
        key = (segment.start_ms, index, segment.speaker_label)
        previous = first_seen.get(segment.speaker_label)
        if previous is None or key < previous:
            first_seen[segment.speaker_label] = key
    ordered = sorted(first_seen, key=lambda label: first_seen[label])
    return {label: index + 1 for index, label in enumerate(ordered)}


def _has_speaker(cue: Cue) -> bool:
    return bool(cue.speaker_id.strip() or cue.speaker_name.strip())


def assign_speakers(
    cues: Sequence[Cue], segments: Sequence[DiarizationSegment]
) -> tuple[list[Cue], SpeakerAssignmentStats]:
    """Assign the speaker with the greatest total overlap to each unlabelled cue."""

    ordered_segments = sorted(
        segments, key=lambda item: (item.start_ms, item.end_ms, item.speaker_label)
    )
    speaker_order = _speaker_order(ordered_segments)
    assigned = 0
    preserved = 0
    unmatched = 0
    output: list[Cue] = []

    for cue in cues:
        if _has_speaker(cue):
            preserved += 1
            output.append(cue)
            continue

        overlaps: dict[str, int] = defaultdict(int)
        first_overlap_start: dict[str, int] = {}
        for segment in ordered_segments:
            if segment.end_ms <= cue.start_ms:
                continue
            if segment.start_ms >= cue.end_ms:
                break
            overlap_ms = min(cue.end_ms, segment.end_ms) - max(cue.start_ms, segment.start_ms)
            if overlap_ms <= 0:
                continue
            overlaps[segment.speaker_label] += overlap_ms
            first_overlap_start.setdefault(segment.speaker_label, segment.start_ms)

        if not overlaps:
            unmatched += 1
            output.append(cue)
            continue

        raw_label = min(
            overlaps,
            key=lambda label: (
                -overlaps[label],
                first_overlap_start[label],
                speaker_order[label],
            ),
        )
        speaker_number = speaker_order[raw_label]
        speaker_color = SPEAKER_COLORS[(speaker_number - 1) % len(SPEAKER_COLORS)]
        output.append(
            cue.model_copy(
                update={
                    "speaker_id": f"SPK_{speaker_number:02d}",
                    "speaker_name": f"Speaker {speaker_number}",
                    "speaker_color": speaker_color,
                }
            )
        )
        assigned += 1

    return output, SpeakerAssignmentStats(
        assigned_cues=assigned,
        preserved_cues=preserved,
        unmatched_cues=unmatched,
        speaker_count=len(speaker_order),
    )


def run_nemo_diarization(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    model_dir: str | Path | None = None,
    device: str | None = None,
    batch_size: int = 32,
    max_speakers: int = 8,
    progress: Callable[[float, str], None] | None = None,
) -> DiarizationRunResult:
    """Run NeMo clustering diarization using only local model archives."""

    audio = Path(audio_path).resolve()
    output = Path(output_dir).resolve()
    if not audio.is_file():
        return DiarizationRunResult(success=False, error=f"audio file not found: {audio}")
    if batch_size < 1:
        return DiarizationRunResult(success=False, error="batch_size must be at least 1")
    if max_speakers < 1:
        return DiarizationRunResult(success=False, error="max_speakers must be at least 1")

    try:
        import torch
        from nemo.collections.asr.models import ClusteringDiarizer
        from nemo.collections.asr.models.configs.diarizer_config import (
            NeuralDiarizerInferenceConfig,
        )
        from omegaconf import OmegaConf

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            return DiarizationRunResult(
                success=False,
                error=f"requested diarization device is unavailable: {selected_device}",
            )

        try:
            vad_model, speaker_model = ensure_model_assets(
                model_dir,
                progress=(
                    (lambda value, message: progress(value * 0.18, message))
                    if progress is not None
                    else None
                ),
            )
        except Exception as exc:
            return DiarizationRunResult(
                success=False,
                error=f"local diarization model setup failed: {type(exc).__name__}: {exc}",
            )

        output.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            progress(0.2, "Loading local VAD and speaker models")
        config = OmegaConf.structured(NeuralDiarizerInferenceConfig())
        config.device = selected_device
        config.verbose = False
        config.batch_size = batch_size
        config.num_workers = 0
        config.diarizer.out_dir = str(output)
        config.diarizer.vad.model_path = str(vad_model)
        config.diarizer.speaker_embeddings.model_path = str(speaker_model)
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        config.diarizer.clustering.parameters.max_num_speakers = max_speakers

        diarizer = ClusteringDiarizer(cfg=config)
        if progress is not None:
            progress(0.35, "Detecting speech and clustering speakers")
        diarizer.diarize(paths2audio_files=[str(audio)], batch_size=batch_size)

        rttm_dir = output / "pred_rttms"
        candidates = sorted(rttm_dir.glob("*.rttm"))
        preferred = rttm_dir / f"{audio.stem}.rttm"
        rttm_path = preferred if preferred.is_file() else (candidates[0] if len(candidates) == 1 else None)
        if rttm_path is None:
            return DiarizationRunResult(
                success=False,
                error=f"NeMo produced no unambiguous RTTM output in {rttm_dir}",
            )
        segments = tuple(parse_rttm(rttm_path))
        if progress is not None:
            progress(1.0, f"Speaker diarization produced {len(segments)} segments")
        return DiarizationRunResult(
            success=True,
            segments=segments,
            rttm_path=rttm_path,
        )
    except Exception as exc:
        return DiarizationRunResult(
            success=False,
            error=f"NeMo speaker diarization failed: {type(exc).__name__}: {exc}",
        )


def diarize_cues(
    audio_path: str | Path,
    cues: Sequence[Cue],
    output_dir: str | Path,
    **kwargs: object,
) -> DiarizationCueResult:
    """Run diarization and apply its segments while preserving all original cues on failure."""

    run = run_nemo_diarization(audio_path, output_dir, **kwargs)
    if not run.success:
        return DiarizationCueResult(
            success=False,
            cues=tuple(cues),
            error=run.error,
        )
    assigned_cues, stats = assign_speakers(cues, run.segments)
    return DiarizationCueResult(
        success=True,
        cues=tuple(assigned_cues),
        segments=run.segments,
        rttm_path=run.rttm_path,
        stats=stats,
    )
