from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from threading import Event
from typing import Callable
from uuid import uuid4

from .audio_pipeline import AudioPipelineError, AsrModelSpec
from .audio_slicer import AudioSliceManifest, slice_audio
from .media import application_root, transcode_audio_for_asr
from .models import Cue
from .worker_runtime import audio_worker_command


ProgressCallback = Callable[[float, str], None]


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    bin_dir = application_root() / "bin"
    if bin_dir.is_dir():
        environment["PATH"] = str(bin_dir) + os.pathsep + environment.get("PATH", "")
    environment["PYTHONUTF8"] = "1"
    return environment


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _terminate(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_stage_worker(
    stage: str,
    request: dict[str, object],
    output_dir: Path,
    *,
    progress: ProgressCallback,
    cancel_event: Event,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    prefix = f"{stage}-worker-{run_id}"
    request_path = output_dir / f"{prefix}.request.json"
    result_path = output_dir / f"{prefix}.result.json"
    progress_path = output_dir / f"{prefix}.progress.json"
    log_path = output_dir / f"{prefix}.log"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command = audio_worker_command([
        stage,
        "--request",
        str(request_path),
        "--result",
        str(result_path),
        "--progress",
        str(progress_path),
    ])
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=application_root(),
            env=_worker_environment(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        last_progress: tuple[float, str] | None = None
        while process.poll() is None:
            if cancel_event.is_set():
                _terminate(process)
                raise InterruptedError("job cancelled")
            snapshot = _read_json(progress_path)
            if snapshot:
                current = (
                    float(snapshot.get("progress", 0.0)),
                    str(snapshot.get("message", f"Processing {stage}")),
                )
                if current != last_progress:
                    progress(*current)
                    last_progress = current
            time.sleep(0.25)

    snapshot = _read_json(progress_path)
    if snapshot:
        current = (
            float(snapshot.get("progress", 0.0)),
            str(snapshot.get("message", f"Processing {stage}")),
        )
        if current != last_progress:
            progress(*current)
    result = _read_json(result_path)
    if process.returncode != 0 or not result or result.get("status") != "ok":
        detail = str(result.get("error")) if result else f"{stage} worker exited without a result"
        if log_path.is_file():
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
            if log_tail and log_tail not in detail:
                detail = f"{detail}; worker log: {log_tail}"
        raise AudioPipelineError(detail)
    return result


def run_uvr_worker(
    input_wav: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    progress: ProgressCallback,
    cancel_event: Event,
) -> Path:
    result = _run_stage_worker(
        "uvr",
        {
            "input_wav": str(input_wav.resolve()),
            "output_dir": str(output_dir.resolve()),
            "device": device,
        },
        output_dir,
        progress=progress,
        cancel_event=cancel_event,
    )
    value = result.get("vocals_path")
    if not isinstance(value, str):
        raise AudioPipelineError("UVR worker result did not contain vocals_path")
    vocals = Path(value).resolve()
    if not vocals.is_file() or vocals.stat().st_size <= 44:
        raise AudioPipelineError("UVR worker produced an invalid vocals WAV")
    return vocals


def run_slicer_stage(
    input_wav: Path,
    output_dir: Path,
    *,
    threshold_db: float = -34.0,
    min_length_ms: int = 4_000,
    min_interval_ms: int = 200,
    hop_size_ms: int = 10,
    max_sil_kept_ms: int = 500,
    max_length_ms: int = 30_000,
    progress: ProgressCallback | None = None,
) -> AudioSliceManifest:
    return slice_audio(
        input_wav,
        output_dir,
        threshold_db=threshold_db,
        min_length_ms=min_length_ms,
        min_interval_ms=min_interval_ms,
        hop_size_ms=hop_size_ms,
        max_sil_kept_ms=max_sil_kept_ms,
        max_length_ms=max_length_ms,
        progress=progress,
    )


def run_asr_worker(
    audio_path: Path,
    slice_manifest: Path,
    output_dir: Path,
    model: AsrModelSpec,
    model_dir: Path,
    *,
    device: str = "auto",
    batch_size: int = 4,
    forced_alignment: bool = True,
    diarization: bool = True,
    checkpoint_path: Path | None = None,
    checkpoint_signature: str = "",
    progress: ProgressCallback,
    cancel_event: Event,
) -> list[Cue]:
    result = _run_stage_worker(
        "asr",
        {
            "audio_path": str(audio_path.resolve()),
            "slice_manifest": str(slice_manifest.resolve()),
            "output_dir": str(output_dir.resolve()),
            "model_id": model.id,
            "model_dir": str(model_dir.resolve()),
            "device": device,
            "batch_size": batch_size,
            "forced_alignment": forced_alignment,
            "diarization": diarization,
            "checkpoint_path": (
                str(checkpoint_path.resolve()) if checkpoint_path else ""
            ),
            "checkpoint_signature": checkpoint_signature,
        },
        output_dir,
        progress=progress,
        cancel_event=cancel_event,
    )
    rows = result.get("cues")
    if not isinstance(rows, list):
        raise AudioPipelineError("ASR worker result did not contain cues")
    cues = [Cue.model_validate(row) for row in rows]
    if not cues:
        raise AudioPipelineError("ASR worker completed with 0 speech cues")
    return cues


def run_speech_worker(
    input_wav: Path,
    output_dir: Path,
    asr_audio: Path,
    model: AsrModelSpec,
    model_dir: Path,
    *,
    device: str,
    separate_vocals: bool,
    forced_alignment: bool = True,
    diarization: bool = True,
    asr_batch_size: int = 4,
    threshold_db: float = -34.0,
    min_length_ms: int = 4_000,
    min_interval_ms: int = 200,
    hop_size_ms: int = 10,
    max_sil_kept_ms: int = 500,
    max_length_ms: int = 30_000,
    progress: ProgressCallback,
    cancel_event: Event,
) -> list[Cue]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if separate_vocals:
        vocals = run_uvr_worker(
            input_wav,
            output_dir,
            device=device,
            progress=lambda value, message: progress(value * 0.55, message),
            cancel_event=cancel_event,
        )
    else:
        vocals = input_wav.resolve()
        progress(0.55, "Using the original audio track for speech recognition")
    if cancel_event.is_set():
        raise InterruptedError("job cancelled")

    progress(0.57, "Preparing 16 kHz mono speech audio")
    transcode_audio_for_asr(vocals, asr_audio)

    def slicer_progress(value: float, message: str) -> None:
        if cancel_event.is_set():
            raise InterruptedError("job cancelled")
        progress(0.59 + value * 0.09, message)

    manifest = run_slicer_stage(
        asr_audio,
        output_dir / "slices",
        threshold_db=threshold_db,
        min_length_ms=min_length_ms,
        min_interval_ms=min_interval_ms,
        hop_size_ms=hop_size_ms,
        max_sil_kept_ms=max_sil_kept_ms,
        max_length_ms=max_length_ms,
        progress=slicer_progress,
    )
    if cancel_event.is_set():
        raise InterruptedError("job cancelled")
    return run_asr_worker(
        asr_audio,
        manifest.manifest_path,
        output_dir,
        model,
        model_dir,
        device=device,
        batch_size=asr_batch_size,
        forced_alignment=forced_alignment,
        diarization=diarization,
        progress=lambda value, message: progress(0.68 + value * 0.32, message),
        cancel_event=cancel_event,
    )
