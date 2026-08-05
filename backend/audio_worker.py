from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _configure() -> None:
    from .media import application_root
    from .runtime import configure_runtime_directories

    configure_runtime_directories()
    bin_dir = application_root() / "bin"
    if bin_dir.is_dir():
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _probe() -> int:
    _configure()
    payload: dict[str, Any] = {
        "torch_available": importlib.util.find_spec("torch") is not None,
        "torch_version": None,
        "torch_cuda_version": None,
        "torchvision_available": False,
        "torchvision_version": None,
        "torchvision_ops_available": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
        "audio_separator_available": False,
        "nemo_available": False,
        "funasr_available": False,
        "module_errors": {},
        "error": None,
    }
    module_errors: dict[str, str] = {}
    if payload["torch_available"]:
        try:
            import torch

            payload["torch_version"] = str(torch.__version__)
            payload["torch_cuda_version"] = (
                str(torch.version.cuda) if torch.version.cuda else None
            )
            payload["cuda_available"] = bool(torch.cuda.is_available())
            if payload["cuda_available"]:
                count = int(torch.cuda.device_count())
                payload["cuda_device_count"] = count
                payload["cuda_device_names"] = [
                    str(torch.cuda.get_device_name(index)) for index in range(count)
                ]
        except Exception as exc:
            module_errors["torch"] = str(exc)
    try:
        import torchvision

        payload["torchvision_available"] = True
        payload["torchvision_version"] = str(torchvision.__version__)
        payload["torchvision_ops_available"] = bool(torchvision.extension._has_ops())
        if not payload["torchvision_ops_available"]:
            raise RuntimeError("torchvision native operators are unavailable")
    except Exception as exc:
        module_errors["torchvision"] = str(exc)
    try:
        from audio_separator.separator import Separator  # noqa: F401
        from audio_separator.separator.architectures.mdxc_separator import (  # noqa: F401
            MDXCSeparator,
        )
        from audio_separator.separator.roformer.roformer_loader import (  # noqa: F401
            RoformerLoader,
        )

        payload["audio_separator_available"] = True
    except Exception as exc:
        module_errors["audio_separator"] = str(exc)
    try:
        from funasr import AutoModel  # noqa: F401

        payload["funasr_available"] = True
    except Exception as exc:
        module_errors["funasr"] = str(exc)
    try:
        import nemo.collections.asr  # noqa: F401
        from nemo.collections.asr.models import ClusteringDiarizer  # noqa: F401
        from nemo.collections.asr.models.configs.diarizer_config import (  # noqa: F401
            NeuralDiarizerInferenceConfig,
        )

        payload["nemo_available"] = True
    except Exception as exc:
        module_errors["nemo"] = str(exc)
    payload["module_errors"] = module_errors
    payload["error"] = "; ".join(
        f"{name}: {message}" for name, message in module_errors.items()
    ) or None
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return 0


def _progress_writer(progress_path: Path):  # type: ignore[no-untyped-def]
    def progress(value: float, message: str) -> None:
        _write_json(
            progress_path,
            {"progress": max(0.0, min(1.0, value)), "message": message},
        )

    return progress


def _run_uvr(request_path: Path, result_path: Path, progress_path: Path) -> int:
    _configure()
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    progress = _progress_writer(progress_path)
    try:
        progress(0.005, "Loading packaged audio runtime")
        from .speech_pipeline import separate_vocals

        input_wav = Path(request["input_wav"])
        output_dir = Path(request["output_dir"])
        device = str(request.get("device") or "auto")
        vocals = separate_vocals(
            input_wav,
            output_dir,
            device=device,
            progress=progress,
        )
        _write_json(
            result_path,
            {"status": "ok", "vocals_path": str(vocals.resolve())},
        )
        progress(1.0, "Vocal separation completed")
        return 0
    except Exception as exc:
        _write_json(
            result_path,
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


def _run_asr(request_path: Path, result_path: Path, progress_path: Path) -> int:
    _configure()
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    progress = _progress_writer(progress_path)
    try:
        from .audio_pipeline import get_asr_model
        from .speech_pipeline import recognize_speech

        audio_path = Path(request["audio_path"])
        output_dir = Path(request["output_dir"])
        model = get_asr_model(str(request["model_id"]))
        device = str(request.get("device") or "auto")
        cues = recognize_speech(
            audio_path,
            model,
            Path(request["model_dir"]),
            device=device,
            slice_manifest=Path(request["slice_manifest"]),
            batch_size=int(request.get("batch_size", 4)),
            forced_alignment=bool(request.get("forced_alignment", True)),
            diarization=bool(request.get("diarization", True)),
            diarization_output_dir=output_dir / "diarization",
            progress=progress,
            checkpoint_path=(
                Path(request["checkpoint_path"])
                if request.get("checkpoint_path")
                else None
            ),
            checkpoint_signature=str(request.get("checkpoint_signature") or ""),
        )
        if not cues:
            raise RuntimeError("speech recognition completed with 0 cues")
        _write_json(
            result_path,
            {
                "status": "ok",
                "model_id": model.id,
                "cues": [cue.model_dump(mode="json") for cue in cues],
            },
        )
        progress(1.0, f"Recognized {len(cues)} speech cues")
        return 0
    except Exception as exc:
        _write_json(
            result_path,
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe")
    for name in ("uvr", "asr"):
        stage = subparsers.add_parser(name)
        stage.add_argument("--request", type=Path, required=True)
        stage.add_argument("--result", type=Path, required=True)
        stage.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "probe":
        return _probe()
    if args.command == "uvr":
        return _run_uvr(args.request, args.result, args.progress)
    return _run_asr(args.request, args.result, args.progress)


if __name__ == "__main__":
    raise SystemExit(main())
