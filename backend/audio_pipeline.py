from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .media import application_root, find_binary
from .diarization import resolve_model_assets as resolve_diarization_model_assets
from .worker_runtime import audio_worker_command


class AudioPipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class AsrModelSpec:
    id: str
    language: str
    language_label: str
    label: str
    engine: str
    repository: str
    description: str
    download_size_mb: int
    supports_word_timestamps: bool = True
    supports_speaker_labels: bool = False


ASR_MODELS: tuple[AsrModelSpec, ...] = (
    AsrModelSpec(
        id="ja-parakeet-tdt-ctc-0.6b",
        language="ja",
        language_label="日本語",
        label="NVIDIA Parakeet TDT-CTC 0.6B Japanese",
        engine="nemo",
        repository="nvidia/parakeet-tdt_ctc-0.6b-ja",
        description="日语专用 FastConformer；标点、长音频和时间戳",
        download_size_mb=2500,
    ),
    AsrModelSpec(
        id="en-parakeet-tdt-0.6b-v2",
        language="en",
        language_label="English",
        label="NVIDIA Parakeet TDT 0.6B v2 English",
        engine="nemo",
        repository="nvidia/parakeet-tdt-0.6b-v2",
        description="英语专用 FastConformer；标点、大小写和时间戳",
        download_size_mb=2500,
    ),
    AsrModelSpec(
        id="zh-paraformer-large-vad-punc-spk",
        language="zh",
        language_label="中文",
        label="FunASR Paraformer Large Chinese",
        engine="funasr",
        repository="iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",
        description="中文专用 Paraformer；内置 VAD、标点和说话人信息",
        download_size_mb=2200,
        supports_speaker_labels=True,
    ),
    AsrModelSpec(
        id="ko-conformer-transducer-large",
        language="ko",
        language_label="한국어",
        label="Conformer Transducer Large Korean",
        engine="nemo",
        repository="eesungkim/stt_kr_conformer_transducer_large",
        description="韩语专用 Conformer Transducer",
        download_size_mb=500,
    ),
    AsrModelSpec(
        id="es-conformer-transducer-large",
        language="es",
        language_label="Español",
        label="NVIDIA Conformer Transducer Large Spanish",
        engine="nemo",
        repository="nvidia/stt_es_conformer_transducer_large",
        description="西班牙语专用 Conformer Transducer",
        download_size_mb=500,
    ),
    AsrModelSpec(
        id="fr-conformer-transducer-large",
        language="fr",
        language_label="Français",
        label="NVIDIA Conformer Transducer Large French",
        engine="nemo",
        repository="nvidia/stt_fr_conformer_transducer_large",
        description="法语专用 Conformer Transducer",
        download_size_mb=500,
    ),
    AsrModelSpec(
        id="de-conformer-transducer-large",
        language="de",
        language_label="Deutsch",
        label="NVIDIA Conformer Transducer Large German",
        engine="nemo",
        repository="nvidia/stt_de_conformer_transducer_large",
        description="德语专用 Conformer Transducer",
        download_size_mb=500,
    ),
    AsrModelSpec(
        id="ru-fastconformer-hybrid-large",
        language="ru",
        language_label="Русский",
        label="NVIDIA FastConformer Hybrid Large Russian",
        engine="nemo",
        repository="nvidia/stt_ru_fastconformer_hybrid_large_pc",
        description="俄语专用 FastConformer Hybrid",
        download_size_mb=500,
    ),
)


LANGUAGE_ALIASES = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "jpn": "ja",
    "eng": "en",
    "kor": "ko",
}


UVR_MODEL_FILENAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
UVR_CONFIG_FILENAME = "model_bs_roformer_ep_317_sdr_12.9755.yaml"
UVR_EXPECTED_SIZE = 639_331_213
UVR_EXPECTED_SHA256 = "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa"


def normalize_language(language: str) -> str:
    value = (language or "").strip().lower().replace("_", "-")
    return LANGUAGE_ALIASES.get(value, value.split("-", 1)[0])


def get_asr_model(model_id: str) -> AsrModelSpec:
    for model in ASR_MODELS:
        if model.id == model_id:
            return model
    raise AudioPipelineError(f"unknown ASR model: {model_id}")


def recommended_asr_model(language: str) -> AsrModelSpec:
    normalized = normalize_language(language)
    for model in ASR_MODELS:
        if model.language == normalized:
            return model
    supported = ", ".join(sorted({item.language for item in ASR_MODELS}))
    raise AudioPipelineError(
        f"no language-specific ASR model is configured for {language!r}; "
        f"supported languages: {supported}"
    )


def asr_model_directory(models_root: Path, model: AsrModelSpec) -> Path:
    return models_root.resolve() / "asr" / model.id


def asr_model_installed(models_root: Path, model: AsrModelSpec) -> bool:
    directory = asr_model_directory(models_root, model)
    marker = directory / ".kaor-model.json"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload.get("repository") == model.repository and any(
        path.is_file() and path.name != marker.name for path in directory.rglob("*")
    )


def _uvr_asset_paths(root: Path) -> tuple[Path, Path]:
    direct_model = root / UVR_MODEL_FILENAME
    direct_config = root / UVR_CONFIG_FILENAME
    if direct_model.is_file() or direct_config.is_file():
        return direct_model, direct_config
    model_root = root / "models" / "MDX_Net_Models"
    return (
        model_root / UVR_MODEL_FILENAME,
        model_root / "model_data" / "mdx_c_configs" / UVR_CONFIG_FILENAME,
    )


def uvr_asset_roots(root: Path | None = None) -> list[Path]:
    if root is not None:
        return [root.expanduser().resolve()]
    return [(application_root() / "models" / "uvr").resolve()]


def resolve_uvr_assets(root: Path | None = None) -> tuple[Path | None, Path | None, str | None]:
    checked: list[str] = []
    for candidate_root in uvr_asset_roots(root):
        model, config = _uvr_asset_paths(candidate_root)
        checked.append(str(model))
        if not model.is_file() or not config.is_file():
            continue
        if model.stat().st_size != UVR_EXPECTED_SIZE:
            return None, None, f"UVR model size mismatch: {model}"
        return model.resolve(), config.resolve(), None
    return (
        None,
        None,
        "bundled UVR model is missing or incomplete; reinstall this Kaor release: "
        + "; ".join(checked),
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def _audio_worker_probe() -> dict[str, object]:
    try:
        command = audio_worker_command(["probe"])
        completed = subprocess.run(
            command,
            cwd=application_root(),
            env={**os.environ, "PYTHONUTF8": "1"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception as exc:
        return {
            "torch_available": _module_available("torch"),
            "error": f"audio runtime probe failed: {exc}",
        }
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if completed.returncode and not payload.get("error"):
        payload["error"] = completed.stderr.strip() or "audio runtime probe failed"
    return payload


def detect_torch_runtime() -> tuple[bool, str | None, bool, int, list[str], str | None]:
    payload = _audio_worker_probe()
    names = payload.get("cuda_device_names")
    return (
        bool(payload.get("torch_available", False)),
        str(payload["torch_version"]) if payload.get("torch_version") else None,
        bool(payload.get("cuda_available", False)),
        int(payload.get("cuda_device_count", 0)),
        [str(name) for name in names] if isinstance(names, list) else [],
        str(payload["error"]) if payload.get("error") else None,
    )


def download_asr_model(
    model: AsrModelSpec,
    models_root: Path,
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise AudioPipelineError(
            "huggingface_hub is required to download the selected ASR model"
        ) from exc

    output_dir = asr_model_directory(models_root, model)
    output_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.02, f"Downloading {model.label}")
    try:
        snapshot_download(
            repo_id=model.repository,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise AudioPipelineError(f"ASR model download failed: {exc}") from exc
    marker = output_dir / ".kaor-model.json"
    marker.write_text(
        json.dumps(
            {
                "id": model.id,
                "repository": model.repository,
                "engine": model.engine,
                "language": model.language,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if progress:
        progress(1.0, f"Downloaded {model.label}")
    return output_dir.resolve()


def asr_model_catalog(models_root: Path) -> list[dict[str, object]]:
    return [
        {
            "id": model.id,
            "language": model.language,
            "language_label": model.language_label,
            "label": model.label,
            "engine": model.engine,
            "repository": model.repository,
            "description": model.description,
            "recommended": True,
            "download_size_mb": model.download_size_mb,
            "installed": asr_model_installed(models_root, model),
            "local_path": str(asr_model_directory(models_root, model)),
            "supports_word_timestamps": model.supports_word_timestamps,
            "supports_speaker_labels": model.supports_speaker_labels,
        }
        for model in ASR_MODELS
    ]


def audio_capabilities_payload(models_root: Path) -> dict[str, object]:
    model_path, config_path, uvr_error = resolve_uvr_assets()
    vad_model, speaker_model, missing_diarization = resolve_diarization_model_assets(
        models_root.resolve() / "diarization"
    )
    torch_available, torch_version, cuda_available, count, names, torch_error = (
        detect_torch_runtime()
    )
    runtime = _audio_worker_probe()
    errors = [message for message in (uvr_error, torch_error) if message]
    return {
        "ffmpeg_available": find_binary("ffmpeg") is not None,
        "torch_available": torch_available,
        "torch_version": torch_version,
        "audio_separator_available": bool(
            runtime.get("audio_separator_available", _module_available("audio_separator"))
        ),
        "nemo_available": bool(runtime.get("nemo_available", _module_available("nemo"))),
        "funasr_available": bool(
            runtime.get("funasr_available", _module_available("funasr"))
        ),
        "cuda_available": cuda_available,
        "cuda_device_count": count,
        "cuda_device_names": names,
        "default_device": "cuda:0" if count else "cpu",
        "uvr_model": {
            "id": "bs-roformer-viperx-1297",
            "label": "BS-Roformer-Viperx-1297",
            "filename": UVR_MODEL_FILENAME,
            "available": model_path is not None and config_path is not None,
            "runtime": "uvr5-local-core",
            "load_mode": "in_place",
            "root_path": str((application_root() / "models" / "uvr").resolve()),
            "path": str(model_path) if model_path else None,
            "config_path": str(config_path) if config_path else None,
            "size_bytes": model_path.stat().st_size if model_path else None,
        },
        "diarization_model": {
            "id": "nemo-titanet-marblenet",
            "label": "NeMo TitaNet + MarbleNet",
            "available": not missing_diarization,
            "runtime": "nemo-clustering",
            "load_mode": "local-or-auto-download",
            "root_path": str((models_root.resolve() / "diarization")),
            "path": str(speaker_model) if speaker_model.is_file() else None,
            "config_path": str(vad_model) if vad_model.is_file() else None,
            "size_bytes": (
                speaker_model.stat().st_size + vad_model.stat().st_size
                if not missing_diarization
                else None
            ),
        },
        "asr_models": asr_model_catalog(models_root),
        "errors": errors,
    }
