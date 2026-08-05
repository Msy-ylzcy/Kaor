from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Event
from typing import BinaryIO, Callable, Iterator

import httpx

from .diarization import resolve_model_assets as resolve_diarization_model_assets
from .media import application_root, find_binary
from .worker_runtime import audio_worker_command


class AudioPipelineError(RuntimeError):
    pass


logger = logging.getLogger("kaor.audio")


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
UVR_CONFIG_EXPECTED_SIZE = 2_273
UVR_CONFIG_EXPECTED_SHA256 = (
    "2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f"
)
UVR_MODEL_URL = (
    "https://github.com/TRvlvr/model_repo/releases/download/"
    "all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
)
UVR_CONFIG_URL = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/"
    "22b79fc01ada8f3b9e3526ad0ed645af414a7cde/mdx_model_data/"
    "mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml"
)


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


def _hash_file(
    path: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancel_event: Event | None = None,
) -> str:
    total = path.stat().st_size
    completed = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("job cancelled")
            digest.update(chunk)
            completed += len(chunk)
            if progress:
                progress(completed, total)
    return digest.hexdigest().lower()


def _lock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _uvr_download_lock(
    path: Path,
    cancel_event: Event | None,
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    deadline = time.monotonic() + 15 * 60
    acquired = False
    try:
        while not acquired:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("job cancelled")
            try:
                _lock_file(handle)
                acquired = True
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise AudioPipelineError(
                        "timed out waiting for another BS-Roformer download to finish"
                    ) from exc
                time.sleep(0.2)
        yield
    finally:
        if acquired:
            _unlock_file(handle)
        handle.close()


def ensure_uvr_config(
    root: Path | None = None,
    *,
    progress: Callable[[float, str], None] | None = None,
    cancel_event: Event | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Install the pinned UVR configuration from its upstream repository."""

    destination = uvr_asset_roots(root)[0] / UVR_CONFIG_FILENAME
    partial = destination.with_suffix(destination.suffix + ".part")
    lock_path = destination.with_suffix(destination.suffix + ".download.lock")

    def report(value: float, message: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, value)), message)

    try:
        with _uvr_download_lock(lock_path, cancel_event):
            if destination.is_file() and destination.stat().st_size == UVR_CONFIG_EXPECTED_SIZE:
                actual = _hash_file(destination, cancel_event=cancel_event)
                if actual == UVR_CONFIG_EXPECTED_SHA256.lower():
                    report(1.0, "BS-Roformer configuration is ready")
                    return destination.resolve()
                logger.warning(
                    "existing UVR configuration failed SHA-256 validation path=%s actual=%s",
                    destination,
                    actual,
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            partial.unlink(missing_ok=True)
            report(0.0, "Downloading the pinned BS-Roformer configuration")
            headers = {
                "Accept": "application/octet-stream",
                "Accept-Encoding": "identity",
                "User-Agent": "Kaor/0.2.0",
            }
            with httpx.Client(
                transport=transport,
                follow_redirects=True,
                timeout=httpx.Timeout(60.0, connect=30.0),
                trust_env=True,
            ) as client:
                with client.stream("GET", UVR_CONFIG_URL, headers=headers) as response:
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise AudioPipelineError(
                            "official BS-Roformer configuration returned unexpected HTTP "
                            f"{response.status_code}"
                        )
                    completed = 0
                    with partial.open("wb") as handle:
                        for chunk in response.iter_bytes(chunk_size=64 * 1024):
                            if cancel_event is not None and cancel_event.is_set():
                                raise InterruptedError("job cancelled")
                            if not chunk:
                                continue
                            handle.write(chunk)
                            completed += len(chunk)
                            if completed > UVR_CONFIG_EXPECTED_SIZE:
                                raise AudioPipelineError(
                                    "official BS-Roformer configuration exceeded the pinned "
                                    f"size of {UVR_CONFIG_EXPECTED_SIZE} bytes"
                                )
                        handle.flush()
                        os.fsync(handle.fileno())

            actual_size = partial.stat().st_size if partial.is_file() else 0
            if actual_size != UVR_CONFIG_EXPECTED_SIZE:
                raise AudioPipelineError(
                    "BS-Roformer configuration download is incomplete: expected "
                    f"{UVR_CONFIG_EXPECTED_SIZE} bytes, received {actual_size}"
                )
            actual_sha256 = _hash_file(partial, cancel_event=cancel_event)
            if actual_sha256 != UVR_CONFIG_EXPECTED_SHA256.lower():
                partial.unlink(missing_ok=True)
                raise AudioPipelineError(
                    "BS-Roformer configuration SHA-256 mismatch: expected "
                    f"{UVR_CONFIG_EXPECTED_SHA256}, received {actual_sha256}"
                )
            os.replace(partial, destination)
            report(1.0, "BS-Roformer configuration downloaded and verified")
            logger.info("installed pinned UVR configuration at %s", destination)
            return destination.resolve()
    except InterruptedError:
        raise
    except AudioPipelineError:
        raise
    except (httpx.HTTPError, OSError) as exc:
        raise AudioPipelineError(
            "BS-Roformer configuration download failed from the fixed upstream: "
            f"{exc}. Retry UVR to download a clean copy"
        ) from exc


def ensure_uvr_checkpoint(
    root: Path | None = None,
    *,
    progress: Callable[[float, str], None] | None = None,
    cancel_event: Event | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Install the pinned UVR checkpoint on first use.

    The public portable package does not redistribute the checkpoint. A
    completed download is never made visible at the runtime path until both its
    byte size and SHA-256 match the pinned values.
    """

    destination = uvr_asset_roots(root)[0] / UVR_MODEL_FILENAME
    partial = destination.with_suffix(destination.suffix + ".part")
    lock_path = destination.with_suffix(destination.suffix + ".download.lock")
    last_reported = -1.0

    def report(value: float, message: str, *, force: bool = False) -> None:
        nonlocal last_reported
        value = max(0.0, min(1.0, value))
        if progress and (force or last_reported < 0 or value - last_reported >= 0.01):
            progress(value, message)
            last_reported = value

    def hash_progress(
        start: float, end: float, label: str
    ) -> Callable[[int, int], None]:
        def update(completed: int, total: int) -> None:
            fraction = completed / max(total, 1)
            report(start + fraction * (end - start), label)

        return update

    try:
        with _uvr_download_lock(lock_path, cancel_event):
            if destination.is_file():
                if destination.stat().st_size == UVR_EXPECTED_SIZE:
                    report(0.0, "Validating the local BS-Roformer checkpoint", force=True)
                    actual = _hash_file(
                        destination,
                        progress=hash_progress(
                            0.0, 0.05, "Validating the local BS-Roformer checkpoint"
                        ),
                        cancel_event=cancel_event,
                    )
                    if actual == UVR_EXPECTED_SHA256.lower():
                        report(1.0, "BS-Roformer checkpoint is ready", force=True)
                        return destination.resolve()
                    logger.warning(
                        "existing UVR checkpoint failed SHA-256 validation path=%s actual=%s",
                        destination,
                        actual,
                    )
                else:
                    logger.warning(
                        "existing UVR checkpoint has the wrong size path=%s size=%d",
                        destination,
                        destination.stat().st_size,
                    )

            destination.parent.mkdir(parents=True, exist_ok=True)
            if partial.is_file() and partial.stat().st_size > UVR_EXPECTED_SIZE:
                partial.unlink()
            if partial.is_file() and partial.stat().st_size == UVR_EXPECTED_SIZE:
                report(0.0, "Validating the resumed BS-Roformer download", force=True)
                actual = _hash_file(
                    partial,
                    progress=hash_progress(
                        0.0, 0.05, "Validating the resumed BS-Roformer download"
                    ),
                    cancel_event=cancel_event,
                )
                if actual == UVR_EXPECTED_SHA256.lower():
                    os.replace(partial, destination)
                    report(1.0, "BS-Roformer checkpoint is ready", force=True)
                    return destination.resolve()
                partial.unlink()

            timeout = httpx.Timeout(120.0, connect=30.0)
            headers = {
                "Accept": "application/octet-stream",
                "Accept-Encoding": "identity",
                "User-Agent": "Kaor/0.2.0",
            }
            logger.info("downloading pinned UVR checkpoint from %s", UVR_MODEL_URL)
            with httpx.Client(
                transport=transport,
                follow_redirects=True,
                timeout=timeout,
                trust_env=True,
            ) as client:
                for _attempt in range(2):
                    offset = partial.stat().st_size if partial.is_file() else 0
                    request_headers = dict(headers)
                    if offset:
                        request_headers["Range"] = f"bytes={offset}-"
                    downloaded_mib = offset / (1024 * 1024)
                    expected_mib = UVR_EXPECTED_SIZE / (1024 * 1024)
                    report(
                        0.05 + 0.85 * offset / UVR_EXPECTED_SIZE,
                        "Downloading BS-Roformer checkpoint "
                        f"({downloaded_mib:.1f}/{expected_mib:.1f} MiB)",
                        force=True,
                    )
                    with client.stream(
                        "GET", UVR_MODEL_URL, headers=request_headers
                    ) as response:
                        if response.status_code == 416 and offset:
                            partial.unlink(missing_ok=True)
                            continue
                        response.raise_for_status()
                        append = response.status_code == 206 and offset > 0
                        if response.status_code == 206:
                            content_range = response.headers.get("content-range", "")
                            match = re.fullmatch(
                                r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range
                            )
                            if not match or int(match.group(1)) != offset:
                                raise AudioPipelineError(
                                    "official BS-Roformer download returned an invalid "
                                    f"Content-Range for offset {offset}: {content_range!r}"
                                )
                            if (
                                match.group(3) != "*"
                                and int(match.group(3)) != UVR_EXPECTED_SIZE
                            ):
                                raise AudioPipelineError(
                                    "official BS-Roformer download size changed: expected "
                                    f"{UVR_EXPECTED_SIZE}, upstream reported {match.group(3)}"
                                )
                        elif response.status_code != 200:
                            raise AudioPipelineError(
                                "official BS-Roformer download returned unexpected HTTP "
                                f"{response.status_code}"
                            )
                        if not append:
                            offset = 0
                        content_length = response.headers.get("content-length", "")
                        if content_length.isdigit():
                            advertised_total = offset + int(content_length)
                            if advertised_total != UVR_EXPECTED_SIZE:
                                raise AudioPipelineError(
                                    "official BS-Roformer download size changed: expected "
                                    f"{UVR_EXPECTED_SIZE}, upstream reported {advertised_total}"
                                )
                        completed = offset
                        with partial.open("ab" if append else "wb") as handle:
                            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                                if cancel_event is not None and cancel_event.is_set():
                                    raise InterruptedError("job cancelled")
                                if not chunk:
                                    continue
                                handle.write(chunk)
                                completed += len(chunk)
                                if completed > UVR_EXPECTED_SIZE:
                                    raise AudioPipelineError(
                                        "official BS-Roformer download exceeded the pinned "
                                        f"size of {UVR_EXPECTED_SIZE} bytes"
                                    )
                                report(
                                    0.05 + 0.85 * completed / UVR_EXPECTED_SIZE,
                                    "Downloading BS-Roformer checkpoint "
                                    f"({completed / (1024 * 1024):.1f}/"
                                    f"{expected_mib:.1f} MiB)",
                                )
                            handle.flush()
                            os.fsync(handle.fileno())
                    break
                else:
                    raise AudioPipelineError(
                        "official BS-Roformer server rejected a clean download request"
                    )

            actual_size = partial.stat().st_size if partial.is_file() else 0
            if actual_size != UVR_EXPECTED_SIZE:
                raise AudioPipelineError(
                    "BS-Roformer checkpoint download is incomplete: expected "
                    f"{UVR_EXPECTED_SIZE} bytes, received {actual_size}. Retry UVR to "
                    f"resume from {partial}"
                )
            report(0.9, "Verifying the downloaded BS-Roformer checkpoint", force=True)
            actual_sha256 = _hash_file(
                partial,
                progress=hash_progress(
                    0.9, 1.0, "Verifying the downloaded BS-Roformer checkpoint"
                ),
                cancel_event=cancel_event,
            )
            if actual_sha256 != UVR_EXPECTED_SHA256.lower():
                partial.unlink(missing_ok=True)
                raise AudioPipelineError(
                    "BS-Roformer checkpoint SHA-256 mismatch: expected "
                    f"{UVR_EXPECTED_SHA256}, received {actual_sha256}. The invalid "
                    "download was removed; retry UVR to download a clean copy"
                )
            os.replace(partial, destination)
            report(1.0, "BS-Roformer checkpoint downloaded and verified", force=True)
            logger.info("installed pinned UVR checkpoint at %s", destination)
            return destination.resolve()
    except InterruptedError:
        raise
    except AudioPipelineError:
        raise
    except (httpx.HTTPError, OSError) as exc:
        raise AudioPipelineError(
            "BS-Roformer checkpoint download failed from the fixed official upstream: "
            f"{exc}. Retry UVR; partial data is kept at {partial} for resume"
        ) from exc


def resolve_uvr_assets(root: Path | None = None) -> tuple[Path | None, Path | None, str | None]:
    checked: list[str] = []
    for candidate_root in uvr_asset_roots(root):
        model, config = _uvr_asset_paths(candidate_root)
        checked.append(str(model))
        if not model.is_file() or not config.is_file():
            continue
        if model.stat().st_size != UVR_EXPECTED_SIZE:
            return None, None, f"UVR model size mismatch: {model}"
        if config.stat().st_size != UVR_CONFIG_EXPECTED_SIZE:
            return None, None, f"UVR configuration size mismatch: {config}"
        if _hash_file(config) != UVR_CONFIG_EXPECTED_SHA256.lower():
            return None, None, f"UVR configuration SHA-256 mismatch: {config}"
        return model.resolve(), config.resolve(), None
    return (
        None,
        None,
        "BS-Roformer assets are missing or incomplete; start UVR to download "
        "and verify them from their fixed upstream locations: "
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
            "load_mode": "local-or-auto-download",
            "root_path": str((application_root() / "models" / "uvr").resolve()),
            "path": str(model_path) if model_path else None,
            "config_path": str(config_path) if config_path else None,
            "size_bytes": model_path.stat().st_size if model_path else None,
            "download_size_mb": round(UVR_EXPECTED_SIZE / (1024 * 1024)),
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
