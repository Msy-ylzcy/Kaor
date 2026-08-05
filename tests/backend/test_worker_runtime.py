from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from backend import audio_pipeline, worker_runtime


def test_audio_worker_command_uses_python_module_in_source_mode(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)

    command = worker_runtime.audio_worker_command(["probe"])

    assert command == [sys.executable, "-m", "backend.audio_worker", "probe"]


def test_audio_worker_command_uses_packaged_executable(monkeypatch, tmp_path):
    executable = tmp_path / "KaorAudioWorker.exe"
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("KAOR_AUDIO_WORKER", str(executable))

    command = worker_runtime.audio_worker_command(["asr", "--request", "job.json"])

    assert command == [str(executable.resolve()), "asr", "--request", "job.json"]


def test_missing_packaged_audio_worker_has_actionable_error(monkeypatch, tmp_path):
    missing = tmp_path / "missing-worker.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("KAOR_AUDIO_WORKER", str(missing))

    with pytest.raises(FileNotFoundError, match="reinstall this Kaor release"):
        worker_runtime.audio_worker_command(["probe"])


def test_uvr_assets_prefer_portable_direct_layout(monkeypatch, tmp_path):
    model = tmp_path / audio_pipeline.UVR_MODEL_FILENAME
    config = tmp_path / audio_pipeline.UVR_CONFIG_FILENAME
    model.write_bytes(b"x" * 32)
    config.write_text("audio: {}\n", encoding="utf-8")
    monkeypatch.setattr(audio_pipeline, "UVR_EXPECTED_SIZE", 32)
    monkeypatch.setattr(audio_pipeline, "UVR_CONFIG_EXPECTED_SIZE", config.stat().st_size)
    monkeypatch.setattr(
        audio_pipeline,
        "UVR_CONFIG_EXPECTED_SHA256",
        hashlib.sha256(config.read_bytes()).hexdigest(),
    )

    resolved_model, resolved_config, error = audio_pipeline.resolve_uvr_assets(tmp_path)

    assert error is None
    assert resolved_model == model.resolve()
    assert resolved_config == config.resolve()
