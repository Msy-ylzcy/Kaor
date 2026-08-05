from __future__ import annotations

import os
from pathlib import Path

from backend.runtime import configure_runtime_directories


def test_runtime_forces_all_ml_caches_under_portable_data(tmp_path, monkeypatch):
    blocked = r"C:\Users\User\.cache"
    for name in (
        "HOME",
        "USERPROFILE",
        "XDG_CACHE_HOME",
        "PADDLE_HOME",
        "PADDLEOCR_HOME",
        "PADDLE_PDX_CACHE_HOME",
        "PADDLE_EXTENSION_DIR",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "MODELSCOPE_CACHE",
    ):
        monkeypatch.setenv(name, blocked)

    paths = configure_runtime_directories(tmp_path)
    cache = (tmp_path / "data" / "cache").resolve()

    assert paths["cache"] == cache
    for name in (
        "HOME",
        "USERPROFILE",
        "XDG_CACHE_HOME",
        "PADDLE_HOME",
        "PADDLEOCR_HOME",
        "PADDLE_PDX_CACHE_HOME",
        "PADDLE_EXTENSION_DIR",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "MODELSCOPE_CACHE",
    ):
        value = Path(os.environ[name]).resolve()
        assert value == cache or cache in value.parents
        assert value.exists() or name == "HUGGINGFACE_HUB_CACHE"

    assert (paths["home"] / ".cache" / "paddle").is_dir()


def test_runtime_honors_kaor_data_dir_for_source_and_portable_launch(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "portable-data"
    app_root = tmp_path / "app"
    monkeypatch.setenv("KAOR_DATA_DIR", str(data_dir))
    monkeypatch.setattr("backend.runtime.application_root", lambda: app_root)
    monkeypatch.setattr("backend.runtime.resource_root", lambda: app_root)

    paths = configure_runtime_directories()

    assert paths["data"] == data_dir.resolve()
    assert paths["cache"] == (data_dir / "cache").resolve()
    assert paths["models"] == (tmp_path / "app" / "models").resolve()


def test_runtime_finds_pyinstaller_bundled_models_without_writing_app_root(
    tmp_path, monkeypatch
):
    app_root = tmp_path / "read-only-app"
    resource_root = tmp_path / "_MEIPASS"
    bundled_models = resource_root / "models"
    bundled_models.mkdir(parents=True)
    monkeypatch.delenv("KAOR_DATA_DIR", raising=False)
    monkeypatch.setattr("backend.runtime.application_root", lambda: app_root)
    monkeypatch.setattr("backend.runtime.resource_root", lambda: resource_root)

    paths = configure_runtime_directories()

    assert paths["models"] == bundled_models.resolve()
    assert not (app_root / "models").exists()
