from __future__ import annotations

import os
from pathlib import Path

from .media import application_root, resource_root


def configure_runtime_directories(root: Path | None = None) -> dict[str, Path]:
    base = (root or application_root()).resolve()
    configured_data = os.environ.get("KAOR_DATA_DIR") if root is None else None
    data = (
        Path(configured_data).expanduser().resolve()
        if configured_data
        else base / "data"
    )
    cache = data / "cache"
    home = cache / "home"
    external_models = base / "models"
    bundled_models = resource_root() / "models" if root is None else external_models
    models = external_models if external_models.is_dir() else bundled_models
    huggingface = cache / "huggingface"
    modelscope = cache / "modelscope"
    paddle = cache / "paddle"
    paddleocr = cache / "paddleocr"
    paddlex = cache / "paddlex"
    paddle_extensions = cache / "paddle_extensions"
    matplotlib = cache / "matplotlib"
    numba = cache / "numba"
    torch = cache / "torch"
    nemo = cache / "nemo"
    for path in (
        data,
        cache,
        home,
        huggingface,
        modelscope,
        paddle,
        paddleocr,
        paddlex,
        paddle_extensions,
        matplotlib,
        numba,
        torch,
        nemo,
        home / ".lhotse" / "tools",
        home / ".cache" / "paddle",
    ):
        path.mkdir(parents=True, exist_ok=True)
    variables = {
        "HOME": home,
        "USERPROFILE": home,
        "XDG_CACHE_HOME": cache,
        "PADDLE_HOME": paddle,
        "PADDLEOCR_HOME": paddleocr,
        "PADDLE_PDX_CACHE_HOME": paddlex,
        "PADDLE_EXTENSION_DIR": paddle_extensions,
        "MPLCONFIGDIR": matplotlib,
        "NUMBA_CACHE_DIR": numba,
        "TORCH_HOME": torch,
        "NEMO_CACHE_DIR": nemo,
        "HF_HOME": huggingface,
        "HUGGINGFACE_HUB_CACHE": huggingface / "hub",
        "MODELSCOPE_CACHE": modelscope,
    }
    for name, path in variables.items():
        # A portable runtime must not inherit machine-wide cache locations. In
        # particular Paddle imports modules that eagerly create ~/.cache/paddle.
        os.environ[name] = str(path)
    bin_dir = base / "bin"
    if bin_dir.is_dir():
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if str(bin_dir) not in path_entries:
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    return {
        "data": data,
        "cache": cache,
        "home": home,
        "models": models,
        "paddle": paddle,
        "paddleocr": paddleocr,
        "paddlex": paddlex,
    }
