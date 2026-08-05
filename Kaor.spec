# -*- mode: python ; coding: utf-8 -*-

import os
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata


root = Path(os.environ.get("KAOR_REPOSITORY_ROOT", Path.cwd())).resolve()
console_enabled = os.environ.get("KAOR_CONSOLE") == "1"
build_profile = os.environ.get("KAOR_BUILD_PROFILE", "cpu")
build_home = root / ".build" / "pyinstaller-home"
build_cache = root / ".build" / "pyinstaller-cache"
build_home.mkdir(parents=True, exist_ok=True)
build_cache.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(build_home)
os.environ["USERPROFILE"] = str(build_home)
os.environ["XDG_CACHE_HOME"] = str(build_cache)
os.environ["PADDLE_HOME"] = str(build_cache / "paddle")
os.environ["PADDLE_PDX_CACHE_HOME"] = str(root / "models" / "paddlex")
os.environ["FLAGS_use_mkldnn"] = "0"
datas = [(str(root / "apps" / "web" / "dist"), "web")]
binaries = []
hiddenimports = []


def filesystem_submodules(package, include_prefixes=()):
    """List lazy modules without importing each one in an isolated process."""
    spec = find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return []
    modules = set()
    for location in spec.submodule_search_locations:
        package_root = Path(location)
        for source in package_root.rglob("*.py"):
            relative = source.relative_to(package_root)
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            suffix = ".".join(parts)
            if include_prefixes and suffix and not any(
                suffix == prefix or suffix.startswith(prefix + ".")
                for prefix in include_prefixes
            ):
                continue
            modules.add(".".join([package, *parts]).rstrip("."))
    return sorted(module for module in modules if module)


def filesystem_sources(package, include_prefixes=()):
    """Preserve source files required by TorchScript's runtime compiler."""
    spec = find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return []
    sources = []
    for location in spec.submodule_search_locations:
        package_root = Path(location)
        for source in package_root.rglob("*.py"):
            relative = source.relative_to(package_root)
            suffix = ".".join(relative.with_suffix("").parts)
            if include_prefixes and suffix and not any(
                suffix == prefix or suffix.startswith(prefix + ".")
                for prefix in include_prefixes
            ):
                continue
            destination = Path(package, *relative.parent.parts)
            sources.append((str(source), str(destination)))
    return sources

for package in ("paddle", "paddleocr", "paddlex"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# Paddle imports Cython's runtime utility templates while initializing its
# extension helpers. PyInstaller sees the Python package but omits these source
# templates unless they are collected explicitly.
datas += collect_data_files("Cython")

# NeMo imports TorchMetrics during ASR/diarization initialization, and
# TorchMetrics conditionally loads torchvision. Bundle the matching wheel in
# full so the frozen worker behaves like the verified build environment.
torchvision_datas, torchvision_binaries, torchvision_hidden = collect_all("torchvision")
datas += torchvision_datas
binaries += torchvision_binaries
hiddenimports += torchvision_hidden

# NeMo's Lightning integration resolves CUDA Python bindings dynamically even
# in the CPU profile. They are optional at runtime, but the package must be
# importable for NeMo's module initialization to complete.
cuda_datas, cuda_binaries, cuda_hidden = collect_all("cuda.bindings")
datas += cuda_datas
binaries += cuda_binaries
hiddenimports += cuda_hidden

for package in ("lightning", "lightning_fabric", "pytorch_lightning"):
    datas += collect_data_files(package)

# UVR5, language-specific ASR, forced alignment, and speaker diarization are
# loaded lazily by isolated jobs. collect_all()/collect_submodules() imports
# every module in helper processes and can stall for many minutes on NeMo.
# Walking package source is deterministic and lets the standard Torch and
# TorchAudio hooks collect native libraries once into this shared COLLECT.
datas += collect_data_files("audio_separator")
hiddenimports += ["audio_separator.separator.architectures.mdxc_separator"]
hiddenimports += filesystem_submodules(
    "audio_separator",
    include_prefixes=("separator.roformer", "separator.uvr_lib_v5"),
)

datas += collect_data_files("funasr")
hiddenimports += filesystem_submodules("funasr")

nemo_prefixes = (
    "collections.asr",
    "collections.audio",
    "collections.common",
    "core",
    "lightning",
    "utils",
    "constants",
    "package_info",
)
datas += collect_data_files("nemo")
datas += filesystem_sources("nemo", include_prefixes=nemo_prefixes)
hiddenimports += filesystem_submodules("nemo", include_prefixes=nemo_prefixes)

if build_profile.startswith("nvidia"):
    # Kaor's OCR path only needs the CUDA runtime, cuBLAS, and cuDNN. The
    # upstream Paddle wheel also declares FFT/random/sparse solver packages,
    # but bundling those adds over 1 GiB and they are not loaded by PP-OCRv6.
    for package in (
        "nvidia.cuda_runtime",
        "nvidia.cublas",
        "nvidia.cudnn",
    ):
        package_datas, package_binaries, package_hidden = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hidden

for distribution in ("paddlepaddle", "paddlepaddle-gpu", "paddleocr", "paddlex"):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

for distribution in (
    "audio-separator",
    "funasr",
    "huggingface-hub",
    "imagesize",
    "nemo-toolkit",
    "opencv-contrib-python",
    "pyclipper",
    "pypdfium2",
    "python-bidi",
    "shapely",
    "torch",
    "torchaudio",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

if build_profile.startswith("nvidia"):
    for distribution in (
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-nvjitlink-cu12",
    ):
        datas += copy_metadata(distribution)

hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("keyring.backends")
# SciPy 1.17 loads its vendored Array API compatibility modules through
# importlib. PyInstaller cannot discover those imports statically, while both
# audio-separator and NeMo reach the NumPy FFT adapter during initialization.
hiddenimports += collect_submodules("scipy._external.array_api_compat")
hiddenimports += ["imagesize", "modulefinder", "torch", "torchaudio"]

a = Analysis(
    [str(root / "scripts" / "kaor_frozen_entry.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tensorflow",
        "jax",
        "nemo.collections.avlm",
        "nemo.collections.diffusion",
        "nemo.collections.llm",
        "nemo.collections.multimodal",
        "nemo.collections.multimodal_autoregressive",
        "nemo.collections.nlp",
        "nemo.collections.tts",
        "nemo.collections.vision",
        "nemo.collections.vlm",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Kaor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=console_enabled,
    disable_windowed_traceback=False,
)
audio_worker_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KaorAudioWorker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    hide_console="hide-early",
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    audio_worker_exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Kaor",
)
