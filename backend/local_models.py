from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock
from typing import Callable, Iterable, Mapping
from urllib.parse import urlparse
from uuid import uuid4

import httpx


LLAMA_RELEASE_TAG = "b10270"
LLAMA_RELEASE_ASSETS: tuple[dict[str, object], ...] = (
    {
        "name": "llama-b10270-bin-win-cpu-x64.zip",
        "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b10270/llama-b10270-bin-win-cpu-x64.zip",
        "size": 18_328_672,
        "digest": "sha256:80406b0faa562ef6268a446cfb4cfd91511770a3a716daf36d9ff1e4e582aea4",
    },
    {
        "name": "llama-b10270-bin-win-vulkan-x64.zip",
        "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b10270/llama-b10270-bin-win-vulkan-x64.zip",
        "size": 34_078_763,
        "digest": "sha256:297b09e04670bf7290d5f6be95bacf4c7c794b1fd9c9e7d972a72daf781f120a",
    },
    {
        "name": "llama-b10270-bin-win-cuda-12.4-x64.zip",
        "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b10270/llama-b10270-bin-win-cuda-12.4-x64.zip",
        "size": 250_420_795,
        "digest": "sha256:ebe3a93102cb0436e12608d4cd1f66db4fd2d4674e097a9b9b501bb75f3d88ac",
    },
    {
        "name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b10270/cudart-llama-bin-win-cuda-12.4-x64.zip",
        "size": 391_443_627,
        "digest": "sha256:8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    },
)
LOCAL_MODEL_PROJECT_ID = "local-model-runtime"
Progress = Callable[[float, str, dict[str, object] | None], None]


@dataclass(frozen=True)
class GpuAdapter:
    name: str
    vendor: str
    memory_bytes: int | None = None
    driver_version: str = ""

    def public(self) -> dict[str, object]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "memory_bytes": self.memory_bytes,
            "driver_version": self.driver_version,
        }


@dataclass(frozen=True)
class HardwareProfile:
    system: str
    architecture: str
    cpu_name: str
    logical_cpus: int
    memory_bytes: int | None
    gpus: tuple[GpuAdapter, ...]
    build_profile: str = ""

    def public(self) -> dict[str, object]:
        return {
            "system": self.system,
            "architecture": self.architecture,
            "cpu_name": self.cpu_name,
            "logical_cpus": self.logical_cpus,
            "memory_bytes": self.memory_bytes,
            "gpus": [gpu.public() for gpu in self.gpus],
            "build_profile": self.build_profile,
        }


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    filename: str
    url: str
    size_bytes: int
    minimum_memory_bytes: int
    description: str
    sha256: str
    revision: str
    license_url: str

    def public(self, installed: bool = False) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "filename": self.filename,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "minimum_memory_bytes": self.minimum_memory_bytes,
            "description": self.description,
            "sha256": self.sha256 or None,
            "revision": self.revision,
            "license_url": self.license_url,
            "installed": installed,
        }


MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="qwen3-1.7b-q8-0",
        label="Qwen3 1.7B Q8_0",
        filename="Qwen3-1.7B-Q8_0.gguf",
        url=(
            "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/"
            "61cf2fab285ddd4cad3dafa9c69dfef96426ee0b/Qwen3-1.7B-Q8_0.gguf?download=true"
        ),
        size_bytes=1_834_426_016,
        minimum_memory_bytes=3 * 1024**3,
        description="Low-memory fallback for CPU systems and integrated graphics.",
        sha256="061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
        revision="61cf2fab285ddd4cad3dafa9c69dfef96426ee0b",
        license_url="https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/61cf2fab285ddd4cad3dafa9c69dfef96426ee0b/LICENSE?download=true",
    ),
    ModelSpec(
        id="qwen3-4b-q4-k-m",
        label="Qwen3 4B Q4_K_M",
        filename="Qwen3-4B-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/"
            "a9a60d009fa7ff9606305047c2bf77ac25dbec49/Qwen3-4B-Q4_K_M.gguf?download=true"
        ),
        size_bytes=2_497_280_256,
        minimum_memory_bytes=4 * 1024**3,
        description="Balanced preset for subtitle translation on mainstream PCs.",
        sha256="7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
        revision="a9a60d009fa7ff9606305047c2bf77ac25dbec49",
        license_url="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/a9a60d009fa7ff9606305047c2bf77ac25dbec49/LICENSE?download=true",
    ),
    ModelSpec(
        id="qwen3-8b-q4-k-m",
        label="Qwen3 8B Q4_K_M",
        filename="Qwen3-8B-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/"
            "6a569868d07d3bd59e8b97fb001bf8c0b254bb20/Qwen3-8B-Q4_K_M.gguf?download=true"
        ),
        size_bytes=5_027_783_488,
        minimum_memory_bytes=7 * 1024**3,
        description="Quality preset for discrete GPUs or high-memory CPU systems.",
        sha256="d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
        revision="6a569868d07d3bd59e8b97fb001bf8c0b254bb20",
        license_url="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/6a569868d07d3bd59e8b97fb001bf8c0b254bb20/LICENSE?download=true",
    ),
    ModelSpec(
        id="qwen3-14b-q4-k-m",
        label="Qwen3 14B Q4_K_M",
        filename="Qwen3-14B-Q4_K_M.gguf",
        url="https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/c75e7b2d0234068f674a1bacf548ea32e27ccd29/Qwen3-14B-Q4_K_M.gguf?download=true",
        size_bytes=9_001_752_960,
        minimum_memory_bytes=12 * 1024**3,
        description="High-quality preset for 12-16 GB GPUs or 32 GB system memory.",
        sha256="500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
        revision="c75e7b2d0234068f674a1bacf548ea32e27ccd29",
        license_url="https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/c75e7b2d0234068f674a1bacf548ea32e27ccd29/LICENSE?download=true",
    ),
    ModelSpec(
        id="qwen3-32b-q4-k-m",
        label="Qwen3 32B Q4_K_M",
        filename="Qwen3-32B-Q4_K_M.gguf",
        url="https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/99caa2c657b2d35e922d903773e5ca3892c3b248/Qwen3-32B-Q4_K_M.gguf?download=true",
        size_bytes=19_762_149_024,
        minimum_memory_bytes=24 * 1024**3,
        description="Maximum-quality preset for 24 GB GPUs or 64 GB system memory.",
        sha256="efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689",
        revision="99caa2c657b2d35e922d903773e5ca3892c3b248",
        license_url="https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/99caa2c657b2d35e922d903773e5ca3892c3b248/LICENSE?download=true",
    ),
)


def _vendor_for_name(name: str) -> str:
    lowered = name.lower()
    if "nvidia" in lowered:
        return "nvidia"
    if any(token in lowered for token in ("amd", "radeon", "advanced micro devices")):
        return "amd"
    if "intel" in lowered:
        return "intel"
    return "unknown"


def _total_memory_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * page_count
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _detect_windows_adapters() -> list[GpuAdapter]:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return []
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=4,
            creationflags=_creation_flags(),
            check=False,
        )
        payload = json.loads(result.stdout.strip() or "[]")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    rows = payload if isinstance(payload, list) else [payload]
    adapters: list[GpuAdapter] = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("Name") or "").strip():
            continue
        name = str(row["Name"]).strip()
        memory = row.get("AdapterRAM")
        adapters.append(
            GpuAdapter(
                name=name,
                vendor=_vendor_for_name(name),
                memory_bytes=int(memory) if isinstance(memory, (int, float)) else None,
                driver_version=str(row.get("DriverVersion") or ""),
            )
        )
    return _apply_windows_registry_memory(adapters, powershell)


def _apply_windows_registry_memory(
    adapters: list[GpuAdapter], powershell: str
) -> list[GpuAdapter]:
    """Refine WMI's 32-bit AdapterRAM value with the 64-bit driver registry value."""
    command = (
        "$rows=@(); Get-ChildItem 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Video' "
        "-ErrorAction SilentlyContinue | ForEach-Object { "
        "$p=Join-Path $_.PSPath '0000'; if(Test-Path $p){ "
        "$v=Get-ItemProperty $p -ErrorAction SilentlyContinue; "
        "$m=$v.'HardwareInformation.qwMemorySize'; "
        "$n=$v.DriverDesc; if(-not $n){$n=$v.'HardwareInformation.AdapterString'}; "
        "if($n -and $m){$rows += [pscustomobject]@{Name=[string]$n;Memory=[uint64]$m}} "
        "}}; $rows | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=4,
            creationflags=_creation_flags(),
            check=False,
        )
        payload = json.loads(result.stdout.strip() or "[]")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return adapters
    rows = payload if isinstance(payload, list) else [payload]
    registry = [
        (str(row.get("Name") or "").strip(), int(row.get("Memory") or 0))
        for row in rows
        if isinstance(row, dict) and int(row.get("Memory") or 0) > 0
    ]
    if not registry:
        return adapters

    def tokens(name: str) -> set[str]:
        ignored = {"amd", "nvidia", "intel", "radeon", "graphics", "adapter", "series"}
        return {
            token
            for token in re.findall(r"[a-z0-9]+", name.casefold())
            if len(token) > 1 and token not in ignored
        }

    refined: list[GpuAdapter] = []
    for adapter in adapters:
        candidates = [
            (len(tokens(adapter.name) & tokens(name)), memory)
            for name, memory in registry
            if _vendor_for_name(name) in {adapter.vendor, "unknown"}
        ]
        best = max(candidates, default=(0, 0))
        memory = max(adapter.memory_bytes or 0, best[1]) or None
        refined.append(
            GpuAdapter(adapter.name, adapter.vendor, memory, adapter.driver_version)
        )
    return refined


def _detect_nvidia_adapters() -> list[GpuAdapter]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            creationflags=_creation_flags(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    adapters: list[GpuAdapter] = []
    for line in result.stdout.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) < 3:
            continue
        try:
            memory = int(float(columns[1])) * 1024**2
        except ValueError:
            memory = None
        adapters.append(
            GpuAdapter(columns[0], "nvidia", memory, columns[2])
        )
    return adapters


def _release_build_profile() -> str:
    configured = os.environ.get("KAOR_BUILD_PROFILE", "").strip().lower()
    if configured:
        return configured
    roots = [Path(sys.executable).resolve().parent]
    if not getattr(sys, "frozen", False):
        roots.append(Path(__file__).resolve().parents[1])
    for root in roots:
        try:
            with (root / "RELEASE.json").open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            profile = str(payload.get("runtime_profile") or "").strip().lower()
            if profile:
                return profile
    return ""


def detect_hardware() -> HardwareProfile:
    adapters = _detect_windows_adapters() if os.name == "nt" else []
    nvidia = _detect_nvidia_adapters()
    if nvidia:
        adapters = [gpu for gpu in adapters if gpu.vendor != "nvidia"] + nvidia
    cpu_name = platform.processor().strip() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    return HardwareProfile(
        system=platform.system() or sys.platform,
        architecture=platform.machine() or "unknown",
        cpu_name=cpu_name or "Unknown CPU",
        logical_cpus=max(1, os.cpu_count() or 1),
        memory_bytes=_total_memory_bytes(),
        gpus=tuple(adapters),
        build_profile=_release_build_profile(),
    )


def recommend_runtime(hardware: HardwareProfile) -> str:
    build_profile = hardware.build_profile
    if build_profile.startswith("nvidia"):
        return "cuda"
    if build_profile == "amd":
        return "vulkan"
    if build_profile == "cpu":
        return "cpu"
    if any(gpu.vendor == "nvidia" for gpu in hardware.gpus):
        return "cuda"
    if any(gpu.vendor == "amd" for gpu in hardware.gpus):
        return "vulkan"
    return "cpu"


def recommend_model(hardware: HardwareProfile, runtime_variant: str) -> ModelSpec:
    if runtime_variant in {"cuda", "vulkan"}:
        gpu_memory = max(
            (gpu.memory_bytes or 0 for gpu in hardware.gpus),
            default=0,
        )
        if gpu_memory >= 22 * 1024**3:
            return MODEL_CATALOG[4]
        if gpu_memory >= 12 * 1024**3:
            return MODEL_CATALOG[3]
        if gpu_memory >= 7 * 1024**3:
            return MODEL_CATALOG[2]
        if gpu_memory >= 4 * 1024**3:
            return MODEL_CATALOG[1]
        return MODEL_CATALOG[0]
    system_memory = hardware.memory_bytes or 0
    if system_memory >= 64 * 1024**3:
        return MODEL_CATALOG[4]
    if system_memory >= 32 * 1024**3:
        return MODEL_CATALOG[3]
    if system_memory >= 16 * 1024**3:
        return MODEL_CATALOG[2]
    if system_memory >= 10 * 1024**3:
        return MODEL_CATALOG[1]
    return MODEL_CATALOG[0]


def _asset_matches(name: str, variant: str) -> bool:
    lowered = name.lower()
    if not lowered.endswith(".zip") or "bin-win-" not in lowered:
        return False
    if variant == "cpu":
        return "bin-win-cpu-x64" in lowered
    if variant == "vulkan":
        return "bin-win-vulkan-x64" in lowered
    if variant == "cuda":
        return bool(re.search(r"bin-win-cuda-\d+(?:\.\d+)?-x64", lowered))
    return False


def select_runtime_assets(
    release: Mapping[str, object], variant: str
) -> tuple[str, list[dict[str, object]]]:
    tag = str(release.get("tag_name") or release.get("name") or "latest")
    raw_assets = release.get("assets")
    if not isinstance(raw_assets, list):
        raise RuntimeError("llama.cpp release metadata does not contain assets")
    assets = [item for item in raw_assets if isinstance(item, dict)]
    candidates = [
        item for item in assets if _asset_matches(str(item.get("name") or ""), variant)
    ]
    if not candidates:
        raise RuntimeError(f"llama.cpp release does not provide a Windows {variant} runtime")
    candidates.sort(key=lambda item: str(item.get("name") or ""), reverse=True)
    selected = [candidates[0]]
    if variant == "cuda":
        name = str(candidates[0].get("name") or "").lower()
        version_match = re.search(r"cuda-(\d+(?:\.\d+)?)", name)
        if version_match:
            marker = f"cuda-{version_match.group(1)}"
            companion = next(
                (
                    item
                    for item in assets
                    if str(item.get("name") or "").lower().endswith(".zip")
                    and "cudart" in str(item.get("name") or "").lower()
                    and marker in str(item.get("name") or "").lower()
                ),
                None,
            )
            if companion is not None:
                selected.append(companion)
    for asset in selected:
        if not str(asset.get("browser_download_url") or "").startswith("https://"):
            raise RuntimeError("llama.cpp release asset URL is missing or insecure")
    return tag, selected


class LocalModelManager:
    def __init__(
        self,
        root: Path,
        *,
        transport: httpx.BaseTransport | None = None,
        hardware_detector: Callable[[], HardwareProfile] = detect_hardware,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.root = root.resolve()
        self.runtime_dir = self.root / "runtime"
        self.models_dir = self.root / "models"
        self.downloads_dir = self.root / "downloads"
        self.config_path = self.root / "config.json"
        self.remote_profile_path = self.root / "remote-profile.json"
        self.local_active_path = self.root / "local-active.json"
        self.logs_dir = self.root / "logs"
        for directory in (
            self.root,
            self.runtime_dir,
            self.models_dir,
            self.downloads_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._transport = transport
        self._hardware_detector = hardware_detector
        self._process_factory = process_factory
        self._process: subprocess.Popen | None = None
        self._log_handle = None
        self._last_error = ""
        self._hardware: HardwareProfile | None = None
        self._lock = RLock()
        self._operation_lock = RLock()
        self._deployment_active = False
        self._rollback_pending = False
        self._rollback_config: dict[str, object] | None = None
        self._rollback_was_running = False

    def _client(self, timeout: float = 30) -> httpx.Client:
        return httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            transport=self._transport,
            headers={"User-Agent": "Kaor-local-model-manager/0.1"},
        )

    def hardware(self, *, refresh: bool = False) -> HardwareProfile:
        with self._lock:
            if self._hardware is None or refresh:
                self._hardware = self._hardware_detector()
            return self._hardware

    def catalog(self) -> list[dict[str, object]]:
        return [
            model.public((self.models_dir / model.filename).is_file())
            for model in MODEL_CATALOG
        ]

    def recommendation(self) -> dict[str, object]:
        hardware = self.hardware()
        runtime_variant = recommend_runtime(hardware)
        model = recommend_model(hardware, runtime_variant)
        return {
            "runtime_variant": runtime_variant,
            "model_id": model.id,
            "model_label": model.label,
            "reason": self._recommendation_reason(hardware, runtime_variant, model),
        }

    @staticmethod
    def _recommendation_reason(
        hardware: HardwareProfile, runtime_variant: str, model: ModelSpec
    ) -> str:
        if runtime_variant == "cuda":
            device = next((gpu.name for gpu in hardware.gpus if gpu.vendor == "nvidia"), "NVIDIA GPU")
            return f"CUDA was selected for {device}; {model.label} fits the detected memory tier."
        if runtime_variant == "vulkan":
            device = next((gpu.name for gpu in hardware.gpus if gpu.vendor == "amd"), "AMD GPU")
            return f"Vulkan was selected for {device}; {model.label} fits the detected memory tier."
        return f"CPU inference was selected; {model.label} fits the detected system memory tier."

    def _read_config(self) -> dict[str, object] | None:
        try:
            with self.config_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            self._last_error = f"local model configuration is unreadable: {exc}"
            return None
        return payload if isinstance(payload, dict) else None

    def _write_config(self, payload: Mapping[str, object]) -> None:
        self._write_json(self.config_path, payload)

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)

    def save_remote_profile(self, payload: Mapping[str, object]) -> None:
        allowed = {
            "api_base_url",
            "api_model",
            "api_reasoning_effort",
            "api_path",
            "custom_headers",
            "request_timeout_seconds",
        }
        with self._lock:
            self._write_json(
                self.remote_profile_path,
                {key: value for key, value in payload.items() if key in allowed},
            )

    def remote_profile(self) -> dict[str, object] | None:
        try:
            with self.remote_profile_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def clear_remote_profile(self) -> None:
        with self._lock:
            self.remote_profile_path.unlink(missing_ok=True)

    def local_active_provider(self) -> dict[str, object] | None:
        try:
            with self.local_active_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def mark_local_active(self, provider: Mapping[str, object]) -> None:
        allowed = {"base_url", "model", "api_path"}
        payload = {key: provider.get(key) for key in allowed}
        payload["schema_version"] = 1
        with self._lock:
            self._write_json(self.local_active_path, payload)

    def clear_local_active(self) -> None:
        with self._lock:
            self.local_active_path.unlink(missing_ok=True)

    def profile_is_active_local(self, profile: Mapping[str, object]) -> bool:
        active = self.local_active_provider()
        if active is None:
            return False
        return self.profile_matches_provider(profile, active)

    @staticmethod
    def profile_matches_provider(
        profile: Mapping[str, object], provider: Mapping[str, object]
    ) -> bool:
        profile_base = str(
            profile.get("api_base_url") or profile.get("base_url") or ""
        ).rstrip("/")
        profile_model = str(profile.get("api_model") or profile.get("model") or "")
        profile_path = str(
            profile.get("api_path") or profile.get("path") or "/chat/completions"
        )
        return (
            profile_base == str(provider.get("base_url") or "").rstrip("/")
            and profile_model == str(provider.get("model") or "")
            and profile_path == str(provider.get("api_path") or "/chat/completions")
        )

    @staticmethod
    def _validate_endpoint(base_url: str) -> str:
        normalized = base_url.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be a valid HTTP or HTTPS endpoint")
        return normalized

    def _configuration_payload(
        self, values: Mapping[str, object]
    ) -> dict[str, object]:
        mode = str(values.get("mode") or "managed")
        if mode not in {"managed", "external"}:
            raise ValueError("mode must be managed or external")
        model = str(values.get("model") or "").strip()
        auto_start = bool(values.get("auto_start", mode == "managed"))
        if mode == "external":
            base_url = self._validate_endpoint(str(values.get("base_url") or ""))
            if not model:
                raise ValueError("model is required for an external endpoint")
            payload: dict[str, object] = {
                "schema_version": 1,
                "mode": mode,
                "base_url": base_url,
                "api_path": str(values.get("api_path") or "/chat/completions"),
                "model": model,
                "auto_start": False,
            }
        else:
            executable = Path(str(values.get("executable_path") or "")).expanduser()
            model_path = Path(str(values.get("model_path") or "")).expanduser()
            if not executable.is_file():
                raise ValueError(f"llama.cpp server executable was not found: {executable}")
            if not model_path.is_file():
                raise ValueError(f"GGUF model was not found: {model_path}")
            if model_path.suffix.lower() != ".gguf":
                raise ValueError("managed local models must use a .gguf file")
            port = int(values.get("port") or 18080)
            if not 1024 <= port <= 65535:
                raise ValueError("port must be between 1024 and 65535")
            runtime_variant = str(values.get("runtime_variant") or "auto")
            if runtime_variant == "auto":
                runtime_variant = recommend_runtime(self.hardware())
            if runtime_variant not in {"cpu", "vulkan", "cuda"}:
                raise ValueError("runtime_variant must be auto, cpu, vulkan, or cuda")
            gpu_layers = int(values.get("gpu_layers", -1))
            if gpu_layers < 0:
                gpu_layers = 999 if runtime_variant != "cpu" else 0
            model = model or model_path.stem
            payload = {
                "schema_version": 1,
                "mode": mode,
                "base_url": f"http://127.0.0.1:{port}/v1",
                "api_path": "/chat/completions",
                "model": model,
                "executable_path": str(executable.resolve()),
                "model_path": str(model_path.resolve()),
                "runtime_variant": runtime_variant,
                "port": port,
                "context_size": int(values.get("context_size") or 8192),
                "gpu_layers": gpu_layers,
                "threads": int(values.get("threads") or max(1, (os.cpu_count() or 2) // 2)),
                "auto_start": auto_start,
            }
            if not 2048 <= int(payload["context_size"]) <= 131072:
                raise ValueError("context_size must be between 2048 and 131072")
            if not 1 <= int(payload["threads"]) <= 256:
                raise ValueError("threads must be between 1 and 256")
        return payload

    def configure(
        self,
        values: Mapping[str, object],
        *,
        probe_external: bool = False,
        stage_for_start: bool = False,
    ) -> dict[str, object]:
        payload = self._configuration_payload(values)
        with self._operation_lock:
            if probe_external and payload.get("mode") == "external" and not self._probe(
                payload, timeout=3.0
            ):
                raise ValueError(
                    "external endpoint is unreachable or does not list the configured model"
                )
            with self._lock:
                current = self._read_config()
                changed = current != payload
                process_running = bool(
                    self._process is not None and self._process.poll() is None
                )
                if changed and stage_for_start and not self._rollback_pending:
                    self._rollback_pending = True
                    self._rollback_config = dict(current) if current is not None else None
                    self._rollback_was_running = process_running
                elif changed and not stage_for_start:
                    self._clear_staged_configuration_locked()
                if changed and self._process is not None:
                    self.stop()
                self._write_config(payload)
                self._last_error = ""
            return payload

    def _clear_staged_configuration_locked(self) -> None:
        self._rollback_pending = False
        self._rollback_config = None
        self._rollback_was_running = False

    def _commit_staged_configuration(self) -> None:
        with self._lock:
            self._clear_staged_configuration_locked()

    def _has_staged_configuration(self) -> bool:
        with self._lock:
            return self._rollback_pending

    def _rollback_staged_configuration(self, error: str) -> None:
        with self._operation_lock:
            with self._lock:
                if not self._rollback_pending:
                    self._last_error = error
                    return
                previous = (
                    dict(self._rollback_config)
                    if self._rollback_config is not None
                    else None
                )
                restart_previous = self._rollback_was_running
            try:
                self._stop_process()
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass
            with self._lock:
                if previous is None:
                    self.config_path.unlink(missing_ok=True)
                else:
                    self._write_config(previous)
                self._clear_staged_configuration_locked()
            rollback_error = ""
            if (
                restart_previous
                and previous is not None
                and previous.get("mode") == "managed"
            ):
                try:
                    self.start()
                except (OSError, RuntimeError, ValueError) as exc:
                    rollback_error = f"; previous local model restart failed: {exc}"
            self._last_error = error + rollback_error

    def provider(self) -> dict[str, object] | None:
        config = self._read_config()
        if config is None:
            return None
        return {
            "base_url": config["base_url"],
            "api_key": "",
            "model": config["model"],
            "api_path": config.get("api_path", "/chat/completions"),
            "custom_headers": {},
            "timeout_seconds": 600,
            "temperature": 0.2,
            "json_mode": True,
            "reasoning_effort": "",
        }

    @staticmethod
    def _listed_model_ids(response: httpx.Response) -> set[str]:
        try:
            payload = response.json()
        except ValueError:
            return set()
        if isinstance(payload, dict):
            rows = payload.get("data", payload.get("models", []))
        else:
            rows = payload
        if not isinstance(rows, list):
            return set()
        model_ids: set[str] = set()
        for row in rows:
            if isinstance(row, str) and row.strip():
                model_ids.add(row.strip())
            elif isinstance(row, dict):
                value = row.get("id") or row.get("model") or row.get("name")
                if value is not None and str(value).strip():
                    model_ids.add(str(value).strip())
        return model_ids

    def _probe(self, config: Mapping[str, object], timeout: float = 0.75) -> bool:
        base_url = str(config.get("base_url") or "").rstrip("/")
        model = str(config.get("model") or "").strip()
        if not base_url or not model:
            return False
        managed = config.get("mode") == "managed"
        if managed:
            process = self._process
            if process is None or process.poll() is not None:
                return False
        try:
            with self._client(timeout) as client:
                if managed:
                    health = client.get(f"{base_url.removesuffix('/v1')}/health")
                    if health.status_code != 200:
                        return False
                models = client.get(f"{base_url}/models")
            return models.status_code == 200 and model in self._listed_model_ids(models)
        except (httpx.HTTPError, ValueError):
            return False

    def _log_tail(self, limit: int = 4000) -> str:
        path = self.logs_dir / "llama-server.log"
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - limit))
                return handle.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def status(self, *, include_hardware: bool = True) -> dict[str, object]:
        config = self._read_config()
        process = self._process
        process_running = bool(process is not None and process.poll() is None)
        ready = bool(config and self._probe(config))
        if config is None:
            state = "not_configured"
        elif ready:
            state = "ready"
        elif process_running:
            state = "starting"
        elif process is not None and process.poll() is not None:
            state = "failed"
            if not self._last_error:
                self._last_error = self._log_tail() or f"llama.cpp exited with code {process.poll()}"
        elif config.get("mode") == "external":
            state = "unreachable"
        else:
            state = "stopped"
        result: dict[str, object] = {
            "state": state,
            "ready": ready,
            "process_running": process_running,
            "managed_process": process is not None,
            "configuration": config,
            "provider": self.provider(),
            "error": self._last_error or None,
            "log_tail": self._log_tail() if state == "failed" else "",
            "recommendation": self.recommendation(),
            "catalog": self.catalog(),
            "remote_profile_available": self.remote_profile() is not None,
        }
        if include_hardware:
            result["hardware"] = self.hardware().public()
        if ready and self._has_staged_configuration():
            self._commit_staged_configuration()
        elif state == "failed" and self._has_staged_configuration():
            self._rollback_staged_configuration(str(result["error"] or "startup failed"))
        return result

    @staticmethod
    def _port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True

    def start(self) -> dict[str, object]:
        with self._operation_lock:
            return self._start()

    def start_with_rollback(self) -> dict[str, object]:
        staged_at_start = self._has_staged_configuration()
        try:
            status = self.start()
            if status.get("state") == "failed":
                raise RuntimeError(str(status.get("error") or "llama.cpp failed to start"))
            return status
        except (OSError, RuntimeError, ValueError) as exc:
            if self._has_staged_configuration():
                self._rollback_staged_configuration(str(exc))
            elif not staged_at_start:
                self._last_error = str(exc)
            raise

    def _start(self) -> dict[str, object]:
        with self._lock:
            config = self._read_config()
            if config is None:
                raise RuntimeError("configure a local model before starting it")
            if config.get("mode") == "external":
                return self.status()
            if self._process is not None and self._process.poll() is None:
                return self.status()
            executable = Path(str(config.get("executable_path") or ""))
            model_path = Path(str(config.get("model_path") or ""))
            if not executable.is_file():
                raise RuntimeError(f"llama.cpp server executable was not found: {executable}")
            if not model_path.is_file():
                raise RuntimeError(f"GGUF model was not found: {model_path}")
            port = int(config.get("port") or 18080)
            if not self._port_available(port):
                raise RuntimeError(f"local model port {port} is already in use")
            command = [
                str(executable),
                "--model",
                str(model_path),
                "--alias",
                str(config["model"]),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--ctx-size",
                str(config.get("context_size", 8192)),
                "--n-gpu-layers",
                str(config.get("gpu_layers", 0)),
                "--threads",
                str(config.get("threads", 4)),
                "--jinja",
                "--chat-template-kwargs",
                '{"enable_thinking":false}',
            ]
            if str(config.get("runtime_variant")) != "cpu":
                command.extend(
                    [
                        "--fit",
                        "on",
                        "--fit-target",
                        "1024",
                        "--flash-attn",
                        "auto",
                    ]
                )
            log_path = self.logs_dir / "llama-server.log"
            if self._log_handle is not None:
                self._log_handle.close()
            self._log_handle = log_path.open("w", encoding="utf-8", errors="replace")
            kwargs: dict[str, object] = {
                "cwd": str(executable.parent),
                "stdin": subprocess.DEVNULL,
                "stdout": self._log_handle,
                "stderr": subprocess.STDOUT,
                "creationflags": _creation_flags(),
            }
            try:
                self._process = self._process_factory(command, **kwargs)
            except Exception:
                self._process = None
                self._log_handle.close()
                self._log_handle = None
                raise
            self._last_error = ""
        time.sleep(0.05)
        if self._process is not None and self._process.poll() is not None:
            self._last_error = self._log_tail() or "llama.cpp stopped during startup"
            raise RuntimeError(self._last_error)
        return self.status()

    def start_and_wait(
        self,
        *,
        timeout: float = 180,
        cancel_event: Event | None = None,
    ) -> dict[str, object]:
        with self._operation_lock:
            configuration = self._read_config()
            if configuration is None:
                raise RuntimeError("configure a local model before starting it")
            staged_at_start = self._has_staged_configuration()
            try:
                if configuration.get("mode") == "external":
                    if not self._probe(configuration, timeout=min(3.0, timeout)):
                        raise RuntimeError(
                            "external endpoint is unreachable or does not list the configured model"
                        )
                    self._commit_staged_configuration()
                    return self.status()
                initial = self.start()
                if initial.get("state") == "failed":
                    raise RuntimeError(
                        str(initial.get("error") or "llama.cpp failed to start")
                    )
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("job cancelled")
                    if self._probe(configuration, timeout=1.0):
                        self._commit_staged_configuration()
                        return self.status()
                    process = self._process
                    if process is not None and process.poll() is not None:
                        raise RuntimeError(
                            self._log_tail()
                            or "llama.cpp stopped while loading the model"
                        )
                    time.sleep(0.5)
                raise RuntimeError(
                    f"local model did not become ready within {int(timeout)} seconds; "
                    f"{self._log_tail()}"
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if self._has_staged_configuration():
                    self._rollback_staged_configuration(str(exc))
                elif not staged_at_start:
                    try:
                        self._stop_process()
                    except (OSError, RuntimeError, subprocess.SubprocessError):
                        pass
                    self._last_error = str(exc)
                raise

    def stop(self) -> dict[str, object]:
        with self._operation_lock:
            return self._stop()

    def _stop(self) -> dict[str, object]:
        self._stop_process()
        # Keep the response shape stable for the WebUI after a stop action.
        return self.status()

    def _stop_process(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            self._process = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def close(self) -> None:
        with self._operation_lock:
            try:
                if self._has_staged_configuration():
                    self._rollback_staged_configuration(
                        "local model startup was interrupted by application shutdown"
                    )
                self._stop_process()
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass

    def autostart(self) -> None:
        config = self._read_config()
        if config and config.get("mode") == "managed" and config.get("auto_start"):
            try:
                self.start()
            except (OSError, RuntimeError, ValueError) as exc:
                self._last_error = str(exc)

    @staticmethod
    def _validate_download_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise RuntimeError("managed downloads require an HTTPS URL")

    @staticmethod
    def _verify_sha256(path: Path, expected: str) -> None:
        if not expected:
            return
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest().lower()
        if actual != expected.lower():
            raise RuntimeError(
                f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
            )

    @classmethod
    def _verify_download(
        cls,
        path: Path,
        *,
        expected_sha256: str = "",
        expected_size: int | None = None,
        gguf: bool = False,
    ) -> None:
        if expected_size is not None and path.stat().st_size != expected_size:
            raise RuntimeError(
                f"size mismatch for {path.name}: expected {expected_size}, "
                f"got {path.stat().st_size}"
            )
        if gguf:
            with path.open("rb") as handle:
                if handle.read(4) != b"GGUF":
                    raise RuntimeError(f"invalid GGUF header: {path.name}")
        cls._verify_sha256(path, expected_sha256)

    def _download(
        self,
        url: str,
        destination: Path,
        *,
        expected_sha256: str = "",
        expected_size: int | None = None,
        gguf: bool = False,
        progress: Callable[[int, int | None], None] | None = None,
        cancel_event: Event | None = None,
    ) -> Path:
        self._validate_download_url(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            try:
                self._verify_download(
                    destination,
                    expected_sha256=expected_sha256,
                    expected_size=expected_size,
                    gguf=gguf,
                )
            except RuntimeError:
                destination.unlink()
            else:
                if progress:
                    progress(destination.stat().st_size, destination.stat().st_size)
                return destination
        partial = destination.with_suffix(destination.suffix + ".part")
        if partial.is_file() and expected_size is not None and partial.stat().st_size > expected_size:
            partial.unlink()
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        response_digest = ""
        with self._client(timeout=120) as client:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code == 416 and offset:
                    try:
                        self._verify_download(
                            partial,
                            expected_sha256=expected_sha256,
                            expected_size=expected_size,
                            gguf=gguf,
                        )
                    except RuntimeError:
                        partial.unlink(missing_ok=True)
                        return self._download(
                            url,
                            destination,
                            expected_sha256=expected_sha256,
                            expected_size=expected_size,
                            gguf=gguf,
                            progress=progress,
                            cancel_event=cancel_event,
                        )
                    partial.replace(destination)
                    return destination
                response.raise_for_status()
                raw_digest = (
                    response.headers.get("x-linked-etag")
                    or response.headers.get("etag")
                    or ""
                ).strip('W/"')
                if re.fullmatch(r"[0-9a-fA-F]{64}", raw_digest):
                    response_digest = raw_digest
                append = response.status_code == 206 and offset > 0
                if not append:
                    offset = 0
                content_length = response.headers.get("content-length")
                total = offset + int(content_length) if content_length and content_length.isdigit() else None
                mode = "ab" if append else "wb"
                completed = offset
                with partial.open(mode) as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        if cancel_event and cancel_event.is_set():
                            raise InterruptedError("job cancelled")
                        if not chunk:
                            continue
                        handle.write(chunk)
                        completed += len(chunk)
                        if progress:
                            progress(completed, total)
                    handle.flush()
                    os.fsync(handle.fileno())
        partial.replace(destination)
        try:
            self._verify_download(
                destination,
                expected_sha256=expected_sha256 or response_digest,
                expected_size=expected_size,
                gguf=gguf,
            )
        except RuntimeError:
            destination.unlink(missing_ok=True)
            raise
        return destination

    @staticmethod
    def _extract_archives(archives: Iterable[Path], destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        for archive in archives:
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    target = (destination / member.filename).resolve()
                    if target != root and root not in target.parents:
                        raise RuntimeError(f"unsafe path in runtime archive: {member.filename}")
                bundle.extractall(destination)

    def _release(self) -> dict[str, object]:
        return {
            "tag_name": LLAMA_RELEASE_TAG,
            "assets": [dict(asset) for asset in LLAMA_RELEASE_ASSETS],
        }

    def deployment_active(self) -> bool:
        with self._lock:
            return self._deployment_active

    def deploy(
        self,
        values: Mapping[str, object],
        progress: Progress,
        cancel_event: Event,
    ) -> dict[str, object]:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("another local model operation is already running")
        with self._lock:
            if self._deployment_active:
                self._operation_lock.release()
                raise RuntimeError("a local model deployment is already running")
            self._deployment_active = True
        try:
            return self._deploy_locked(values, progress, cancel_event)
        finally:
            with self._lock:
                self._deployment_active = False
            self._operation_lock.release()

    def _deploy_locked(
        self,
        values: Mapping[str, object],
        progress: Progress,
        cancel_event: Event,
    ) -> dict[str, object]:
        hardware = self.hardware()
        variant = str(values.get("runtime_variant") or "auto")
        if variant == "auto":
            variant = recommend_runtime(hardware)
        if variant not in {"cpu", "vulkan", "cuda"}:
            raise ValueError("runtime_variant must be auto, cpu, vulkan, or cuda")
        model_id = str(values.get("model_id") or "")
        if not model_id:
            model_id = recommend_model(hardware, variant).id
        model = next((item for item in MODEL_CATALOG if item.id == model_id), None)
        if model is None:
            raise ValueError(f"unknown local model: {model_id}")

        startup_timeout = int(values.get("startup_timeout_seconds") or 180)

        def start_and_wait(
            configuration: Mapping[str, object], *, reused: bool
        ) -> dict[str, object]:
            try:
                progress(
                    0.95,
                    "Starting local model server",
                    {"reused": reused},
                )
                self.start_and_wait(
                    timeout=startup_timeout,
                    cancel_event=cancel_event,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if self._has_staged_configuration():
                    self._rollback_staged_configuration(str(exc))
                raise
            progress(0.99, "Local model is ready", {"reused": reused})
            return {
                "runtime_variant": variant,
                "model_id": model.id,
                "configuration": dict(configuration),
                "provider": self.provider(),
                "reused": reused,
            }

        existing = self._read_config()
        existing_model_path = Path(str(existing.get("model_path") or "")) if existing else None
        existing_valid = False
        if existing_model_path and existing_model_path.is_file():
            try:
                self._verify_download(
                    existing_model_path,
                    expected_sha256=model.sha256,
                    expected_size=model.size_bytes,
                    gguf=True,
                )
                existing_valid = True
            except RuntimeError:
                try:
                    managed_path = (self.models_dir / model.filename).resolve()
                    candidate_path = existing_model_path.resolve()
                except OSError:
                    managed_path = None
                    candidate_path = None
                if candidate_path is not None and candidate_path == managed_path:
                    existing_model_path.unlink(missing_ok=True)
        if (
            existing
            and existing.get("mode") == "managed"
            and existing.get("model") == model.id
            and existing.get("runtime_variant") == variant
            and Path(str(existing.get("executable_path") or "")).is_file()
            and existing_valid
        ):
            progress(0.9, "Validated existing local model artifacts", {"reused": True})
            configuration = self.configure(
                {
                    "mode": "managed",
                    "model": model.id,
                    "executable_path": str(existing["executable_path"]),
                    "model_path": str(existing_model_path),
                    "runtime_variant": variant,
                    "port": int(values.get("port") or 18080),
                    "context_size": int(values.get("context_size") or 8192),
                    "gpu_layers": int(
                        values.get("gpu_layers", -1 if variant != "cpu" else 0)
                    ),
                    "threads": int(
                        values.get("threads")
                        or max(1, hardware.logical_cpus // 2)
                    ),
                    "auto_start": bool(values.get("auto_start", True)),
                },
                stage_for_start=True,
            )
            return start_and_wait(configuration, reused=True)

        progress(0.01, "Resolving llama.cpp runtime", {"runtime_variant": variant})
        release = self._release()
        tag, assets = select_runtime_assets(release, variant)
        runtime_target = self.runtime_dir / tag / variant
        executable_name = "llama-server.exe" if os.name == "nt" else "llama-server"
        executable = next(runtime_target.rglob(executable_name), None) if runtime_target.is_dir() else None

        asset_sizes = [int(asset.get("size") or 0) for asset in assets]
        estimated_total = max(1, sum(asset_sizes) + model.size_bytes)
        completed_before = 0
        archives: list[Path] = []
        if executable is None:
            for index, asset in enumerate(assets):
                url = str(asset["browser_download_url"])
                name = str(asset.get("name") or f"llama-runtime-{index}.zip")
                destination = self.downloads_dir / name

                def runtime_progress(done: int, _total: int | None, base: int = completed_before) -> None:
                    progress(
                        min(0.35, (base + done) / estimated_total),
                        f"Downloading llama.cpp runtime ({index + 1}/{len(assets)})",
                        {"filename": name, "downloaded_bytes": done},
                    )

                self._download(
                    url,
                    destination,
                    expected_sha256=(
                        str(asset.get("digest") or "").removeprefix("sha256:")
                        if str(asset.get("digest") or "").startswith("sha256:")
                        else ""
                    ),
                    expected_size=int(asset.get("size") or 0) or None,
                    progress=runtime_progress,
                    cancel_event=cancel_event,
                )
                archives.append(destination)
                completed_before += destination.stat().st_size
            progress(0.36, "Extracting llama.cpp runtime", None)
            staging = self.runtime_dir / f".staging-{uuid4().hex}"
            try:
                self._extract_archives(archives, staging)
                executable = next(staging.rglob(executable_name), None)
                if executable is None:
                    raise RuntimeError(f"{executable_name} was not found in the runtime archive")
                if runtime_target.is_dir():
                    shutil.rmtree(runtime_target)
                runtime_target.parent.mkdir(parents=True, exist_ok=True)
                staging.replace(runtime_target)
                executable = next(runtime_target.rglob(executable_name))
            finally:
                if staging.is_dir():
                    shutil.rmtree(staging)
        else:
            completed_before = sum(asset_sizes)

        model_path = self.models_dir / model.filename

        def model_progress(done: int, total: int | None) -> None:
            denominator = max(estimated_total, completed_before + (total or model.size_bytes))
            value = 0.36 + 0.54 * min(1.0, done / max(1, total or model.size_bytes))
            progress(
                min(0.9, value),
                "Downloading translation model",
                {
                    "filename": model.filename,
                    "downloaded_bytes": done,
                    "total_bytes": total,
                    "overall_bytes": min(denominator, completed_before + done),
                },
            )

        self._download(
            model.url,
            model_path,
            expected_sha256=model.sha256,
            expected_size=model.size_bytes,
            gguf=True,
            progress=model_progress,
            cancel_event=cancel_event,
        )
        license_path = self.models_dir / "licenses" / f"{model.id}-LICENSE.txt"
        progress(0.91, "Downloading model license", {"filename": license_path.name})
        self._download(
            model.license_url,
            license_path,
            cancel_event=cancel_event,
        )
        manifest_path = self.models_dir / f"{model.id}.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "id": model.id,
                    "filename": model.filename,
                    "source": model.url,
                    "revision": model.revision,
                    "size_bytes": model.size_bytes,
                    "sha256": model.sha256,
                    "license_file": str(license_path.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        progress(0.92, "Configuring local model", None)
        configuration = self.configure(
            {
                "mode": "managed",
                "model": model.id,
                "executable_path": str(executable),
                "model_path": str(model_path),
                "runtime_variant": variant,
                "port": int(values.get("port") or 18080),
                "context_size": int(values.get("context_size") or 8192),
                "gpu_layers": int(values.get("gpu_layers", -1 if variant != "cpu" else 0)),
                "threads": int(values.get("threads") or max(1, hardware.logical_cpus // 2)),
                "auto_start": bool(values.get("auto_start", True)),
            },
            stage_for_start=True,
        )
        return start_and_wait(configuration, reused=False)
