from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from .media import application_root


LOG_FORMAT = "%(asctime)sZ | %(levelname)s | %(name)s | %(message)s"
_STRUCTURED_LINE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[^|]*)\s*\|\s*"
    r"(?P<level>[A-Z]+)\s*\|\s*(?P<logger>[^|]+)\|\s*(?P<message>.*)$"
)
_ERROR_WORDS = re.compile(
    r"\b(error|exception|failed|failure|fatal|traceback|denied|mismatch|missing)\b",
    re.IGNORECASE,
)
_WARNING_WORDS = re.compile(r"\b(warn(?:ing)?|retry|fallback|slow)\b", re.IGNORECASE)
_SENSITIVE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)([\"']?(?:proxy-)?authorization[\"']?\s*[:=]\s*[\"']?"
            r"(?:(?:bearer|basic|token)\s+)?)([^\"'\s,;}]+)"
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret)"
            r"\s*[\"']?\s*[:=]\s*[\"']?)([^\"'\s,}]+)"
        ),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(?i)\bsk-[a-z0-9_-]{8,}\b"), "[REDACTED]"),
    (
        re.compile(r"(?i)(https?://)([^/\s:@]+):([^@\s/]+)@"),
        r"\1[REDACTED]@",
    ),
)


def redact_log_text(value: str) -> str:
    redacted = value
    for pattern, replacement in _SENSITIVE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


@dataclass(frozen=True)
class RepairGuide:
    id: str
    title: str
    summary: str
    patterns: tuple[str, ...]
    steps: tuple[str, ...]
    anchor: str

    def public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "patterns": list(self.patterns),
            "steps": list(self.steps),
            "anchor": self.anchor,
        }


REPAIR_GUIDES: tuple[RepairGuide, ...] = (
    RepairGuide(
        "ocr-runtime-missing",
        "OCR 运行时缺失",
        "发行包缺少 PaddleOCR/PaddleX，或解压时文件被安全软件隔离。",
        ("PaddleOCR is not installed", "PaddleX is not installed", "No module named 'paddle'"),
        ("重新完整解压对应硬件版本。", "检查安全软件隔离区。", "确认 Kaor.exe 旁的 _internal/paddle 与 _internal/paddlex 存在。"),
        "ocr-runtime-missing",
    ),
    RepairGuide(
        "cuda-unavailable",
        "CUDA/GPU 不可用",
        "当前阶段请求了 CUDA，但驱动、运行时或显卡版本不匹配。",
        ("CUDA was requested", "CUDA device", "cuda driver", "cudnn", "no kernel image"),
        ("在日志页确认失败的是 OCR、音频还是本地翻译。", "更新 NVIDIA 驱动后重启。", "仍失败时在该阶段临时选择 CPU，并核对发行包是否为 NVIDIA 版。"),
        "cuda-unavailable",
    ),
    RepairGuide(
        "gpu-out-of-memory",
        "显存不足",
        "批量、模型或上下文超过当前可用显存。",
        ("out of memory", "ResourceExhausted", "CUDNN_STATUS_ALLOC_FAILED", "failed to allocate"),
        ("关闭占用显存的其他程序。", "降低 OCR/ASR batch。", "本地翻译改用更小模型或缩小 context size。", "不要让 OCR、ASR 与本地 LLM 同时运行。"),
        "gpu-out-of-memory",
    ),
    RepairGuide(
        "uvr-model-damaged",
        "BS-Roformer 下载失败或校验异常",
        "首次人声分离会从固定上游下载 CKPT，校验后只读取 models/uvr 内的 CKPT 与 YAML。",
        ("UVR model download", "UVR model size mismatch", "SHA-256 mismatch", "BS-Roformer", "model_bs_roformer"),
        ("检查 VPN、代理和 GitHub Release 是否可访问。", "确认程序目录可写并保留 .part 后重试。", "若哈希仍不一致，删除错误的 CKPT 与 .part，再从日志中的固定上游重新下载。"),
        "uvr-model-damaged",
    ),
    RepairGuide(
        "audio-worker-missing",
        "音频 Worker 缺失",
        "KaorAudioWorker.exe 不在主程序旁，UVR/ASR 无法隔离运行。",
        ("packaged audio worker was not found", "KaorAudioWorker.exe", "worker exited without a result"),
        ("确认 KaorAudioWorker.exe 与 Kaor.exe 位于同一目录。", "检查安全软件隔离记录。", "重新完整解压当前发行包。"),
        "audio-worker-missing",
    ),
    RepairGuide(
        "pytorch-audio-runtime",
        "本地音频推理运行时异常",
        "PyTorch、TorchAudio、NeMo、FunASR 或 audio-separator 没有被正确装入发行包。",
        ("PyTorch is required", "No module named 'torch'", "audio-separator is required", "nemo", "funasr"),
        ("先在日志中确认缺失模块名。", "使用对应 CPU/AMD/NVIDIA 完整发行包。", "不要混合复制不同版本的 _internal 目录。"),
        "pytorch-audio-runtime",
    ),
    RepairGuide(
        "ffmpeg-runtime",
        "FFmpeg 或媒体文件异常",
        "媒体工具缺失，或输入视频/音频无法解析。",
        ("ffmpeg was not found", "ffprobe", "Invalid data found", "did not find a video stream"),
        ("确认 bin/ffmpeg.exe 存在。", "将片源复制到短路径后重试。", "用播放器确认片源可完整播放；必要时先转封装为 MP4/MKV。"),
        "ffmpeg-runtime",
    ),
    RepairGuide(
        "translation-auth",
        "翻译接口认证失败",
        "API Key、Base URL 或中转站认证头不正确。",
        ("HTTP 401", "HTTP 403", "Unauthorized", "invalid api key", "authentication"),
        ("重新保存 API Key。", "检查中转站要求的是 Authorization 还是自定义 Header。", "确认 Base URL 包含服务要求的 /v1。"),
        "translation-auth",
    ),
    RepairGuide(
        "translation-rate-limit",
        "接口限流",
        "上游返回 429，当前并发或频率超过额度。",
        ("HTTP 429", "rate limit", "Too Many Requests"),
        ("降低翻译并发。", "等待上游冷却窗口。", "检查账户余额、RPM 与 TPM 限制。"),
        "translation-rate-limit",
    ),
    RepairGuide(
        "translation-timeout",
        "接口或中转站超时",
        "连接已建立，但上游未在代理时限内完成整批字幕。",
        ("HTTP 524", "timed out", "origin_response_timeout", "ReadTimeout"),
        ("保留完整 CSV 参考不变，由程序自动缩小输出批次续传。", "提高客户端超时只对非 524 情况有效。", "中转站固定 120 秒时换更快模型或直连上游。"),
        "translation-timeout",
    ),
    RepairGuide(
        "translation-invalid-json",
        "模型返回的不是有效 JSON",
        "上游返回空页面、HTML 错误页或模型没有遵守结构化输出。",
        ("Expecting value: line 1 column 1", "response was not valid JSON", "JSONDecodeError"),
        ("在日志中查看 HTTP 状态和响应预览。", "确认请求路径为 /chat/completions。", "关闭不兼容的中转站流式改写，或选择支持 JSON mode 的模型。"),
        "translation-invalid-json",
    ),
    RepairGuide(
        "fusion-input-missing",
        "混合校准输入不完整",
        "OCR 或语音 CSV 没有成功落盘，校准任务因此停止。",
        ("fusion requires completed OCR and speech CSV", "missing=['speech.csv']", "missing=['ocr.csv']"),
        ("分别检查 OCR 与 ASR 任务状态。", "在工作区确认 ocr.csv 和 speech.csv 非空。", "对缺失阶段点重新运行；完整阶段会自动跳过。"),
        "fusion-input-missing",
    ),
    RepairGuide(
        "local-model-runtime",
        "本地翻译模型启动失败",
        "llama.cpp 后端、Vulkan/CUDA 驱动、GGUF 或端口存在问题。",
        ("llama.cpp", "llama-server", "GGUF", "Vulkan", "local model port", "within 180 seconds"),
        ("先查看 llama-server 日志源。", "确认 GGUF 校验通过且磁盘空间充足。", "端口占用时改用 18081。", "AMD Vulkan 失败时更新显卡驱动；再切 CPU 运行时验证模型本身。"),
        "local-model-runtime",
    ),
    RepairGuide(
        "download-integrity",
        "下载文件校验失败",
        "断点文件、代理缓存或镜像返回内容与固定清单不一致。",
        ("SHA-256 mismatch", "size mismatch", "invalid GGUF header", "unsafe path in runtime archive"),
        ("删除 data/local-models/downloads 下对应 .part。", "不要关闭 HTTPS 校验或跳过 SHA-256。", "确认代理没有返回登录页/限流页后重新一键部署。"),
        "download-integrity",
    ),
    RepairGuide(
        "network-download",
        "模型或运行时下载失败",
        "DNS、代理、GitHub/Hugging Face 连通性或断点续传失败。",
        ("NameResolutionError", "ConnectError", "connection reset", "model download failed", "huggingface"),
        ("保持现有 .part 文件，恢复网络后再次点击部署即可续传。", "检查系统代理能访问 GitHub Releases 与 huggingface.co。", "校准系统时间，避免 HTTPS 证书错误。"),
        "network-download",
    ),
    RepairGuide(
        "disk-space",
        "磁盘空间不足",
        "模型、缓存或视频导出超过当前磁盘剩余容量。",
        ("No space left on device", "disk full", "There is not enough space"),
        ("清理 data/cache 与失败导出。", "为本地模型预留模型体积至少 1.5 倍空间。", "重新运行任务会从完整检查点继续。"),
        "disk-space",
    ),
    RepairGuide(
        "permission-denied",
        "文件访问被拒绝",
        "程序目录不可写、文件被占用或安全软件拦截。",
        ("PermissionError", "Access is denied", "拒绝访问", "being used by another process"),
        ("把完整目录移到普通可写路径。", "关闭占用 CSV/视频的表格软件或播放器。", "检查安全软件拦截记录后重试。"),
        "permission-denied",
    ),
)


def configure_runtime_logging(data_dir: Path) -> Path:
    log_dir = data_dir.resolve() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "kaor.log"
    root = logging.getLogger()
    marker = str(log_path).casefold()
    if not any(
        isinstance(handler, RotatingFileHandler)
        and str(getattr(handler, "baseFilename", "")).casefold() == marker
        for handler in root.handlers
    ):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=4,
            encoding="utf-8",
        )
        formatter = logging.Formatter(LOG_FORMAT)
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)
    _install_exception_hooks()
    return log_path


def _install_exception_hooks() -> None:
    original = sys.excepthook

    def log_exception(exc_type, exc_value, exc_traceback):  # type: ignore[no-untyped-def]
        logging.getLogger("kaor.crash").critical(
            "uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )
        original(exc_type, exc_value, exc_traceback)

    if not getattr(sys.excepthook, "_kaor_logging_hook", False):
        setattr(log_exception, "_kaor_logging_hook", True)
        sys.excepthook = log_exception

    if hasattr(threading, "excepthook") and not getattr(
        threading.excepthook, "_kaor_logging_hook", False
    ):
        original_thread = threading.excepthook

        def log_thread_exception(args):  # type: ignore[no-untyped-def]
            logging.getLogger("kaor.crash").critical(
                "uncaught thread exception",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            original_thread(args)

        setattr(log_thread_exception, "_kaor_logging_hook", True)
        threading.excepthook = log_thread_exception


def matching_guides(text: str) -> list[str]:
    lowered = text.casefold()
    return [
        guide.id
        for guide in REPAIR_GUIDES
        if any(pattern.casefold() in lowered for pattern in guide.patterns)
    ]


def _tail_lines(path: Path, limit: int, max_bytes: int = 2 * 1024 * 1024) -> list[str]:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        raw = handle.read()
    if size > max_bytes:
        _, _, raw = raw.partition(b"\n")
    return raw.decode("utf-8", errors="replace").splitlines()[-limit:]


class DiagnosticLogService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()

    def _paths(self) -> list[Path]:
        candidates: list[Path] = []
        patterns = (
            "logs/*.log*",
            "local-models/logs/*.log*",
            "projects/*/cache/**/*.log",
        )
        for pattern in patterns:
            candidates.extend(self.data_dir.glob(pattern))
        startup = application_root() / "startup-trace.log"
        if startup.is_file():
            candidates.append(startup)
        existing: list[tuple[float, Path]] = []
        for path in {candidate.resolve() for candidate in candidates}:
            try:
                if path.is_file():
                    existing.append((path.stat().st_mtime, path))
            except OSError:
                continue
        existing.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in existing[:200]]

    def _source(self, path: Path) -> dict[str, object]:
        try:
            relative = path.relative_to(self.data_dir).as_posix()
        except ValueError:
            relative = path.name
        source_id = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
        stat = path.stat()
        return {
            "id": source_id,
            "name": relative,
            "size_bytes": stat.st_size,
            "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "path": str(path),
        }

    def sources(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for path in self._paths():
            try:
                rows.append(self._source(path))
            except OSError:
                continue
        return rows

    def entries(
        self,
        *,
        source: str = "all",
        tail: int = 1000,
        query: str = "",
        levels: Iterable[str] = (),
    ) -> dict[str, object]:
        source_rows = self.sources()
        by_id = {str(row["id"]): row for row in source_rows}
        selected = (
            [by_id[source]] if source != "all" and source in by_id else source_rows
        )
        allowed_levels = {value.upper() for value in levels}
        query_folded = query.casefold().strip()
        rows: list[dict[str, object]] = []
        row_limit = max(1, min(5000, tail))
        remaining_read_budget = row_limit if source != "all" else row_limit * 2
        for source_row in selected:
            if remaining_read_budget <= 0:
                break
            path = Path(str(source_row["path"]))
            try:
                lines = _tail_lines(path, min(row_limit, remaining_read_budget))
            except OSError:
                continue
            remaining_read_budget -= len(lines)
            fallback_time = str(source_row["updated_at"])
            for index, line in enumerate(lines):
                if query_folded and query_folded not in line.casefold():
                    continue
                match = _STRUCTURED_LINE.match(line)
                if match:
                    timestamp = match.group("timestamp").strip()
                    level = match.group("level").upper()
                    logger_name = match.group("logger").strip()
                    message = match.group("message")
                else:
                    timestamp = fallback_time
                    level = "ERROR" if _ERROR_WORDS.search(line) else (
                        "WARNING" if _WARNING_WORDS.search(line) else "INFO"
                    )
                    logger_name = str(source_row["name"])
                    message = line
                if allowed_levels and level not in allowed_levels:
                    continue
                entry_key = f"{source_row['id']}:{index}:{line}"
                rows.append(
                    {
                        "id": hashlib.sha1(entry_key.encode("utf-8")).hexdigest(),
                        "timestamp": timestamp,
                        "level": level,
                        "logger": logger_name,
                        "message": redact_log_text(message),
                        "source_id": source_row["id"],
                        "source_name": source_row["name"],
                        "guide_ids": matching_guides(line),
                    }
                )
        rows.sort(key=lambda row: str(row["timestamp"]))
        return {
            "sources": [
                {key: value for key, value in row.items() if key != "path"}
                for row in source_rows
            ],
            "entries": rows[-row_limit:],
            "guides": [guide.public() for guide in REPAIR_GUIDES],
        }

    def export_bundle(self) -> Path:
        export_dir = self.data_dir / "diagnostics"
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        destination = export_dir / f"kaor-diagnostics-{stamp}-{uuid4().hex[:8]}.zip"
        system = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": Path(sys.executable).name,
            "application_root": application_root().name,
            "logical_cpus": os.cpu_count(),
        }
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("system.json", json.dumps(system, ensure_ascii=False, indent=2))
            bundle.writestr(
                "repair-guides.json",
                json.dumps([guide.public() for guide in REPAIR_GUIDES], ensure_ascii=False, indent=2),
            )
            for index, path in enumerate(self._paths(), start=1):
                try:
                    contents = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                bundle.writestr(
                    f"logs/{index:03d}-{path.name}", redact_log_text(contents)
                )
        exports: list[tuple[float, Path]] = []
        for path in export_dir.glob("kaor-diagnostics-*.zip"):
            try:
                exports.append((path.stat().st_mtime, path))
            except OSError:
                continue
        exports.sort(key=lambda item: item[0], reverse=True)
        for _, stale in exports[5:]:
            try:
                stale.unlink()
            except OSError:
                pass
        return destination
