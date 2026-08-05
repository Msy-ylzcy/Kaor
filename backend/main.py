from __future__ import annotations

import os
import logging
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import TextIO

import uvicorn

from .startup_trace import trace_startup


trace_startup("backend.main module imports started")

from .app import create_app
from .media import application_root
from .runtime import configure_runtime_directories
from .diagnostics import configure_runtime_logging


trace_startup("backend.main dependencies imported")


_fallback_standard_streams: list[TextIO] = []


def ensure_standard_streams() -> None:
    """Provide streams for PyInstaller's windowed mode before logging starts."""
    for stream_name in ("stdout", "stderr"):
        if getattr(sys, stream_name, None) is not None:
            continue

        original_name = f"__{stream_name}__"
        stream = getattr(sys, original_name, None)
        if stream is None:
            stream = open(os.devnull, "w", encoding="utf-8", errors="replace")
            _fallback_standard_streams.append(stream)
            setattr(sys, original_name, stream)
        setattr(sys, stream_name, stream)


def find_free_port(preferred: int = 8765) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def portable_data_directory() -> Path:
    override = os.environ.get("KAOR_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return application_root() / "data"


def main() -> None:
    ensure_standard_streams()
    trace_startup("main entered")
    runtime_paths = configure_runtime_directories()
    log_root = (
        runtime_paths["data"]
        if isinstance(runtime_paths, dict) and "data" in runtime_paths
        else portable_data_directory()
    )
    log_path = configure_runtime_logging(log_root)
    logger = logging.getLogger("kaor.main")
    logger.info("Kaor starting log=%s", log_path)
    trace_startup("runtime directories configured")
    port = find_free_port(int(os.environ.get("KAOR_PORT", "8765")))
    url = f"http://127.0.0.1:{port}"
    app = create_app(portable_data_directory())
    trace_startup(f"FastAPI app created on port {port}")
    if os.environ.get("KAOR_NO_BROWSER") != "1":
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    trace_startup("starting uvicorn")
    logger.info("local WebUI listening at %s", url)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
