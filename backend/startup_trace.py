from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def trace_startup(message: str) -> None:
    if os.environ.get("KAOR_TRACE_STARTUP") != "1":
        return
    root = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[1]
    )
    try:
        with (root / "startup-trace.log").open("a", encoding="utf-8") as handle:
            timestamp = datetime.now(timezone.utc).isoformat()
            handle.write(f"{timestamp} {message}\n")
    except OSError:
        pass
