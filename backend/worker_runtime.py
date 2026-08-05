from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from .media import application_root


def audio_worker_executable() -> Path | None:
    """Return the frozen audio worker executable, or None in source mode."""
    if not getattr(sys, "frozen", False):
        return None
    configured = os.environ.get("KAOR_AUDIO_WORKER", "").strip()
    candidate = (
        Path(configured).expanduser()
        if configured
        else application_root() / "KaorAudioWorker.exe"
    )
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"packaged audio worker was not found: {candidate}; reinstall this Kaor release"
        )
    return candidate


def audio_worker_command(arguments: Sequence[str]) -> list[str]:
    worker = audio_worker_executable()
    if worker is not None:
        return [str(worker), *arguments]
    return [sys.executable, "-m", "backend.audio_worker", *arguments]
