from __future__ import annotations

import os
import subprocess
from pathlib import Path
from threading import Event
from typing import Callable

from .media import MediaToolError, application_root, require_binary, resource_root


ProgressCallback = Callable[[float, str], None]


def _default_fonts_directory() -> Path | None:
    override = os.environ.get("KAOR_FONTS_DIR")
    candidates = [
        Path(override).expanduser() if override else None,
        application_root() / "fonts" / "NotoSansSC",
        resource_root() / "fonts" / "NotoSansSC",
    ]
    return next(
        (
            candidate.resolve()
            for candidate in candidates
            if candidate is not None and candidate.is_dir()
        ),
        None,
    )


def _escape_filter_path(path: Path) -> str:
    return (
        path.resolve()
        .as_posix()
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
    )


def render_video(
    input_path: Path,
    subtitle_path: Path,
    output_path: Path,
    *,
    duration_ms: int,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
    start_ms: int | None = None,
    clip_duration_ms: int | None = None,
    video_encoder: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    fonts_directory: Path | None = None,
) -> Path:
    ffmpeg = require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ffmpeg), "-y", "-v", "error"]
    if start_ms is not None:
        command.extend(["-ss", f"{start_ms / 1000:.3f}"])
    command.extend(["-i", str(input_path.resolve())])
    if clip_duration_ms is not None:
        command.extend(["-t", f"{clip_duration_ms / 1000:.3f}"])
    subtitle_filter = f"ass={subtitle_path.name}"
    selected_fonts = fonts_directory or _default_fonts_directory()
    if selected_fonts is not None:
        subtitle_filter += f":fontsdir='{_escape_filter_path(selected_fonts)}'"
    command.extend(
        [
            "-vf",
            subtitle_filter,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            video_encoder,
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-c:a",
            "copy",
            "-progress",
            "pipe:1",
            "-nostats",
            str(output_path.resolve()),
        ]
    )
    process = subprocess.Popen(
        command,
        cwd=subtitle_path.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    effective_duration = clip_duration_ms or duration_ms
    assert process.stdout is not None
    while True:
        if cancel_event and cancel_event.is_set():
            process.terminate()
            process.wait(timeout=10)
            raise InterruptedError("render cancelled")
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        key, separator, value = line.strip().partition("=")
        if not separator or not effective_duration:
            continue
        if key in {"out_time_us", "out_time_ms"}:
            try:
                elapsed_ms = int(value) / 1000
            except ValueError:
                continue
            if progress:
                progress(
                    min(0.99, elapsed_ms / effective_duration),
                    f"Rendering {elapsed_ms / 1000:.1f}s",
                )
    stderr = process.stderr.read() if process.stderr else ""
    if process.returncode:
        raise MediaToolError(stderr.strip() or "ffmpeg render failed")
    if progress:
        progress(1.0, "Render complete")
    return output_path.resolve()
