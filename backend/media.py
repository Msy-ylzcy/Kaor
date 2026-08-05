from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class MediaToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    duration_ms: int
    width: int
    height: int
    frame_rate: float
    video_codec: str
    audio_codec: str
    has_subtitle_stream: bool
    subtitle_streams: tuple[int, ...]


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled).resolve() if bundled else application_root()


def find_binary(name: str, extra_roots: Iterable[Path] = ()) -> Path | None:
    executable = f"{name}.exe" if os.name == "nt" else name
    roots = [
        *extra_roots,
        application_root() / "bin",
        application_root() / "runtime" / "bin",
        resource_root() / "bin",
    ]
    for root in roots:
        candidate = root / executable
        if candidate.is_file():
            return candidate.resolve()
    resolved = shutil.which(name)
    return Path(resolved).resolve() if resolved else None


def require_binary(name: str) -> Path:
    binary = find_binary(name)
    if binary is None:
        raise MediaToolError(
            f"{name} was not found; place {name}.exe in the portable bin directory"
        )
    return binary


def _parse_fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0


def _run_media_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def _probe_with_ffprobe(source: Path, ffprobe: Path) -> VideoMetadata:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(source),
    ]
    completed = _run_media_command(command)
    if completed.returncode:
        raise MediaToolError(completed.stderr.strip() or "ffprobe failed")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    subtitles = tuple(
        int(item["index"])
        for item in streams
        if item.get("codec_type") == "subtitle" and "index" in item
    )
    duration_seconds = payload.get("format", {}).get("duration") or video.get(
        "duration", 0
    )
    return VideoMetadata(
        path=str(source),
        duration_ms=max(0, round(float(duration_seconds or 0) * 1000)),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        frame_rate=_parse_fraction(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        video_codec=str(video.get("codec_name") or ""),
        audio_codec=str(audio.get("codec_name") or ""),
        has_subtitle_stream=bool(subtitles),
        subtitle_streams=subtitles,
    )


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_STREAM_RE = re.compile(
    r"Stream #\d+:(\d+)(?:\[[^\]]+\])?(?:\([^)]*\))?:\s*"
    r"(Video|Audio|Subtitle):\s*([^\r\n]+)"
)
_DIMENSION_RE = re.compile(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s+fps\b")


def _probe_with_ffmpeg(source: Path, ffmpeg: Path) -> VideoMetadata:
    completed = _run_media_command(
        [str(ffmpeg), "-hide_banner", "-i", str(source)]
    )
    diagnostic = completed.stderr + "\n" + completed.stdout
    if "No such file or directory" in diagnostic or "Invalid data found" in diagnostic:
        raise MediaToolError(diagnostic.strip() or "ffmpeg probe failed")

    duration_ms = 0
    duration_match = _DURATION_RE.search(diagnostic)
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        duration_ms = round(
            (int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000
        )

    width = height = 0
    frame_rate = 0.0
    video_codec = audio_codec = ""
    subtitle_streams: list[int] = []
    for index, kind, details in _STREAM_RE.findall(diagnostic):
        codec = details.split(",", 1)[0].strip().split()[0]
        if kind == "Video" and not video_codec:
            video_codec = codec
            dimensions = _DIMENSION_RE.search(details)
            if dimensions:
                width, height = (int(value) for value in dimensions.groups())
            fps = _FPS_RE.search(details)
            if fps:
                frame_rate = float(fps.group(1))
        elif kind == "Audio" and not audio_codec:
            audio_codec = codec
        elif kind == "Subtitle":
            subtitle_streams.append(int(index))

    if not video_codec or not duration_ms or not width or not height or not frame_rate:
        import cv2

        capture = cv2.VideoCapture(str(source))
        if capture.isOpened():
            detected_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = height or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            frame_rate = frame_rate or detected_fps
            if not duration_ms and detected_fps and frame_count:
                duration_ms = round(frame_count / detected_fps * 1000)
        capture.release()

    if not video_codec and not width:
        raise MediaToolError(diagnostic.strip() or "ffmpeg did not find a video stream")
    return VideoMetadata(
        path=str(source),
        duration_ms=max(0, duration_ms),
        width=width,
        height=height,
        frame_rate=frame_rate,
        video_codec=video_codec,
        audio_codec=audio_codec,
        has_subtitle_stream=bool(subtitle_streams),
        subtitle_streams=tuple(subtitle_streams),
    )


def probe_video(
    path: Path,
    ffprobe: Path | None = None,
    ffmpeg: Path | None = None,
) -> VideoMetadata:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    selected_ffprobe = ffprobe or find_binary("ffprobe")
    if selected_ffprobe is not None:
        return _probe_with_ffprobe(source, selected_ffprobe)
    return _probe_with_ffmpeg(source, ffmpeg or require_binary("ffmpeg"))


def extract_embedded_subtitle(
    video_path: Path,
    output_path: Path,
    stream_index: int,
    ffmpeg: Path | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg or require_binary("ffmpeg")),
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path.resolve()),
        "-map",
        f"0:{stream_index}",
        str(output_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if completed.returncode:
        raise MediaToolError(completed.stderr.strip() or "subtitle extraction failed")
    return output_path.resolve()


def extract_audio_track(
    video_path: Path,
    output_path: Path,
    *,
    sample_rate: int = 44_100,
    channels: int = 2,
    ffmpeg: Path | None = None,
) -> Path:
    source = video_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg or require_binary("ffmpeg")),
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path.resolve()),
    ]
    completed = _run_media_command(command)
    if completed.returncode:
        raise MediaToolError(completed.stderr.strip() or "audio extraction failed")
    if not output_path.is_file() or output_path.stat().st_size <= 44:
        raise MediaToolError("audio extraction did not create a valid WAV file")
    return output_path.resolve()


def transcode_audio_for_asr(
    input_path: Path,
    output_path: Path,
    *,
    ffmpeg: Path | None = None,
) -> Path:
    source = input_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg or require_binary("ffmpeg")),
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path.resolve()),
    ]
    completed = _run_media_command(command)
    if completed.returncode:
        raise MediaToolError(completed.stderr.strip() or "ASR audio conversion failed")
    if not output_path.is_file() or output_path.stat().st_size <= 44:
        raise MediaToolError("ASR audio conversion did not create a valid WAV file")
    return output_path.resolve()
