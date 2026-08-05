from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import NormalizedRegion
from .ocr_engines import ConsensusOcrEngine, PaddleOcrEngine
from .recognition import TextDetection


class FrameOcrError(RuntimeError):
    """Raised when a single-frame OCR request cannot be completed."""


@dataclass(frozen=True)
class FrameOcrResult:
    timestamp_ms: int
    frame_index: int
    text: str
    confidence: float
    detections: list[dict[str, Any]]


def read_frame(path: Path, timestamp_ms: int, fps: float) -> tuple[np.ndarray, int, int]:
    """Read the frame nearest to ``timestamp_ms`` without decoding the whole video."""
    capture = cv2.VideoCapture(str(path.resolve()))
    if not capture.isOpened():
        raise FrameOcrError(f"could not open video: {path}")
    try:
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or float(fps or 30.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_index = max(0, round(max(0, timestamp_ms) / 1000 * actual_fps))
        if frame_count > 0:
            frame_index = min(frame_index, frame_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise FrameOcrError(f"could not decode video frame {frame_index}")
        actual_timestamp_ms = round(frame_index / actual_fps * 1000)
        return frame, frame_index, actual_timestamp_ms
    finally:
        capture.release()


def crop_region(frame: np.ndarray, region: NormalizedRegion) -> np.ndarray:
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, round(region.x * width)))
    top = max(0, min(height - 1, round(region.y * height)))
    right = max(left + 1, min(width, round((region.x + region.width) * width)))
    bottom = max(top + 1, min(height, round((region.y + region.height) * height)))
    return frame[top:bottom, left:right].copy()


def _line_text(detections: list[TextDetection]) -> str:
    """Keep visual line breaks while removing OCR whitespace noise."""
    if not detections:
        return ""
    ordered = sorted(detections, key=lambda item: (item.box.y, item.box.x))
    lines: list[list[TextDetection]] = []
    for detection in ordered:
        center = detection.box.y + detection.box.height / 2
        line = next(
            (
                current
                for current in lines
                if abs(
                    center
                    - (current[0].box.y + current[0].box.height / 2)
                )
                <= max(detection.box.height, current[0].box.height) * 0.7
            ),
            None,
        )
        if line is None:
            lines.append([detection])
        else:
            line.append(detection)
    return "\n".join(
        " ".join(item.text.strip() for item in sorted(line, key=lambda value: value.box.x))
        for line in lines
    ).strip()


def recognize_frame(
    path: Path,
    *,
    timestamp_ms: int,
    fps: float,
    region: NormalizedRegion,
    language: str,
    device: str = "auto",
    high_accuracy: bool = True,
    engine: Any | None = None,
) -> FrameOcrResult:
    frame, frame_index, actual_timestamp_ms = read_frame(path, timestamp_ms, fps)
    crop = crop_region(frame, region)
    selected = engine or PaddleOcrEngine(language=language, device=device)
    selected = ConsensusOcrEngine(selected) if high_accuracy and not isinstance(selected, ConsensusOcrEngine) else selected
    detections = selected.detect(crop)
    text = _line_text(detections)
    confidence = (
        round(sum(max(0.0, min(1.0, item.confidence)) for item in detections) / len(detections), 4)
        if detections
        else 0.0
    )
    payload = [
        {
            "text": item.text,
            "confidence": round(float(item.confidence), 4),
            "bbox": {
                "x": round(float(item.box.x), 6),
                "y": round(float(item.box.y), 6),
                "width": round(float(item.box.width), 6),
                "height": round(float(item.box.height), 6),
            },
            "color": "#%02X%02X%02X" % item.color,
        }
        for item in detections
    ]
    return FrameOcrResult(
        timestamp_ms=actual_timestamp_ms,
        frame_index=frame_index,
        text=text,
        confidence=confidence,
        detections=payload,
    )
