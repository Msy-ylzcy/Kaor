from __future__ import annotations

from importlib import import_module

import cv2
import numpy as np


def _load_native_module():
    for name in ("backend._kaor_ocr_native", "_kaor_ocr_native"):
        try:
            return import_module(name)
        except (ImportError, OSError):
            continue
    return None


_NATIVE = _load_native_module()
_NATIVE_DISABLED = False


def implementation_name() -> str:
    """Return the active frame-change implementation for runtime diagnostics."""

    return (
        "kaor_native+opencv"
        if _NATIVE is not None and not _NATIVE_DISABLED
        else "opencv"
    )


def frame_signature(image: np.ndarray) -> np.ndarray:
    """Build the same low-resolution signature used by the OCR change gate.

    Resize, color conversion, and blur are OpenCV operations, so this fallback
    already executes in OpenCV's compiled C++ implementation.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frame signature expects a BGR image")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("frame signature expects a non-empty image")
    target_width = min(320, max(64, width))
    target_height = max(32, round(height * target_width / max(width, 1)))
    resized = cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _opencv_changed_pixel_ratio(
    previous: np.ndarray,
    current: np.ndarray,
    pixel_threshold: int,
) -> float:
    difference = cv2.absdiff(previous, current)
    # uint8 pixels satisfy value >= threshold exactly when value > threshold - 1.
    _, changed = cv2.threshold(
        difference,
        float(pixel_threshold - 1),
        255,
        cv2.THRESH_BINARY,
    )
    return float(cv2.countNonZero(changed)) / max(changed.size, 1)


def changed_pixel_ratio(
    previous: np.ndarray,
    current: np.ndarray,
    pixel_threshold: int,
) -> float:
    """Return the ratio of pixels whose absolute delta meets the threshold.

    The optional extension fuses absdiff, comparison, and countNonZero into one
    native call. Any load or runtime incompatibility falls back to the same
    OpenCV operations instead of changing OCR behavior.
    """

    if previous.dtype != np.uint8 or current.dtype != np.uint8:
        raise ValueError("frame signatures must use uint8 pixels")
    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError("frame signatures must be two-dimensional")
    if previous.shape != current.shape:
        raise ValueError("frame signatures must have matching shapes")
    if pixel_threshold <= 0:
        return 1.0
    if pixel_threshold > 255:
        return 0.0

    global _NATIVE_DISABLED
    native = _NATIVE
    if native is not None and not _NATIVE_DISABLED:
        try:
            result = float(
                native.changed_pixel_ratio(previous, current, pixel_threshold)
            )
            if 0.0 <= result <= 1.0:
                return result
        except (BufferError, RuntimeError, TypeError, ValueError):
            pass
        _NATIVE_DISABLED = True
    return _opencv_changed_pixel_ratio(previous, current, pixel_threshold)
