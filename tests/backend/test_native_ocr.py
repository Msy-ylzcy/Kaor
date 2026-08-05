from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend import native_ocr


def _legacy_signature(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    target_width = min(320, max(64, width))
    target_height = max(32, round(height * target_width / max(width, 1)))
    resized = cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


@pytest.mark.parametrize("shape", [(19, 27, 3), (80, 160, 3), (720, 1280, 3)])
def test_frame_signature_is_bit_exact_with_previous_algorithm(shape):
    image = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)

    actual = native_ocr.frame_signature(image)

    np.testing.assert_array_equal(actual, _legacy_signature(image))


@pytest.mark.parametrize("threshold", [1, 18, 128, 255])
def test_opencv_change_ratio_matches_numpy_algorithm(threshold):
    random = np.random.default_rng(9)
    previous = random.integers(0, 256, (67, 113), dtype=np.uint8)
    current = random.integers(0, 256, (67, 113), dtype=np.uint8)
    expected_difference = cv2.absdiff(previous, current)
    expected = float(np.count_nonzero(expected_difference >= threshold)) / max(
        expected_difference.size, 1
    )

    actual = native_ocr._opencv_changed_pixel_ratio(previous, current, threshold)

    assert actual == expected


def test_native_change_ratio_is_used_when_available(monkeypatch):
    calls = []

    class FakeNative:
        @staticmethod
        def changed_pixel_ratio(previous, current, threshold):
            calls.append((previous.shape, current.shape, threshold))
            return 0.375

    monkeypatch.setattr(native_ocr, "_NATIVE", FakeNative())
    monkeypatch.setattr(native_ocr, "_NATIVE_DISABLED", False)
    image = np.zeros((8, 12), dtype=np.uint8)

    assert native_ocr.changed_pixel_ratio(image, image.copy(), 18) == 0.375
    assert calls == [((8, 12), (8, 12), 18)]
    assert native_ocr.implementation_name() == "kaor_native+opencv"


def test_native_runtime_failure_falls_back_to_equivalent_opencv(monkeypatch):
    class BrokenNative:
        @staticmethod
        def changed_pixel_ratio(previous, current, threshold):
            raise BufferError("incompatible extension")

    monkeypatch.setattr(native_ocr, "_NATIVE", BrokenNative())
    monkeypatch.setattr(native_ocr, "_NATIVE_DISABLED", False)
    previous = np.zeros((4, 5), dtype=np.uint8)
    current = previous.copy()
    current[0, :2] = 30

    assert native_ocr.changed_pixel_ratio(previous, current, 18) == 0.1
    assert native_ocr.implementation_name() == "opencv"


def test_change_ratio_validates_signature_contract():
    image = np.zeros((4, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="matching shapes"):
        native_ocr.changed_pixel_ratio(image, np.zeros((5, 4), dtype=np.uint8), 18)
    with pytest.raises(ValueError, match="two-dimensional"):
        native_ocr.changed_pixel_ratio(image[..., None], image[..., None], 18)
