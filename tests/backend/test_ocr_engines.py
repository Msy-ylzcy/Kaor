import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

import backend.ocr_engines as ocr_engines
from backend.ocr_engines import (
    OcrRuntimeCapabilities,
    OcrUnavailableError,
    PaddleOcrEngine,
    _local_model_directories,
    parse_paddle_result,
    recommended_ocr_batch_sizes,
    resolve_ocr_device,
)


def test_parse_current_paddle_result_shape():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = {
        "res": {
            "rec_texts": ["First", "Second"],
            "rec_scores": [0.98, 0.91],
            "rec_polys": [
                [[10, 20], [190, 20], [190, 40], [10, 40]],
                [[20, 60], [180, 60], [180, 82], [20, 82]],
            ],
        }
    }

    detections = parse_paddle_result([result], image)

    assert [item.text for item in detections] == ["First", "Second"]
    assert detections[0].box.x == 0.05
    assert detections[1].confidence == 0.91


def test_parse_legacy_paddle_result_shape():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = [
        [
            [[10, 20], [190, 20], [190, 40], [10, 40]],
            ("Legacy", 0.88),
        ]
    ]

    detections = parse_paddle_result(result, image)

    assert len(detections) == 1
    assert detections[0].text == "Legacy"


def capabilities(*, cuda: bool = False, count: int = 0) -> OcrRuntimeCapabilities:
    return OcrRuntimeCapabilities(
        paddle_available=True,
        paddle_version="3.3.1",
        paddleocr_available=True,
        paddleocr_version="3.7.0",
        cuda_compiled=cuda,
        cuda_device_count=count,
        cuda_device_names=tuple(f"GPU {index}" for index in range(count)),
    )


def test_device_resolution_for_auto_cpu_and_cuda():
    assert resolve_ocr_device("Auto", capabilities()) == "cpu"
    assert resolve_ocr_device("CPU", capabilities()) == "cpu"
    assert resolve_ocr_device("auto", capabilities(cuda=True, count=2)) == "gpu:0"
    assert resolve_ocr_device("CUDA:1", capabilities(cuda=True, count=2)) == "gpu:1"

    with pytest.raises(OcrUnavailableError, match="CPU-only"):
        resolve_ocr_device("cuda", capabilities())
    with pytest.raises(OcrUnavailableError, match="unavailable"):
        resolve_ocr_device("cuda:2", capabilities(cuda=True, count=2))
    with pytest.raises(OcrUnavailableError, match="invalid"):
        resolve_ocr_device("cuda:bad", capabilities(cuda=True, count=2))


def test_batch_sizes_scale_with_gpu_memory():
    assert recommended_ocr_batch_sizes("cpu") == (2, 4)
    assert recommended_ocr_batch_sizes("gpu:0", 4 * 1024**3) == (6, 24)
    assert recommended_ocr_batch_sizes("gpu:0", 8 * 1024**3) == (24, 48)
    assert recommended_ocr_batch_sizes("gpu:0", 12 * 1024**3) == (32, 64)
    assert recommended_ocr_batch_sizes("gpu:0", 16 * 1024**3) == (40, 80)
    assert recommended_ocr_batch_sizes("gpu:0", 24 * 1024**3) == (48, 96)


def test_windows_cpu_engine_disables_mkldnn(monkeypatch):
    captured: dict[str, object] = {}

    class FakePaddleOcr:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOcr))
    monkeypatch.setattr(
        "backend.ocr_engines.detect_ocr_capabilities", lambda: capabilities()
    )

    engine = PaddleOcrEngine(device="auto")

    assert engine.device == "cpu"
    assert captured["device"] == "cpu"
    assert captured["enable_mkldnn"] is False


def test_paddle_import_stubs_modelscope_torch_probe(monkeypatch):
    paddleocr = types.ModuleType("paddleocr")

    class FakePaddleOcr:
        pass

    paddleocr.PaddleOCR = FakePaddleOcr
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "modelscope.utils.torch_utils", raising=False)

    imported = ocr_engines._import_paddle_ocr_class()

    assert imported is FakePaddleOcr
    assert "modelscope.utils.torch_utils" not in sys.modules


def test_engine_preserves_paddle_import_failure(monkeypatch):
    broken = OcrRuntimeCapabilities(
        paddle_available=False,
        paddle_version="3.3.1",
        paddleocr_available=True,
        paddleocr_version="3.7.0",
        cuda_compiled=False,
        cuda_device_count=0,
        cuda_device_names=(),
        error="PermissionError: denied: C:\\Users\\User\\.cache\\paddle",
    )
    monkeypatch.setattr("backend.ocr_engines.detect_ocr_capabilities", lambda: broken)

    with pytest.raises(OcrUnavailableError, match=r"PermissionError.*\.cache\\paddle"):
        PaddleOcrEngine()


def test_local_model_discovery_uses_bundled_official_models(tmp_path):
    official = tmp_path / "paddlex" / "official_models"
    detection = official / "PP-OCRv6_medium_det"
    recognition = official / "PP-OCRv6_medium_rec"
    detection.mkdir(parents=True)
    recognition.mkdir(parents=True)

    assert _local_model_directories(None, tmp_path) == (detection, recognition)
