from __future__ import annotations

import json
import math
import os
import sys
import types
from dataclasses import dataclass
from difflib import SequenceMatcher
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

from .recognition import Box, TextDetection, normalize_text
from .runtime import configure_runtime_directories


class OcrUnavailableError(RuntimeError):
    pass


def _import_paddle_ocr_class() -> Any:
    """Import PaddleOCR without letting ModelScope probe the Torch runtime.

    PaddleX imports ModelScope even when all OCR models are local. ModelScope's
    logger imports Torch only to detect distributed training, which mixes the
    Paddle and Torch cuDNN DLL sets in one Windows process. Audio inference is
    already isolated in a child process, so the OCR process disables that probe.
    """

    module_name = "modelscope.utils.torch_utils"
    inserted_stub = module_name not in sys.modules and "torch" not in sys.modules
    if inserted_stub:
        stub = types.ModuleType(module_name)
        stub.is_dist = lambda: False  # type: ignore[attr-defined]
        stub.is_master = lambda: True  # type: ignore[attr-defined]
        sys.modules[module_name] = stub
    try:
        from paddleocr import PaddleOCR
    finally:
        if inserted_stub:
            sys.modules.pop(module_name, None)
    return PaddleOCR


@dataclass(frozen=True)
class OcrRuntimeCapabilities:
    paddle_available: bool
    paddle_version: str | None
    paddleocr_available: bool
    paddleocr_version: str | None
    cuda_compiled: bool
    cuda_device_count: int
    cuda_device_names: tuple[str, ...]
    error: str | None = None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def detect_ocr_capabilities() -> OcrRuntimeCapabilities:
    """Inspect the local Paddle runtime without requiring a working OCR model."""
    configure_runtime_directories()
    paddle_version = _package_version("paddlepaddle") or _package_version(
        "paddlepaddle-gpu"
    )
    paddleocr_version = _package_version("paddleocr")
    if paddle_version is None:
        return OcrRuntimeCapabilities(
            paddle_available=False,
            paddle_version=None,
            paddleocr_available=paddleocr_version is not None,
            paddleocr_version=paddleocr_version,
            cuda_compiled=False,
            cuda_device_count=0,
            cuda_device_names=(),
            error="PaddlePaddle is not installed",
        )
    try:
        import paddle

        cuda_compiled = bool(paddle.device.is_compiled_with_cuda())
        device_count = int(paddle.device.cuda.device_count()) if cuda_compiled else 0
        names: list[str] = []
        for index in range(device_count):
            try:
                names.append(str(paddle.device.cuda.get_device_name(index)))
            except Exception:
                names.append(f"CUDA device {index}")
        return OcrRuntimeCapabilities(
            paddle_available=True,
            paddle_version=paddle_version,
            paddleocr_available=paddleocr_version is not None,
            paddleocr_version=paddleocr_version,
            cuda_compiled=cuda_compiled,
            cuda_device_count=device_count,
            cuda_device_names=tuple(names),
        )
    except Exception as exc:
        return OcrRuntimeCapabilities(
            paddle_available=False,
            paddle_version=paddle_version,
            paddleocr_available=paddleocr_version is not None,
            paddleocr_version=paddleocr_version,
            cuda_compiled=False,
            cuda_device_count=0,
            cuda_device_names=(),
            error=f"{type(exc).__name__}: {exc}",
        )


def resolve_ocr_device(
    requested: str,
    capabilities: OcrRuntimeCapabilities | None = None,
) -> str:
    """Normalize UI device names to PaddleOCR's cpu/gpu:N syntax."""
    value = requested.strip().lower() or "auto"
    caps = capabilities or detect_ocr_capabilities()
    if value == "auto":
        return "gpu:0" if caps.cuda_compiled and caps.cuda_device_count > 0 else "cpu"
    if value == "cpu":
        return "cpu"
    if value in {"cuda", "gpu"}:
        index = 0
    elif value.startswith("cuda:") or value.startswith("gpu:"):
        _, raw_index = value.split(":", 1)
        if not raw_index.isdigit():
            raise OcrUnavailableError(f"invalid OCR device: {requested}")
        index = int(raw_index)
    else:
        raise OcrUnavailableError(f"unsupported OCR device: {requested}")
    if not caps.cuda_compiled:
        raise OcrUnavailableError(
            "CUDA OCR requested, but the installed PaddlePaddle runtime is CPU-only"
        )
    if index >= caps.cuda_device_count:
        raise OcrUnavailableError(
            f"CUDA device {index} is unavailable; found {caps.cuda_device_count} device(s)"
        )
    return f"gpu:{index}"


def recommended_ocr_batch_sizes(
    device: str, total_memory_bytes: int = 0
) -> tuple[int, int]:
    """Return outer-frame and text-recognition batch sizes for the device."""
    if not device.startswith("gpu"):
        return 2, 4
    gibibytes = total_memory_bytes / (1024**3)
    if gibibytes >= 20:
        return 48, 96
    if gibibytes >= 14:
        return 40, 80
    if gibibytes >= 10:
        return 32, 64
    if gibibytes >= 8:
        return 24, 48
    if gibibytes >= 6:
        return 12, 32
    return 6, 24


def _local_model_directories(
    model_root: Path | None, bundled_models: Path
) -> tuple[Path | None, Path | None]:
    roots = [model_root] if model_root is not None else [bundled_models / "paddlex"]
    for root in roots:
        if root is None:
            continue
        candidates = (
            (root / "detection", root / "recognition"),
            (
                root / "official_models" / "PP-OCRv6_medium_det",
                root / "official_models" / "PP-OCRv6_medium_rec",
            ),
        )
        for detection, recognition in candidates:
            if detection.is_dir() and recognition.is_dir():
                return detection, recognition
    return None, None


class OcrEngine(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[TextDetection]: ...

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[TextDetection]]: ...


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    payload = getattr(value, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _box_from_polygon(polygon: Sequence[Sequence[float]], width: int, height: int) -> Box:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return Box(
        left / max(width, 1),
        top / max(height, 1),
        max(1.0, right - left) / max(width, 1),
        max(1.0, bottom - top) / max(height, 1),
    )


def estimate_text_color(image: np.ndarray, box: Box) -> tuple[int, int, int]:
    height, width = image.shape[:2]
    left = max(0, min(width - 1, round(box.x * width)))
    top = max(0, min(height - 1, round(box.y * height)))
    right = max(left + 1, min(width, round(box.right * width)))
    bottom = max(top + 1, min(height, round(box.bottom * height)))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return (255, 255, 255)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate(
        [lab[0], lab[-1], lab[:, 0], lab[:, -1]], axis=0
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    threshold = max(12.0, float(np.percentile(distance, 72)))
    foreground = rgb[distance >= threshold]
    if len(foreground) < 8:
        foreground = rgb.reshape(-1, 3)
    hsv = cv2.cvtColor(foreground.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    score = hsv[:, 1].astype(np.float32) * 0.65 + hsv[:, 2].astype(np.float32) * 0.35
    selected = foreground[score >= np.percentile(score, 55)]
    color = np.median(selected if len(selected) else foreground, axis=0)
    return tuple(int(np.clip(channel, 0, 255)) for channel in color)  # type: ignore[return-value]


def parse_paddle_result(raw: Any, image: np.ndarray) -> list[TextDetection]:
    height, width = image.shape[:2]
    detections: list[TextDetection] = []
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    for value in values:
        mapping = _as_mapping(value)
        if mapping is not None:
            data = mapping.get("res", mapping)
            texts = data.get("rec_texts") or data.get("texts") or []
            scores = data.get("rec_scores") or data.get("scores") or []
            polygons = (
                data.get("rec_polys")
                or data.get("dt_polys")
                or data.get("polys")
                or []
            )
            for text, score, polygon in zip(texts, scores, polygons):
                if not str(text).strip():
                    continue
                box = _box_from_polygon(polygon, width, height)
                detections.append(
                    TextDetection(
                        str(text), box, float(score), estimate_text_color(image, box)
                    )
                )
            continue
        if not isinstance(value, (list, tuple)):
            continue
        if (
            len(value) >= 2
            and isinstance(value[0], (list, tuple))
            and len(value[0]) >= 3
            and all(
                isinstance(point, (list, tuple)) and len(point) >= 2
                for point in value[0]
            )
            and isinstance(value[1], (list, tuple))
            and len(value[1]) >= 2
            and isinstance(value[1][0], str)
        ):
            polygon, text_score = value[0], value[1]
            text, score = text_score[0], text_score[1]
            if str(text).strip():
                box = _box_from_polygon(polygon, width, height)
                detections.append(
                    TextDetection(
                        str(text), box, float(score), estimate_text_color(image, box)
                    )
                )
            continue
        for row in value:
            if not isinstance(row, (list, tuple)):
                continue
            if len(row) == 1 and isinstance(row[0], (list, tuple)):
                nested = parse_paddle_result(row, image)
                detections.extend(nested)
                continue
            if len(row) < 2 or not isinstance(row[1], (list, tuple)):
                continue
            polygon, text_score = row[0], row[1]
            if len(text_score) < 2:
                continue
            text, score = text_score[0], text_score[1]
            if not str(text).strip():
                continue
            box = _box_from_polygon(polygon, width, height)
            detections.append(
                TextDetection(str(text), box, float(score), estimate_text_color(image, box))
            )
    return detections


class PaddleOcrEngine:
    name = "PaddleOCR"

    def __init__(
        self,
        *,
        language: str = "en",
        device: str = "auto",
        model_root: Path | None = None,
    ) -> None:
        runtime_paths = configure_runtime_directories()
        # PaddleOCR 3.7 enables oneDNN by default. On Windows CPU, Paddle 3.3's
        # PIR executor can fail while lowering oneDNN operators, so keep the
        # stable plain Paddle inference path. CUDA remains unaffected.
        os.environ["FLAGS_use_mkldnn"] = "0"
        capabilities = detect_ocr_capabilities()
        if (
            capabilities.paddle_version is not None
            and not capabilities.paddle_available
            and capabilities.error
        ):
            raise OcrUnavailableError(
                f"PaddlePaddle import failed: {capabilities.error}"
            )
        resolved_device = resolve_ocr_device(device, capabilities)
        total_memory_bytes = 0
        if resolved_device.startswith("gpu"):
            try:
                import paddle

                device_index = int(resolved_device.split(":", 1)[1])
                properties = paddle.device.cuda.get_device_properties(device_index)
                total_memory_bytes = int(getattr(properties, "total_memory", 0) or 0)
            except Exception:
                total_memory_bytes = 0
        frame_batch_size, recognition_batch_size = recommended_ocr_batch_sizes(
            resolved_device, total_memory_bytes
        )
        try:
            PaddleOCR = _import_paddle_ocr_class()
        except Exception as exc:
            raise OcrUnavailableError(
                f"PaddleOCR import failed ({type(exc).__name__}): {exc}"
            ) from exc
        kwargs: dict[str, Any] = {
            "lang": language,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": resolved_device,
            "enable_mkldnn": False,
            "text_recognition_batch_size": (
                recognition_batch_size
            ),
        }
        detection_model, recognition_model = _local_model_directories(
            model_root, runtime_paths["models"]
        )
        if detection_model and recognition_model:
            kwargs["text_detection_model_dir"] = str(detection_model)
            kwargs["text_recognition_model_dir"] = str(recognition_model)
        try:
            self._engine = PaddleOCR(**kwargs)
        except TypeError:
            legacy = {
                "lang": language,
                "use_angle_cls": False,
                "show_log": False,
                "use_gpu": resolved_device.startswith("gpu"),
                "enable_mkldnn": False,
            }
            self._engine = PaddleOCR(**legacy)
        self.device = resolved_device
        self.recommended_batch_size = frame_batch_size
        self.text_recognition_batch_size = recognition_batch_size

    def detect(self, image: np.ndarray) -> list[TextDetection]:
        return self.detect_batch([image])[0]

    def detect_batch(
        self, images: Sequence[np.ndarray]
    ) -> list[list[TextDetection]]:
        if not images:
            return []
        if hasattr(self._engine, "predict"):
            raw = list(self._engine.predict(list(images)))
            if len(raw) != len(images):
                raise RuntimeError(
                    f"PaddleOCR returned {len(raw)} results for {len(images)} images"
                )
            return [
                parse_paddle_result(result, image)
                for result, image in zip(raw, images)
            ]
        if len(images) > 1:
            return [self.detect(image) for image in images]
        image = images[0]
        if hasattr(self._engine, "ocr"):
            raw = self._engine.ocr(image, cls=False)
        else:
            raise OcrUnavailableError("PaddleOCR runtime has no prediction method")
        return [parse_paddle_result(raw, image)]


def enhanced_variants(image: np.ndarray) -> list[np.ndarray]:
    upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    blurred = cv2.GaussianBlur(upscaled, (0, 0), 1.2)
    sharpened = cv2.addWeighted(upscaled, 1.65, blurred, -0.65, 0)
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)
    luminance = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(luminance)
    contrast = cv2.cvtColor(cv2.merge((luminance, a, b)), cv2.COLOR_LAB2BGR)
    return [image, sharpened, contrast]


def _candidate_similarity(left: TextDetection, right: TextDetection) -> float:
    text = SequenceMatcher(
        None, normalize_text(left.text), normalize_text(right.text)
    ).ratio()
    return 0.6 * left.box.intersection_over_union(right.box) + 0.4 * text


class ConsensusOcrEngine:
    name = "Multi-frame OCR consensus"

    def __init__(self, primary: OcrEngine, verifier: OcrEngine | None = None) -> None:
        self.primary = primary
        self.verifier = verifier
        self.device = getattr(primary, "device", "cpu")
        self.recommended_batch_size = getattr(primary, "recommended_batch_size", 0)

    def detect(self, image: np.ndarray) -> list[TextDetection]:
        return self.detect_batch([image])[0]

    def detect_batch(
        self, images: Sequence[np.ndarray]
    ) -> list[list[TextDetection]]:
        if not images:
            return []
        variants_by_image = [enhanced_variants(image) for image in images]
        flattened_variants = [
            variant for variants in variants_by_image for variant in variants
        ]
        detector = getattr(self.primary, "detect_batch", None)
        if callable(detector):
            flattened_results = detector(flattened_variants)
        else:
            flattened_results = [
                self.primary.detect(variant) for variant in flattened_variants
            ]
        if len(flattened_results) != len(flattened_variants):
            raise RuntimeError("OCR consensus batch result count mismatch")

        results: list[list[TextDetection]] = []
        offset = 0
        for image, variants in zip(images, variants_by_image):
            batches = flattened_results[offset : offset + len(variants)]
            offset += len(variants)
            if self.verifier is not None:
                batches = [*batches, self.verifier.detect(image)]
            results.append(self._merge_batches(batches, image))
        return results

    @staticmethod
    def _merge_batches(
        batches: Sequence[Sequence[TextDetection]], image: np.ndarray
    ) -> list[TextDetection]:
        flattened = [item for batch in batches for item in batch]
        clusters: list[list[TextDetection]] = []
        for detection in sorted(flattened, key=lambda item: item.confidence, reverse=True):
            cluster = next(
                (
                    group
                    for group in clusters
                    if max(_candidate_similarity(detection, member) for member in group)
                    >= 0.52
                ),
                None,
            )
            if cluster is None:
                clusters.append([detection])
            else:
                cluster.append(detection)
        results: list[TextDetection] = []
        source_count = len(batches)
        for cluster in clusters:
            winner = max(
                cluster,
                key=lambda item: (
                    sum(
                        other.confidence
                        * SequenceMatcher(
                            None, normalize_text(item.text), normalize_text(other.text)
                        ).ratio()
                        for other in cluster
                    ),
                    len(item.text),
                ),
            )
            agreement = min(1.0, len(cluster) / max(source_count, 1))
            confidence = min(1.0, winner.confidence * (0.72 + 0.28 * agreement))
            results.append(
                TextDetection(
                    winner.text,
                    winner.box,
                    confidence,
                    estimate_text_color(image, winner.box),
                )
            )
        return sorted(results, key=lambda item: (item.box.y, item.box.x))


@dataclass(frozen=True)
class StaticOcrEngine:
    detections: list[TextDetection]
    name: str = "Static OCR"

    def detect(self, image: np.ndarray) -> list[TextDetection]:
        return list(self.detections)

    def detect_batch(
        self, images: Sequence[np.ndarray]
    ) -> list[list[TextDetection]]:
        return [list(self.detections) for _ in images]
