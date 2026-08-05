from __future__ import annotations

import gc
import math
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from .models import Cue
from .native_ocr import (
    changed_pixel_ratio,
    frame_signature,
    implementation_name as frame_change_implementation,
)
from .ocr_engines import OcrEngine
from .recognition import (
    FrameDetections,
    RecognizedCue,
    TemporalTextTracker,
    TextDetection,
    normalize_text,
)


ProgressCallback = Callable[..., None]


@dataclass(frozen=True)
class Region:
    x: float
    y: float
    width: float
    height: float

    def validate(self) -> "Region":
        values = (self.x, self.y, self.width, self.height)
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("ROI values must be normalized between 0 and 1")
        if self.width <= 0 or self.height <= 0 or self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("ROI must fit inside the video frame")
        return self


@dataclass(frozen=True)
class RecognitionOptions:
    source_roi: Region
    sample_fps: float = 4.0
    minimum_observations: int = 2
    confidence_threshold: float = 0.82
    filter_noise: bool = True
    batch_size: int = 0
    queue_size: int = 16
    change_detection: bool = True
    change_pixel_threshold: int = 18
    change_ratio_threshold: float = 0.006
    force_ocr_interval_ms: int = 5000
    transition_lookahead_ms: int = 500
    transition_confidence_threshold: float = 0.92
    start_ms: int = 0


@dataclass(frozen=True)
class _SampledFrame:
    timestamp_ms: int
    image: np.ndarray


def _same_subtitle_line(left: TextDetection, right: TextDetection) -> bool:
    vertical = left.box.vertical_affinity(right.box)
    horizontal_overlap = max(
        0.0,
        min(left.box.right, right.box.right) - max(left.box.x, right.box.x),
    ) / max(min(left.box.width, right.box.width), 1e-6)
    return vertical >= 0.72 and (
        left.box.intersection_over_union(right.box) >= 0.05
        or horizontal_overlap >= 0.35
    )


def _transition_text_compatible(left: str, right: str) -> bool:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return True
    return SequenceMatcher(None, normalized_left, normalized_right).ratio() >= 0.45


def _replacement_key(detection: TextDetection) -> tuple[str, int]:
    center_y = detection.box.y + detection.box.height / 2
    return normalize_text(detection.text), round(center_y * 40)


class _TemporalDetectionStabilizer:
    """Delay low-confidence transition frames until a clearer nearby frame exists."""

    def __init__(self, lookahead_ms: int, confidence_threshold: float) -> None:
        self.lookahead_ms = max(0, lookahead_ms)
        self.confidence_threshold = confidence_threshold
        self.pending: list[FrameDetections] = []
        self.repaired_detections = 0

    def push(self, frame: FrameDetections) -> list[FrameDetections]:
        self.pending.append(frame)
        return self._drain(frame.timestamp_ms - self.lookahead_ms)

    def finish(self) -> list[FrameDetections]:
        return self._drain(math.inf)

    def _drain(self, cutoff_ms: float) -> list[FrameDetections]:
        ready: list[FrameDetections] = []
        while self.pending and self.pending[0].timestamp_ms <= cutoff_ms:
            current = self.pending.pop(0)
            future = [
                frame
                for frame in self.pending
                if frame.timestamp_ms - current.timestamp_ms <= self.lookahead_ms
            ]
            stabilized, repaired = self._stabilize(current, future)
            self.repaired_detections += repaired
            ready.append(stabilized)
        return ready

    def _stabilize(
        self,
        current: FrameDetections,
        future: list[FrameDetections],
    ) -> tuple[FrameDetections, int]:
        repaired = 0
        detections: list[TextDetection] = []
        used_replacements: set[tuple[str, int]] = set()
        for detection in current.detections:
            if detection.confidence >= self.confidence_threshold:
                detections.append(detection)
                continue
            candidates = [
                candidate
                for frame in future
                for candidate in frame.detections
                if candidate.confidence
                >= max(self.confidence_threshold, detection.confidence + 0.08)
                and _same_subtitle_line(detection, candidate)
                and _transition_text_compatible(detection.text, candidate.text)
            ]
            if candidates:
                replacement = max(
                    candidates,
                    key=lambda candidate: (
                        candidate.confidence,
                        len(candidate.text.strip()),
                    ),
                )
                key = _replacement_key(replacement)
                if key not in used_replacements and not any(
                    _replacement_key(existing) == key for existing in detections
                ):
                    detections.append(replacement)
                    used_replacements.add(key)
                repaired += 1
            else:
                detections.append(detection)
        return FrameDetections(current.timestamp_ms, tuple(detections)), repaired


class _FrameChangeGate:
    def __init__(self, options: RecognitionOptions) -> None:
        self.options = options
        self.previous: np.ndarray | None = None
        self.last_ocr_ms = -options.force_ocr_interval_ms

    @staticmethod
    def _signature(image: np.ndarray) -> np.ndarray:
        return frame_signature(image)

    def requires_ocr(self, image: np.ndarray, timestamp_ms: int) -> bool:
        signature = self._signature(image)
        previous = self.previous
        self.previous = signature
        forced = timestamp_ms - self.last_ocr_ms >= self.options.force_ocr_interval_ms
        if not self.options.change_detection or previous is None or forced:
            self.last_ocr_ms = timestamp_ms
            return True
        changed_ratio = changed_pixel_ratio(
            previous,
            signature,
            self.options.change_pixel_threshold,
        )
        if changed_ratio >= self.options.change_ratio_threshold:
            self.last_ocr_ms = timestamp_ms
            return True
        return False


def _crop(frame, region: Region):
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, round(region.x * width)))
    top = max(0, min(height - 1, round(region.y * height)))
    right = max(left + 1, min(width, round((region.x + region.width) * width)))
    bottom = max(top + 1, min(height, round((region.y + region.height) * height)))
    return frame[top:bottom, left:right]


def _color_hex(color: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02X}" for channel in color)


def _speaker_ids(colors: list[tuple[int, int, int]], threshold: float = 54.0) -> list[str]:
    centers: list[list[float]] = []
    counts: list[int] = []
    result: list[str] = []
    for color in colors:
        nearest = None
        nearest_distance = math.inf
        for index, center in enumerate(centers):
            distance = math.sqrt(sum((left - right) ** 2 for left, right in zip(color, center)))
            if distance < nearest_distance:
                nearest, nearest_distance = index, distance
        if nearest is None or nearest_distance > threshold:
            centers.append([float(value) for value in color])
            counts.append(1)
            nearest = len(centers) - 1
        else:
            counts[nearest] += 1
            weight = counts[nearest]
            centers[nearest] = [
                center + (value - center) / weight
                for center, value in zip(centers[nearest], color)
            ]
        result.append(f"SPK_{nearest + 1:02d}")
    return result


def _detect_batch(
    engine: OcrEngine, images: list[np.ndarray]
) -> list[list[TextDetection]]:
    detector = getattr(engine, "detect_batch", None)
    if callable(detector):
        return detector(images)
    return [engine.detect(image) for image in images]


def _is_gpu_memory_error(exc: BaseException) -> bool:
    detail = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in detail
        for marker in (
            "out of memory",
            "cuda_error_out_of_memory",
            "resource exhausted",
            "memory allocation failed",
        )
    )


def _release_gpu_cache() -> None:
    gc.collect()
    paddle = sys.modules.get("paddle")
    if paddle is None:
        return
    try:
        empty_cache = getattr(getattr(paddle.device, "cuda", None), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception:
        pass


def _detect_batch_with_memory_backoff(
    engine: OcrEngine, images: list[np.ndarray]
) -> tuple[list[list[TextDetection]], int, int, int]:
    try:
        return _detect_batch(engine, images), 1, 0, len(images)
    except Exception as exc:
        if len(images) <= 1 or not _is_gpu_memory_error(exc):
            raise
        _release_gpu_cache()
        midpoint = len(images) // 2
        left, left_calls, left_backoffs, left_size = _detect_batch_with_memory_backoff(
            engine, images[:midpoint]
        )
        right, right_calls, right_backoffs, right_size = _detect_batch_with_memory_backoff(
            engine, images[midpoint:]
        )
        return (
            [*left, *right],
            left_calls + right_calls,
            left_backoffs + right_backoffs + 1,
            max(left_size, right_size),
        )


def _recognized_to_cues(recognized: list[RecognizedCue]) -> list[Cue]:
    speakers = _speaker_ids([cue.color for cue in recognized])
    cues: list[Cue] = []
    for cue, speaker_id in zip(recognized, speakers):
        cues.append(
            Cue(
                cue_id=cue.cue_id,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                group_id=cue.group_id,
                layer=cue.layer,
                track_id=cue.track_id,
                speaker_id=speaker_id,
                speaker_color=_color_hex(cue.color),
                source_kind="ocr",
                source_text=cue.text,
                ocr_confidence=round(cue.confidence, 4),
                review_status="needs_review" if cue.review_required else "ocr_ok",
            )
        )
    return cues


def recognize_video(
    path: Path,
    engine: OcrEngine,
    options: RecognitionOptions,
    progress: ProgressCallback | None = None,
) -> list[Cue]:
    region = options.source_roi.validate()
    metadata_capture = cv2.VideoCapture(str(path.resolve()))
    if not metadata_capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = float(metadata_capture.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    frame_count = int(metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    metadata_capture.release()
    duration_ms = round(frame_count / fps * 1000) if frame_count else 0
    step = max(1, round(fps / max(options.sample_fps, 0.1)))
    first_frame = max(0, round(max(0, options.start_ms) / 1000 * fps))
    if frame_count > 0:
        first_frame = min(first_frame, frame_count - 1)
    first_frame -= first_frame % step
    total_sampled_frames = (
        (max(0, frame_count - first_frame) + step - 1) // step
        if frame_count > 0
        else 0
    )
    device = str(getattr(engine, "device", "cpu"))
    recommended_batch_size = int(
        getattr(engine, "recommended_batch_size", 0)
        or (6 if device.startswith("gpu") else 2)
    )
    batch_size = options.batch_size or recommended_batch_size
    batch_size = max(1, min(batch_size, 64))
    tracker = TemporalTextTracker(
        minimum_observations=options.minimum_observations,
        confidence_threshold=options.confidence_threshold,
        filter_noise=options.filter_noise,
        max_missed_frames=max(1, round(options.sample_fps * 0.6)),
    )
    change_gate = _FrameChangeGate(options)
    stabilizer = _TemporalDetectionStabilizer(
        options.transition_lookahead_ms,
        options.transition_confidence_threshold,
    )
    frame_queue: Queue[object] = Queue(maxsize=max(batch_size, options.queue_size))
    sentinel = object()
    stop_event = Event()
    producer_errors: list[BaseException] = []
    metrics: dict[str, float | int | str] = {
        "device": device,
        "batch_size": batch_size,
        "batch_mode": "manual" if options.batch_size else "auto",
        "filter_noise": options.filter_noise,
        "force_ocr_interval_ms": options.force_ocr_interval_ms,
        "frame_change_implementation": frame_change_implementation(),
        "transition_lookahead_ms": options.transition_lookahead_ms,
        "stabilized_detections": 0,
        "video_frames_total": frame_count,
        "sampled_frames_total": total_sampled_frames,
        "resume_start_ms": round(first_frame / fps * 1000),
        "decoded_frames": 0,
        "sampled_frames": 0,
        "processed_frames": 0,
        "ocr_frames": 0,
        "reused_frames": 0,
        "ocr_batches": 0,
        "ocr_batch_backoffs": 0,
        "effective_ocr_batch_size": batch_size,
        "decode_seconds": 0.0,
        "ocr_seconds": 0.0,
    }
    wall_started = perf_counter()

    def enqueue(value: object) -> bool:
        while not stop_event.is_set():
            try:
                frame_queue.put(value, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def produce_frames() -> None:
        capture = cv2.VideoCapture(str(path.resolve()))
        if not capture.isOpened():
            producer_errors.append(RuntimeError(f"could not open video: {path}"))
            enqueue(sentinel)
            return
        frame_index = first_frame
        if first_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        try:
            while not stop_event.is_set():
                decode_started = perf_counter()
                ok = capture.grab()
                metrics["decode_seconds"] = float(metrics["decode_seconds"]) + (
                    perf_counter() - decode_started
                )
                if not ok:
                    break
                metrics["decoded_frames"] = int(metrics["decoded_frames"]) + 1
                if frame_index % step:
                    frame_index += 1
                    continue
                decode_started = perf_counter()
                ok, frame = capture.retrieve()
                metrics["decode_seconds"] = float(metrics["decode_seconds"]) + (
                    perf_counter() - decode_started
                )
                if not ok:
                    break
                timestamp_ms = round(frame_index / fps * 1000)
                crop = _crop(frame, region).copy()
                metrics["sampled_frames"] = int(metrics["sampled_frames"]) + 1
                if not enqueue(_SampledFrame(timestamp_ms, crop)):
                    break
                frame_index += 1
        except BaseException as exc:
            producer_errors.append(exc)
        finally:
            capture.release()
            enqueue(sentinel)

    producer = Thread(target=produce_frames, name="kaor-video-decode", daemon=True)
    producer.start()
    previous_detections = ()

    def process_batch(batch: list[_SampledFrame]) -> None:
        nonlocal previous_detections
        if not batch:
            return
        sources: list[int | None] = []
        changed_images: list[np.ndarray] = []
        latest_changed: int | None = None
        for item in batch:
            if change_gate.requires_ocr(item.image, item.timestamp_ms):
                latest_changed = len(changed_images)
                changed_images.append(item.image)
                metrics["ocr_frames"] = int(metrics["ocr_frames"]) + 1
            else:
                metrics["reused_frames"] = int(metrics["reused_frames"]) + 1
            sources.append(latest_changed)

        changed_results: list[list[TextDetection]] = []
        if changed_images:
            ocr_started = perf_counter()
            (
                changed_results,
                successful_batches,
                batch_backoffs,
                effective_batch_size,
            ) = _detect_batch_with_memory_backoff(engine, changed_images)
            metrics["ocr_seconds"] = float(metrics["ocr_seconds"]) + (
                perf_counter() - ocr_started
            )
            metrics["ocr_batches"] = int(metrics["ocr_batches"]) + successful_batches
            metrics["ocr_batch_backoffs"] = int(metrics["ocr_batch_backoffs"]) + batch_backoffs
            metrics["effective_ocr_batch_size"] = min(
                int(metrics["effective_ocr_batch_size"]), effective_batch_size
            )
            if len(changed_results) != len(changed_images):
                raise RuntimeError("OCR frame batch result count mismatch")

        for item, source in zip(batch, sources):
            detections = (
                tuple(changed_results[source])
                if source is not None
                else previous_detections
            )
            if source is not None:
                previous_detections = detections
            ready_frames = stabilizer.push(
                FrameDetections(item.timestamp_ms, detections)
            )
            for ready_frame in ready_frames:
                tracker.update(ready_frame)
            metrics["processed_frames"] = int(metrics["processed_frames"]) + 1
        metrics["stabilized_detections"] = stabilizer.repaired_detections

        latest = batch[-1]
        live_cues = _recognized_to_cues(tracker.snapshot(end_ms=latest.timestamp_ms))
        current = [
            {
                "text": detection.text,
                "color": _color_hex(detection.color),
                "confidence": round(detection.confidence, 4),
            }
            for detection in previous_detections
        ]
        metrics["wall_seconds"] = round(perf_counter() - wall_started, 4)
        metrics["effective_sample_fps"] = round(
            int(metrics["processed_frames"])
            / max(float(metrics["wall_seconds"]), 1e-6),
            3,
        )
        snapshot = {
            "timestamp_ms": latest.timestamp_ms,
            "cues": [cue.model_dump() for cue in live_cues],
            "current": current,
            "metrics": dict(metrics),
        }
        if progress:
            processed_frames = int(metrics["processed_frames"])
            value = (
                min(1.0, processed_frames / total_sampled_frames)
                if total_sampled_frames
                else min(1.0, latest.timestamp_ms / max(duration_ms, 1))
            )
            progress(
                value,
                f"OCR sampled {processed_frames}/{total_sampled_frames or '?'} · "
                f"actual {metrics['ocr_frames']} · reused {metrics['reused_frames']}",
                snapshot,
            )

    try:
        while True:
            batch: list[_SampledFrame] = []
            finished = False
            while len(batch) < batch_size:
                try:
                    queued = frame_queue.get(timeout=0.2)
                except Empty:
                    if not producer.is_alive():
                        finished = True
                        break
                    continue
                if queued is sentinel:
                    finished = True
                    break
                if isinstance(queued, _SampledFrame):
                    batch.append(queued)
            process_batch(batch)
            if finished:
                break
        if producer_errors:
            raise producer_errors[0]
        for ready_frame in stabilizer.finish():
            tracker.update(ready_frame)
        metrics["stabilized_detections"] = stabilizer.repaired_detections
    finally:
        stop_event.set()
        producer.join(timeout=2.0)
    recognized = tracker.finalize(end_ms=duration_ms or None)
    cues = _recognized_to_cues(recognized)
    metrics["wall_seconds"] = round(perf_counter() - wall_started, 4)
    if progress:
        progress(
            1.0,
            "OCR complete",
            {
                "timestamp_ms": duration_ms,
                "cues": [cue.model_dump() for cue in cues],
                "current": [],
                "metrics": dict(metrics),
            },
        )
    return cues
