from __future__ import annotations

import cv2
import numpy as np

from backend.recognition import Box, TextDetection
from backend.video_recognition import RecognitionOptions, Region, recognize_video


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float = 4.0) -> None:
        self.frames = frames
        self.fps = fps
        self.index = 0
        self.current: np.ndarray | None = None

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def grab(self):
        if self.index >= len(self.frames):
            return False
        self.current = self.frames[self.index]
        self.index += 1
        return True

    def retrieve(self):
        return True, self.current.copy()

    def release(self):
        return None


class BatchEngine:
    device = "gpu:0"
    recommended_batch_size = 16

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def detect(self, image):
        return self.detect_batch([image])[0]

    def detect_batch(self, images):
        self.batch_sizes.append(len(images))
        return [
            [TextDetection("HELLO", Box(0.1, 0.2, 0.8, 0.3), 0.96)]
            for _ in images
        ]


class SequenceEngine:
    device = "cpu"
    recommended_batch_size = 1

    def __init__(self, results: list[list[TextDetection]]) -> None:
        self.results = results
        self.index = 0

    def detect(self, image):
        return self.detect_batch([image])[0]

    def detect_batch(self, images):
        batch = self.results[self.index : self.index + len(images)]
        self.index += len(images)
        return batch


class MemoryBoundEngine(BatchEngine):
    recommended_batch_size = 4

    def detect_batch(self, images):
        self.batch_sizes.append(len(images))
        if len(images) > 2:
            raise RuntimeError("CUDA out of memory while allocating tensor")
        return [
            [TextDetection("HELLO", Box(0.1, 0.2, 0.8, 0.3), 0.96)]
            for _ in images
        ]


def test_video_recognition_batches_frames_and_publishes_live_cues(tmp_path, monkeypatch):
    frames = [np.full((80, 160, 3), value, dtype=np.uint8) for value in (0, 40, 80, 120)]
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda path: FakeCapture([frame.copy() for frame in frames])
    )
    engine = BatchEngine()
    snapshots = []

    cues = recognize_video(
        tmp_path / "fixture.mp4",
        engine,
        RecognitionOptions(
            source_roi=Region(0, 0, 1, 1),
            sample_fps=4,
            batch_size=3,
            change_detection=False,
        ),
        lambda value, message, snapshot=None: snapshots.append(snapshot),
    )

    assert engine.batch_sizes == [3, 1]
    assert [cue.source_text for cue in cues] == ["HELLO"]
    assert any(snapshot and snapshot["current"] for snapshot in snapshots)
    assert snapshots[-1]["cues"][0]["source_text"] == "HELLO"


def test_unchanged_frames_reuse_previous_ocr_result(tmp_path, monkeypatch):
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    frames = [frame.copy() for _ in range(4)]
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda path: FakeCapture([item.copy() for item in frames])
    )
    engine = BatchEngine()
    snapshots = []

    recognize_video(
        tmp_path / "fixture.mp4",
        engine,
        RecognitionOptions(
            source_roi=Region(0, 0, 1, 1),
            sample_fps=4,
            batch_size=4,
            force_ocr_interval_ms=10_000,
        ),
        lambda value, message, snapshot=None: snapshots.append(snapshot),
    )

    assert engine.batch_sizes == [1]
    assert snapshots[-1]["metrics"]["ocr_frames"] == 1
    assert snapshots[-1]["metrics"]["reused_frames"] == 3
    assert snapshots[-1]["metrics"]["processed_frames"] == 4
    assert snapshots[-1]["metrics"]["sampled_frames_total"] == 4
    assert snapshots[-1]["metrics"]["video_frames_total"] == 4


def test_auto_batch_size_uses_engine_device_recommendation(tmp_path, monkeypatch):
    frames = [np.full((80, 160, 3), index, dtype=np.uint8) for index in range(20)]
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda path: FakeCapture([item.copy() for item in frames])
    )
    engine = BatchEngine()

    recognize_video(
        tmp_path / "fixture.mp4",
        engine,
        RecognitionOptions(
            source_roi=Region(0, 0, 1, 1),
            sample_fps=4,
            change_detection=False,
        ),
    )

    assert engine.batch_sizes == [16, 4]


def test_gpu_memory_error_halves_batch_and_reports_effective_size(
    tmp_path, monkeypatch
):
    frames = [np.full((80, 160, 3), index * 30, dtype=np.uint8) for index in range(4)]
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda path: FakeCapture([item.copy() for item in frames])
    )
    engine = MemoryBoundEngine()
    snapshots = []

    cues = recognize_video(
        tmp_path / "fixture.mp4",
        engine,
        RecognitionOptions(
            source_roi=Region(0, 0, 1, 1),
            sample_fps=4,
            change_detection=False,
        ),
        lambda _value, _message, snapshot=None: snapshots.append(snapshot),
    )

    assert [cue.source_text for cue in cues] == ["HELLO"]
    assert engine.batch_sizes == [4, 2, 2]
    assert snapshots[-1]["metrics"]["ocr_batch_backoffs"] == 1
    assert snapshots[-1]["metrics"]["effective_ocr_batch_size"] == 2


def test_gpu_cache_release_does_not_import_paddle(monkeypatch):
    import sys

    from backend import video_recognition

    monkeypatch.delitem(sys.modules, "paddle", raising=False)

    video_recognition._release_gpu_cache()

    assert "paddle" not in sys.modules


def test_long_unchanged_subtitle_uses_five_second_verification_interval(
    tmp_path, monkeypatch
):
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    frames = [frame.copy() for _ in range(24)]
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda path: FakeCapture([item.copy() for item in frames])
    )
    engine = BatchEngine()
    snapshots = []

    recognize_video(
        tmp_path / "fixture.mp4",
        engine,
        RecognitionOptions(
            source_roi=Region(0, 0, 1, 1),
            sample_fps=4,
        ),
        lambda value, message, snapshot=None: snapshots.append(snapshot),
    )

    metrics = snapshots[-1]["metrics"]
    assert engine.batch_sizes == [1, 1]
    assert metrics["ocr_frames"] == 2
    assert metrics["reused_frames"] == 22
    assert metrics["force_ocr_interval_ms"] == 5000


def test_fade_in_low_confidence_frame_uses_clearer_future_detection(
    tmp_path, monkeypatch
):
    frames = [np.full((80, 160, 3), value, dtype=np.uint8) for value in (30, 60, 90)]
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda path: FakeCapture([item.copy() for item in frames])
    )
    box = Box(0.1, 0.2, 0.8, 0.3)
    engine = SequenceEngine(
        [
            [TextDetection("Hcllo", box, 0.48)],
            [TextDetection("Hello", box, 0.97)],
            [TextDetection("Hello", box, 0.99)],
        ]
    )
    snapshots = []

    cues = recognize_video(
        tmp_path / "fixture.mp4",
        engine,
        RecognitionOptions(
            source_roi=Region(0, 0, 1, 1),
            sample_fps=4,
            batch_size=1,
            change_detection=False,
            transition_lookahead_ms=500,
        ),
        lambda value, message, snapshot=None: snapshots.append(snapshot),
    )

    assert [(cue.source_text, cue.start_ms) for cue in cues] == [("Hello", 0)]
    assert snapshots[-1]["metrics"]["stabilized_detections"] == 1


def test_fade_in_fragments_share_one_clear_replacement(tmp_path, monkeypatch):
    frames = [np.full((80, 160, 3), value, dtype=np.uint8) for value in (30, 60, 90)]
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda path: FakeCapture([item.copy() for item in frames])
    )
    box = Box(0.1, 0.2, 0.8, 0.3)
    engine = SequenceEngine(
        [
            [
                TextDetection("Hcllo", box, 0.48),
                TextDetection("Hello", box, 0.52),
            ],
            [TextDetection("Hello", box, 0.97)],
            [TextDetection("Hello", box, 0.99)],
        ]
    )
    snapshots = []

    cues = recognize_video(
        tmp_path / "fixture.mp4",
        engine,
        RecognitionOptions(
            source_roi=Region(0, 0, 1, 1),
            sample_fps=4,
            batch_size=1,
            change_detection=False,
            transition_lookahead_ms=500,
        ),
        lambda value, message, snapshot=None: snapshots.append(snapshot),
    )

    assert [cue.source_text for cue in cues] == ["Hello"]
    assert snapshots[-1]["metrics"]["stabilized_detections"] == 2
