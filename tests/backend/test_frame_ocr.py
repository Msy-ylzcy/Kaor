from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.frame_ocr import FrameOcrResult, crop_region, recognize_frame
from backend.models import NormalizedRegion
from backend.recognition import Box, TextDetection


class FixtureEngine:
    def detect(self, _image):
        return [
            TextDetection("left", Box(0.05, 0.1, 0.2, 0.2), 0.8),
            TextDetection("right", Box(0.35, 0.11, 0.2, 0.2), 1.0),
            TextDetection("second", Box(0.1, 0.6, 0.3, 0.2), 0.7),
        ]


def test_crop_region_uses_normalized_video_coordinates():
    frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape((100, 200, 3))
    crop = crop_region(
        frame,
        NormalizedRegion(x=0.25, y=0.2, width=0.5, height=0.4),
    )
    assert crop.shape == (40, 100, 3)
    assert np.array_equal(crop[0, 0], frame[20, 50])


def test_recognize_frame_groups_visual_lines_and_confidence(monkeypatch, tmp_path):
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "backend.frame_ocr.read_frame",
        lambda *_args, **_kwargs: (frame, 30, 1000),
    )
    result = recognize_frame(
        tmp_path / "fixture.mp4",
        timestamp_ms=999,
        fps=30,
        region=NormalizedRegion(x=0, y=0, width=1, height=1),
        language="en",
        high_accuracy=False,
        engine=FixtureEngine(),
    )
    assert result.text == "left right\nsecond"
    assert result.confidence == 0.8333
    assert result.frame_index == 30
    assert len(result.detections) == 3


def test_frame_ocr_api_returns_selected_frame_result(tmp_path, monkeypatch):
    video_path = tmp_path / "fixture.mp4"
    video_path.write_bytes(b"video-fixture")
    constructed_engines = []

    def make_engine(**_kwargs):
        engine = object()
        constructed_engines.append(engine)
        return engine

    monkeypatch.setattr("backend.app.PaddleOcrEngine", make_engine)
    monkeypatch.setattr(
        "backend.app.recognize_frame",
        lambda *_args, **_kwargs: FrameOcrResult(
            timestamp_ms=1000,
            frame_index=30,
            text="corrected text",
            confidence=0.91,
            detections=[],
        ),
    )
    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        project = client.post(
            "/api/projects",
            json={
                "title": "Frame OCR",
                "video_filename": video_path.name,
                "video_path": str(video_path),
                "fps": 30,
            },
        ).json()
        response = client.post(
            f"/api/projects/{project['project_id']}/frame-ocr",
            json={
                "timestamp_ms": 999,
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.7, "height": 0.3},
                "language": "en",
                "device": "auto",
                "high_accuracy": True,
            },
        )
        cached_response = client.post(
            f"/api/projects/{project['project_id']}/frame-ocr",
            json={
                "timestamp_ms": 999,
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.7, "height": 0.3},
                "language": "en",
                "device": "auto",
                "high_accuracy": True,
            },
        )
    assert response.status_code == 200
    assert cached_response.status_code == 200
    assert len(constructed_engines) == 1
    assert response.json()["text"] == "corrected text"
    assert response.json()["confidence"] == 0.91
