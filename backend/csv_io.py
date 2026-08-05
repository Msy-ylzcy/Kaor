from __future__ import annotations

import csv
import os
from pathlib import Path

from .models import Cue


CUE_COLUMNS = [
    "cue_id",
    "start_ms",
    "end_ms",
    "group_id",
    "layer",
    "track_id",
    "speaker_id",
    "speaker_name",
    "speaker_color",
    "source_kind",
    "source_text",
    "ocr_confidence",
    "target_text",
    "review_status",
]


def read_cues(path: Path) -> list[Cue]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CUE_COLUMNS:
            raise ValueError("invalid cue CSV header")
        cues: list[Cue] = []
        for row in reader:
            payload = dict(row)
            payload["ocr_confidence"] = payload["ocr_confidence"] or None
            cues.append(Cue.model_validate(payload))
        return cues


def write_cues(path: Path, cues: list[Cue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CUE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for cue in sorted(
            cues, key=lambda item: (item.start_ms, item.layer, item.track_id, item.cue_id)
        ):
            row = cue.model_dump()
            row["ocr_confidence"] = (
                "" if cue.ocr_confidence is None else cue.ocr_confidence
            )
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)
