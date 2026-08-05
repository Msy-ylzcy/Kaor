from __future__ import annotations

import re
from pathlib import Path

from .models import Cue


_TIMING_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def _milliseconds(value: str) -> int:
    hours, minutes, remainder = value.replace(",", ".").split(":")
    seconds, milliseconds = remainder.split(".")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1000
        + int(milliseconds)
    )


def parse_srt(value: str) -> list[Cue]:
    blocks = re.split(r"\r?\n\s*\r?\n", value.strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = block.splitlines()
        timing_index = next(
            (index for index, line in enumerate(lines) if _TIMING_RE.search(line)), None
        )
        if timing_index is None:
            continue
        match = _TIMING_RE.search(lines[timing_index])
        assert match is not None
        text = "\\N".join(
            _TAG_RE.sub("", line).strip()
            for line in lines[timing_index + 1 :]
            if line.strip()
        )
        if not text:
            continue
        cue_id = f"{len(cues) + 1:06d}"
        cues.append(
            Cue(
                cue_id=cue_id,
                start_ms=_milliseconds(match.group("start")),
                end_ms=_milliseconds(match.group("end")),
                track_id="embedded",
                speaker_id="SPK_01",
                source_kind="imported",
                source_text=text,
                review_status="ocr_ok",
            )
        )
    for left_index, left in enumerate(cues):
        overlapping = [
            right_index
            for right_index, right in enumerate(cues)
            if right_index != left_index
            and min(left.end_ms, right.end_ms) > max(left.start_ms, right.start_ms)
        ]
        if not overlapping:
            continue
        group_members = sorted({left_index, *overlapping})
        group_id = f"G{group_members[0] + 1:04d}"
        for layer, member_index in enumerate(group_members):
            cues[member_index].group_id = group_id
            cues[member_index].layer = layer
    return cues


def read_srt(path: Path) -> list[Cue]:
    return parse_srt(path.read_text(encoding="utf-8-sig"))
