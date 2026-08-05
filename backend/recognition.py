from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import count
from typing import Iterable, Sequence


_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip()).casefold()


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def intersection_over_union(self, other: "Box") -> float:
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        area = max(0.0, right - left) * max(0.0, bottom - top)
        union = self.width * self.height + other.width * other.height - area
        return area / union if union > 0 else 0.0

    def vertical_affinity(self, other: "Box") -> float:
        center_a = self.y + self.height / 2
        center_b = other.y + other.height / 2
        scale = max(self.height, other.height, 1e-6)
        return max(0.0, 1.0 - abs(center_a - center_b) / (scale * 2))


@dataclass(frozen=True)
class TextDetection:
    text: str
    box: Box
    confidence: float
    color: tuple[int, int, int] = (255, 255, 255)


@dataclass(frozen=True)
class FrameDetections:
    timestamp_ms: int
    detections: tuple[TextDetection, ...]


@dataclass
class _Observation:
    timestamp_ms: int
    detection: TextDetection


@dataclass
class TextTrack:
    track_id: str
    start_ms: int
    last_seen_ms: int
    observations: list[_Observation] = field(default_factory=list)
    missed_frames: int = 0

    @property
    def latest(self) -> TextDetection:
        return self.observations[-1].detection

    def add(self, timestamp_ms: int, detection: TextDetection) -> None:
        self.observations.append(_Observation(timestamp_ms, detection))
        self.last_seen_ms = timestamp_ms
        self.missed_frames = 0

    def consensus_text(self) -> str:
        candidates: dict[str, tuple[str, float]] = {}
        for observation in self.observations:
            text = observation.detection.text.strip()
            normalized = normalize_text(text)
            if not normalized:
                continue
            current_text, current_weight = candidates.get(normalized, (text, 0.0))
            candidates[normalized] = (
                text if len(text) > len(current_text) else current_text,
                current_weight + max(0.01, observation.detection.confidence),
            )
        if not candidates:
            return ""
        values = list(candidates.values())
        normalized_values = [normalize_text(item[0]) for item in values]
        if all(
            left.startswith(right) or right.startswith(left)
            for index, left in enumerate(normalized_values)
            for right in normalized_values[index + 1 :]
        ):
            return max(values, key=lambda item: len(normalize_text(item[0])))[0]
        return max(
            values,
            key=lambda item: (
                sum(
                    item[1]
                    * SequenceMatcher(None, normalize_text(item[0]), normalize_text(other[0])).ratio()
                    for other in values
                ),
                len(item[0]),
            ),
        )[0]

    def confidence(self) -> float:
        if not self.observations:
            return 0.0
        values = sorted(
            observation.detection.confidence for observation in self.observations
        )
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2

    def median_color(self) -> tuple[int, int, int]:
        channels = zip(*(observation.detection.color for observation in self.observations))
        result = []
        for channel in channels:
            values = sorted(channel)
            result.append(values[len(values) // 2])
        return tuple(result)  # type: ignore[return-value]


@dataclass(frozen=True)
class RecognizedCue:
    cue_id: str
    track_id: str
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    color: tuple[int, int, int]
    box: Box
    group_id: str = ""
    layer: int = 0
    review_required: bool = False


def _color_similarity(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    distance = math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))
    return max(0.0, 1.0 - distance / 441.673)


def _text_similarity(a: str, b: str) -> float:
    left = normalize_text(a)
    right = normalize_text(b)
    if not left or not right:
        return 0.0
    if left.startswith(right) or right.startswith(left):
        return 0.96
    return SequenceMatcher(None, left, right).ratio()


def _match_score(track: TextTrack, detection: TextDetection) -> float:
    previous = track.latest
    iou = previous.box.intersection_over_union(detection.box)
    text = _text_similarity(previous.text, detection.text)
    color = _color_similarity(previous.color, detection.color)
    vertical = previous.box.vertical_affinity(detection.box)
    return 0.38 * iou + 0.42 * text + 0.14 * color + 0.06 * vertical


class TemporalTextTracker:
    def __init__(
        self,
        *,
        match_threshold: float = 0.53,
        max_missed_frames: int = 2,
        minimum_observations: int = 2,
        confidence_threshold: float = 0.82,
        filter_noise: bool = True,
        single_observation_confidence_threshold: float = 0.94,
        fragment_window_ms: int = 250,
        repeat_merge_window_ms: int = 1500,
    ) -> None:
        self.match_threshold = match_threshold
        self.max_missed_frames = max_missed_frames
        self.minimum_observations = minimum_observations
        self.confidence_threshold = confidence_threshold
        self.filter_noise = filter_noise
        self.single_observation_confidence_threshold = (
            single_observation_confidence_threshold
        )
        self.fragment_window_ms = fragment_window_ms
        self.repeat_merge_window_ms = repeat_merge_window_ms
        self._ids = count(1)
        self._active: list[TextTrack] = []
        self._closed: list[TextTrack] = []
        self._last_timestamp_ms = 0

    def update(self, frame: FrameDetections) -> None:
        if frame.timestamp_ms < self._last_timestamp_ms:
            raise ValueError("frames must be provided in chronological order")
        self._last_timestamp_ms = frame.timestamp_ms
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._active):
            for detection_index, detection in enumerate(frame.detections):
                score = _match_score(track, detection)
                if score >= self.match_threshold:
                    candidates.append((score, track_index, detection_index))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            self._active[track_index].add(
                frame.timestamp_ms, frame.detections[detection_index]
            )
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        survivors: list[TextTrack] = []
        for track_index, track in enumerate(self._active):
            if track_index not in matched_tracks:
                track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                self._closed.append(track)
            else:
                survivors.append(track)
        self._active = survivors

        for detection_index, detection in enumerate(frame.detections):
            if detection_index in matched_detections or not normalize_text(detection.text):
                continue
            track = TextTrack(
                track_id=f"TRK_{next(self._ids):06d}",
                start_ms=frame.timestamp_ms,
                last_seen_ms=frame.timestamp_ms,
            )
            track.add(frame.timestamp_ms, detection)
            self._active.append(track)

    def finalize(self, end_ms: int | None = None) -> list[RecognizedCue]:
        active_ids = {track.track_id for track in self._active}
        self._closed.extend(self._active)
        self._active = []
        return self._build_cues(self._closed, active_ids, end_ms)

    def snapshot(self, end_ms: int | None = None) -> list[RecognizedCue]:
        """Return current closed and active tracks without consuming the tracker."""
        tracks = [*self._closed, *self._active]
        active_ids = {track.track_id for track in self._active}
        return self._build_cues(tracks, active_ids, end_ms)

    def _build_cues(
        self,
        tracks: Sequence[TextTrack],
        active_ids: set[str],
        end_ms: int | None,
    ) -> list[RecognizedCue]:
        terminal = max(self._last_timestamp_ms, end_ms or 0)
        tracks, active_ids = _coalesce_repeated_tracks(
            tracks,
            active_ids,
            self.repeat_merge_window_ms,
        )
        tracks = sorted(tracks, key=lambda item: (item.start_ms, item.track_id))
        if self.filter_noise:
            stable_tracks = [track for track in tracks if len(track.observations) >= 2]
            tracks = [
                track
                for track in tracks
                if not self._is_noise_track(track, stable_tracks)
            ]
        cues: list[RecognizedCue] = []
        for index, track in enumerate(tracks, 1):
            text = track.consensus_text()
            confidence = track.confidence()
            cue_end = max(track.start_ms + 1, track.last_seen_ms)
            if track.track_id in active_ids:
                cue_end = max(track.start_ms + 1, terminal)
            cues.append(
                RecognizedCue(
                    cue_id=f"{index:06d}",
                    track_id=track.track_id,
                    start_ms=track.start_ms,
                    end_ms=cue_end,
                    text=text,
                    confidence=confidence,
                    color=track.median_color(),
                    box=track.latest.box,
                    review_required=(
                        len(track.observations) < self.minimum_observations
                        or confidence < self.confidence_threshold
                    ),
                )
            )
        return assign_overlap_groups(cues)

    def _is_noise_track(
        self,
        track: TextTrack,
        stable_tracks: Sequence[TextTrack],
    ) -> bool:
        confidence = track.confidence()
        if any(
            stable.track_id != track.track_id
            and _is_redundant_track(track, stable, self.fragment_window_ms)
            for stable in stable_tracks
        ):
            return True
        if len(track.observations) >= 2:
            compact = "".join(track.consensus_text().split())
            visible = [
                character
                for character in track.consensus_text().strip()
                if not character.isspace()
            ]
            return bool(
                (
                    confidence < 0.94
                    and re.fullmatch(r"[A-Za-z0-9_:.-]{1,2}", compact)
                )
                or (
                    confidence < 0.60
                    and re.fullmatch(r"[A-Za-z0-9_:.-]{3,8}", compact)
                )
                or (len(visible) == 1 and confidence < 0.985)
            )
        threshold = max(
            self.confidence_threshold,
            self.single_observation_confidence_threshold,
        )
        if confidence < threshold or not _is_meaningful_flash_text(
            track.consensus_text(), confidence
        ):
            return True

        candidate = normalize_text(track.consensus_text())
        for stable in stable_tracks:
            if _track_time_gap_ms(track, stable) > self.fragment_window_ms:
                continue
            reference = normalize_text(stable.consensus_text())
            if candidate and candidate != reference and candidate in reference:
                return True
        return False


def _is_meaningful_flash_text(text: str, confidence: float) -> bool:
    visible = [character for character in text.strip() if not character.isspace()]
    readable = [character for character in visible if character.isalnum()]
    if not readable:
        return False
    if len(visible) >= 2:
        return True
    # Preserve exceptionally clear single-glyph CJK dialogue, but reject the
    # common one-frame ASCII counters and UI labels found around subtitle ROIs.
    return ord(readable[0]) > 127 and confidence >= 0.985


def _track_time_gap_ms(left: TextTrack, right: TextTrack) -> int:
    if left.last_seen_ms < right.start_ms:
        return right.start_ms - left.last_seen_ms
    if right.last_seen_ms < left.start_ms:
        return left.start_ms - right.last_seen_ms
    return 0


def _same_track_region(left: TextTrack, right: TextTrack) -> bool:
    left_box = left.latest.box
    right_box = right.latest.box
    horizontal_overlap = max(
        0.0,
        min(left_box.right, right_box.right) - max(left_box.x, right_box.x),
    ) / max(min(left_box.width, right_box.width), 1e-6)
    return left_box.vertical_affinity(right_box) >= 0.72 and (
        left_box.intersection_over_union(right_box) >= 0.05
        or horizontal_overlap >= 0.35
    )


def _track_is_stronger(reference: TextTrack, candidate: TextTrack) -> bool:
    reference_rank = (
        len(reference.observations),
        reference.last_seen_ms - reference.start_ms,
        reference.confidence(),
        len(normalize_text(reference.consensus_text())),
    )
    candidate_rank = (
        len(candidate.observations),
        candidate.last_seen_ms - candidate.start_ms,
        candidate.confidence(),
        len(normalize_text(candidate.consensus_text())),
    )
    if reference_rank != candidate_rank:
        return reference_rank > candidate_rank
    return reference.track_id < candidate.track_id


def _is_redundant_track(
    candidate: TextTrack,
    reference: TextTrack,
    window_ms: int,
) -> bool:
    candidate_text = normalize_text(candidate.consensus_text())
    reference_text = normalize_text(reference.consensus_text())
    if not candidate_text or not reference_text:
        return False
    overlap_ms = max(
        0,
        min(candidate.last_seen_ms, reference.last_seen_ms)
        - max(candidate.start_ms, reference.start_ms),
    )
    candidate_duration = max(1, candidate.last_seen_ms - candidate.start_ms)
    if (
        candidate_text != reference_text
        and candidate_text in reference_text
        and overlap_ms / candidate_duration >= 0.60
        and (
            overlap_ms / candidate_duration >= 0.90
            or reference.confidence() >= candidate.confidence() - 0.10
        )
    ):
        return True
    if _track_time_gap_ms(candidate, reference) > window_ms:
        return False
    if not _same_track_region(candidate, reference):
        return False
    if candidate_text == reference_text:
        return _track_is_stronger(reference, candidate)
    return bool(
        candidate_text in reference_text
        and len(reference_text) > len(candidate_text)
        and reference.confidence() >= candidate.confidence() - 0.05
    )


def _coalesce_repeated_tracks(
    tracks: Sequence[TextTrack],
    active_ids: set[str],
    window_ms: int,
) -> tuple[list[TextTrack], set[str]]:
    merged: list[TextTrack] = []
    merged_active = set(active_ids)
    for source in sorted(tracks, key=lambda item: (item.start_ms, item.track_id)):
        current = TextTrack(
            track_id=source.track_id,
            start_ms=source.start_ms,
            last_seen_ms=source.last_seen_ms,
            observations=list(source.observations),
            missed_frames=source.missed_frames,
        )
        reference = next(
            (
                previous
                for previous in reversed(merged)
                if normalize_text(previous.consensus_text())
                == normalize_text(current.consensus_text())
                and _track_time_gap_ms(previous, current) <= window_ms
                and _same_track_region(previous, current)
                and _color_similarity(
                    previous.median_color(), current.median_color()
                )
                >= 0.72
            ),
            None,
        )
        if reference is None:
            merged.append(current)
            continue
        reference.observations.extend(current.observations)
        reference.observations.sort(key=lambda item: item.timestamp_ms)
        reference.start_ms = min(reference.start_ms, current.start_ms)
        reference.last_seen_ms = max(reference.last_seen_ms, current.last_seen_ms)
        if current.track_id in merged_active:
            merged_active.add(reference.track_id)
        merged_active.discard(current.track_id)
    return merged, merged_active


def assign_overlap_groups(cues: Sequence[RecognizedCue]) -> list[RecognizedCue]:
    parent = list(range(len(cues)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(cues):
        for right_index in range(left_index + 1, len(cues)):
            right = cues[right_index]
            overlap = min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms)
            if overlap > 0:
                union(left_index, right_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(cues)):
        groups.setdefault(find(index), []).append(index)
    result = list(cues)
    group_number = 1
    for indices in groups.values():
        if len(indices) < 2:
            continue
        group_id = f"G{group_number:04d}"
        group_number += 1
        ordered = sorted(indices, key=lambda index: (cues[index].box.y, cues[index].start_ms))
        for layer, cue_index in enumerate(ordered):
            current = result[cue_index]
            result[cue_index] = RecognizedCue(
                **{
                    **current.__dict__,
                    "group_id": group_id,
                    "layer": layer,
                }
            )
    return result


def track_frames(frames: Iterable[FrameDetections], **tracker_options: object) -> list[RecognizedCue]:
    tracker = TemporalTextTracker(**tracker_options)
    for frame in frames:
        tracker.update(frame)
    return tracker.finalize()
