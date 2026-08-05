from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from .models import Cue, ProjectManifest
from .translation import (
    CheckpointCallback,
    OpenAICompatibleTranslator,
    ProviderEventCallback,
    TranslationError,
    TranslationProvider,
    TranslationRequestError,
    _extract_json,
    cues_to_csv,
)


SYSTEM_PROMPT = """You are a subtitle evidence fusion and source-language correction engine.

You receive two explicitly labelled evidence streams:
- OCR_CSV: text observed in video frames.
- SPEECH_CSV: text recognized from separated speech audio.

Produce a single corrected source-language cue table. This is a correction step only. Never
translate source_text, and never emit target_text.

Evidence characteristics:
1. OCR_CSV can repeat a long-lived subtitle across sampled frames, capture a fade-in before
   glyphs are fully visible, retain stale text, confuse low-contrast glyphs, or produce isolated
   noise such as "0" and "1". Consecutive near-duplicates can describe one real cue.
2. SPEECH_CSV avoids frame-sampling duplicates but can contain homophone substitutions,
   missing punctuation, voice-activity boundary cuts, timestamp drift, speaker swaps, and
   omissions during overlapping speech or imperfect vocal separation.
3. Use agreement between streams, confidence, adjacent rows, title/story context, character
   context, and language context. Treat neither stream as universally authoritative.

Timeline and overlap rules:
1. Merge rows only when they are duplicate observations of the same utterance. Split a row
   when evidence clearly contains multiple utterances.
2. Preserve simultaneous speech and staggered overlays as distinct cues. If one subtitle starts
   first and a second subtitle appears while it remains visible, keep the first cue's earlier
   start and the second cue's later start. Do not flatten both to one start time, concatenate
   them, or discard either speaker.
3. Use a shared non-empty group_id for cues that intentionally overlap, with distinct layers
   and stable track_id values. Keep independent speaker identity and color when evidence allows.
4. Times are integer milliseconds, start_ms >= 0, and end_ms > start_ms.
5. Assign unique deterministic cue_id values and order cues chronologically.
6. FULL_OCR_CSV_READ_ONLY and FULL_SPEECH_CSV_READ_ONLY always contain the complete evidence
   tables. They are global reference for continuity and cross-checking only. Emit corrected cues
   only for evidence IDs listed in output_contract and the primary OCR_CSV/SPEECH_CSV subset.
7. OCR_CONTEXT_CSV and SPEECH_CONTEXT_CSV, when present, are read-only adjacent context.
   Use them to understand continuity, but never emit a cue solely from a context row.

Confidence and review rules:
1. ocr_confidence is retained as the current CSV compatibility field. In the final table it is
   the combined confidence in the corrected source cue, from 0 through 1, or null only when the
   evidence cannot support a numeric estimate.
2. Set review_status to "ocr_ok" when the corrected source cue is reliable, otherwise set it to
   "needs_review". Never use translated or approved status in this step.
3. source_kind identifies the primary supporting evidence and must be one of "ocr", "speech",
   "manual", or "imported".

Return valid JSON only, without Markdown or commentary, using exactly this root schema:
{"cues":[{"cue_id":"F000001","start_ms":0,"end_ms":1000,"group_id":"",
"layer":0,"track_id":"main","speaker_id":"","speaker_name":"",
"speaker_color":"#FFFFFF","source_kind":"ocr","source_text":"source language text",
"ocr_confidence":0.95,"review_status":"ocr_ok"}]}
"""


ProgressCallback = Callable[[float], None]


@dataclass(frozen=True)
class FusionOptions:
    batch_size: int = 80
    context_cues: int = 3
    retries: int = 2
    retry_backoff_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.context_cues < 0:
            raise ValueError("context_cues must be non-negative")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")


def _project_context(manifest: ProjectManifest) -> dict[str, Any]:
    return {
        "video_title": manifest.title,
        "video_filename": manifest.video_filename,
        "story_synopsis": manifest.synopsis,
        "genre_and_tone": manifest.genre_and_tone,
        "characters_context": manifest.characters_context,
        "glossary_context": manifest.glossary_context,
        "character_profiles": [
            profile.model_dump() for profile in manifest.character_profiles
        ],
        "glossary": manifest.glossary,
    }


def build_fusion_prompt(
    manifest: ProjectManifest,
    ocr_cues: list[Cue],
    speech_cues: list[Cue],
    *,
    ocr_context: list[Cue] | None = None,
    speech_context: list[Cue] | None = None,
    batch_index: int = 1,
    batch_count: int = 1,
    full_ocr_cues: list[Cue] | None = None,
    full_speech_cues: list[Cue] | None = None,
    subset_label: str = "1",
) -> str:
    payload = {
        "task": (
            "Compare OCR_CSV with SPEECH_CSV and return one corrected source cue table. "
            "Do not translate it."
        ),
        "project_context": _project_context(manifest),
        "language_context": {
            "source_language": manifest.source_language,
            "output_language": manifest.source_language,
            "future_translation_target_not_for_this_step": manifest.target_language,
        },
        "batch": {
            "index": batch_index,
            "count": batch_count,
            "subset_label": subset_label,
            "instruction": (
                "Return corrected cues for OCR_CSV and SPEECH_CSV only. The FULL_* "
                "tables are read-only global reference."
            ),
        },
        "output_contract": {
            "OCR_cue_ids": [cue.cue_id for cue in ocr_cues],
            "SPEECH_cue_ids": [cue.cue_id for cue in speech_cues],
            "instruction": (
                "Use all evidence for reasoning, but emit only the corrected timeline "
                "represented by these primary evidence rows."
            ),
        },
        "evidence_notes": {
            "OCR_CSV": (
                "Frame OCR: watch for sampling duplicates, incomplete fade-in glyphs, "
                "stale overlays, low-confidence substitutions, and isolated noise."
            ),
            "SPEECH_CSV": (
                "Separated-audio ASR: watch for phonetic substitutions, VAD boundary "
                "errors, timing drift, overlap omissions, and diarization mistakes."
            ),
            "overlap": (
                "Preserve concurrent speakers and staggered subtitle appearances as "
                "separate timed rows, grouped and layered where appropriate."
            ),
        },
        "OCR_CSV": cues_to_csv(ocr_cues),
        "SPEECH_CSV": cues_to_csv(speech_cues),
        "OCR_CONTEXT_CSV": cues_to_csv(ocr_context or []),
        "SPEECH_CONTEXT_CSV": cues_to_csv(speech_context or []),
        "FULL_OCR_CSV_READ_ONLY": cues_to_csv(
            ocr_cues if full_ocr_cues is None else full_ocr_cues
        ),
        "FULL_SPEECH_CSV_READ_ONLY": cues_to_csv(
            speech_cues if full_speech_cues is None else full_speech_cues
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class _FusionBatch:
    ocr_cues: list[Cue]
    speech_cues: list[Cue]
    ocr_context: list[Cue]
    speech_context: list[Cue]

    @property
    def evidence_count(self) -> int:
        return len(self.ocr_cues) + len(self.speech_cues)


def _split_fusion_batch(batch: _FusionBatch) -> tuple[_FusionBatch, _FusionBatch] | None:
    rows = sorted(
        [("ocr", cue) for cue in batch.ocr_cues]
        + [("speech", cue) for cue in batch.speech_cues],
        key=lambda item: (
            (item[1].start_ms + item[1].end_ms) / 2,
            item[1].start_ms,
            item[0],
            item[1].cue_id,
        ),
    )
    if len(rows) < 2:
        return None

    # Prefer a real timeline gap near the middle so matching OCR/ASR observations stay together.
    middle = len(rows) / 2
    lower = max(1, len(rows) // 4)
    upper = min(len(rows) - 1, len(rows) - len(rows) // 4)
    candidates = range(lower, upper + 1)
    split_at = max(
        candidates,
        key=lambda index: (
            (
                (rows[index][1].start_ms + rows[index][1].end_ms) / 2
                - (rows[index - 1][1].start_ms + rows[index - 1][1].end_ms) / 2
            ),
            -abs(index - middle),
        ),
    )
    left_rows = rows[:split_at]
    right_rows = rows[split_at:]

    def make(part: list[tuple[str, Cue]]) -> _FusionBatch:
        return _FusionBatch(
            ocr_cues=[cue for source, cue in part if source == "ocr"],
            speech_cues=[cue for source, cue in part if source == "speech"],
            ocr_context=batch.ocr_context,
            speech_context=batch.speech_context,
        )

    return make(left_rows), make(right_rows)


def _build_fusion_batches(
    ocr_cues: list[Cue],
    speech_cues: list[Cue],
    *,
    batch_size: int,
    context_cues: int,
) -> list[_FusionBatch]:
    evidence = sorted(
        [("ocr", cue) for cue in ocr_cues]
        + [("speech", cue) for cue in speech_cues],
        key=lambda item: (item[1].start_ms, item[1].end_ms, item[0], item[1].cue_id),
    )
    if not evidence:
        return []

    components: list[list[tuple[str, Cue]]] = []
    current: list[tuple[str, Cue]] = []
    current_end = -1
    for row in evidence:
        cue = row[1]
        if current and cue.start_ms > current_end + 500:
            components.append(current)
            current = []
            current_end = -1
        current.append(row)
        current_end = max(current_end, cue.end_ms)
    if current:
        components.append(current)

    packed: list[list[tuple[str, Cue]]] = []
    current = []
    for component in components:
        if current and len(current) + len(component) > batch_size:
            packed.append(current)
            current = []
        current.extend(component)
    if current:
        packed.append(current)

    batches: list[_FusionBatch] = []
    evidence_keys = [(source, cue.cue_id) for source, cue in evidence]
    for rows in packed:
        primary_keys = {(source, cue.cue_id) for source, cue in rows}
        indices = [
            index for index, key in enumerate(evidence_keys) if key in primary_keys
        ]
        context_rows: list[tuple[str, Cue]] = []
        if context_cues and indices:
            first = min(indices)
            last = max(indices)
            context_rows.extend(evidence[max(0, first - context_cues) : first])
            context_rows.extend(evidence[last + 1 : last + 1 + context_cues])
        batches.append(
            _FusionBatch(
                ocr_cues=[cue for source, cue in rows if source == "ocr"],
                speech_cues=[cue for source, cue in rows if source == "speech"],
                ocr_context=[cue for source, cue in context_rows if source == "ocr"],
                speech_context=[cue for source, cue in context_rows if source == "speech"],
            )
        )
    return batches


def _normalize_fused_batches(rows: list[tuple[int, Cue]]) -> list[Cue]:
    ordered = sorted(
        rows,
        key=lambda item: (
            item[1].start_ms,
            item[1].layer,
            item[1].track_id,
            item[0],
            item[1].cue_id,
        ),
    )
    group_ids: dict[tuple[int, str], str] = {}
    normalized: list[Cue] = []
    for index, (batch_index, cue) in enumerate(ordered, start=1):
        group_id = ""
        if cue.group_id:
            group_key = (batch_index, cue.group_id)
            group_id = group_ids.setdefault(
                group_key, f"G{len(group_ids) + 1:04d}"
            )
        normalized.append(
            cue.model_copy(
                update={
                    "cue_id": f"F{index:06d}",
                    "group_id": group_id,
                }
            )
        )
    return normalized


_REQUIRED_CUE_FIELDS = {
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
    "review_status",
}


def parse_fusion_response(value: str) -> list[Cue]:
    payload = _extract_json(value)
    rows = payload.get("cues")
    if not isinstance(rows, list):
        raise TranslationError("fusion response is missing cues")
    if not rows:
        raise TranslationError("fusion response contained no cues")

    cues: list[Cue] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TranslationError(f"fusion cue at index {index} must be an object")
        missing = sorted(_REQUIRED_CUE_FIELDS - row.keys())
        if missing:
            raise TranslationError(
                f"fusion cue at index {index} is missing fields: {missing}"
            )
        if not isinstance(row.get("source_text"), str) or not row["source_text"].strip():
            raise TranslationError(
                f"fusion cue at index {index} has empty source_text"
            )
        if row.get("target_text") not in (None, ""):
            raise TranslationError("fusion response must not contain translated text")

        cue_payload = {field: row[field] for field in _REQUIRED_CUE_FIELDS}
        cue_payload["target_text"] = ""
        try:
            cue = Cue.model_validate(cue_payload)
        except ValidationError as exc:
            raise TranslationError(
                f"fusion cue at index {index} is invalid: {exc.errors()}"
            ) from exc
        if cue.review_status not in {"ocr_ok", "needs_review"}:
            raise TranslationError(
                "fusion review_status must be ocr_ok or needs_review"
            )
        if cue.cue_id in seen_ids:
            raise TranslationError(
                f"fusion response contained duplicate cue_id: {cue.cue_id}"
            )
        seen_ids.add(cue.cue_id)
        cues.append(cue)

    return sorted(
        cues,
        key=lambda cue: (cue.start_ms, cue.layer, cue.track_id, cue.cue_id),
    )


class OpenAICompatibleFusionEngine:
    def __init__(
        self,
        provider: TranslationProvider,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.provider = provider
        self.transport = transport
        self.translator = OpenAICompatibleTranslator(provider, transport=transport)

    def fuse(
        self,
        manifest: ProjectManifest,
        ocr_cues: list[Cue],
        speech_cues: list[Cue],
        options: FusionOptions | None = None,
        progress: ProgressCallback | None = None,
        stream_event: ProviderEventCallback | None = None,
        checkpoint: CheckpointCallback | None = None,
        full_ocr_reference: list[Cue] | None = None,
        full_speech_reference: list[Cue] | None = None,
    ) -> list[Cue]:
        if not ocr_cues and not speech_cues:
            raise TranslationError("fusion requires OCR or speech cues")

        settings = options or FusionOptions()
        global_ocr_cues = (
            ocr_cues if full_ocr_reference is None else full_ocr_reference
        )
        global_speech_cues = (
            speech_cues if full_speech_reference is None else full_speech_reference
        )
        batches = _build_fusion_batches(
            ocr_cues,
            speech_cues,
            batch_size=settings.batch_size,
            context_cues=settings.context_cues,
        )
        if progress:
            progress(0.05)

        fused_batches: list[tuple[int, Cue]] = []
        batch_count = len(batches)
        total_evidence = sum(batch.evidence_count for batch in batches)
        completed_evidence = 0
        leaf_serial = 0

        def run_subset(
            batch: _FusionBatch,
            batch_index: int,
            subset_label: str,
        ) -> None:
            nonlocal completed_evidence, leaf_serial
            prompt = build_fusion_prompt(
                manifest,
                batch.ocr_cues,
                batch.speech_cues,
                ocr_context=batch.ocr_context,
                speech_context=batch.speech_context,
                batch_index=batch_index + 1,
                batch_count=batch_count,
                full_ocr_cues=global_ocr_cues,
                full_speech_cues=global_speech_cues,
                subset_label=subset_label,
            )
            last_error: TranslationError | None = None
            for attempt in range(settings.retries + 1):
                self.translator._emit_provider_event(
                    stream_event,
                    {
                        "type": "provider_request_started",
                        "operation": "fusion",
                        "subset": subset_label,
                        "ocr_cue_ids": [cue.cue_id for cue in batch.ocr_cues],
                        "speech_cue_ids": [cue.cue_id for cue in batch.speech_cues],
                        "attempt": attempt + 1,
                    },
                )
                try:
                    content = self.translator._request(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        stream_event=stream_event,
                    )
                    cues = parse_fusion_response(content)
                    break
                except TranslationError as exc:
                    last_error = exc
                    if (
                        isinstance(exc, TranslationRequestError)
                        and exc.should_split_batch
                    ):
                        split = _split_fusion_batch(batch)
                        if split is not None:
                            self.translator._emit_provider_event(
                                stream_event,
                                {
                                    "type": "batch_split",
                                    "operation": "fusion",
                                    "subset": subset_label,
                                    "reason": str(exc),
                                    "sizes": [
                                        split[0].evidence_count,
                                        split[1].evidence_count,
                                    ],
                                },
                            )
                            run_subset(split[0], batch_index, f"{subset_label}.1")
                            run_subset(split[1], batch_index, f"{subset_label}.2")
                            return
                    if attempt >= settings.retries:
                        raise
                    self.translator._emit_provider_event(
                        stream_event,
                        {
                            "type": "provider_retry",
                            "operation": "fusion",
                            "subset": subset_label,
                            "attempt": attempt + 2,
                            "reason": str(exc),
                        },
                    )
                    if settings.retry_backoff_seconds:
                        delay = settings.retry_backoff_seconds * min(2**attempt, 4)
                        time.sleep(delay)
            else:
                raise TranslationError(str(last_error or "fusion failed"))

            leaf_serial += 1
            fused_batches.extend((leaf_serial, cue) for cue in cues)
            completed_evidence += batch.evidence_count
            self.translator._emit_provider_event(
                stream_event,
                {
                    "type": "subset_completed",
                    "operation": "fusion",
                    "subset": subset_label,
                    "ocr_cue_ids": [cue.cue_id for cue in batch.ocr_cues],
                    "speech_cue_ids": [cue.cue_id for cue in batch.speech_cues],
                    "completed": completed_evidence,
                    "total": total_evidence,
                },
            )
            if checkpoint:
                checkpoint(_normalize_fused_batches(fused_batches))
            if progress:
                progress(
                    0.05
                    + 0.9
                    * (completed_evidence / max(total_evidence, 1))
                )

        for batch_index, batch in enumerate(batches):
            run_subset(batch, batch_index, str(batch_index + 1))

        cues = _normalize_fused_batches(fused_batches)
        if progress:
            progress(1.0)
        return cues


def fuse_cues(
    manifest: ProjectManifest,
    ocr_cues: list[Cue],
    speech_cues: list[Cue],
    provider: TranslationProvider,
    *,
    options: FusionOptions | None = None,
    transport: httpx.BaseTransport | None = None,
    progress: ProgressCallback | None = None,
    stream_event: ProviderEventCallback | None = None,
    checkpoint: CheckpointCallback | None = None,
    full_ocr_reference: list[Cue] | None = None,
    full_speech_reference: list[Cue] | None = None,
) -> list[Cue]:
    engine = OpenAICompatibleFusionEngine(provider, transport=transport)
    return engine.fuse(
        manifest,
        ocr_cues,
        speech_cues,
        options=options,
        progress=progress,
        stream_event=stream_event,
        checkpoint=checkpoint,
        full_ocr_reference=full_ocr_reference,
        full_speech_reference=full_speech_reference,
    )
