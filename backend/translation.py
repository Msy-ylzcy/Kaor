from __future__ import annotations

import csv
import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
from urllib.parse import urljoin

import httpx

from .csv_io import CUE_COLUMNS
from .models import Cue, ProjectManifest, TranslationModelInfo


SYSTEM_PROMPT = """You are a professional audiovisual subtitle translator and consistency editor.

Translate source_text from the supplied CSV into the requested target language. The source
rows may come from OCR and can contain recognition mistakes. Use ocr_confidence together
with the project title, synopsis, character profiles, speaker identity, glossary, and
adjacent cues to resolve the intended source text, tone, address forms, pronouns, jokes,
and terminology.

Strict rules:
1. Return every cue_id requested by output_contract exactly once. Never merge, split, omit,
   reorder, invent, or return additional read-only reference cue IDs.
2. Do not change timing, speaker fields, tags, variables, numbers, or IDs.
3. Write concise, natural dialogue rather than a word-for-word rendering.
4. Preserve emotion, register, honorifics, profanity, humor, and character voice.
5. Follow the glossary and keep names and terms consistent across the whole batch.
6. Return target_text as plain single-line text. Never output /N, \\N, a literal \\n,
   newline characters, HTML <br> tags, or any other line-break control marker. Line wrapping
   is handled locally during rendering. Respect the requested length without inserting breaks.
7. Treat low ocr_confidence as a warning, not proof that the row is wrong. Infer the most
   likely wording from adjacent cues and project context, then translate that corrected
   meaning. Do not alter clear high-confidence text without strong contextual evidence.
8. When you correct OCR, return the full corrected source in source_correction. If the
   intended wording is still ambiguous, set uncertain=true and explain why briefly.
9. Output valid JSON only, without Markdown or commentary.
10. ALL_SOURCE_CSV_READ_ONLY is the complete source subtitle table and is supplied only for
    global story, terminology, speaker, and continuity reference. Return translations only for
    the cue IDs listed in output_contract and csv_to_translate. Never emit rows merely because
    they appear in the read-only table.

Output schema:
{"translations":[{"cue_id":"000001","target_text":"...","source_correction":null,
"uncertain":false,"uncertainty_reason":null}]}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_LINE_BREAK_MARKER_RE = re.compile(
    r"\s*(?:[/\\][Nn]|<br\s*/?>|\r\n?|\n)\s*",
    re.IGNORECASE,
)


class TranslationError(RuntimeError):
    pass


class TranslationRequestError(TranslationError):
    """Provider transport/status failure with enough detail for adaptive batching."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.timed_out = timed_out

    @property
    def should_split_batch(self) -> bool:
        return self.timed_out or self.status_code == 524


ProviderEventCallback = Callable[[dict[str, Any]], None]
CheckpointCallback = Callable[[list[Cue]], None]


@dataclass(frozen=True)
class TranslationProvider:
    base_url: str
    api_key: str
    model: str
    api_path: str = "/chat/completions"
    custom_headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 120
    temperature: float = 0.2
    json_mode: bool = True
    reasoning_effort: str = ""

    def endpoint(self) -> str:
        base = self.base_url.rstrip("/") + "/"
        return urljoin(base, self.api_path.lstrip("/"))

    def models_endpoint(self) -> str:
        base = self.base_url.rstrip("/") + "/"
        return urljoin(base, "models")


@dataclass(frozen=True)
class TranslationOptions:
    max_lines: int = 2
    max_chars_per_line: int = 24
    batch_size: int = 80
    context_cues: int = 3
    retries: int = 2


@dataclass(frozen=True)
class TranslationItem:
    cue_id: str
    target_text: str
    source_correction: str | None = None
    uncertain: bool = False
    uncertainty_reason: str | None = None


def normalize_target_text(value: str) -> str:
    normalized = _LINE_BREAK_MARKER_RE.sub(" ", value.strip())
    return re.sub(r"[ \t]{2,}", " ", normalized).strip()


def cues_to_csv(cues: Iterable[Cue]) -> str:
    buffer = io.StringIO(newline="")
    columns = [
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
    ]
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for cue in cues:
        writer.writerow(cue.model_dump())
    return buffer.getvalue()


def _project_context(manifest: ProjectManifest) -> dict[str, Any]:
    return {
        "video_title": manifest.title,
        "video_filename": manifest.video_filename,
        "source_language": manifest.source_language,
        "target_language": manifest.target_language,
        "synopsis": manifest.synopsis,
        "genre_and_tone": manifest.genre_and_tone,
        "characters_context": manifest.characters_context,
        "glossary_context": manifest.glossary_context,
        "character_profiles": [
            profile.model_dump() for profile in manifest.character_profiles
        ],
        "glossary": manifest.glossary,
    }


def build_user_prompt(
    manifest: ProjectManifest,
    batch: list[Cue],
    previous: list[Cue],
    following: list[Cue],
    options: TranslationOptions,
    *,
    all_cues: list[Cue] | None = None,
    subset_label: str = "1",
) -> str:
    global_reference = all_cues
    if global_reference is None:
        by_id = {
            cue.cue_id: cue for cue in [*previous, *batch, *following]
        }
        global_reference = list(by_id.values())
    sections = {
        "translation_directive": (
            f"Translate every source_text in csv_to_translate into "
            f"{manifest.target_language or 'zh-CN'}."
        ),
        "project_context": _project_context(manifest),
        "layout_constraints": {
            "max_lines": options.max_lines,
            "max_chars_per_line": options.max_chars_per_line,
        },
        "output_contract": {
            "subset_label": subset_label,
            "cue_ids": [cue.cue_id for cue in batch],
            "instruction": (
                "Return exactly these cue IDs. ALL_SOURCE_CSV_READ_ONLY is context only."
            ),
        },
        "ocr_guidance": {
            "description": (
                "source_text is OCR output and may be inaccurate; use context to "
                "proofread it before translating"
            ),
            "low_confidence_threshold": 0.82,
            "instructions": (
                "Prioritize contextual review for rows below the threshold. Translate "
                "the corrected intended meaning, return source_correction when changed, "
                "and mark uncertain when context is insufficient."
            ),
        },
        "previous_context_csv": cues_to_csv(previous),
        "csv_to_translate": cues_to_csv(batch),
        "next_context_csv": cues_to_csv(following),
        "ALL_SOURCE_CSV_READ_ONLY": cues_to_csv(global_reference),
    }
    return json.dumps(sections, ensure_ascii=False, indent=2)


def _extract_json(value: str) -> dict[str, Any]:
    cleaned = _FENCE_RE.sub("", value.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise TranslationError("translation response did not contain JSON")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise TranslationError("translation response contained invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise TranslationError("translation response root must be an object")
    return parsed


def parse_translation_response(value: str, expected_ids: list[str]) -> list[TranslationItem]:
    payload = _extract_json(value)
    rows = payload.get("translations")
    if not isinstance(rows, list):
        raise TranslationError("translation response is missing translations")
    items: list[TranslationItem] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TranslationError("translation entries must be objects")
        cue_id = str(row.get("cue_id", ""))
        target_text = row.get("target_text")
        if not cue_id or not isinstance(target_text, str):
            raise TranslationError("translation entry is missing cue_id or target_text")
        items.append(
            TranslationItem(
                cue_id=cue_id,
                target_text=normalize_target_text(target_text),
                source_correction=(
                    str(row["source_correction"]).strip()
                    if row.get("source_correction")
                    else None
                ),
                uncertain=bool(row.get("uncertain", False)),
                uncertainty_reason=(
                    str(row["uncertainty_reason"]).strip()
                    if row.get("uncertainty_reason")
                    else None
                ),
            )
        )
    actual_ids = [item.cue_id for item in items]
    if len(actual_ids) != len(set(actual_ids)):
        raise TranslationError("translation response contained duplicate cue IDs")
    if set(actual_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        raise TranslationError(f"translation ID mismatch; missing={missing}, extra={extra}")
    by_id = {item.cue_id: item for item in items}
    return [by_id[cue_id] for cue_id in expected_ids]


class OpenAICompatibleTranslator:
    def __init__(
        self,
        provider: TranslationProvider,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.provider = provider
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **self.provider.custom_headers,
        }
        if self.provider.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.provider.api_key}"
        return headers

    def _redact_provider_secrets(self, value: str) -> str:
        redacted = value
        secrets = [self.provider.api_key, *self.provider.custom_headers.values()]
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, "[redacted]")
        return re.sub(
            r"(?i)(bearer\s+)[^\s,;\"']+",
            r"\1[redacted]",
            redacted,
        )

    def _response_preview(self, response: httpx.Response, limit: int = 500) -> str:
        preview = self._redact_provider_secrets(response.text.strip())
        if not preview:
            return "<empty response>"
        if len(preview) > limit:
            return preview[:limit] + "..."
        return preview

    @staticmethod
    def _provider_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text", item.get("content"))
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _emit_provider_event(
        callback: ProviderEventCallback | None,
        event: dict[str, Any],
    ) -> None:
        if callback is not None:
            callback(event)

    def _content_from_payload(
        self,
        payload: Any,
        stream_event: ProviderEventCallback | None = None,
    ) -> str:
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TranslationError(
                "provider response did not contain message content"
            ) from exc
        if not isinstance(message, dict):
            raise TranslationError("provider response did not contain message content")
        reasoning = self._provider_text(
            message.get("reasoning_content", message.get("reasoning"))
        )
        content = self._provider_text(message.get("content"))
        if reasoning:
            self._emit_provider_event(
                stream_event,
                {
                    "type": "provider_delta",
                    "field": "reasoning_content",
                    "delta": reasoning,
                    "reasoning_content": reasoning,
                    "content": "",
                },
            )
        if content:
            self._emit_provider_event(
                stream_event,
                {
                    "type": "provider_delta",
                    "field": "content",
                    "delta": content,
                    "reasoning_content": reasoning,
                    "content": content,
                },
            )
        if not content:
            raise TranslationError("provider returned non-text message content")
        return content

    def _content_from_event_stream(
        self,
        response: httpx.Response,
        stream_event: ProviderEventCallback,
    ) -> str:
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for line in response.iter_lines():
            stripped = line.strip()
            if not stripped or stripped.startswith(":"):
                continue
            if not stripped.startswith("data:"):
                continue
            data = stripped[5:].strip()
            if data == "[DONE]":
                break
            try:
                event_payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise TranslationError(
                    "provider stream contained invalid JSON"
                ) from exc
            if isinstance(event_payload, dict) and event_payload.get("error"):
                detail = self._redact_provider_secrets(str(event_payload["error"]))
                raise TranslationError(f"translation request failed: {detail}")
            try:
                choice = event_payload["choices"][0]
            except (KeyError, IndexError, TypeError):
                continue
            delta = choice.get("delta", choice.get("message", {}))
            if not isinstance(delta, dict):
                continue
            reasoning_delta = self._provider_text(
                delta.get("reasoning_content", delta.get("reasoning"))
            )
            content_delta = self._provider_text(delta.get("content"))
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
                self._emit_provider_event(
                    stream_event,
                    {
                        "type": "provider_delta",
                        "field": "reasoning_content",
                        "delta": reasoning_delta,
                        "reasoning_content": "".join(reasoning_parts),
                        "content": "".join(content_parts),
                    },
                )
            if content_delta:
                content_parts.append(content_delta)
                self._emit_provider_event(
                    stream_event,
                    {
                        "type": "provider_delta",
                        "field": "content",
                        "delta": content_delta,
                        "reasoning_content": "".join(reasoning_parts),
                        "content": "".join(content_parts),
                    },
                )
        content = "".join(content_parts)
        if not content:
            raise TranslationError("provider stream did not contain message content")
        return content

    def list_models(self) -> list[TranslationModelInfo]:
        try:
            with httpx.Client(
                timeout=self.provider.timeout_seconds,
                transport=self.transport,
                follow_redirects=True,
            ) as client:
                response = client.get(
                    self.provider.models_endpoint(), headers=self._headers()
                )
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as exc:
                    preview = self._response_preview(response)
                    raise TranslationError(
                        "model list request failed: "
                        f"HTTP {response.status_code}: response was not valid JSON: "
                        f"{preview}"
                    ) from exc
        except httpx.HTTPStatusError as exc:
            preview = self._response_preview(exc.response)
            raise TranslationError(
                f"model list request failed: HTTP {exc.response.status_code}: {preview}"
            ) from exc
        except httpx.HTTPError as exc:
            detail = self._redact_provider_secrets(str(exc))
            raise TranslationError(f"model list request failed: {detail}") from exc

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("data", payload.get("models"))
        else:
            rows = None
        if not isinstance(rows, list):
            preview = self._response_preview(response)
            raise TranslationError(
                "model list request failed: "
                f"HTTP {response.status_code}: response did not contain a model array: "
                f"{preview}"
            )

        by_id: dict[str, TranslationModelInfo] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_id = row.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            normalized_id = model_id.strip()
            owned_by = row.get("owned_by")
            if not isinstance(owned_by, str):
                owned_by = None
            by_id.setdefault(
                normalized_id,
                TranslationModelInfo(id=normalized_id, owned_by=owned_by),
            )
        return [by_id[model_id] for model_id in sorted(by_id, key=str.casefold)]

    def _request(
        self,
        messages: list[dict[str, str]],
        stream_event: ProviderEventCallback | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.provider.model,
            "messages": messages,
            "temperature": self.provider.temperature,
        }
        if self.provider.reasoning_effort:
            body["reasoning_effort"] = self.provider.reasoning_effort
        if self.provider.json_mode:
            body["response_format"] = {"type": "json_object"}
        if stream_event is not None:
            body["stream"] = True
        try:
            with httpx.Client(
                timeout=self.provider.timeout_seconds,
                transport=self.transport,
                follow_redirects=True,
            ) as client:
                if stream_event is not None:
                    with client.stream(
                        "POST",
                        self.provider.endpoint(),
                        headers=self._headers(),
                        json=body,
                    ) as response:
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "").lower()
                        if "text/event-stream" in content_type:
                            return self._content_from_event_stream(response, stream_event)
                        try:
                            payload = json.loads(response.read())
                        except (ValueError, json.JSONDecodeError) as exc:
                            preview = self._response_preview(response)
                            raise TranslationError(
                                "translation request failed: "
                                f"HTTP {response.status_code}: response was not valid JSON: "
                                f"{preview}"
                            ) from exc
                else:
                    response = client.post(
                        self.provider.endpoint(), headers=self._headers(), json=body
                    )
                    response.raise_for_status()
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        preview = self._response_preview(response)
                        raise TranslationError(
                            "translation request failed: "
                            f"HTTP {response.status_code}: response was not valid JSON: "
                            f"{preview}"
                        ) from exc
        except httpx.HTTPStatusError as exc:
            preview = self._response_preview(exc.response)
            raise TranslationRequestError(
                f"translation request failed: HTTP {exc.response.status_code}: {preview}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.TimeoutException as exc:
            detail = self._redact_provider_secrets(str(exc)) or "request timed out"
            raise TranslationRequestError(
                f"translation request failed: {detail}", timed_out=True
            ) from exc
        except httpx.HTTPError as exc:
            detail = self._redact_provider_secrets(str(exc))
            raise TranslationRequestError(
                f"translation request failed: {detail}"
            ) from exc
        return self._content_from_payload(payload, stream_event)

    def test_connection(self) -> str:
        result = self._request(
            [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": '{"status":"reply with ok"}'},
            ]
        )
        return result

    def translate(
        self,
        manifest: ProjectManifest,
        cues: list[Cue],
        options: TranslationOptions | None = None,
        progress: Any | None = None,
        stream_event: ProviderEventCallback | None = None,
        checkpoint: CheckpointCallback | None = None,
        reference_cues: list[Cue] | None = None,
    ) -> list[Cue]:
        settings = options or TranslationOptions()
        translated = [cue.model_copy(deep=True) for cue in cues]
        cue_positions = {cue.cue_id: index for index, cue in enumerate(cues)}
        global_cues = list(reference_cues) if reference_cues is not None else list(cues)
        global_ids = {cue.cue_id for cue in global_cues}
        global_cues.extend(cue for cue in cues if cue.cue_id not in global_ids)
        global_positions = {
            cue.cue_id: index for index, cue in enumerate(global_cues)
        }
        completed = 0

        def run_subset(batch: list[Cue], subset_label: str) -> None:
            nonlocal completed
            first = global_positions[batch[0].cue_id]
            last = global_positions[batch[-1].cue_id] + 1
            previous = global_cues[
                max(0, first - settings.context_cues) : first
            ]
            following = global_cues[last : last + settings.context_cues]
            prompt = build_user_prompt(
                manifest,
                batch,
                previous,
                following,
                settings,
                all_cues=global_cues,
                subset_label=subset_label,
            )
            expected_ids = [cue.cue_id for cue in batch]
            last_error: Exception | None = None
            for attempt in range(settings.retries + 1):
                self._emit_provider_event(
                    stream_event,
                    {
                        "type": "provider_request_started",
                        "operation": "translation",
                        "subset": subset_label,
                        "cue_ids": expected_ids,
                        "attempt": attempt + 1,
                    },
                )
                try:
                    content = self._request(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        stream_event=stream_event,
                    )
                    items = parse_translation_response(content, expected_ids)
                    break
                except TranslationError as exc:
                    last_error = exc
                    if (
                        isinstance(exc, TranslationRequestError)
                        and exc.should_split_batch
                        and len(batch) > 1
                    ):
                        midpoint = len(batch) // 2
                        self._emit_provider_event(
                            stream_event,
                            {
                                "type": "batch_split",
                                "operation": "translation",
                                "subset": subset_label,
                                "reason": str(exc),
                                "sizes": [midpoint, len(batch) - midpoint],
                            },
                        )
                        run_subset(batch[:midpoint], f"{subset_label}.1")
                        run_subset(batch[midpoint:], f"{subset_label}.2")
                        return
                    if attempt >= settings.retries:
                        raise
                    self._emit_provider_event(
                        stream_event,
                        {
                            "type": "provider_retry",
                            "operation": "translation",
                            "subset": subset_label,
                            "attempt": attempt + 2,
                            "reason": str(exc),
                        },
                    )
                    time.sleep(min(2**attempt, 4))
            else:
                raise TranslationError(str(last_error or "translation failed"))

            for item in items:
                cue = translated[cue_positions[item.cue_id]]
                if item.source_correction:
                    cue.source_text = item.source_correction
                cue.target_text = item.target_text
                cue.review_status = (
                    "needs_review"
                    if item.uncertain or item.source_correction
                    else "translated"
                )
            completed += len(batch)
            self._emit_provider_event(
                stream_event,
                {
                    "type": "subset_completed",
                    "operation": "translation",
                    "subset": subset_label,
                    "cue_ids": expected_ids,
                    "completed": completed,
                    "total": len(cues),
                },
            )
            if checkpoint:
                checkpoint([cue.model_copy(deep=True) for cue in translated])
            if progress:
                progress(min(1.0, completed / max(len(cues), 1)))

        for start in range(0, len(cues), settings.batch_size):
            batch = cues[start : start + settings.batch_size]
            run_subset(batch, str(start // settings.batch_size + 1))
        return translated
