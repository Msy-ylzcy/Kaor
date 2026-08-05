import json

import httpx
import pytest

from backend.fusion import FusionOptions, OpenAICompatibleFusionEngine
from backend.models import Cue, ProjectManifest
from backend.translation import (
    OpenAICompatibleTranslator,
    TranslationOptions,
    TranslationProvider,
)


def cue(cue_id: str, start_ms: int, text: str, *, source_kind: str = "ocr") -> Cue:
    return Cue(
        cue_id=cue_id,
        start_ms=start_ms,
        end_ms=start_ms + 900,
        track_id="main",
        source_kind=source_kind,
        source_text=text,
        ocr_confidence=0.9,
        review_status="ocr_ok",
    )


def provider() -> TranslationProvider:
    return TranslationProvider(
        base_url="https://relay.example/v1",
        api_key="secret",
        model="fixture-model",
    )


@pytest.mark.parametrize("failure", ["524", "timeout"])
def test_translation_adaptively_splits_timeout_and_keeps_full_reference(failure):
    source = [cue(f"C{index}", index * 1000, f"line {index}") for index in range(4)]
    prompts: list[dict] = []
    failed = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal failed
        body = json.loads(request.content)
        prompt = json.loads(body["messages"][1]["content"])
        prompts.append(prompt)
        ids = prompt["output_contract"]["cue_ids"]
        if len(ids) == 4 and not failed:
            failed = True
            if failure == "timeout":
                raise httpx.ReadTimeout("origin was slow", request=request)
            return httpx.Response(524, text="origin timeout")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {"cue_id": cue_id, "target_text": f"T-{cue_id}"}
                                        for cue_id in ids
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    checkpoints: list[list[Cue]] = []
    progress: list[float] = []
    events: list[dict] = []
    result = OpenAICompatibleTranslator(
        provider(), transport=httpx.MockTransport(handler)
    ).translate(
        ProjectManifest(project_id="p", title="Demo"),
        source,
        TranslationOptions(batch_size=4, retries=0),
        progress.append,
        events.append,
        lambda rows: checkpoints.append(rows),
    )

    assert [item.target_text for item in result] == [
        "T-C0",
        "T-C1",
        "T-C2",
        "T-C3",
    ]
    assert [prompt["output_contract"]["cue_ids"] for prompt in prompts] == [
        ["C0", "C1", "C2", "C3"],
        ["C0", "C1"],
        ["C2", "C3"],
    ]
    assert all(
        all(cue_id in prompt["ALL_SOURCE_CSV_READ_ONLY"] for cue_id in ["C0", "C1", "C2", "C3"])
        for prompt in prompts
    )
    assert progress == [0.5, 1.0]
    assert len(checkpoints) == 2
    assert checkpoints[0][0].target_text == "T-C0"
    assert checkpoints[0][2].target_text == ""
    assert any(event["type"] == "batch_split" for event in events)


def test_provider_stream_reports_reasoning_and_content_deltas():
    stream = "\n".join(
        [
            'data: {"choices":[{"delta":{"reasoning_content":"checking "}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"context"}}]}',
            'data: {"choices":[{"delta":{"content":"{\\"translations\\":["}}]}',
            'data: {"choices":[{"delta":{"content":"{\\"cue_id\\":\\"C1\\",\\"target_text\\":\\"done\\"}] }"}}]}',
            "data: [DONE]",
            "",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=stream,
        )

    events: list[dict] = []
    result = OpenAICompatibleTranslator(
        provider(), transport=httpx.MockTransport(handler)
    ).translate(
        ProjectManifest(project_id="p", title="Demo"),
        [cue("C1", 0, "hello")],
        TranslationOptions(retries=0),
        stream_event=events.append,
    )

    deltas = [event for event in events if event["type"] == "provider_delta"]
    assert [event["field"] for event in deltas] == [
        "reasoning_content",
        "reasoning_content",
        "content",
        "content",
    ]
    assert deltas[-1]["reasoning_content"] == "checking context"
    assert result[0].target_text == "done"


def test_translation_can_resume_subset_with_complete_reference_table():
    source = [cue(f"C{index}", index * 1000, f"line {index}") for index in range(3)]

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert prompt["output_contract"]["cue_ids"] == ["C2"]
        assert all(
            cue_id in prompt["ALL_SOURCE_CSV_READ_ONLY"]
            for cue_id in ["C0", "C1", "C2"]
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {"cue_id": "C2", "target_text": "finished"}
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    result = OpenAICompatibleTranslator(
        provider(), transport=httpx.MockTransport(handler)
    ).translate(
        ProjectManifest(project_id="p", title="Demo"),
        source[2:],
        TranslationOptions(retries=0),
        reference_cues=source,
    )

    assert [item.cue_id for item in result] == ["C2"]
    assert result[0].target_text == "finished"


def test_fusion_split_requests_keep_both_complete_evidence_tables():
    ocr = [
        cue("O1", 1000, "first"),
        cue("O2", 10_000, "second"),
    ]
    speech = [
        cue("S1", 1020, "first", source_kind="speech"),
        cue("S2", 10_020, "second", source_kind="speech"),
    ]
    starts = {item.cue_id: item.start_ms for item in [*ocr, *speech]}
    prompts: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(json.loads(request.content)["messages"][1]["content"])
        prompts.append(prompt)
        ids = [
            *prompt["output_contract"]["OCR_cue_ids"],
            *prompt["output_contract"]["SPEECH_cue_ids"],
        ]
        if len(ids) == 4:
            return httpx.Response(524, text="origin timeout")
        start_ms = min(starts[cue_id] for cue_id in ids)
        row = {
            "cue_id": "temporary",
            "start_ms": start_ms,
            "end_ms": start_ms + 900,
            "group_id": "",
            "layer": 0,
            "track_id": "main",
            "speaker_id": "",
            "speaker_name": "",
            "speaker_color": "#FFFFFF",
            "source_kind": "imported",
            "source_text": "corrected",
            "ocr_confidence": 0.95,
            "review_status": "ocr_ok",
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"cues": [row]})}}
                ]
            },
        )

    checkpoints: list[list[Cue]] = []
    result = OpenAICompatibleFusionEngine(
        provider(), transport=httpx.MockTransport(handler)
    ).fuse(
        ProjectManifest(project_id="p", title="Demo"),
        ocr,
        speech,
        FusionOptions(batch_size=10, retries=0),
        checkpoint=lambda rows: checkpoints.append(rows),
    )

    assert len(result) == 2
    assert len(checkpoints) == 2
    assert all(
        all(cue_id in prompt["FULL_OCR_CSV_READ_ONLY"] for cue_id in ["O1", "O2"])
        and all(cue_id in prompt["FULL_SPEECH_CSV_READ_ONLY"] for cue_id in ["S1", "S2"])
        for prompt in prompts
    )
    assert [
        len(prompt["output_contract"]["OCR_cue_ids"])
        + len(prompt["output_contract"]["SPEECH_cue_ids"])
        for prompt in prompts
    ] == [4, 2, 2]
