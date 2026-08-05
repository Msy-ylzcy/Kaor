import json

import httpx
import pytest

from backend.fusion import (
    SYSTEM_PROMPT,
    FusionOptions,
    OpenAICompatibleFusionEngine,
    build_fusion_prompt,
    parse_fusion_response,
)
from backend.models import Cue, ProjectManifest
from backend.translation import TranslationError, TranslationProvider


def sample_manifest() -> ProjectManifest:
    return ProjectManifest(
        project_id="project-1",
        title="Episode 7: Crossed Signals",
        video_filename="episode-07.mkv",
        source_language="ja",
        target_language="zh-CN",
        synopsis="Two pilots reunite during an evacuation.",
        genre_and_tone="tense science fiction",
        characters_context="Aki speaks formally; Ren interrupts when alarmed.",
        glossary_context="Keep the source terminology unchanged in this step.",
    )


def sample_ocr_cues() -> list[Cue]:
    return [
        Cue(
            cue_id="O000001",
            start_ms=1000,
            end_ms=3000,
            group_id="G0001",
            track_id="visual-main",
            speaker_id="SPK_A",
            speaker_name="Aki",
            speaker_color="#FFCC00",
            source_text="We need to leave now.",
            ocr_confidence=0.94,
            review_status="ocr_ok",
        ),
        Cue(
            cue_id="O000002",
            start_ms=1400,
            end_ms=2400,
            group_id="G0001",
            layer=1,
            track_id="visual-second",
            speaker_id="SPK_B",
            speaker_name="Ren",
            speaker_color="#00CCFF",
            source_text="0",
            ocr_confidence=0.18,
            review_status="needs_review",
        ),
    ]


def sample_speech_cues() -> list[Cue]:
    return [
        Cue(
            cue_id="S000001",
            start_ms=980,
            end_ms=2920,
            track_id="speech-a",
            speaker_id="SPK_A",
            speaker_name="Aki",
            source_kind="speech",
            source_text="We need to leave now",
            ocr_confidence=0.96,
            review_status="ocr_ok",
        ),
        Cue(
            cue_id="S000002",
            start_ms=1380,
            end_ms=2300,
            layer=1,
            track_id="speech-b",
            speaker_id="SPK_B",
            speaker_name="Ren",
            source_kind="speech",
            source_text="Wait for me",
            ocr_confidence=0.91,
            review_status="ocr_ok",
        ),
    ]


def fused_row(
    cue_id: str = "F000001",
    *,
    start_ms: int = 1000,
    end_ms: int = 3000,
    layer: int = 0,
    track_id: str = "main-a",
    speaker_id: str = "SPK_A",
    speaker_name: str = "Aki",
    speaker_color: str = "#FFCC00",
    source_text: str = "We need to leave now.",
) -> dict:
    return {
        "cue_id": cue_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "group_id": "G0001",
        "layer": layer,
        "track_id": track_id,
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "speaker_color": speaker_color,
        "source_kind": "imported",
        "source_text": source_text,
        "ocr_confidence": 0.95,
        "review_status": "ocr_ok",
    }


def provider_response(cues: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"cues": cues}, ensure_ascii=False)
                }
            }
        ]
    }


def test_prompt_labels_both_streams_and_includes_project_context():
    prompt = build_fusion_prompt(
        sample_manifest(), sample_ocr_cues(), sample_speech_cues()
    )

    assert "OCR_CSV" in SYSTEM_PROMPT
    assert "SPEECH_CSV" in SYSTEM_PROMPT
    assert "fade-in" in SYSTEM_PROMPT
    assert "homophone" in SYSTEM_PROMPT
    assert "staggered overlays" in SYSTEM_PROMPT
    assert "Never\ntranslate" in SYSTEM_PROMPT
    assert '"OCR_CSV"' in prompt
    assert '"SPEECH_CSV"' in prompt
    assert "Episode 7: Crossed Signals" in prompt
    assert "Two pilots reunite during an evacuation." in prompt
    assert '"source_language": "ja"' in prompt
    assert '"future_translation_target_not_for_this_step": "zh-CN"' in prompt
    assert "sampling duplicates" in prompt
    assert "phonetic substitutions" in prompt


def test_engine_returns_deterministic_order_and_preserves_staggered_overlap():
    second = fused_row(
        "F000002",
        start_ms=1400,
        end_ms=2400,
        layer=1,
        track_id="main-b",
        speaker_id="SPK_B",
        speaker_name="Ren",
        speaker_color="#00CCFF",
        source_text="Wait for me!",
    )
    first = fused_row()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][0]["content"] == SYSTEM_PROMPT
        assert "OCR_CSV" in body["messages"][1]["content"]
        assert "SPEECH_CSV" in body["messages"][1]["content"]
        assert "Episode 7: Crossed Signals" in body["messages"][1]["content"]
        return httpx.Response(200, json=provider_response([second, first]))

    engine = OpenAICompatibleFusionEngine(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="secret",
            model="fusion-model",
        ),
        transport=httpx.MockTransport(handler),
    )
    result = engine.fuse(
        sample_manifest(),
        sample_ocr_cues(),
        sample_speech_cues(),
        FusionOptions(retries=0),
    )

    assert [cue.cue_id for cue in result] == ["F000001", "F000002"]
    assert result[0].start_ms == 1000
    assert result[0].end_ms == 3000
    assert result[1].start_ms == 1400
    assert result[1].end_ms == 2400
    assert result[0].group_id == result[1].group_id == "G0001"
    assert result[0].layer == 0
    assert result[1].layer == 1
    assert result[0].speaker_id != result[1].speaker_id
    assert all(cue.target_text == "" for cue in result)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "did not contain JSON"),
        ('{"cues":[]}', "contained no cues"),
        (json.dumps({"cues": [{"cue_id": "only-an-id"}]}), "missing fields"),
        (
            json.dumps(
                {
                    "cues": [
                        fused_row("F000001"),
                        fused_row("F000001", start_ms=3100, end_ms=4000),
                    ]
                }
            ),
            "duplicate cue_id",
        ),
        (
            json.dumps(
                {"cues": [fused_row(start_ms=1000, end_ms=1000)]}
            ),
            "is invalid",
        ),
        (
            json.dumps({"cues": [fused_row(source_text=" ")]}),
            "empty source_text",
        ),
        (
            json.dumps(
                {"cues": [{**fused_row(), "target_text": "translated text"}]}
            ),
            "must not contain translated text",
        ),
    ],
)
def test_parser_rejects_malformed_empty_duplicate_and_invalid_rows(
    payload: str, message: str
):
    with pytest.raises(TranslationError, match=message):
        parse_fusion_response(payload)


def test_engine_retries_invalid_response_and_reports_progress():
    attempts = 0
    progress_values: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, json=provider_response([]))
        return httpx.Response(200, json=provider_response([fused_row()]))

    engine = OpenAICompatibleFusionEngine(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="secret",
            model="fusion-model",
        ),
        transport=httpx.MockTransport(handler),
    )
    result = engine.fuse(
        sample_manifest(),
        sample_ocr_cues(),
        sample_speech_cues(),
        FusionOptions(retries=1, retry_backoff_seconds=0),
        progress_values.append,
    )

    assert attempts == 2
    assert result[0].cue_id == "F000001"
    assert progress_values[0] == 0.05
    assert progress_values[-1] == 1.0
    assert progress_values == sorted(progress_values)


def test_engine_batches_distant_timeline_groups_and_sends_adjacent_context():
    first_ocr = sample_ocr_cues()[0]
    first_speech = sample_speech_cues()[0]
    second_ocr = first_ocr.model_copy(
        update={"cue_id": "O000010", "start_ms": 10_000, "end_ms": 12_000}
    )
    second_speech = first_speech.model_copy(
        update={"cue_id": "S000010", "start_ms": 10_020, "end_ms": 11_950}
    )
    prompts: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = json.loads(body["messages"][1]["content"])
        prompts.append(prompt)
        if prompt["batch"]["index"] == 1:
            row = fused_row(start_ms=1000, end_ms=3000, source_text="First")
        else:
            row = fused_row(start_ms=10_000, end_ms=12_000, source_text="Second")
        return httpx.Response(200, json=provider_response([row]))

    engine = OpenAICompatibleFusionEngine(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="secret",
            model="fusion-model",
        ),
        transport=httpx.MockTransport(handler),
    )
    result = engine.fuse(
        sample_manifest(),
        [first_ocr, second_ocr],
        [first_speech, second_speech],
        FusionOptions(batch_size=2, context_cues=2, retries=0),
    )

    assert len(prompts) == 2
    assert [prompt["batch"]["index"] for prompt in prompts] == [1, 2]
    assert all(prompt["batch"]["count"] == 2 for prompt in prompts)
    assert "O000010" in prompts[0]["OCR_CONTEXT_CSV"]
    assert "S000001" in prompts[1]["SPEECH_CONTEXT_CSV"]
    assert [cue.cue_id for cue in result] == ["F000001", "F000002"]
    assert [cue.source_text for cue in result] == ["First", "Second"]


def test_engine_rejects_two_empty_evidence_streams_before_request():
    engine = OpenAICompatibleFusionEngine(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="secret",
            model="fusion-model",
        ),
        transport=httpx.MockTransport(
            lambda request: pytest.fail("request should not be sent")
        ),
    )

    with pytest.raises(TranslationError, match="requires OCR or speech cues"):
        engine.fuse(sample_manifest(), [], [])
