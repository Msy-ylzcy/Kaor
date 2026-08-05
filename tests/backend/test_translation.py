import json

import httpx
import pytest

from backend.models import Cue, ProjectManifest
from backend.translation import (
    OpenAICompatibleTranslator,
    SYSTEM_PROMPT,
    TranslationError,
    TranslationOptions,
    TranslationProvider,
    build_user_prompt,
    parse_translation_response,
)


def sample_cues():
    return [
        Cue(
            cue_id="000001",
            start_ms=1000,
            end_ms=3000,
            source_text="Hello",
            ocr_confidence=0.99,
        ),
        Cue(
            cue_id="000002",
            start_ms=2200,
            end_ms=4200,
            group_id="G0001",
            layer=1,
            speaker_id="SPK_02",
            source_text="Wait!",
            ocr_confidence=0.61,
        ),
    ]


def test_prompt_contains_title_context_target_language_and_csv():
    manifest = ProjectManifest(
        project_id="p",
        title="Episode 7",
        synopsis="A reunion",
        target_language="ja",
    )
    prompt = build_user_prompt(
        manifest, sample_cues(), [], [], TranslationOptions(max_chars_per_line=20)
    )

    assert "Episode 7" in prompt
    assert "A reunion" in prompt
    assert "Translate every source_text in csv_to_translate into ja" in prompt
    assert '"target_language": "ja"' in prompt
    assert "cue_id,start_ms,end_ms" in prompt
    assert "ocr_confidence" in prompt
    assert "0.61" in prompt
    assert "source_text is OCR output and may be inaccurate" in prompt
    assert "Wait!" in prompt


def test_parser_rejects_missing_or_duplicate_ids():
    with pytest.raises(TranslationError, match="mismatch"):
        parse_translation_response(
            '{"translations":[{"cue_id":"000001","target_text":"x"}]}',
            ["000001", "000002"],
        )
    with pytest.raises(TranslationError, match="duplicate"):
        parse_translation_response(
            '{"translations":[{"cue_id":"000001","target_text":"x"},'
            '{"cue_id":"000001","target_text":"y"}]}',
            ["000001"],
        )


def test_translation_forbids_and_sanitizes_line_break_markers():
    assert "Never output /N" in SYSTEM_PROMPT
    items = parse_translation_response(
        json.dumps(
            {
                "translations": [
                    {"cue_id": "000001", "target_text": "第一行/N第二行"},
                    {"cue_id": "000002", "target_text": "A\\NB<br>B\nC"},
                ]
            },
            ensure_ascii=False,
        ),
        ["000001", "000002"],
    )

    assert [item.target_text for item in items] == ["第一行 第二行", "A B B C"]
    assert all("/N" not in item.target_text and "\\N" not in item.target_text for item in items)


def test_openai_compatible_translation_preserves_timing_and_overlap():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["model"] == "relay-model"
        assert body["reasoning_effort"] == "high"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {"cue_id": "000001", "target_text": "你好"},
                                        {"cue_id": "000002", "target_text": "等等！"},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    translator = OpenAICompatibleTranslator(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="secret",
            model="relay-model",
            reasoning_effort="high",
        ),
        transport=httpx.MockTransport(handler),
    )
    source = sample_cues()
    result = translator.translate(
        ProjectManifest(project_id="p", title="Demo"),
        source,
        TranslationOptions(batch_size=10, retries=0),
    )

    assert [cue.target_text for cue in result] == ["你好", "等等！"]
    assert result[1].group_id == "G0001"
    assert result[1].start_ms == 2200
    assert source[0].target_text == ""


def test_model_list_uses_upstream_endpoint_and_returns_sorted_unique_models():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://relay.example/v1/models"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "zeta", "owned_by": "relay"},
                    {"id": "alpha", "owned_by": "upstream"},
                    {"id": "alpha", "owned_by": "duplicate"},
                    {"object": "model"},
                ]
            },
        )

    translator = OpenAICompatibleTranslator(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="secret",
            model="",
        ),
        transport=httpx.MockTransport(handler),
    )

    models = translator.list_models()

    assert [model.id for model in models] == ["alpha", "zeta"]
    assert models[0].owned_by == "upstream"


@pytest.mark.parametrize(
    "payload",
    [
        {"models": [{"id": "relay-model", "owned_by": "relay"}]},
        [{"id": "relay-model", "owned_by": "relay"}],
    ],
)
def test_model_list_accepts_common_relay_response_shapes(payload):
    translator = OpenAICompatibleTranslator(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="",
            model="",
        ),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        ),
    )

    models = translator.list_models()

    assert [(model.id, model.owned_by) for model in models] == [
        ("relay-model", "relay")
    ]


def test_model_list_http_error_redacts_credentials():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            text="rejected secret-key and relay-token",
        )

    translator = OpenAICompatibleTranslator(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="secret-key",
            model="",
            custom_headers={"X-Relay-Token": "relay-token"},
        ),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(TranslationError) as captured:
        translator.list_models()

    message = str(captured.value)
    assert "HTTP 401" in message
    assert "[redacted]" in message
    assert "secret-key" not in message
    assert "relay-token" not in message


def test_model_list_invalid_json_includes_status_and_short_response():
    translator = OpenAICompatibleTranslator(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="",
            model="",
        ),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="upstream returned HTML")
        ),
    )

    with pytest.raises(TranslationError) as captured:
        translator.list_models()

    message = str(captured.value)
    assert "HTTP 200" in message
    assert "upstream returned HTML" in message


def test_translation_invalid_json_includes_status_and_short_response():
    translator = OpenAICompatibleTranslator(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="",
            model="relay-model",
        ),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="upstream returned HTML")
        ),
    )

    with pytest.raises(TranslationError) as captured:
        translator.test_connection()

    message = str(captured.value)
    assert "HTTP 200" in message
    assert "upstream returned HTML" in message


def test_translation_http_error_redacts_credentials():
    translator = OpenAICompatibleTranslator(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="secret-key",
            model="relay-model",
            custom_headers={"X-Relay-Token": "relay-token"},
        ),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                502, text="rejected secret-key and relay-token"
            )
        ),
    )

    with pytest.raises(TranslationError) as captured:
        translator.test_connection()

    message = str(captured.value)
    assert "HTTP 502" in message
    assert "[redacted]" in message
    assert "secret-key" not in message
    assert "relay-token" not in message


def test_translation_applies_ocr_correction_only_to_translated_copy():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user_prompt = body["messages"][1]["content"]
        assert "ocr_confidence" in user_prompt
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {
                                            "cue_id": "000001",
                                            "target_text": "你好",
                                            "source_correction": None,
                                            "uncertain": False,
                                        },
                                        {
                                            "cue_id": "000002",
                                            "target_text": "等等！",
                                            "source_correction": "Wait!",
                                            "uncertain": False,
                                        },
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    translator = OpenAICompatibleTranslator(
        TranslationProvider(
            base_url="https://relay.example/v1",
            api_key="secret",
            model="relay-model",
        ),
        transport=httpx.MockTransport(handler),
    )
    source = sample_cues()
    source[1].source_text = "Walt!"

    result = translator.translate(
        ProjectManifest(project_id="p", title="Demo"),
        source,
        TranslationOptions(batch_size=10, retries=0),
    )

    assert source[1].source_text == "Walt!"
    assert result[1].source_text == "Wait!"
    assert result[1].target_text == "等等！"
    assert result[1].review_status == "needs_review"
