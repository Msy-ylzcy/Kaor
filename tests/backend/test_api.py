from __future__ import annotations

import csv
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.csv_io import CUE_COLUMNS
from backend.models import Cue


def cue_payload(cue_id: str = "000001", **overrides):
    payload = {
        "cue_id": cue_id,
        "start_ms": 1000,
        "end_ms": 2500,
        "group_id": "overlap-1",
        "layer": 0,
        "track_id": "subtitle-primary",
        "speaker_id": "SPK_01",
        "speaker_name": "Alice",
        "speaker_color": "#f4d35e",
        "source_kind": "ocr",
        "source_text": "Hello, world",
        "ocr_confidence": 0.98,
        "target_text": "你好，世界",
        "review_status": "translated",
    }
    payload.update(overrides)
    return payload


def create_project(client: TestClient) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "title": "Demo",
            "video_filename": "demo.mp4",
            "source_language": "en",
            "target_language": "zh-CN",
            "synopsis": "A demo story",
            "glossary": {"Kaor": "烤肉"},
        },
    )
    assert response.status_code == 201
    return response.json()


def test_health_and_config(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["version"] == "0.2.0"

        config = {
            "api_base_url": "https://api.example.test/v1",
            "api_model": "model-name",
            "api_reasoning_effort": "high",
            "api_path": "/chat/completions",
            "custom_headers": {},
            "request_timeout_seconds": 30,
            "translation_batch_size": 50,
            "static_directory": "static",
        }
        assert client.put("/api/config", json=config).json() == config
        assert client.get("/api/config").json() == config
        assert "api_key" not in (tmp_path / "config.json").read_text("utf-8")


def test_reset_workspace_preserves_media_and_rebuilds_mix(tmp_path, monkeypatch):
    video_path = tmp_path / "fixture.mp4"
    video_path.write_bytes(b"source-video")

    monkeypatch.setattr(
        "backend.app.probe_video",
        lambda _path: SimpleNamespace(
            duration_ms=4321,
            width=1280,
            height=720,
            frame_rate=30.0,
        ),
    )

    def fake_extract(_source, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF" + b"0" * 64)
        return output

    monkeypatch.setattr("backend.app.extract_audio_track", fake_extract)
    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        created = client.post(
            "/api/projects",
            json={
                "title": "Reset fixture",
                "video_filename": video_path.name,
                "video_path": str(video_path),
            },
        ).json()
        project_id = created["project_id"]
        client.post(
            f"/api/projects/{project_id}/cues",
            json=cue_payload(),
        )
        project_dir = tmp_path / "data" / "projects" / project_id
        (project_dir / "ocr.csv").write_text("generated", encoding="utf-8")
        (project_dir / "exports").mkdir()
        (project_dir / "exports" / "render.mp4").write_bytes(b"render")
        finished = app.state.jobs.submit(
            project_id, "ocr", lambda _progress, _cancel: {}
        )
        while app.state.jobs.get(finished["id"])["status"] not in {
            "completed",
            "failed",
        }:
            pass

        response = client.post(f"/api/projects/{project_id}/reset")

        assert response.status_code == 200
        assert response.json()["audio_ready"] is True
        assert video_path.read_bytes() == b"source-video"
        assert client.get(f"/api/projects/{project_id}/cues").json() == []
        assert not (project_dir / "ocr.csv").exists()
        assert not (project_dir / "exports").exists()
        assert (project_dir / "cache" / "audio" / "mix.wav").is_file()
        assert app.state.jobs.list(project_id) == []


def test_old_config_defaults_reasoning_effort(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"api_base_url":"https://relay.example/v1","api_model":"legacy"}',
        encoding="utf-8",
    )

    with TestClient(create_app(tmp_path)) as client:
        config = client.get("/api/config")

    assert config.status_code == 200
    assert config.json()["api_reasoning_effort"] == ""


def test_translation_models_endpoint_and_reasoning_profile(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://relay.example/v1/models"
        return httpx.Response(
            200,
            json={"data": [{"id": "model-b"}, {"id": "model-a"}]},
        )

    with TestClient(
        create_app(tmp_path, translation_transport=httpx.MockTransport(handler))
    ) as client:
        project_id = create_project(client)["project_id"]
        models = client.post(
            "/api/translation/models",
            json={
                "base_url": "https://relay.example/v1",
                "api_key": "",
                "custom_headers": {},
                "timeout_seconds": 30,
            },
        )
        profile = client.put(
            f"/api/projects/{project_id}/translation-profile",
            json={
                "base_url": "https://relay.example/v1",
                "api_key": "",
                "model": "model-a",
                "path": "/chat/completions",
                "custom_headers": "{}",
                "timeout_seconds": 30,
                "concurrency": 1,
                "reasoning_effort": "high",
                "send_title": True,
                "send_story_context": True,
                "send_character_profiles": True,
                "send_glossary": True,
            },
        )

        saved_profile = client.get(
            f"/api/projects/{project_id}/translation-profile"
        ).json()
        saved_config = client.get("/api/config").json()

    assert models.status_code == 200
    assert [model["id"] for model in models.json()["models"]] == [
        "model-a",
        "model-b",
    ]
    assert profile.status_code == 200
    assert saved_profile["reasoning_effort"] == "high"
    assert saved_config["api_reasoning_effort"] == "high"


def test_ocr_capabilities_api(tmp_path, monkeypatch):
    from backend.ocr_engines import OcrRuntimeCapabilities

    monkeypatch.setattr(
        "backend.app.detect_ocr_capabilities",
        lambda: OcrRuntimeCapabilities(
            paddle_available=True,
            paddle_version="3.3.1",
            paddleocr_available=True,
            paddleocr_version="3.7.0",
            cuda_compiled=False,
            cuda_device_count=0,
            cuda_device_names=(),
        ),
    )
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/ocr/capabilities")
        assert response.status_code == 200
        payload = response.json()
        assert payload["default_device"] == "cpu"
        assert payload["cpu_onednn_enabled"] is False
        assert payload["devices"][-1]["id"] == "cuda:0"
        assert payload["devices"][-1]["available"] is False


def test_ocr_capabilities_api_preserves_runtime_import_error(tmp_path, monkeypatch):
    from backend.ocr_engines import OcrRuntimeCapabilities

    detail = r"PermissionError: denied: C:\Users\User\.cache\paddle"
    monkeypatch.setattr(
        "backend.app.detect_ocr_capabilities",
        lambda: OcrRuntimeCapabilities(
            paddle_available=False,
            paddle_version="3.3.1",
            paddleocr_available=True,
            paddleocr_version="3.7.0",
            cuda_compiled=False,
            cuda_device_count=0,
            cuda_device_names=(),
            error=detail,
        ),
    )
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/ocr/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["error"] == detail
    assert payload["devices"][0]["available"] is False
    assert payload["devices"][0]["reason"] == detail


def test_project_crud_persists_manifest_and_csv(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project = create_project(client)
        project_id = project["project_id"]

        listed = client.get("/api/projects").json()
        assert [item["project_id"] for item in listed] == [project_id]

        updated = client.patch(
            f"/api/projects/{project_id}", json={"title": "Renamed"}
        )
        assert updated.status_code == 200
        assert updated.json()["title"] == "Renamed"

        language = client.patch(
            f"/api/projects/{project_id}", json={"target_language": "ja"}
        )
        assert language.status_code == 200
        assert language.json()["target_language"] == "ja"

        project_dir = tmp_path / "projects" / project_id
        assert (project_dir / "manifest.json").exists()
        assert (project_dir / "source.csv").exists()

        deleted = client.delete(f"/api/projects/{project_id}")
        assert deleted.status_code == 204
        assert client.get(f"/api/projects/{project_id}").status_code == 404


def test_cue_crud_and_csv_columns(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project_id = create_project(client)["project_id"]
        cue = cue_payload()

        created = client.post(f"/api/projects/{project_id}/cues", json=cue)
        assert created.status_code == 201
        assert created.json()["speaker_color"] == "#F4D35E"
        assert client.post(f"/api/projects/{project_id}/cues", json=cue).status_code == 409

        replacement = cue_payload(source_text="Updated", layer=1)
        response = client.put(
            f"/api/projects/{project_id}/cues/000001", json=replacement
        )
        assert response.status_code == 200
        assert response.json()["source_text"] == "Updated"

        csv_path = tmp_path / "projects" / project_id / "source.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == CUE_COLUMNS
            assert list(reader)[0]["group_id"] == "overlap-1"

        assert client.delete(
            f"/api/projects/{project_id}/cues/000001"
        ).status_code == 204
        assert client.get(f"/api/projects/{project_id}/cues").json() == []


def test_cue_color_update_persists_to_source_and_translated_csv(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        project_id = create_project(client)["project_id"]
        source = Cue.model_validate(cue_payload(target_text=""))
        client.post(f"/api/projects/{project_id}/cues", json=source.model_dump())
        app.state.storage.save_translated_cues(
            project_id,
            [source.model_copy(update={"target_text": "你好", "review_status": "translated"})],
        )

        response = client.patch(
            f"/api/projects/{project_id}/cues/000001/color",
            json={"speaker_color": "#12abef"},
        )

        assert response.status_code == 200
        assert response.json()["speaker_color"] == "#12ABEF"
        assert app.state.storage.list_cues(project_id)[0].speaker_color == "#12ABEF"
        assert (
            app.state.storage.list_translated_cues(project_id)[0].speaker_color
            == "#12ABEF"
        )
        assert client.get("/api/workspace").json()["cues"][0]["speaker_color"] == "#12ABEF"

        invalid = client.patch(
            f"/api/projects/{project_id}/cues/000001/color",
            json={"speaker_color": "not-a-color"},
        )
        assert invalid.status_code == 422


def test_speaker_color_update_changes_all_matching_cues_only(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        project_id = create_project(client)["project_id"]
        source = [
            Cue.model_validate(cue_payload("000001", target_text="")),
            Cue.model_validate(cue_payload("000002", start_ms=3000, end_ms=4200, target_text="")),
            Cue.model_validate(
                cue_payload(
                    "000003",
                    start_ms=4300,
                    end_ms=5500,
                    speaker_id="SPK_02",
                    speaker_name="Bob",
                    speaker_color="#55AAFF",
                    target_text="",
                )
            ),
        ]
        app.state.storage.replace_cues(project_id, source)
        app.state.storage.save_translated_cues(
            project_id,
            [cue.model_copy(update={"target_text": f"T-{cue.cue_id}"}) for cue in source],
        )

        response = client.patch(
            f"/api/projects/{project_id}/speakers/color",
            json={
                "speaker_id": "SPK_01",
                "speaker_name": "Alice",
                "speaker_color": "#11cc88",
            },
        )

        assert response.status_code == 200
        assert {cue["cue_id"] for cue in response.json()} == {"000001", "000002"}
        for rows in (
            app.state.storage.list_cues(project_id),
            app.state.storage.list_translated_cues(project_id),
        ):
            colors = {cue.cue_id: cue.speaker_color for cue in rows}
            assert colors == {
                "000001": "#11CC88",
                "000002": "#11CC88",
                "000003": "#55AAFF",
            }


def test_manual_cue_create_update_delete_keeps_translation_table_in_sync(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        project_id = create_project(client)["project_id"]
        source = Cue.model_validate(cue_payload("000001", target_text=""))
        app.state.storage.replace_cues(project_id, [source])
        app.state.storage.save_translated_cues(
            project_id,
            [source.model_copy(update={"target_text": "已有译文"})],
        )
        manual = cue_payload(
            "MANUAL000001",
            start_ms=3000,
            end_ms=4500,
            group_id="",
            speaker_id="",
            speaker_name="",
            source_kind="manual",
            source_text="手动字幕",
            ocr_confidence=None,
            target_text="手动译文",
            review_status="pending",
        )

        created = client.post(f"/api/projects/{project_id}/cues", json=manual)
        assert created.status_code == 201
        assert len(app.state.storage.list_cues(project_id)) == 2
        assert len(app.state.storage.list_translated_cues(project_id)) == 2
        assert app.state.storage.list_cues(project_id)[1].target_text == ""
        assert app.state.storage.list_translated_cues(project_id)[1].target_text == "手动译文"

        manual["source_text"] = "校对后的手动字幕"
        manual["target_text"] = "校对后的手动译文"
        updated = client.put(
            f"/api/projects/{project_id}/cues/MANUAL000001", json=manual
        )
        assert updated.status_code == 200
        assert app.state.storage.list_cues(project_id)[1].source_text == "校对后的手动字幕"
        assert app.state.storage.list_translated_cues(project_id)[1].target_text == "校对后的手动译文"

        deleted = client.delete(f"/api/projects/{project_id}/cues/000001")
        assert deleted.status_code == 204
        assert [cue.cue_id for cue in app.state.storage.list_cues(project_id)] == ["MANUAL000001"]
        assert [cue.cue_id for cue in app.state.storage.list_translated_cues(project_id)] == ["MANUAL000001"]


def test_source_csv_download_is_independent_from_translation(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project_id = create_project(client)["project_id"]
        client.post(
            f"/api/projects/{project_id}/cues",
            json=cue_payload(target_text=""),
        )

        source = client.get(f"/api/projects/{project_id}/source.csv")

        assert source.status_code == 200
        assert "text/csv" in source.headers["content-type"]
        assert "source.csv" in source.headers["content-disposition"]
        assert source.content.startswith(b"\xef\xbb\xbf")
        assert b"Hello, world" in source.content
        assert client.get(f"/api/projects/{project_id}/translated.csv").status_code == 404


def test_batch_replace_preserves_staggered_overlapping_tracks(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project_id = create_project(client)["project_id"]
        cues = [
            cue_payload(
                "000002",
                start_ms=2800,
                end_ms=5000,
                layer=1,
                track_id="subtitle-secondary",
                speaker_id="SPK_02",
                speaker_name="Bob",
                source_text="B appears later",
            ),
            cue_payload(
                "000001",
                start_ms=1000,
                end_ms=5000,
                layer=0,
                track_id="subtitle-primary",
                source_text="A remains visible",
            ),
        ]
        response = client.put(
            f"/api/projects/{project_id}/cues", json={"cues": cues}
        )
        assert response.status_code == 200
        saved = response.json()
        assert [cue["cue_id"] for cue in saved] == ["000001", "000002"]
        assert {cue["group_id"] for cue in saved} == {"overlap-1"}
        assert [cue["layer"] for cue in saved] == [0, 1]
        assert [cue["track_id"] for cue in saved] == [
            "subtitle-primary",
            "subtitle-secondary",
        ]
        assert [(cue["start_ms"], cue["end_ms"]) for cue in saved] == [
            (1000, 5000),
            (2800, 5000),
        ]


def test_validation_and_missing_resources(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/projects/not-a-valid-id").status_code == 404
        project_id = create_project(client)["project_id"]
        invalid = cue_payload(end_ms=999)
        assert client.post(
            f"/api/projects/{project_id}/cues", json=invalid
        ).status_code == 422
        mismatch = client.put(
            f"/api/projects/{project_id}/cues/different", json=cue_payload()
        )
        assert mismatch.status_code == 400


def test_static_mount_and_restart_persistence(tmp_path):
    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True)
    (static_dir / "ready.txt").write_text("ready", encoding="utf-8")

    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/static/ready.txt").text == "ready"
        project_id = create_project(client)["project_id"]
        assert client.post(
            f"/api/projects/{project_id}/cues", json=cue_payload()
        ).status_code == 201

    with TestClient(create_app(tmp_path)) as restarted_client:
        project = restarted_client.get(f"/api/projects/{project_id}")
        assert project.status_code == 200
        cues = restarted_client.get(f"/api/projects/{project_id}/cues")
        assert cues.status_code == 200
        assert cues.json()[0]["cue_id"] == "000001"
