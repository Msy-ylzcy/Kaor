from __future__ import annotations

import hashlib
import json
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import audio_pipeline, speech_pipeline
from backend.app import create_app
from backend.audio_pipeline import download_asr_model, recommended_asr_model
from backend.models import Cue, ProjectCreate
from backend.storage import Storage


def make_cue(
    cue_id: str,
    text: str,
    *,
    source_kind: str = "ocr",
    start_ms: int = 1000,
    end_ms: int = 2200,
) -> Cue:
    return Cue(
        cue_id=cue_id,
        start_ms=start_ms,
        end_ms=end_ms,
        track_id="speech" if source_kind == "speech" else "visual-main",
        source_kind=source_kind,
        source_text=text,
        ocr_confidence=0.93,
        review_status="ocr_ok",
    )


def create_video_project(client: TestClient, tmp_path: Path) -> str:
    video_path = tmp_path / "fixture.mp4"
    video_path.write_bytes(b"local video fixture")
    response = client.post(
        "/api/projects",
        json={
            "title": "Fixture episode",
            "video_filename": video_path.name,
            "video_path": str(video_path),
            "duration_ms": 5000,
            "source_language": "ja",
            "target_language": "zh-CN",
        },
    )
    assert response.status_code == 201
    return response.json()["project_id"]


def wait_for_job(client: TestClient, job_id: str, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not finish: {job_id}")


def install_fake_audio_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    calls: dict[str, object] = {}
    model_dir = tmp_path / "downloaded-asr-model"
    model_dir.mkdir()

    monkeypatch.setattr("backend.app.asr_model_installed", lambda *_: False)

    def fake_download(model, models_root, progress=None):
        calls["download"] = (model.id, Path(models_root))
        if progress:
            progress(1.0, "fixture model ready")
        return model_dir

    def fake_extract(video_path, output_path):
        calls["extract"] = Path(video_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"R" * 96)
        return Path(output_path)

    def fake_worker(
        input_path,
        output_dir,
        audio_path,
        model,
        model_path,
        *,
        device,
        separate_vocals,
        forced_alignment,
        diarization,
        progress,
        cancel_event,
        **stage_options,
    ):
        calls["separate"] = (Path(input_path), device, separate_vocals)
        calls["timing"] = (forced_alignment, diarization)
        calls["stage_options"] = stage_options
        calls["recognize"] = (Path(audio_path), model.id, Path(model_path), device)
        progress(1.0, "fixture speech ready")
        return [make_cue("ASR000001", "spoken evidence", source_kind="speech")]

    monkeypatch.setattr("backend.app.download_asr_model", fake_download)
    monkeypatch.setattr("backend.app.extract_audio_track", fake_extract)
    monkeypatch.setattr("backend.app.run_speech_worker", fake_worker)
    return calls


def fusion_request() -> dict:
    return {
        "provider": {
            "base_url": "https://relay.example.test/v1",
            "api_key": "fixture-key",
            "model": "fixture-fusion-model",
            "api_path": "/chat/completions",
            "timeout_seconds": 5,
        },
        "options": {"retries": 0, "retry_backoff_seconds": 0},
    }


def test_audio_capabilities_api_is_offline_and_marks_recommended_models(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audio_pipeline, "resolve_uvr_assets", lambda: (None, None, "missing fixture"))
    monkeypatch.setattr(
        audio_pipeline,
        "detect_torch_runtime",
        lambda: (False, None, False, 0, [], None),
    )
    monkeypatch.setattr(audio_pipeline, "_module_available", lambda _name: False)
    monkeypatch.setattr(audio_pipeline, "find_binary", lambda _name: None)
    monkeypatch.setattr(audio_pipeline, "asr_model_installed", lambda *_args: False)

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/audio/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ffmpeg_available"] is False
    assert payload["torch_available"] is False
    assert payload["cuda_available"] is False
    assert payload["default_device"] == "cpu"
    assert payload["uvr_model"]["available"] is False
    assert payload["uvr_model"]["load_mode"] == "local-or-auto-download"
    assert payload["uvr_model"]["download_size_mb"] == 610
    assert "diarization_model" in payload
    assert payload["errors"] == ["missing fixture"]
    assert payload["asr_models"]
    assert all(model["recommended"] is True for model in payload["asr_models"])
    assert {model["language"] for model in payload["asr_models"]} >= {
        "ja",
        "en",
        "zh",
        "ko",
    }


def test_audio_model_catalog_does_not_probe_torch_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audio_pipeline,
        "detect_torch_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime probe must not run")),
    )
    monkeypatch.setattr(audio_pipeline, "asr_model_installed", lambda *_args: False)

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/audio/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert {model["language"] for model in payload} >= {"ja", "en", "zh", "ko"}
    assert all(model["installed"] is False for model in payload)


def test_language_model_download_uses_injected_local_snapshot(tmp_path, monkeypatch):
    calls: dict[str, object] = {}
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id, local_dir, local_dir_use_symlinks):
        calls["request"] = (repo_id, Path(local_dir), local_dir_use_symlinks)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "fixture.nemo").write_bytes(b"model")

    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    model = recommended_asr_model("ja-JP")

    installed = download_asr_model(model, tmp_path / "models")

    assert calls["request"][0] == model.repository
    assert calls["request"][2] is False
    marker = json.loads((installed / ".kaor-model.json").read_text("utf-8"))
    assert marker["id"] == model.id
    assert marker["repository"] == model.repository
    assert audio_pipeline.asr_model_installed(tmp_path / "models", model) is True


def test_uvr_checkpoint_download_resumes_and_installs_atomically(tmp_path, monkeypatch):
    payload = b"hello world"
    monkeypatch.setattr(audio_pipeline, "UVR_EXPECTED_SIZE", len(payload))
    monkeypatch.setattr(
        audio_pipeline, "UVR_EXPECTED_SHA256", hashlib.sha256(payload).hexdigest()
    )
    destination = tmp_path / audio_pipeline.UVR_MODEL_FILENAME
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(payload[:6])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url) == audio_pipeline.UVR_MODEL_URL
        assert request.headers["range"] == "bytes=6-"
        return httpx.Response(
            206,
            content=payload[6:],
            headers={
                "content-length": str(len(payload) - 6),
                "content-range": "bytes 6-10/11",
            },
        )

    events: list[tuple[float, str]] = []
    installed = audio_pipeline.ensure_uvr_checkpoint(
        tmp_path,
        progress=lambda value, message: events.append((value, message)),
        transport=httpx.MockTransport(handler),
    )

    assert installed == destination.resolve()
    assert installed.read_bytes() == payload
    assert not partial.exists()
    assert len(requests) == 1
    assert events[-1] == (1.0, "BS-Roformer checkpoint downloaded and verified")


def test_uvr_config_download_uses_pinned_source_and_installs_atomically(
    tmp_path, monkeypatch
):
    payload = b"audio: {}\n"
    monkeypatch.setattr(audio_pipeline, "UVR_CONFIG_EXPECTED_SIZE", len(payload))
    monkeypatch.setattr(
        audio_pipeline,
        "UVR_CONFIG_EXPECTED_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == audio_pipeline.UVR_CONFIG_URL
        return httpx.Response(200, content=payload)

    installed = audio_pipeline.ensure_uvr_config(
        tmp_path, transport=httpx.MockTransport(handler)
    )

    assert installed == (tmp_path / audio_pipeline.UVR_CONFIG_FILENAME).resolve()
    assert installed.read_bytes() == payload
    assert not installed.with_suffix(installed.suffix + ".part").exists()


def test_uvr_checkpoint_hash_failure_does_not_replace_existing_file(
    tmp_path, monkeypatch
):
    expected = b"expected"
    existing = b"old-data"
    downloaded = b"tampered"
    monkeypatch.setattr(audio_pipeline, "UVR_EXPECTED_SIZE", len(expected))
    monkeypatch.setattr(
        audio_pipeline, "UVR_EXPECTED_SHA256", hashlib.sha256(expected).hexdigest()
    )
    destination = tmp_path / audio_pipeline.UVR_MODEL_FILENAME
    destination.write_bytes(existing)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=downloaded,
            headers={"content-length": str(len(downloaded))},
        )

    with pytest.raises(audio_pipeline.AudioPipelineError, match="SHA-256 mismatch"):
        audio_pipeline.ensure_uvr_checkpoint(
            tmp_path, transport=httpx.MockTransport(handler)
        )

    assert destination.read_bytes() == existing
    assert not destination.with_suffix(destination.suffix + ".part").exists()


def test_uvr_checkpoint_reuses_a_verified_local_file_without_network(
    tmp_path, monkeypatch
):
    payload = b"verified"
    monkeypatch.setattr(audio_pipeline, "UVR_EXPECTED_SIZE", len(payload))
    monkeypatch.setattr(
        audio_pipeline, "UVR_EXPECTED_SHA256", hashlib.sha256(payload).hexdigest()
    )
    destination = tmp_path / audio_pipeline.UVR_MODEL_FILENAME
    destination.write_bytes(payload)

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("a verified checkpoint must not use the network")

    installed = audio_pipeline.ensure_uvr_checkpoint(
        tmp_path, transport=httpx.MockTransport(handler)
    )

    assert installed == destination.resolve()


def test_speech_pipeline_separation_adapter_uses_local_assets(tmp_path, monkeypatch):
    input_wav = tmp_path / "mix.wav"
    input_wav.write_bytes(b"M" * 96)
    model_path = tmp_path / audio_pipeline.UVR_MODEL_FILENAME
    config_path = tmp_path / audio_pipeline.UVR_CONFIG_FILENAME
    model_path.write_bytes(b"model")
    config_path.write_text("audio: {}\n", encoding="utf-8")
    observed: dict[str, object] = {}

    class FakeSeparator:
        def __init__(self, **kwargs):
            observed["init"] = kwargs
            self.output_dir = Path(kwargs["output_dir"])

        def load_model(self, *, model_filename):
            observed["model_filename"] = model_filename

        def separate(self, input_path, custom_output_names=None):
            observed["input_path"] = input_path
            observed["custom_output_names"] = custom_output_names
            output = self.output_dir / "fixture-vocals.wav"
            output.write_bytes(b"V" * 96)
            return [str(output)]

    package = types.ModuleType("audio_separator")
    package.__path__ = []
    separator_module = types.ModuleType("audio_separator.separator")
    separator_module.Separator = FakeSeparator
    monkeypatch.setitem(sys.modules, "audio_separator", package)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", separator_module)
    monkeypatch.setattr(
        speech_pipeline,
        "resolve_uvr_assets",
        lambda: (model_path, config_path, None),
    )
    monkeypatch.setattr(
        speech_pipeline, "ensure_uvr_checkpoint", lambda **_kwargs: model_path
    )
    monkeypatch.setattr(
        speech_pipeline, "ensure_uvr_config", lambda **_kwargs: config_path
    )
    monkeypatch.setattr(speech_pipeline, "_resolve_device", lambda _value: "cuda:0")

    vocals = speech_pipeline.separate_vocals(
        input_wav, tmp_path / "separated", device="auto"
    )

    assert vocals == (tmp_path / "separated" / "vocals.wav").resolve()
    assert vocals.read_bytes() == b"V" * 96
    assert observed["model_filename"] == audio_pipeline.UVR_MODEL_FILENAME
    assert observed["init"]["use_autocast"] is True


def test_speech_pipeline_dispatches_without_importing_pytorch(tmp_path, monkeypatch):
    model = recommended_asr_model("en")
    expected = [make_cue("ASR000001", "hello", source_kind="speech")]
    observed: dict[str, object] = {}
    monkeypatch.setattr(speech_pipeline, "_resolve_device", lambda _value: "cpu")

    def fake_nemo(audio_path, model_dir, device, progress):
        observed["args"] = (audio_path, model_dir, device, progress)
        return expected

    monkeypatch.setattr(speech_pipeline, "_recognize_nemo", fake_nemo)

    result = speech_pipeline.recognize_speech(
        tmp_path / "speech.wav",
        model,
        tmp_path / "model",
        device="auto",
        forced_alignment=False,
        diarization=False,
    )

    assert result == expected
    assert observed["args"][2] == "cpu"


def test_speech_pipeline_applies_local_diarization(tmp_path, monkeypatch):
    model = recommended_asr_model("en")
    expected = [make_cue("ASR000001", "hello", source_kind="speech")]
    expected[0].speaker_id = ""
    expected[0].speaker_name = ""
    expected[0].speaker_color = "#FFFFFF"
    assigned = expected[0].model_copy(
        update={
            "speaker_id": "SPK_01",
            "speaker_name": "Speaker 1",
            "speaker_color": "#F4D35E",
        }
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(speech_pipeline, "_resolve_device", lambda _value: "cuda:0")
    monkeypatch.setattr(speech_pipeline, "_recognize_nemo", lambda *_args: expected)

    def fake_diarize(audio_path, cues, output_dir, **kwargs):
        observed["args"] = (audio_path, cues, output_dir, kwargs)
        return type(
            "Result",
            (),
            {
                "success": True,
                "cues": (assigned,),
                "stats": type(
                    "Stats", (), {"assigned_cues": 1, "speaker_count": 1}
                )(),
                "error": "",
            },
        )()

    monkeypatch.setattr(speech_pipeline, "diarize_cues", fake_diarize)
    output_dir = tmp_path / "diarization"

    result = speech_pipeline.recognize_speech(
        tmp_path / "speech.wav",
        model,
        tmp_path / "model",
        device="auto",
        forced_alignment=False,
        diarization=True,
        diarization_output_dir=output_dir,
    )

    assert result == [assigned]
    assert observed["args"][2] == output_dir
    assert observed["args"][3]["device"] == "cuda:0"


def test_speech_confidence_rejects_raw_sequence_scores():
    assert speech_pipeline._confidence(-1256.0755615234375) is None
    assert speech_pipeline._confidence(1.01) is None
    assert speech_pipeline._confidence(0.87654) == 0.8765


def test_nemo_confidence_is_aggregated_per_timed_segment():
    hypothesis = types.SimpleNamespace(
        score=-1256.0,
        word_confidence=[0.92, 0.74, 0.88],
        token_confidence=[0.99],
        timestamp={
            "word": [
                {"word": "first", "start": 0.0, "end": 0.4},
                {"word": "line", "start": 0.4, "end": 0.8},
                {"word": "second", "start": 1.0, "end": 1.5},
            ]
        },
    )

    confidences = speech_pipeline._nemo_segment_confidences(
        hypothesis,
        [(0.0, 0.8, "first line"), (1.0, 1.5, "second")],
    )

    assert confidences == [0.83, 0.88]


def test_nemo_confidence_does_not_fall_back_to_hypothesis_score():
    hypothesis = types.SimpleNamespace(
        score=-1256.0,
        word_confidence=None,
        token_confidence=None,
        timestamp={"segment": [{"segment": "text", "start": 0.0, "end": 1.0}]},
    )

    assert speech_pipeline._nemo_segment_confidences(
        hypothesis, [(0.0, 1.0, "text")]
    ) == [None]


def test_nemo_native_confidence_is_enabled_when_decoder_supports_it(monkeypatch):
    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    def attr_dict(value):
        if isinstance(value, dict):
            return AttrDict({key: attr_dict(item) for key, item in value.items()})
        return value

    fake_omegaconf = types.ModuleType("omegaconf")
    fake_omegaconf.OmegaConf = types.SimpleNamespace(
        to_container=lambda value, resolve=False: dict(value),
        create=attr_dict,
    )
    fake_omegaconf.open_dict = nullcontext
    monkeypatch.setitem(sys.modules, "omegaconf", fake_omegaconf)

    observed: dict[str, object] = {}

    class FakeModel:
        cfg = types.SimpleNamespace(decoding=AttrDict(strategy="greedy_batch"))

        def change_decoding_strategy(self, decoding_cfg, verbose=True):
            observed["config"] = decoding_cfg
            observed["verbose"] = verbose

    assert speech_pipeline._enable_nemo_confidence(FakeModel()) is True
    config = observed["config"]
    assert config.compute_timestamps is True
    assert config.confidence_cfg["preserve_frame_confidence"] is True
    assert config.confidence_cfg["preserve_token_confidence"] is True
    assert config.confidence_cfg["preserve_word_confidence"] is True
    assert config.confidence_cfg["exclude_blank"] is True
    assert config.confidence_cfg["aggregation"] == "mean"
    assert config.confidence_cfg["method_cfg"]["name"] == "max_prob"
    assert config.confidence_cfg["method_cfg"]["alpha"] == 1.0
    assert observed["verbose"] is False


def test_storage_keeps_raw_evidence_independent_until_promoted(tmp_path):
    store = Storage(tmp_path)
    project = store.create_project(ProjectCreate(title="Evidence fixture"))
    original = make_cue("MANUAL001", "manually reviewed", source_kind="manual")
    translated = original.model_copy(
        update={"target_text": "reviewed translation", "review_status": "translated"}
    )
    store.replace_cues(project.project_id, [original])
    store.save_translated_cues(project.project_id, [translated])

    ocr = make_cue("OCR000001", "visual evidence")
    speech = make_cue("ASR000001", "spoken evidence", source_kind="speech")
    store.save_ocr_cues(project.project_id, [ocr], promote=False)
    store.save_speech_cues(project.project_id, [speech], promote=False)

    assert store.list_ocr_cues(project.project_id) == [ocr]
    assert store.list_speech_cues(project.project_id) == [speech]
    assert store.list_cues(project.project_id) == [original]
    assert store.list_translated_cues(project.project_id) == [translated]


def test_audio_job_promotes_speech_csv_to_source(tmp_path, monkeypatch):
    calls = install_fake_audio_runtime(monkeypatch, tmp_path)
    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        project_id = create_video_project(client, tmp_path)
        app.state.storage.replace_cues(
            project_id, [make_cue("OLD001", "old source", source_kind="manual")]
        )

        response = client.post(
            f"/api/projects/{project_id}/audio-jobs",
            json={
                "language": "ja",
                "model_id": "ja-parakeet-tdt-ctc-0.6b",
                "device": "cuda:0",
                "separate_vocals": True,
            },
        )
        assert response.status_code == 202
        job = wait_for_job(client, response.json()["id"])

        assert job["status"] == "completed", job
        assert job["result"]["source"] == "speech.csv"
        assert app.state.storage.list_speech_cues(project_id)[0].source_text == "spoken evidence"
        assert app.state.storage.list_cues(project_id)[0].source_text == "spoken evidence"
        assert client.get(f"/api/projects/{project_id}/speech.csv").status_code == 200

    assert calls["download"][0] == "ja-parakeet-tdt-ctc-0.6b"
    assert calls["separate"][1] == "cuda:0"
    assert calls["timing"] == (True, True)


def test_hybrid_job_preserves_source_and_writes_both_evidence_tables(
    tmp_path, monkeypatch
):
    calls = install_fake_audio_runtime(monkeypatch, tmp_path)
    ocr = make_cue("OCR000001", "visual evidence")
    monkeypatch.setattr("backend.app.PaddleOcrEngine", lambda **_kwargs: object())
    monkeypatch.setattr("backend.app.recognize_video", lambda *_args, **_kwargs: [ocr])

    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        project_id = create_video_project(client, tmp_path)
        original = make_cue("MANUAL001", "approved source", source_kind="manual")
        app.state.storage.replace_cues(project_id, [original])

        response = client.post(
            f"/api/projects/{project_id}/hybrid-jobs",
            json={
                "language": "ja",
                "model_id": "ja-parakeet-tdt-ctc-0.6b",
                "device": "cuda:0",
                "separate_vocals": True,
                "prefer_embedded": False,
                "high_accuracy": False,
            },
        )
        assert response.status_code == 202
        job = wait_for_job(client, response.json()["id"])

        assert job["status"] == "completed", job
        assert job["result"]["artifacts"] == ["ocr.csv", "speech.csv"]
        assert app.state.storage.list_ocr_cues(project_id)[0].source_text == "visual evidence"
        assert app.state.storage.list_speech_cues(project_id)[0].source_text == "spoken evidence"
        assert app.state.storage.list_cues(project_id) == [original]
        assert ("ja", "cuda:0") in app.state.frame_ocr_engines
        assert client.get(f"/api/projects/{project_id}/ocr.csv").status_code == 200
        assert client.get(f"/api/projects/{project_id}/speech.csv").status_code == 200

    assert calls["timing"] == (True, True)


def test_independent_uvr_slicer_and_asr_jobs_reuse_stage_artifacts(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr("backend.app.asr_model_installed", lambda *_args: True)
    monkeypatch.setattr("backend.app.asr_model_directory", lambda *_args: model_dir)

    def fake_uvr(input_wav, output_dir, **options):
        assert Path(input_wav).name == "mix.wav"
        options["progress"](0.5, "fixture UVR")
        vocals = Path(output_dir) / "vocals.wav"
        vocals.write_bytes(b"V" * 96)
        return vocals

    def fake_transcode(_source, destination):
        Path(destination).write_bytes(b"A" * 96)
        return Path(destination)

    def fake_slicer(_source, output_dir, **options):
        options["progress"](0.5, "fixture slicer")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "slices.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return types.SimpleNamespace(
            manifest_path=manifest_path,
            slices=(object(), object()),
            duration_ms=5_000,
            to_dict=lambda: {"parameters": {"min_interval_ms": 200}},
        )

    def fake_asr(*_args, **options):
        options["progress"](0.5, "fixture ASR 1/2")
        return [make_cue("ASR000001", "spoken evidence", source_kind="speech")]

    monkeypatch.setattr("backend.app.run_uvr_worker", fake_uvr)
    monkeypatch.setattr("backend.app.transcode_audio_for_asr", fake_transcode)
    monkeypatch.setattr("backend.app.run_slicer_stage", fake_slicer)
    monkeypatch.setattr("backend.app.run_asr_worker", fake_asr)

    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        project_id = create_video_project(client, tmp_path)
        audio_dir = app.state.storage.projects_dir / project_id / "cache" / "audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "mix.wav").write_bytes(b"M" * 96)

        uvr = client.post(
            f"/api/projects/{project_id}/uvr-jobs",
            json={"device": "cuda:0"},
        )
        assert wait_for_job(client, uvr.json()["id"])["status"] == "completed"

        slicer = client.post(
            f"/api/projects/{project_id}/slicer-jobs",
            json={
                "slicer_threshold_db": -34,
                "slicer_min_length_ms": 4_000,
                "slicer_min_interval_ms": 200,
                "slicer_hop_size_ms": 10,
                "slicer_max_sil_kept_ms": 500,
            },
        )
        slicer_job = wait_for_job(client, slicer.json()["id"])
        assert slicer_job["status"] == "completed"
        assert slicer_job["result"]["slice_count"] == 2

        asr = client.post(
            f"/api/projects/{project_id}/asr-jobs",
            json={
                "language": "ja",
                "model_id": "ja-parakeet-tdt-ctc-0.6b",
                "device": "cuda:0",
                "asr_batch_size": 4,
            },
        )
        asr_job = wait_for_job(client, asr.json()["id"])

        assert asr_job["status"] == "completed", asr_job
        assert asr_job["result"]["cue_count"] == 1
        assert app.state.storage.list_speech_cues(project_id)[0].source_text == "spoken evidence"
        assert client.get(f"/api/projects/{project_id}/speech.csv").status_code == 200


def test_slicer_job_requires_completed_uvr_artifact(tmp_path):
    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        project_id = create_video_project(client, tmp_path)
        response = client.post(
            f"/api/projects/{project_id}/slicer-jobs",
            json={"slicer_min_interval_ms": 200},
        )

    assert response.status_code == 409
    assert "run UVR5" in response.json()["detail"]


def test_audio_job_fails_instead_of_saving_an_empty_speech_table(
    tmp_path, monkeypatch
):
    install_fake_audio_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr("backend.app.run_speech_worker", lambda *_args, **_kwargs: [])
    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        project_id = create_video_project(client, tmp_path)
        response = client.post(
            f"/api/projects/{project_id}/audio-jobs",
            json={
                "language": "ja",
                "model_id": "ja-parakeet-tdt-ctc-0.6b",
            },
        )
        job = wait_for_job(client, response.json()["id"])

    assert job["status"] == "failed"
    assert "0 cues" in job["error"]["detail"]
    assert not app.state.storage.speech_csv_path(project_id).exists()


def test_fusion_reports_existing_but_empty_evidence_tables(tmp_path):
    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        project_id = create_video_project(client, tmp_path)
        app.state.storage.save_ocr_cues(project_id, [], promote=False)
        app.state.storage.save_speech_cues(project_id, [], promote=False)

        response = client.post(
            f"/api/projects/{project_id}/fusion-jobs", json=fusion_request()
        )

    assert response.status_code == 409
    assert "empty=['ocr.csv', 'speech.csv']" in response.json()["detail"]


@pytest.mark.parametrize("present", ["ocr", "speech"])
def test_fusion_job_requires_both_raw_tables(tmp_path, present):
    app = create_app(tmp_path / present)
    with TestClient(app) as client:
        project_id = create_video_project(client, tmp_path)
        if present == "ocr":
            app.state.storage.save_ocr_cues(
                project_id, [make_cue("OCR000001", "visual evidence")], promote=False
            )
            expected_missing = "speech.csv"
        else:
            app.state.storage.save_speech_cues(
                project_id,
                [make_cue("ASR000001", "spoken evidence", source_kind="speech")],
                promote=False,
            )
            expected_missing = "ocr.csv"

        response = client.post(
            f"/api/projects/{project_id}/fusion-jobs", json=fusion_request()
        )

    assert response.status_code == 409
    assert expected_missing in response.json()["detail"]


def test_fusion_job_uses_mock_transport_and_preserves_raw_tables(tmp_path):
    fused_row = {
        "cue_id": "F000001",
        "start_ms": 990,
        "end_ms": 2210,
        "group_id": "",
        "layer": 0,
        "track_id": "main",
        "speaker_id": "",
        "speaker_name": "",
        "speaker_color": "#FFFFFF",
        "source_kind": "imported",
        "source_text": "corrected source",
        "ocr_confidence": 0.97,
        "review_status": "ocr_ok",
    }
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert "OCR_CSV" in body["messages"][1]["content"]
        assert "SPEECH_CSV" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"cues": [fused_row]})}}
                ]
            },
        )

    app = create_app(
        tmp_path / "data", translation_transport=httpx.MockTransport(handler)
    )
    with TestClient(app) as client:
        project_id = create_video_project(client, tmp_path)
        ocr = make_cue("OCR000001", "visual evidence")
        speech = make_cue("ASR000001", "spoken evidence", source_kind="speech")
        app.state.storage.save_ocr_cues(project_id, [ocr], promote=False)
        app.state.storage.save_speech_cues(project_id, [speech], promote=False)

        response = client.post(
            f"/api/projects/{project_id}/fusion-jobs", json=fusion_request()
        )
        assert response.status_code == 202
        job = wait_for_job(client, response.json()["id"])

        assert job["status"] == "completed", job
        assert job["result"]["inputs"] == ["ocr.csv", "speech.csv"]
        assert app.state.storage.list_cues(project_id)[0].source_text == "corrected source"
        assert app.state.storage.list_ocr_cues(project_id) == [ocr]
        assert app.state.storage.list_speech_cues(project_id) == [speech]

    assert len(requests) == 1
