from __future__ import annotations

import html
import json
import os
import subprocess
import wave
from pathlib import Path
from threading import RLock
from urllib.parse import unquote

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    AppConfig,
    AudioCapabilitiesResponse,
    AudioJobRequest,
    AsrModelOption,
    Cue,
    CueBatch,
    CueColorUpdate,
    SpeakerColorUpdate,
    ExportJobRequest,
    FrameOcrRequest,
    FrameOcrResponse,
    FusionJobRequest,
    HealthResponse,
    HybridJobRequest,
    LocalModelConfigureRequest,
    LocalModelDeployRequest,
    OcrCapabilitiesResponse,
    OcrDeviceOption,
    OcrJobRequest,
    ProjectCreate,
    ProjectContextUpdate,
    ProjectManifest,
    ProjectUpdate,
    RegionUpdate,
    SubtitleLayout,
    TranslateProjectRequest,
    TranslationTestResponse,
    TranslationJobRequest,
    TranslationProfile,
    TranslationModelsRequest,
    TranslationModelsResponse,
)
from .fusion import FusionOptions, OpenAICompatibleFusionEngine
from .diagnostics import DiagnosticLogService, REPAIR_GUIDES
from .frame_ocr import FrameOcrError, recognize_frame
from .ai_trace import AiTraceAccumulator
from .audio_pipeline import (
    AudioPipelineError,
    asr_model_catalog,
    asr_model_directory,
    asr_model_installed,
    audio_capabilities_payload,
    download_asr_model,
    get_asr_model,
    normalize_language,
    recommended_asr_model,
)
from .audio_slicer import DEFAULT_MANIFEST_FILENAME, load_slice_manifest
from .jobs import JobManager
from .local_models import LOCAL_MODEL_PROJECT_ID, LocalModelManager
from .audio_process import (
    run_asr_worker,
    run_slicer_stage,
    run_speech_worker,
    run_uvr_worker,
)
from .media import (
    application_root,
    extract_embedded_subtitle,
    extract_audio_track,
    find_binary,
    probe_video,
    resource_root,
    transcode_audio_for_asr,
)
from .ocr_engines import (
    ConsensusOcrEngine,
    PaddleOcrEngine,
    detect_ocr_capabilities,
    resolve_ocr_device,
)
from .render import render_video
from .secrets import SecretStore
from .storage import (
    CueNotFoundError,
    DuplicateCueError,
    ProjectNotFoundError,
    Storage,
)
from .subtitle_import import read_srt
from .subtitles import SubtitleEvent, write_ass
from .task_state import TaskStateStore, atomic_write_json, read_json, task_signature
from .translation import (
    OpenAICompatibleTranslator,
    TranslationError,
    TranslationOptions,
    TranslationProvider,
)
from .video_recognition import RecognitionOptions, Region, recognize_video


VERSION = "0.2.0"
_DEFAULT_SPEECH_WORKER = run_speech_worker
_DEFAULT_UVR_WORKER = run_uvr_worker


def create_app(
    data_dir: Path | str | None = None,
    translation_transport: httpx.BaseTransport | None = None,
    local_model_transport: httpx.BaseTransport | None = None,
    local_model_manager: LocalModelManager | None = None,
) -> FastAPI:
    resolved_data_dir = Path(
        data_dir or os.environ.get("KAOR_DATA_DIR", Path.cwd() / "data")
    )
    storage = Storage(resolved_data_dir)
    static_dir = resolved_data_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Kaor Local API", version=VERSION)
    app.state.storage = storage
    app.state.jobs = JobManager(max_workers=2)
    app.state.secrets = SecretStore()
    app.state.local_models = local_model_manager or LocalModelManager(
        resolved_data_dir / "local-models", transport=local_model_transport
    )
    app.state.local_activation_lock = RLock()
    app.state.pending_local_activation = None
    app.state.diagnostics = DiagnosticLogService(resolved_data_dir)
    app.state.frame_ocr_engines = {}
    app.state.frame_ocr_lock = RLock()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1", "http://localhost"],
        allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    def startup_services() -> None:
        local_models = app.state.local_models
        provider = local_models.provider()
        remote_profile = local_models.remote_profile()
        current_profile = storage.get_config().model_dump()
        if (
            local_models.local_active_provider() is None
            and provider is not None
            and remote_profile is not None
            and local_models.profile_matches_provider(current_profile, provider)
            and not local_models.profile_matches_provider(remote_profile, provider)
        ):
            local_models.mark_local_active(provider)
        local_models.autostart()

    app.router.add_event_handler("startup", startup_services)

    def shutdown_services() -> None:
        try:
            app.state.jobs.shutdown(cancel=True, wait=True)
        finally:
            app.state.local_models.close()

    app.router.add_event_handler("shutdown", shutdown_services)

    def get_storage() -> Storage:
        return app.state.storage

    def get_jobs() -> JobManager:
        return app.state.jobs

    def get_secrets() -> SecretStore:
        return app.state.secrets

    def get_local_models() -> LocalModelManager:
        return app.state.local_models

    def get_diagnostics() -> DiagnosticLogService:
        return app.state.diagnostics

    def ensure_local_model_mutation_allowed(
        jobs: JobManager, local_models: LocalModelManager
    ) -> None:
        if jobs.has_active(LOCAL_MODEL_PROJECT_ID) or local_models.deployment_active():
            raise HTTPException(
                status_code=409,
                detail="a local model deployment is already running",
            )

    def stage_local_activation(provider: dict[str, object] | None) -> None:
        with app.state.local_activation_lock:
            app.state.pending_local_activation = provider

    def finalize_local_activation(
        runtime_status: dict[str, object],
        store: Storage,
        secrets: SecretStore,
        local_models: LocalModelManager,
    ) -> dict[str, object]:
        with app.state.local_activation_lock:
            if runtime_status.get("state") == "failed":
                app.state.pending_local_activation = None
            elif runtime_status.get("ready") and app.state.pending_local_activation:
                activate_local_provider(
                    app.state.pending_local_activation,
                    store,
                    secrets,
                    local_models,
                )
                app.state.pending_local_activation = None
                runtime_status["remote_profile_available"] = (
                    local_models.remote_profile() is not None
                )
        return runtime_status

    @app.get(
        "/api/ocr/capabilities",
        response_model=OcrCapabilitiesResponse,
        tags=["recognition"],
    )
    def get_ocr_capabilities() -> OcrCapabilitiesResponse:
        # Paddle's first import and CUDA runtime initialization are process-global.
        # Serialize them with single-frame model construction to avoid a cold-start
        # race when the WebUI loads capabilities while a correction starts.
        with app.state.frame_ocr_lock:
            capabilities = detect_ocr_capabilities()
            default_device = resolve_ocr_device("auto", capabilities)
        runtime_ready = (
            capabilities.paddle_available and capabilities.paddleocr_available
        )
        runtime_reason = capabilities.error
        if runtime_reason is None and not capabilities.paddleocr_available:
            runtime_reason = "PaddleOCR is not installed"
        devices = [
            OcrDeviceOption(
                id="auto",
                label="Auto",
                available=runtime_ready,
                reason=None if runtime_ready else runtime_reason,
            ),
            OcrDeviceOption(
                id="cpu",
                label="CPU",
                available=runtime_ready,
                reason=None if runtime_ready else runtime_reason,
            ),
        ]
        if capabilities.cuda_device_count and runtime_ready:
            devices.extend(
                OcrDeviceOption(
                    id=f"cuda:{index}",
                    label=name or f"CUDA {index}",
                    available=True,
                )
                for index, name in enumerate(capabilities.cuda_device_names)
            )
        else:
            devices.append(
                OcrDeviceOption(
                    id="cuda:0",
                    label="CUDA",
                    available=False,
                    reason="GPU PaddlePaddle is not installed or no CUDA device was found",
                )
            )
        return OcrCapabilitiesResponse(
            paddle_available=capabilities.paddle_available,
            paddle_version=capabilities.paddle_version,
            paddleocr_available=capabilities.paddleocr_available,
            paddleocr_version=capabilities.paddleocr_version,
            cuda_compiled=capabilities.cuda_compiled,
            cuda_device_count=capabilities.cuda_device_count,
            default_device=default_device,
            devices=devices,
            error=capabilities.error,
        )

    @app.get(
        "/api/audio/capabilities",
        response_model=AudioCapabilitiesResponse,
        tags=["recognition"],
    )
    def get_audio_capabilities() -> AudioCapabilitiesResponse:
        return AudioCapabilitiesResponse.model_validate(
            audio_capabilities_payload(application_root() / "models")
        )

    @app.get(
        "/api/audio/models",
        response_model=list[AsrModelOption],
        tags=["recognition"],
    )
    def get_audio_models() -> list[AsrModelOption]:
        return [
            AsrModelOption.model_validate(model)
            for model in asr_model_catalog(application_root() / "models")
        ]

    def translation_profile(store: Storage, secrets: SecretStore) -> TranslationProfile:
        config = store.get_config()
        return TranslationProfile(
            base_url=config.api_base_url,
            api_key=secrets.get("translation-api-key"),
            model=config.api_model,
            path=config.api_path,
            custom_headers=json.dumps(config.custom_headers, ensure_ascii=False),
            timeout_seconds=config.request_timeout_seconds,
            reasoning_effort=config.api_reasoning_effort,
        )

    def activate_local_provider(
        provider: dict[str, object],
        store: Storage,
        secrets: SecretStore,
        local_models: LocalModelManager,
    ) -> None:
        current = store.get_config()
        current_is_local = local_models.profile_is_active_local(current.model_dump())
        if current.api_base_url and not current_is_local:
            local_models.save_remote_profile(current.model_dump())
            secrets.set(
                "translation-api-key-remote",
                secrets.get("translation-api-key"),
            )
        store.set_config(
            current.model_copy(
                update={
                    "api_base_url": str(provider["base_url"]),
                    "api_model": str(provider["model"]),
                    "api_reasoning_effort": "",
                    "api_path": str(provider.get("api_path") or "/chat/completions"),
                    "custom_headers": {},
                    "request_timeout_seconds": max(
                        current.request_timeout_seconds,
                        int(provider.get("timeout_seconds") or 600),
                    ),
                }
            )
        )
        secrets.set("translation-api-key", "")
        local_models.mark_local_active(provider)

    def project_view(project: ProjectManifest) -> dict[str, object]:
        audio_dir = storage.projects_dir / project.project_id / "cache" / "audio"
        return {
            "id": project.project_id,
            "title": project.title,
            "video_name": project.video_filename,
            "video_url": (
                f"/api/projects/{project.project_id}/video" if project.video_path else None
            ),
            "duration_ms": project.duration_ms,
            "width": project.width,
            "height": project.height,
            "fps": project.fps,
            "source_language": project.source_language,
            "target_language": project.target_language,
            "source_roi": project.source_roi.model_dump(),
            "target_roi": project.target_roi.model_dump(),
            "context": {
                "synopsis": project.synopsis,
                "characters": project.characters_context
                or json.dumps(
                    [item.model_dump() for item in project.character_profiles],
                    ensure_ascii=False,
                    indent=2,
                ),
                "glossary": project.glossary_context
                or json.dumps(project.glossary, ensure_ascii=False, indent=2),
                "translation_style": project.genre_and_tone,
            },
            "updated_at": project.updated_at.isoformat(),
            "audio_ready": (audio_dir / "mix.wav").is_file(),
        }

    def prepare_project_audio(project: ProjectManifest) -> tuple[ProjectManifest, str | None]:
        video_path = Path(project.video_path)
        metadata = probe_video(video_path)
        updated = storage.update_project(
            project.project_id,
            ProjectUpdate(
                duration_ms=metadata.duration_ms,
                width=metadata.width,
                height=metadata.height,
                fps=metadata.frame_rate,
            ),
        )
        audio_dir = storage.projects_dir / project.project_id / "cache" / "audio"
        try:
            extract_audio_track(video_path, audio_dir / "mix.wav")
        except Exception as exc:
            return updated, str(exc)
        return updated, None

    def cue_view(cue: Cue) -> dict[str, object]:
        payload = cue.model_dump()
        payload.update(
            {
                "group_id": cue.group_id or None,
                "overlap_group_id": cue.group_id or None,
                "bbox": None,
                "source_bbox": None,
                "warnings": (
                    ["需要人工复核"] if cue.review_status == "needs_review" else []
                ),
            }
        )
        return payload

    def create_provider(provider_request) -> TranslationProvider:
        return TranslationProvider(
            base_url=provider_request.base_url,
            api_key=provider_request.api_key,
            model=provider_request.model,
            api_path=provider_request.api_path,
            custom_headers=provider_request.custom_headers,
            timeout_seconds=provider_request.timeout_seconds,
            temperature=provider_request.temperature,
            json_mode=provider_request.json_mode,
            reasoning_effort=provider_request.reasoning_effort,
        )

    def create_translator(request: TranslateProjectRequest) -> OpenAICompatibleTranslator:
        return OpenAICompatibleTranslator(
            create_provider(request.provider), transport=translation_transport
        )

    def task_state(store: Storage, project_id: str) -> TaskStateStore:
        return TaskStateStore(store.projects_dir / project_id)

    def provider_task_identity(provider_request) -> dict[str, object]:
        return {
            "base_url": provider_request.base_url.rstrip("/"),
            "api_path": provider_request.api_path,
            "model": provider_request.model,
            "reasoning_effort": provider_request.reasoning_effort,
            "temperature": provider_request.temperature,
            "json_mode": provider_request.json_mode,
            "custom_header_names": sorted(provider_request.custom_headers),
        }

    def ai_checkpoint_path(store: Storage, project_id: str, kind: str) -> Path:
        return store.projects_dir / project_id / "cache" / "ai" / f"{kind}-checkpoint.json"

    def normalized_fusion_output(cues: list[Cue]) -> list[Cue]:
        ordered = sorted(
            cues,
            key=lambda cue: (
                cue.start_ms,
                cue.layer,
                cue.track_id,
                cue.end_ms,
                cue.cue_id,
            ),
        )
        return [
            cue.model_copy(update={"cue_id": f"F{index:06d}"})
            for index, cue in enumerate(ordered, start=1)
        ]

    def normalized_ocr_output(cues: list[Cue]) -> list[Cue]:
        ordered = sorted(
            cues,
            key=lambda cue: (cue.start_ms, cue.layer, cue.end_ms, cue.cue_id),
        )
        return [
            cue.model_copy(update={"cue_id": f"C{index:06d}"})
            for index, cue in enumerate(ordered, start=1)
        ]

    def valid_wav(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size <= 44:
            return False
        try:
            with wave.open(str(path.resolve()), "rb") as handle:
                return handle.getnframes() > 0 and handle.getframerate() > 0
        except (OSError, wave.Error):
            return False

    def valid_slice_manifest(path: Path):
        try:
            return load_slice_manifest(path)
        except (OSError, ValueError, AudioPipelineError):
            return None

    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            version=VERSION, data_directory=str(storage.data_dir)
        )

    @app.get("/api/diagnostics/logs", tags=["diagnostics"])
    def diagnostic_logs(
        source: str = "all",
        tail: int = 1000,
        query: str = "",
        level: str = "",
        diagnostics: DiagnosticLogService = Depends(get_diagnostics),
    ) -> dict[str, object]:
        levels = [item.strip() for item in level.split(",") if item.strip()]
        return diagnostics.entries(
            source=source,
            tail=max(1, min(5000, tail)),
            query=query,
            levels=levels,
        )

    @app.get("/api/diagnostics/guides", tags=["diagnostics"])
    def diagnostic_guides() -> list[dict[str, object]]:
        return [guide.public() for guide in REPAIR_GUIDES]

    @app.get(
        "/api/diagnostics/troubleshooting",
        response_class=HTMLResponse,
        tags=["diagnostics"],
    )
    def diagnostic_troubleshooting() -> HTMLResponse:
        document_path = application_root() / "docs" / "TROUBLESHOOTING.zh-CN.md"
        if not document_path.is_file():
            raise HTTPException(status_code=404, detail="local troubleshooting document is missing")

        document = html.escape(document_path.read_text(encoding="utf-8"))
        for guide in REPAIR_GUIDES:
            marker = html.escape(f'<a id="{guide.anchor}"></a>')
            anchor = html.escape(guide.anchor, quote=True)
            document = document.replace(
                marker,
                f'<span class="doc-anchor" id="{anchor}"></span>',
            )

        page = (
            "<!doctype html><html lang=\"zh-CN\"><head>"
            "<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\">"
            "<title>Kaor 中文故障排查</title><style>"
            ":root{color-scheme:dark}body{margin:0;background:#101513;color:#dfe9e4;"
            "font:14px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace}"
            "main{max-width:1080px;margin:0 auto;padding:32px 28px 64px}"
            "header{position:sticky;top:0;padding:12px 0;background:#101513;border-bottom:1px solid #2a3430}"
            "header a{color:#70dbc4;text-decoration:none}pre{margin:22px 0;white-space:pre-wrap;overflow-wrap:anywhere}"
            ".doc-anchor{display:block;position:relative;top:-58px;visibility:hidden}"
            "</style></head><body><main><header><a href=\"#repair-guide-index\">Kaor / 完整故障排查</a></header><pre>"
            + document
            + "</pre></main></body></html>"
        )
        return HTMLResponse(
            page,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/diagnostics/export", tags=["diagnostics"])
    def export_diagnostics(
        diagnostics: DiagnosticLogService = Depends(get_diagnostics),
    ) -> FileResponse:
        bundle = diagnostics.export_bundle()
        return FileResponse(
            bundle,
            media_type="application/zip",
            filename=bundle.name,
        )

    @app.get("/api/workspace", tags=["system"])
    def get_workspace(
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
        secrets: SecretStore = Depends(get_secrets),
    ) -> dict[str, object]:
        projects = store.list_projects()
        if projects:
            project = projects[0]
        else:
            project = store.create_project(ProjectCreate(title="新项目"))
        visible_cues = store.list_translated_cues(project.project_id) or store.list_cues(
            project.project_id
        )
        return {
            "project": project_view(project),
            "cues": [cue_view(cue) for cue in visible_cues],
            "ocr_cues": [
                cue_view(cue) for cue in store.list_ocr_cues(project.project_id)
            ],
            "speech_cues": [
                cue_view(cue) for cue in store.list_speech_cues(project.project_id)
            ],
            "jobs": jobs.list(project.project_id),
            "translation_settings": translation_profile(store, secrets).model_dump(),
            "layout_settings": project.subtitle_layout.model_dump(),
        }

    @app.get("/api/jobs", tags=["jobs"])
    def list_jobs(
        project_id: str | None = None, jobs: JobManager = Depends(get_jobs)
    ) -> list[dict[str, object]]:
        return jobs.list(project_id)

    @app.get("/api/jobs/{job_id}", tags=["jobs"])
    def get_job(job_id: str, jobs: JobManager = Depends(get_jobs)) -> dict[str, object]:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/api/jobs/{job_id}/cancel", tags=["jobs"])
    def cancel_job(
        job_id: str, jobs: JobManager = Depends(get_jobs)
    ) -> dict[str, object]:
        try:
            return jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/api/config", response_model=AppConfig, tags=["system"])
    def get_config(store: Storage = Depends(get_storage)) -> AppConfig:
        return store.get_config()

    @app.put("/api/config", response_model=AppConfig, tags=["system"])
    def set_config(
        config: AppConfig, store: Storage = Depends(get_storage)
    ) -> AppConfig:
        return store.set_config(config)

    @app.get("/api/local-models/status", tags=["local-models"])
    def get_local_model_status(
        refresh_hardware: bool = False,
        store: Storage = Depends(get_storage),
        secrets: SecretStore = Depends(get_secrets),
        local_models: LocalModelManager = Depends(get_local_models),
    ) -> dict[str, object]:
        if refresh_hardware:
            local_models.hardware(refresh=True)
        return finalize_local_activation(
            local_models.status(), store, secrets, local_models
        )

    @app.get("/api/local-models/catalog", tags=["local-models"])
    def get_local_model_catalog(
        local_models: LocalModelManager = Depends(get_local_models),
    ) -> dict[str, object]:
        return {
            "hardware": local_models.hardware().public(),
            "recommendation": local_models.recommendation(),
            "models": local_models.catalog(),
        }

    @app.get("/api/local-models/provider", tags=["local-models"])
    def get_local_model_provider(
        local_models: LocalModelManager = Depends(get_local_models),
    ) -> dict[str, object]:
        provider = local_models.provider()
        if provider is None:
            raise HTTPException(status_code=404, detail="local model is not configured")
        return provider

    @app.put("/api/local-models/configuration", tags=["local-models"])
    def configure_local_model(
        request: LocalModelConfigureRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
        secrets: SecretStore = Depends(get_secrets),
        local_models: LocalModelManager = Depends(get_local_models),
    ) -> dict[str, object]:
        ensure_local_model_mutation_allowed(jobs, local_models)
        try:
            configuration = local_models.configure(
                request.model_dump(),
                probe_external=request.mode == "external",
                stage_for_start=request.mode == "managed",
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        provider = local_models.provider()
        if request.mode == "managed":
            stage_local_activation(provider if request.make_default else None)
        else:
            stage_local_activation(None)
        if request.mode == "external" and request.make_default and provider is not None:
            activate_local_provider(provider, store, secrets, local_models)
        runtime_status = finalize_local_activation(
            local_models.status(), store, secrets, local_models
        )
        return {
            "configuration": configuration,
            "provider": provider,
            "status": runtime_status,
        }

    @app.post("/api/local-models/start", tags=["local-models"])
    def start_local_model(
        jobs: JobManager = Depends(get_jobs),
        store: Storage = Depends(get_storage),
        secrets: SecretStore = Depends(get_secrets),
        local_models: LocalModelManager = Depends(get_local_models),
    ) -> dict[str, object]:
        ensure_local_model_mutation_allowed(jobs, local_models)
        try:
            return finalize_local_activation(
                local_models.start_with_rollback(), store, secrets, local_models
            )
        except (OSError, RuntimeError, ValueError) as exc:
            stage_local_activation(None)
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/local-models/stop", tags=["local-models"])
    def stop_local_model(
        jobs: JobManager = Depends(get_jobs),
        local_models: LocalModelManager = Depends(get_local_models),
    ) -> dict[str, object]:
        ensure_local_model_mutation_allowed(jobs, local_models)
        try:
            return local_models.stop()
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/local-models/activate", tags=["local-models"])
    def activate_configured_local_model(
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
        secrets: SecretStore = Depends(get_secrets),
        local_models: LocalModelManager = Depends(get_local_models),
    ) -> dict[str, object]:
        ensure_local_model_mutation_allowed(jobs, local_models)
        stage_local_activation(None)
        provider = local_models.provider()
        if provider is None:
            raise HTTPException(status_code=404, detail="local model is not configured")
        activate_local_provider(provider, store, secrets, local_models)
        return {"provider": provider, "active": True}

    @app.post("/api/local-models/deactivate", tags=["local-models"])
    def restore_remote_translation_provider(
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
        secrets: SecretStore = Depends(get_secrets),
        local_models: LocalModelManager = Depends(get_local_models),
    ) -> dict[str, object]:
        ensure_local_model_mutation_allowed(jobs, local_models)
        stage_local_activation(None)
        profile = local_models.remote_profile()
        if profile is None:
            raise HTTPException(status_code=404, detail="no previous remote profile is saved")
        try:
            restored = store.get_config().model_copy(update=profile)
            restored = AppConfig.model_validate(restored.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        store.set_config(restored)
        remote_key = secrets.get("translation-api-key-remote")
        secrets.set("translation-api-key", remote_key)
        secrets.set("translation-api-key-remote", "")
        local_models.clear_local_active()
        local_models.clear_remote_profile()
        return {"provider": restored.model_dump(), "active": False}

    @app.post("/api/local-models/deploy-jobs", tags=["local-models"])
    def deploy_local_model(
        request: LocalModelDeployRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
        secrets: SecretStore = Depends(get_secrets),
        local_models: LocalModelManager = Depends(get_local_models),
    ) -> dict[str, object]:
        def runner(progress, cancel_event):
            stage_local_activation(None)
            result = local_models.deploy(request.model_dump(), progress, cancel_event)
            provider = local_models.provider()
            if request.make_default and provider is not None:
                activate_local_provider(provider, store, secrets, local_models)
            return result

        try:
            return jobs.submit_unique(
                LOCAL_MODEL_PROJECT_ID, "local-model-deploy", runner
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail="a local model deployment is already running",
            ) from exc

    @app.get(
        "/api/projects/{project_id}/translation-profile",
        response_model=TranslationProfile,
        tags=["translation"],
    )
    def get_translation_profile(
        project_id: str,
        store: Storage = Depends(get_storage),
        secrets: SecretStore = Depends(get_secrets),
    ) -> TranslationProfile:
        try:
            store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        return translation_profile(store, secrets)

    @app.put(
        "/api/projects/{project_id}/translation-profile",
        response_model=TranslationProfile,
        tags=["translation"],
    )
    def set_translation_profile(
        project_id: str,
        profile: TranslationProfile,
        store: Storage = Depends(get_storage),
        secrets: SecretStore = Depends(get_secrets),
    ) -> TranslationProfile:
        try:
            store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        try:
            custom_headers = json.loads(profile.custom_headers or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="custom_headers must be JSON") from exc
        if not isinstance(custom_headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in custom_headers.items()
        ):
            raise HTTPException(status_code=422, detail="custom_headers must map strings")
        current = store.get_config()
        store.set_config(
            current.model_copy(
                update={
                    "api_base_url": profile.base_url,
                    "api_model": profile.model,
                    "api_reasoning_effort": profile.reasoning_effort,
                    "api_path": profile.path,
                    "custom_headers": custom_headers,
                    "request_timeout_seconds": profile.timeout_seconds,
                }
            )
        )
        secrets.set("translation-api-key", profile.api_key)
        return profile

    @app.get("/api/projects", response_model=list[ProjectManifest], tags=["projects"])
    def list_projects(store: Storage = Depends(get_storage)) -> list[ProjectManifest]:
        return store.list_projects()

    @app.post(
        "/api/projects",
        response_model=ProjectManifest,
        status_code=status.HTTP_201_CREATED,
        tags=["projects"],
    )
    def create_project(
        project: ProjectCreate, store: Storage = Depends(get_storage)
    ) -> ProjectManifest:
        return store.create_project(project)

    @app.post(
        "/api/media/register",
        status_code=status.HTTP_201_CREATED,
        tags=["media"],
    )
    async def register_media(
        request: Request,
        x_kaor_filename: str = Header(default="video.mp4"),
        store: Storage = Depends(get_storage),
    ) -> dict[str, object]:
        filename = Path(unquote(x_kaor_filename)).name or "video.mp4"
        existing = store.list_projects()
        if existing and not existing[0].video_path and not store.list_cues(existing[0].project_id):
            project = store.update_project(
                existing[0].project_id,
                ProjectUpdate(
                    title=Path(filename).stem or "Untitled", video_filename=filename
                ),
            )
        else:
            project = store.create_project(
                ProjectCreate(title=Path(filename).stem or "Untitled", video_filename=filename)
            )
        project_dir = store.projects_dir / project.project_id
        media_dir = project_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        video_path = media_dir / filename
        try:
            with video_path.open("wb") as handle:
                async for chunk in request.stream():
                    handle.write(chunk)
            if video_path.stat().st_size == 0:
                raise ValueError("uploaded video is empty")
            updated = store.update_project(
                project.project_id,
                ProjectUpdate(
                    video_path=str(video_path.resolve()),
                ),
            )
            updated, audio_error = prepare_project_audio(updated)
            payload = project_view(updated)
            payload["audio_error"] = audio_error
            return payload
        except ValueError as exc:
            if (store.projects_dir / project.project_id / "manifest.json").exists():
                store.delete_project(project.project_id)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            if (store.projects_dir / project.project_id / "manifest.json").exists():
                store.delete_project(project.project_id)
            raise

    @app.get(
        "/api/projects/{project_id}",
        response_model=ProjectManifest,
        tags=["projects"],
    )
    def get_project(
        project_id: str, store: Storage = Depends(get_storage)
    ) -> ProjectManifest:
        try:
            return store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.post(
        "/api/projects/{project_id}/reset",
        tags=["projects"],
    )
    def reset_project_workspace(
        project_id: str,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        try:
            project = store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        if jobs.has_active(project_id):
            raise HTTPException(
                status_code=409,
                detail="cancel or wait for active jobs before resetting the workspace",
            )
        video_path = Path(project.video_path)
        if not project.video_path or not video_path.is_file():
            raise HTTPException(status_code=409, detail="project video is missing")
        project = store.reset_project_workspace(project_id)
        project, audio_error = prepare_project_audio(project)
        jobs.clear_project(project_id)
        payload = project_view(project)
        payload["audio_error"] = audio_error
        return payload

    @app.patch(
        "/api/projects/{project_id}",
        response_model=ProjectManifest,
        tags=["projects"],
    )
    def update_project(
        project_id: str,
        project: ProjectUpdate,
        store: Storage = Depends(get_storage),
    ) -> ProjectManifest:
        try:
            return store.update_project(project_id, project)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.put(
        "/api/projects/{project_id}/regions",
        response_model=ProjectManifest,
        tags=["projects"],
    )
    def update_regions(
        project_id: str,
        regions: RegionUpdate,
        store: Storage = Depends(get_storage),
    ) -> ProjectManifest:
        try:
            return store.update_project(
                project_id, ProjectUpdate(**regions.model_dump(exclude_none=True))
            )
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.put(
        "/api/projects/{project_id}/context",
        tags=["projects"],
    )
    def update_context(
        project_id: str,
        context: ProjectContextUpdate,
        store: Storage = Depends(get_storage),
    ) -> dict[str, str]:
        try:
            store.update_project(
                project_id,
                ProjectUpdate(
                    synopsis=context.synopsis,
                    characters_context=context.characters,
                    glossary_context=context.glossary,
                    genre_and_tone=context.translation_style,
                ),
            )
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        return context.model_dump()

    @app.get(
        "/api/projects/{project_id}/video",
        response_class=FileResponse,
        tags=["media"],
    )
    def get_video(
        project_id: str, store: Storage = Depends(get_storage)
    ) -> FileResponse:
        try:
            project = store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        path = Path(project.video_path)
        if not project.video_path or not path.is_file():
            raise HTTPException(status_code=404, detail="video file not found")
        return FileResponse(path)

    @app.post(
        "/api/projects/{project_id}/frame-ocr",
        response_model=FrameOcrResponse,
        tags=["recognition"],
    )
    def recognize_selected_frame(
        project_id: str,
        request: FrameOcrRequest,
        store: Storage = Depends(get_storage),
    ) -> FrameOcrResponse:
        try:
            project = store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        video_path = Path(project.video_path)
        if not project.video_path or not video_path.is_file():
            raise HTTPException(status_code=409, detail="project video is missing")

        engine_key = (request.language, request.device)
        try:
            # Paddle model construction is expensive, so single-frame corrections
            # reuse one local engine per language/device pair.
            with app.state.frame_ocr_lock:
                engine = app.state.frame_ocr_engines.get(engine_key)
                if engine is None:
                    engine = PaddleOcrEngine(
                        language=request.language,
                        device=request.device,
                    )
                    app.state.frame_ocr_engines[engine_key] = engine
                result = recognize_frame(
                    video_path,
                    timestamp_ms=request.timestamp_ms,
                    fps=project.fps,
                    region=request.bbox,
                    language=request.language,
                    device=request.device,
                    high_accuracy=request.high_accuracy,
                    engine=engine,
                )
        except FrameOcrError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"single-frame OCR failed: {exc}",
            ) from exc
        return FrameOcrResponse.model_validate(result.__dict__)

    @app.get(
        "/api/projects/{project_id}/subtitle-layout",
        response_model=SubtitleLayout,
        tags=["subtitles"],
    )
    def get_subtitle_layout(
        project_id: str, store: Storage = Depends(get_storage)
    ) -> SubtitleLayout:
        try:
            return store.get_project(project_id).subtitle_layout
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.put(
        "/api/projects/{project_id}/subtitle-layout",
        response_model=SubtitleLayout,
        tags=["subtitles"],
    )
    def set_subtitle_layout(
        project_id: str,
        layout: SubtitleLayout,
        store: Storage = Depends(get_storage),
    ) -> SubtitleLayout:
        try:
            store.update_project(project_id, ProjectUpdate(subtitle_layout=layout))
            return layout
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.get(
        "/api/projects/{project_id}/exports/{filename}",
        response_class=FileResponse,
        tags=["subtitles"],
    )
    def download_export(
        project_id: str,
        filename: str,
        store: Storage = Depends(get_storage),
    ) -> FileResponse:
        try:
            store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        export_dir = (store.projects_dir / project_id / "exports").resolve()
        path = (export_dir / Path(filename).name).resolve()
        if export_dir not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="export not found")
        return FileResponse(path, filename=path.name)

    def run_ocr_recognition(
        project_id: str,
        project: ProjectManifest,
        request: OcrJobRequest,
        store: Storage,
        progress,
        cancel_event,
        *,
        promote: bool,
    ) -> tuple[list[Cue], str, dict[str, object]]:
        video_path = Path(project.video_path)
        ocr_signature = task_signature(
            sources=[video_path],
            options={
                "request": request.model_dump(mode="json"),
                "source_roi": project.source_roi.model_dump(mode="json"),
            },
        )
        ocr_state = task_state(store, project_id)
        existing_ocr = store.list_ocr_cues(project_id)
        ocr_record = ocr_state.get("ocr")
        reusable_ocr = bool(
            existing_ocr
            and (
                ocr_state.matches("ocr", ocr_signature, "complete")
                or ocr_record is None
            )
        )
        if reusable_ocr:
            store.save_ocr_cues(project_id, existing_ocr, promote=promote)
            metrics = {"source": "ocr", "reused": True}
            ocr_state.update(
                "ocr",
                ocr_signature,
                status="complete",
                artifact="ocr.csv",
                cue_count=len(existing_ocr),
            )
            progress(
                1.0,
                f"Validated and reused {len(existing_ocr)} OCR cues",
                {
                    "timestamp_ms": project.duration_ms,
                    "cues": [cue.model_dump() for cue in existing_ocr],
                    "current": [],
                    "metrics": metrics,
                },
            )
            return existing_ocr, "ocr", metrics

        checkpoint_path = (
            store.projects_dir / project_id / "cache" / "ocr-checkpoint.json"
        )
        checkpoint_payload = read_json(checkpoint_path)
        resume_start_ms = 0
        retained_cues: list[Cue] = []
        if (
            isinstance(checkpoint_payload, dict)
            and checkpoint_payload.get("signature") == ocr_signature
        ):
            try:
                checkpoint_ms = int(checkpoint_payload.get("timestamp_ms", 0))
                resume_start_ms = max(0, checkpoint_ms - 2_000)
                retained_cues = [
                    Cue.model_validate(row)
                    for row in checkpoint_payload.get("cues", [])
                    if int(row.get("end_ms", 0)) <= resume_start_ms
                ]
            except (AttributeError, TypeError, ValueError):
                resume_start_ms = 0
                retained_cues = []
        if request.prefer_embedded:
            ffprobe = find_binary("ffprobe")
            ffmpeg = find_binary("ffmpeg")
            if ffmpeg:
                progress(0.01, "Checking embedded subtitle streams")
                metadata = probe_video(video_path, ffprobe=ffprobe, ffmpeg=ffmpeg)
                if metadata.subtitle_streams:
                    subtitle_path = store.projects_dir / project_id / "cache" / "embedded.srt"
                    try:
                        extract_embedded_subtitle(
                            video_path,
                            subtitle_path,
                            metadata.subtitle_streams[0],
                            ffmpeg,
                        )
                        cues = read_srt(subtitle_path)
                        if cues:
                            store.save_ocr_cues(project_id, cues, promote=promote)
                            checkpoint_path.unlink(missing_ok=True)
                            ocr_state.update(
                                "ocr",
                                ocr_signature,
                                status="complete",
                                artifact="ocr.csv",
                                cue_count=len(cues),
                            )
                            progress(
                                1.0,
                                "Imported embedded subtitles",
                                {
                                    "timestamp_ms": project.duration_ms,
                                    "cues": [cue.model_dump() for cue in cues],
                                    "current": [],
                                    "metrics": {"source": "embedded"},
                                },
                            )
                            return cues, "embedded", {"source": "embedded"}
                    except Exception:
                        pass
        if cancel_event.is_set():
            raise InterruptedError("job cancelled")
        progress(0.02, "Loading local PaddleOCR model")
        engine = PaddleOcrEngine(language=request.language, device=request.device)
        selected_engine = ConsensusOcrEngine(engine) if request.high_accuracy else engine
        latest_metrics: dict[str, object] = {}

        def guarded_progress(
            value: float,
            message: str,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            if cancel_event.is_set():
                raise InterruptedError("job cancelled")
            if snapshot and isinstance(snapshot.get("metrics"), dict):
                latest_metrics.clear()
                latest_metrics.update(snapshot["metrics"])
            output_snapshot = snapshot
            if snapshot and isinstance(snapshot.get("cues"), list):
                try:
                    resumed = [Cue.model_validate(row) for row in snapshot["cues"]]
                    combined = normalized_ocr_output([*retained_cues, *resumed])
                    output_snapshot = {
                        **snapshot,
                        "cues": [cue.model_dump() for cue in combined],
                    }
                    atomic_write_json(
                        checkpoint_path,
                        {
                            "signature": ocr_signature,
                            "timestamp_ms": int(snapshot.get("timestamp_ms", 0)),
                            "cues": [cue.model_dump(mode="json") for cue in combined],
                        },
                    )
                    ocr_state.update(
                        "ocr",
                        ocr_signature,
                        status="running",
                        checkpoint=str(checkpoint_path),
                        timestamp_ms=int(snapshot.get("timestamp_ms", 0)),
                    )
                except (TypeError, ValueError):
                    output_snapshot = snapshot
            resume_fraction = resume_start_ms / max(project.duration_ms, 1)
            overall = resume_fraction + value * (1.0 - resume_fraction)
            progress(0.03 + overall * 0.97, message, output_snapshot)

        cues = recognize_video(
            video_path,
            selected_engine,
            RecognitionOptions(
                source_roi=Region(**project.source_roi.model_dump()),
                sample_fps=request.sample_fps,
                filter_noise=request.filter_noise,
                batch_size=request.batch_size,
                start_ms=resume_start_ms,
            ),
            guarded_progress,
        )
        cues = normalized_ocr_output([*retained_cues, *cues])
        store.save_ocr_cues(project_id, cues, promote=promote)
        # A proofreading correction normally follows full-video OCR. Reuse the
        # already-loaded base engine so that frame OCR does not pay another model
        # cold start. Keep the first cached engine if another request won the race.
        with app.state.frame_ocr_lock:
            app.state.frame_ocr_engines.setdefault(
                (request.language, request.device), engine
            )
        checkpoint_path.unlink(missing_ok=True)
        ocr_state.update(
            "ocr",
            ocr_signature,
            status="complete",
            artifact="ocr.csv",
            cue_count=len(cues),
        )
        return cues, "ocr", latest_metrics

    @app.post(
        "/api/projects/{project_id}/ocr-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["recognition"],
    )
    def start_ocr_job(
        project_id: str,
        request: OcrJobRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        try:
            project = store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        video_path = Path(project.video_path)
        if not project.video_path or not video_path.is_file():
            raise HTTPException(status_code=409, detail="project video is missing")

        def runner(progress, cancel_event):
            cues, source, metrics = run_ocr_recognition(
                project_id,
                project,
                request,
                store,
                progress,
                cancel_event,
                promote=True,
            )
            return {"cue_count": len(cues), "source": source, "metrics": metrics}

        return jobs.submit(project_id, "ocr", runner)

    def run_audio_recognition(
        project_id: str,
        project: ProjectManifest,
        request: AudioJobRequest,
        store: Storage,
        progress,
        cancel_event,
        *,
        promote: bool,
    ) -> tuple[list[Cue], str]:
        video_path = Path(project.video_path)
        language = normalize_language(request.language or project.source_language)
        model = (
            get_asr_model(request.model_id)
            if request.model_id
            else recommended_asr_model(language)
        )
        if model.language != language:
            raise AudioPipelineError(
                f"ASR model {model.id} is for {model.language}, not {language}"
            )
        # Keep the worker seam injectable for embedders and tests. The built-in
        # worker uses the resumable three-stage implementation below.
        if run_speech_worker is not _DEFAULT_SPEECH_WORKER:
            models_root = application_root() / "models"
            if not asr_model_installed(models_root, model):
                model_dir = download_asr_model(
                    model,
                    models_root,
                    lambda value, message: progress(value * 0.15, message),
                )
            else:
                model_dir = asr_model_directory(models_root, model)
            audio_dir = store.projects_dir / project_id / "cache" / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            mix_path = audio_dir / "mix.wav"
            if not mix_path.is_file() or mix_path.stat().st_size <= 44:
                progress(0.17, "Extracting the local audio track")
                extract_audio_track(video_path, mix_path)
            asr_audio_path = audio_dir / "speech-16k-mono.wav"
            cues = run_speech_worker(
                mix_path,
                audio_dir,
                asr_audio_path,
                model,
                model_dir,
                device=request.device,
                separate_vocals=request.separate_vocals,
                forced_alignment=request.forced_alignment,
                diarization=request.diarization,
                asr_batch_size=request.asr_batch_size,
                threshold_db=request.slicer_threshold_db,
                min_length_ms=request.slicer_min_length_ms,
                min_interval_ms=request.slicer_min_interval_ms,
                hop_size_ms=request.slicer_hop_size_ms,
                max_sil_kept_ms=request.slicer_max_sil_kept_ms,
                max_length_ms=request.slicer_max_length_ms,
                progress=lambda value, message: progress(0.2 + value * 0.78, message),
                cancel_event=cancel_event,
            )
            if not cues:
                raise AudioPipelineError(
                    "speech recognition produced 0 cues; speech.csv was not replaced"
                )
            store.save_speech_cues(project_id, cues, promote=promote)
            progress(1.0, f"Saved {len(cues)} speech cues")
            return cues, model.id

        audio_dir = store.projects_dir / project_id / "cache" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        mix_path = audio_dir / "mix.wav"
        if not valid_wav(mix_path):
            progress(0.02, "Extracting the local audio track")
            extract_audio_track(video_path, mix_path)
        else:
            progress(0.02, "Using the audio track prepared during import")
        if not valid_wav(mix_path):
            raise AudioPipelineError("video audio extraction produced an invalid WAV")

        state = task_state(store, project_id)
        vocals_path = audio_dir / "vocals.wav"
        if request.separate_vocals:
            uvr_signature = task_signature(
                sources=[mix_path],
                options={"device": request.device, "model": "bs-roformer-viperx-1297"},
            )
            uvr_record = state.get("uvr")
            if valid_wav(vocals_path) and (
                state.matches("uvr", uvr_signature, "complete")
                or uvr_record is None
            ):
                state.update(
                    "uvr",
                    uvr_signature,
                    status="complete",
                    artifact="vocals.wav",
                    size_bytes=vocals_path.stat().st_size,
                )
                progress(0.55, "Validated and reused vocals.wav")
            else:
                state.update("uvr", uvr_signature, status="running")
                vocals_path = run_uvr_worker(
                    mix_path,
                    audio_dir,
                    device=request.device,
                    progress=lambda value, message: progress(
                        0.03 + value * 0.52, message
                    ),
                    cancel_event=cancel_event,
                )
                if not valid_wav(vocals_path) and not (
                    run_uvr_worker is not _DEFAULT_UVR_WORKER
                    and vocals_path.is_file()
                    and vocals_path.stat().st_size > 44
                ):
                    raise AudioPipelineError("UVR completed without a valid vocals WAV")
                state.update(
                    "uvr",
                    uvr_signature,
                    status="complete",
                    artifact="vocals.wav",
                    size_bytes=vocals_path.stat().st_size,
                )
        else:
            vocals_path = mix_path
            progress(0.55, "Using the original audio track for speech recognition")

        if cancel_event.is_set():
            raise InterruptedError("job cancelled")
        asr_audio_path = audio_dir / "speech-16k-mono.wav"
        slices_path = audio_dir / "slices"
        manifest_path = slices_path / DEFAULT_MANIFEST_FILENAME
        slicer_parameters = {
            "threshold_db": request.slicer_threshold_db,
            "min_length_ms": request.slicer_min_length_ms,
            "min_interval_ms": request.slicer_min_interval_ms,
            "hop_size_ms": request.slicer_hop_size_ms,
            "max_sil_kept_ms": request.slicer_max_sil_kept_ms,
            "max_length_ms": request.slicer_max_length_ms,
        }
        slicer_signature = task_signature(
            sources=[vocals_path], options=slicer_parameters
        )
        slicer_record = state.get("slicer")
        slice_manifest = valid_slice_manifest(manifest_path)
        can_reuse_slices = bool(
            slice_manifest is not None
            and valid_wav(asr_audio_path)
            and slice_manifest.source_path == asr_audio_path.resolve()
            and slice_manifest.settings.__dict__ == slicer_parameters
            and (
                state.matches("slicer", slicer_signature, "complete")
                or slicer_record is None
            )
        )
        if can_reuse_slices:
            state.update(
                "slicer",
                slicer_signature,
                status="complete",
                artifact=DEFAULT_MANIFEST_FILENAME,
                slice_count=len(slice_manifest.slices),
            )
            progress(0.68, "Validated and reused the speech slices")
        else:
            progress(0.56, "Preparing 16 kHz mono speech audio")
            transcode_audio_for_asr(vocals_path, asr_audio_path)
            state.update("slicer", slicer_signature, status="running")
            slice_manifest = run_slicer_stage(
                asr_audio_path,
                slices_path,
                threshold_db=request.slicer_threshold_db,
                min_length_ms=request.slicer_min_length_ms,
                min_interval_ms=request.slicer_min_interval_ms,
                hop_size_ms=request.slicer_hop_size_ms,
                max_sil_kept_ms=request.slicer_max_sil_kept_ms,
                max_length_ms=request.slicer_max_length_ms,
                progress=lambda value, message: progress(
                    0.57 + value * 0.11, message
                ),
            )
            state.update(
                "slicer",
                slicer_signature,
                status="complete",
                artifact=DEFAULT_MANIFEST_FILENAME,
                slice_count=len(slice_manifest.slices),
            )

        asr_signature = task_signature(
            sources=[asr_audio_path, manifest_path],
            options={
                "model_id": model.id,
                "repository": model.repository,
                "device": request.device,
                "batch_size": request.asr_batch_size,
                "forced_alignment": request.forced_alignment,
                "diarization": request.diarization,
            },
        )
        existing_speech = store.list_speech_cues(project_id)
        asr_record = state.get("asr")
        if existing_speech and (
            state.matches("asr", asr_signature, "complete") or asr_record is None
        ):
            store.save_speech_cues(project_id, existing_speech, promote=promote)
            state.update(
                "asr",
                asr_signature,
                status="complete",
                artifact="speech.csv",
                cue_count=len(existing_speech),
            )
            progress(
                1.0,
                f"Validated and reused {len(existing_speech)} speech cues",
                {
                    "timestamp_ms": project.duration_ms,
                    "cues": [cue.model_dump() for cue in existing_speech],
                    "current": [],
                    "metrics": {"source": "speech", "model": model.id, "reused": True},
                },
            )
            return existing_speech, model.id

        models_root = application_root() / "models"
        if not asr_model_installed(models_root, model):
            model_dir = download_asr_model(
                model,
                models_root,
                lambda value, message: progress(0.68 + value * 0.1, message),
            )
        else:
            model_dir = asr_model_directory(models_root, model)
        state.update("asr", asr_signature, status="running")
        cues = run_asr_worker(
            asr_audio_path,
            manifest_path,
            audio_dir,
            model,
            model_dir,
            device=request.device,
            batch_size=request.asr_batch_size,
            forced_alignment=request.forced_alignment,
            diarization=request.diarization,
            checkpoint_path=audio_dir / "asr-checkpoint.json",
            checkpoint_signature=asr_signature,
            progress=lambda value, message: progress(0.78 + value * 0.21, message),
            cancel_event=cancel_event,
        )
        if not cues:
            raise AudioPipelineError(
                "speech recognition produced 0 cues; speech.csv was not replaced"
            )
        store.save_speech_cues(project_id, cues, promote=promote)
        (audio_dir / "asr-checkpoint.json").unlink(missing_ok=True)
        state.update(
            "asr",
            asr_signature,
            status="complete",
            artifact="speech.csv",
            cue_count=len(cues),
        )
        progress(
            1.0,
            f"Saved {len(cues)} speech cues",
            {
                "timestamp_ms": project.duration_ms,
                "cues": [cue.model_dump() for cue in cues],
                "current": [],
                "metrics": {"source": "speech", "model": model.id},
            },
        )
        return cues, model.id

    def project_audio_paths(project_id: str) -> dict[str, Path]:
        audio_dir = storage.projects_dir / project_id / "cache" / "audio"
        return {
            "directory": audio_dir,
            "mix": audio_dir / "mix.wav",
            "vocals": audio_dir / "vocals.wav",
            "asr_audio": audio_dir / "speech-16k-mono.wav",
            "slices": audio_dir / "slices",
            "manifest": audio_dir / "slices" / DEFAULT_MANIFEST_FILENAME,
        }

    def require_audio_project(
        project_id: str, store: Storage
    ) -> tuple[ProjectManifest, Path]:
        try:
            project = store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        video_path = Path(project.video_path)
        if not project.video_path or not video_path.is_file():
            raise HTTPException(status_code=409, detail="project video is missing")
        return project, video_path

    @app.post(
        "/api/projects/{project_id}/uvr-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["recognition"],
    )
    def start_uvr_job(
        project_id: str,
        request: AudioJobRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        _project, video_path = require_audio_project(project_id, store)
        paths = project_audio_paths(project_id)

        def runner(progress, cancel_event):
            paths["directory"].mkdir(parents=True, exist_ok=True)
            if not paths["mix"].is_file() or paths["mix"].stat().st_size <= 44:
                progress(0.03, "Extracting the local audio track")
                extract_audio_track(video_path, paths["mix"])
            signature = task_signature(
                sources=[paths["mix"]],
                options={"device": request.device, "model": "bs-roformer-viperx-1297"},
            )
            state = task_state(store, project_id)
            record = state.get("uvr")
            if valid_wav(paths["vocals"]) and (
                state.matches("uvr", signature, "complete") or record is None
            ):
                state.update(
                    "uvr",
                    signature,
                    status="complete",
                    artifact="vocals.wav",
                    size_bytes=paths["vocals"].stat().st_size,
                )
                progress(1.0, "Validated and reused vocals.wav")
                return {
                    "artifact": paths["vocals"].name,
                    "size_bytes": paths["vocals"].stat().st_size,
                    "reused": True,
                }
            state.update("uvr", signature, status="running")
            vocals = run_uvr_worker(
                paths["mix"],
                paths["directory"],
                device=request.device,
                progress=progress,
                cancel_event=cancel_event,
            )
            if not valid_wav(vocals) and not (
                run_uvr_worker is not _DEFAULT_UVR_WORKER
                and vocals.is_file()
                and vocals.stat().st_size > 44
            ):
                raise AudioPipelineError("UVR completed without a valid vocals WAV")
            state.update(
                "uvr",
                signature,
                status="complete",
                artifact=vocals.name,
                size_bytes=vocals.stat().st_size,
            )
            return {
                "artifact": vocals.name,
                "size_bytes": vocals.stat().st_size,
            }

        return jobs.submit(project_id, "uvr", runner)

    @app.post(
        "/api/projects/{project_id}/slicer-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["recognition"],
    )
    def start_slicer_job(
        project_id: str,
        request: AudioJobRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        require_audio_project(project_id, store)
        paths = project_audio_paths(project_id)
        if not paths["vocals"].is_file():
            raise HTTPException(
                status_code=409,
                detail="vocals.wav is missing; run UVR5 vocal separation first",
            )

        def runner(progress, cancel_event):
            parameters = {
                "threshold_db": request.slicer_threshold_db,
                "min_length_ms": request.slicer_min_length_ms,
                "min_interval_ms": request.slicer_min_interval_ms,
                "hop_size_ms": request.slicer_hop_size_ms,
                "max_sil_kept_ms": request.slicer_max_sil_kept_ms,
                "max_length_ms": request.slicer_max_length_ms,
            }
            signature = task_signature(sources=[paths["vocals"]], options=parameters)
            state = task_state(store, project_id)
            existing_manifest = valid_slice_manifest(paths["manifest"])
            record = state.get("slicer")
            if (
                existing_manifest is not None
                and valid_wav(paths["asr_audio"])
                and existing_manifest.source_path == paths["asr_audio"].resolve()
                and existing_manifest.settings.__dict__ == parameters
                and (
                    state.matches("slicer", signature, "complete")
                    or record is None
                )
            ):
                state.update(
                    "slicer",
                    signature,
                    status="complete",
                    artifact=DEFAULT_MANIFEST_FILENAME,
                    slice_count=len(existing_manifest.slices),
                )
                progress(1.0, "Validated and reused the speech slice manifest")
                return {
                    "artifact": DEFAULT_MANIFEST_FILENAME,
                    "slice_count": len(existing_manifest.slices),
                    "segment_count": len(existing_manifest.slices),
                    "duration_ms": existing_manifest.duration_ms,
                    "parameters": parameters,
                    "reused": True,
                }
            progress(0.03, "Preparing 16 kHz mono speech audio")
            transcode_audio_for_asr(paths["vocals"], paths["asr_audio"])
            state.update("slicer", signature, status="running")

            def slicer_progress(value: float, message: str) -> None:
                if cancel_event.is_set():
                    raise InterruptedError("job cancelled")
                progress(0.08 + value * 0.92, message)

            manifest = run_slicer_stage(
                paths["asr_audio"],
                paths["slices"],
                threshold_db=request.slicer_threshold_db,
                min_length_ms=request.slicer_min_length_ms,
                min_interval_ms=request.slicer_min_interval_ms,
                hop_size_ms=request.slicer_hop_size_ms,
                max_sil_kept_ms=request.slicer_max_sil_kept_ms,
                max_length_ms=request.slicer_max_length_ms,
                progress=slicer_progress,
            )
            state.update(
                "slicer",
                signature,
                status="complete",
                artifact=DEFAULT_MANIFEST_FILENAME,
                slice_count=len(manifest.slices),
            )
            return {
                "artifact": DEFAULT_MANIFEST_FILENAME,
                "slice_count": len(manifest.slices),
                "segment_count": len(manifest.slices),
                "duration_ms": manifest.duration_ms,
                "parameters": manifest.to_dict()["parameters"],
            }

        return jobs.submit(project_id, "slicer", runner)

    @app.post(
        "/api/projects/{project_id}/asr-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["recognition"],
    )
    def start_asr_job(
        project_id: str,
        request: AudioJobRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        project, _video_path = require_audio_project(project_id, store)
        paths = project_audio_paths(project_id)
        if not paths["asr_audio"].is_file() or not paths["manifest"].is_file():
            raise HTTPException(
                status_code=409,
                detail="audio slices are missing; run silence slicing first",
            )
        language = normalize_language(request.language or project.source_language)
        try:
            model = (
                get_asr_model(request.model_id)
                if request.model_id
                else recommended_asr_model(language)
            )
        except AudioPipelineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if model.language != language:
            raise HTTPException(
                status_code=422,
                detail=f"ASR model {model.id} does not match {language}",
            )

        def runner(progress, cancel_event):
            signature = task_signature(
                sources=[paths["asr_audio"], paths["manifest"]],
                options={
                    "model_id": model.id,
                    "repository": model.repository,
                    "device": request.device,
                    "batch_size": request.asr_batch_size,
                    "forced_alignment": request.forced_alignment,
                    "diarization": request.diarization,
                },
            )
            state = task_state(store, project_id)
            existing_speech = store.list_speech_cues(project_id)
            record = state.get("asr")
            if existing_speech and (
                state.matches("asr", signature, "complete") or record is None
            ):
                store.save_speech_cues(project_id, existing_speech, promote=True)
                state.update(
                    "asr",
                    signature,
                    status="complete",
                    artifact="speech.csv",
                    cue_count=len(existing_speech),
                )
                progress(
                    1.0,
                    f"Validated and reused {len(existing_speech)} speech cues",
                    {
                        "timestamp_ms": project.duration_ms,
                        "cues": [cue.model_dump() for cue in existing_speech],
                        "current": [],
                        "metrics": {"source": "speech", "model": model.id, "reused": True},
                    },
                )
                return {
                    "cue_count": len(existing_speech),
                    "source": "speech.csv",
                    "model_id": model.id,
                    "reused": True,
                }
            state.update("asr", signature, status="running")
            models_root = application_root() / "models"
            if not asr_model_installed(models_root, model):
                model_dir = download_asr_model(
                    model,
                    models_root,
                    lambda value, message: progress(value * 0.15, message),
                )
            else:
                model_dir = asr_model_directory(models_root, model)
            cues = run_asr_worker(
                paths["asr_audio"],
                paths["manifest"],
                paths["directory"],
                model,
                model_dir,
                device=request.device,
                batch_size=request.asr_batch_size,
                forced_alignment=request.forced_alignment,
                diarization=request.diarization,
                checkpoint_path=paths["directory"] / "asr-checkpoint.json",
                checkpoint_signature=signature,
                progress=lambda value, message: progress(
                    0.15 + value * 0.83, message
                ),
                cancel_event=cancel_event,
            )
            store.save_speech_cues(project_id, cues, promote=True)
            (paths["directory"] / "asr-checkpoint.json").unlink(missing_ok=True)
            state.update(
                "asr",
                signature,
                status="complete",
                artifact="speech.csv",
                cue_count=len(cues),
            )
            progress(
                1.0,
                f"Saved {len(cues)} speech cues",
                {
                    "timestamp_ms": project.duration_ms,
                    "cues": [cue.model_dump() for cue in cues],
                    "current": [],
                    "metrics": {"source": "speech", "model": model.id},
                },
            )
            return {
                "cue_count": len(cues),
                "source": "speech.csv",
                "model_id": model.id,
            }

        return jobs.submit(project_id, "asr", runner)

    @app.post(
        "/api/projects/{project_id}/audio-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["recognition"],
    )
    def start_audio_job(
        project_id: str,
        request: AudioJobRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        try:
            project = store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        video_path = Path(project.video_path)
        if not project.video_path or not video_path.is_file():
            raise HTTPException(status_code=409, detail="project video is missing")
        try:
            model = (
                get_asr_model(request.model_id)
                if request.model_id
                else recommended_asr_model(request.language)
            )
        except AudioPipelineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if model.language != normalize_language(request.language):
            raise HTTPException(
                status_code=422,
                detail=f"ASR model {model.id} does not match {request.language}",
            )

        def runner(progress, cancel_event):
            cues, model_id = run_audio_recognition(
                project_id,
                project,
                request,
                store,
                progress,
                cancel_event,
                promote=True,
            )
            return {
                "cue_count": len(cues),
                "source": "speech.csv",
                "model_id": model_id,
            }

        return jobs.submit(project_id, "audio", runner)

    @app.post(
        "/api/projects/{project_id}/hybrid-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["recognition"],
    )
    def start_hybrid_job(
        project_id: str,
        request: HybridJobRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        try:
            project = store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        video_path = Path(project.video_path)
        if not project.video_path or not video_path.is_file():
            raise HTTPException(status_code=409, detail="project video is missing")
        try:
            model = (
                get_asr_model(request.model_id)
                if request.model_id
                else recommended_asr_model(request.language)
            )
        except AudioPipelineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if model.language != normalize_language(request.language):
            raise HTTPException(
                status_code=422,
                detail=f"ASR model {model.id} does not match {request.language}",
            )

        def runner(progress, cancel_event):
            def ocr_progress(value, message, snapshot=None):
                progress(value * 0.45, message, snapshot)

            ocr_request = OcrJobRequest(
                language=request.language,
                device=request.device,
                sample_fps=request.sample_fps,
                high_accuracy=request.high_accuracy,
                prefer_embedded=request.prefer_embedded,
                filter_noise=request.filter_noise,
                batch_size=request.batch_size,
            )
            ocr_cues, ocr_source, ocr_metrics = run_ocr_recognition(
                project_id,
                project,
                ocr_request,
                store,
                ocr_progress,
                cancel_event,
                promote=False,
            )

            def audio_progress(value, message, snapshot=None):
                progress(0.45 + value * 0.55, message, snapshot)

            speech_cues, model_id = run_audio_recognition(
                project_id,
                project,
                AudioJobRequest(**request.model_dump()),
                store,
                audio_progress,
                cancel_event,
                promote=False,
            )
            return {
                "ocr_cue_count": len(ocr_cues),
                "speech_cue_count": len(speech_cues),
                "ocr_source": ocr_source,
                "ocr_metrics": ocr_metrics,
                "model_id": model_id,
                "artifacts": ["ocr.csv", "speech.csv"],
            }

        return jobs.submit(project_id, "hybrid", runner)

    def submit_render_job(
        project_id: str,
        request: ExportJobRequest,
        store: Storage,
        jobs: JobManager,
    ) -> dict[str, object]:
        try:
            project = store.get_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        video_path = Path(project.video_path)
        if not project.video_path or not video_path.is_file():
            raise HTTPException(status_code=409, detail="project video is missing")
        cues = store.list_translated_cues(project_id) or store.list_cues(project_id)
        if not cues:
            raise HTTPException(status_code=409, detail="project has no subtitles")

        def runner(progress, cancel_event):
            export_dir = store.projects_dir / project_id / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            ass_path = export_dir / "translated.ass"
            events = [
                SubtitleEvent(
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    text=cue.target_text or cue.source_text,
                    speaker_id=cue.speaker_id or "SPK_00",
                    color=(
                        cue.speaker_color
                        if project.subtitle_layout.color_mode == "speaker"
                        else "#FFFFFF"
                    ),
                    layer=cue.layer,
                )
                for cue in cues
            ]
            region = project.target_roi
            write_ass(
                ass_path,
                events,
                width=project.width or 1920,
                height=project.height or 1080,
                font_name=project.subtitle_layout.font_family,
                font_size=project.subtitle_layout.font_size,
                outline=project.subtitle_layout.outline,
                target_region=(region.x, region.y, region.width, region.height),
            )
            progress(0.02, "ASS subtitle generated")
            if request.preview:
                output_path = export_dir / f"preview-{request.start_ms}.mp4"
                clip_duration = request.preview_duration_ms
            else:
                output_path = export_dir / f"{video_path.stem}.kaor.mp4"
                clip_duration = None

            def render_progress(value: float, message: str) -> None:
                progress(0.02 + value * 0.98, message)

            render_video(
                video_path,
                ass_path,
                output_path,
                duration_ms=project.duration_ms,
                progress=render_progress,
                cancel_event=cancel_event,
                start_ms=request.start_ms if request.preview else None,
                clip_duration_ms=clip_duration,
                video_encoder=request.video_encoder,
                crf=request.crf,
                preset=request.preset,
            )
            return {
                "filename": output_path.name,
                "download_url": f"/api/projects/{project_id}/exports/{output_path.name}",
                "ass_filename": ass_path.name,
            }

        return jobs.submit(project_id, "preview" if request.preview else "export", runner)

    @app.post(
        "/api/projects/{project_id}/preview-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["subtitles"],
    )
    def start_preview_job(
        project_id: str,
        request: ExportJobRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        return submit_render_job(
            project_id, request.model_copy(update={"preview": True}), store, jobs
        )

    @app.post(
        "/api/projects/{project_id}/export-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["subtitles"],
    )
    def start_export_job(
        project_id: str,
        request: ExportJobRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        return submit_render_job(
            project_id, request.model_copy(update={"preview": False}), store, jobs
        )

    @app.delete(
        "/api/projects/{project_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["projects"],
    )
    def delete_project(
        project_id: str, store: Storage = Depends(get_storage)
    ) -> Response:
        try:
            store.delete_project(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get(
        "/api/projects/{project_id}/translated-cues",
        response_model=list[Cue],
        tags=["translation"],
    )
    def list_translated_cues(
        project_id: str, store: Storage = Depends(get_storage)
    ) -> list[Cue]:
        try:
            return store.list_translated_cues(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.get(
        "/api/projects/{project_id}/source.csv",
        response_class=FileResponse,
        tags=["cues"],
    )
    def download_source_csv(
        project_id: str, store: Storage = Depends(get_storage)
    ) -> FileResponse:
        try:
            path = store.source_csv_path(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        return FileResponse(path, media_type="text/csv", filename="source.csv")

    @app.get(
        "/api/projects/{project_id}/ocr.csv",
        response_class=FileResponse,
        tags=["recognition"],
    )
    def download_ocr_csv(
        project_id: str, store: Storage = Depends(get_storage)
    ) -> FileResponse:
        try:
            path = store.ocr_csv_path(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="OCR CSV not found")
        return FileResponse(path, media_type="text/csv", filename="ocr.csv")

    @app.get(
        "/api/projects/{project_id}/speech.csv",
        response_class=FileResponse,
        tags=["recognition"],
    )
    def download_speech_csv(
        project_id: str, store: Storage = Depends(get_storage)
    ) -> FileResponse:
        try:
            path = store.speech_csv_path(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="speech CSV not found")
        return FileResponse(path, media_type="text/csv", filename="speech.csv")

    @app.get(
        "/api/projects/{project_id}/translated.csv",
        response_class=FileResponse,
        tags=["translation"],
    )
    def download_translated_csv(
        project_id: str, store: Storage = Depends(get_storage)
    ) -> FileResponse:
        try:
            path = store.translated_csv_path(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="translated CSV not found")
        return FileResponse(path, media_type="text/csv", filename="translated.csv")

    @app.post(
        "/api/translation/models",
        response_model=TranslationModelsResponse,
        tags=["translation"],
    )
    def list_translation_models(
        request: TranslationModelsRequest,
    ) -> TranslationModelsResponse:
        provider = TranslationProvider(
            base_url=request.base_url,
            api_key=request.api_key,
            model="",
            custom_headers=request.custom_headers,
            timeout_seconds=request.timeout_seconds,
        )
        try:
            models = OpenAICompatibleTranslator(
                provider, transport=translation_transport
            ).list_models()
        except TranslationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return TranslationModelsResponse(models=models)

    @app.post(
        "/api/translation/test",
        response_model=TranslationTestResponse,
        tags=["translation"],
    )
    def test_translation_provider(
        request: TranslateProjectRequest,
    ) -> TranslationTestResponse:
        try:
            preview = create_translator(request).test_connection()
        except TranslationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return TranslationTestResponse(response_preview=preview[:500])

    @app.post(
        "/api/projects/{project_id}/translate",
        response_model=list[Cue],
        tags=["translation"],
    )
    def translate_project(
        project_id: str,
        request: TranslateProjectRequest,
        store: Storage = Depends(get_storage),
    ) -> list[Cue]:
        try:
            manifest = store.get_project(project_id)
            cues = store.list_cues(project_id)
            if not cues:
                raise HTTPException(status_code=409, detail="project has no source cues")
            options = TranslationOptions(**request.options.model_dump())
            translated = create_translator(request).translate(manifest, cues, options)
            return store.save_translated_cues(project_id, translated)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except TranslationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/api/projects/{project_id}/fusion-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["recognition"],
    )
    def start_fusion_job(
        project_id: str,
        request: FusionJobRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        try:
            manifest = store.get_project(project_id)
            ocr_cues = store.list_ocr_cues(project_id)
            speech_cues = store.list_speech_cues(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        evidence = (
            ("ocr.csv", store.ocr_csv_path(project_id), ocr_cues),
            ("speech.csv", store.speech_csv_path(project_id), speech_cues),
        )
        missing = [name for name, path, _cues in evidence if not path.is_file()]
        empty = [
            name
            for name, path, cues in evidence
            if path.is_file() and not cues
        ]
        if missing:
            raise HTTPException(
                status_code=409,
                detail=f"fusion requires completed OCR and speech CSV files; missing={missing}",
            )
        if empty:
            raise HTTPException(
                status_code=409,
                detail=f"fusion evidence CSV files contain no subtitle rows; empty={empty}",
            )

        fusion_signature = task_signature(
            sources=[path for _name, path, _cues in evidence],
            options={
                "provider": provider_task_identity(request.provider),
                "options": request.options.model_dump(mode="json"),
                "source_language": manifest.source_language,
                "title": manifest.title,
                "synopsis": manifest.synopsis,
                "characters_context": manifest.characters_context,
                "glossary_context": manifest.glossary_context,
            },
        )
        fusion_state = task_state(store, project_id)
        source_path = store.source_csv_path(project_id)
        existing_source = store.list_cues(project_id)
        fusion_record = fusion_state.get("fusion")
        legacy_complete = bool(
            fusion_record is None
            and existing_source
            and source_path.is_file()
            and source_path.stat().st_mtime_ns
            >= max(path.stat().st_mtime_ns for _name, path, _cues in evidence)
        )
        reusable = bool(
            existing_source
            and (
                fusion_state.matches("fusion", fusion_signature, "complete")
                or legacy_complete
            )
        )

        def runner(progress, cancel_event):
            trace = AiTraceAccumulator("fusion")
            trace.total = len(ocr_cues) + len(speech_cues)
            if reusable:
                trace.phase = "reused"
                trace.status = "Validated and reused the completed fusion artifact"
                trace.completed = trace.total
                fusion_state.update(
                    "fusion",
                    fusion_signature,
                    status="complete",
                    artifact="source.csv",
                    cue_count=len(existing_source),
                )
                progress(1.0, trace.status, {"ai_trace": trace.snapshot()})
                return {
                    "cue_count": len(existing_source),
                    "source": "source.csv",
                    "inputs": ["ocr.csv", "speech.csv"],
                    "reused": True,
                }

            engine = OpenAICompatibleFusionEngine(
                create_provider(request.provider), transport=translation_transport
            )
            checkpoint_path = ai_checkpoint_path(store, project_id, "fusion")
            checkpoint_payload = read_json(checkpoint_path)
            resumed_cues: list[Cue] = []
            completed_keys: set[tuple[str, str]] = set()
            if (
                isinstance(checkpoint_payload, dict)
                and checkpoint_payload.get("signature") == fusion_signature
            ):
                try:
                    resumed_cues = [
                        Cue.model_validate(row)
                        for row in checkpoint_payload.get("cues", [])
                    ]
                    completed_keys = {
                        (str(row[0]), str(row[1]))
                        for row in checkpoint_payload.get("completed_evidence", [])
                        if isinstance(row, list) and len(row) == 2
                    }
                except (TypeError, ValueError):
                    resumed_cues = []
                    completed_keys = set()

            pending_ocr = [
                cue for cue in ocr_cues if ("ocr", cue.cue_id) not in completed_keys
            ]
            pending_speech = [
                cue
                for cue in speech_cues
                if ("speech", cue.cue_id) not in completed_keys
            ]
            initial_completed = len(completed_keys)
            pending_total = len(pending_ocr) + len(pending_speech)
            current_progress = [
                initial_completed / max(trace.total, 1)
            ]

            def publish_trace(message: str | None = None) -> None:
                if cancel_event.is_set():
                    raise InterruptedError("job cancelled")
                trace.completed = len(completed_keys)
                progress(
                    current_progress[0],
                    message or trace.status,
                    {"ai_trace": trace.snapshot()},
                )

            def fusion_progress(value: float) -> None:
                if cancel_event.is_set():
                    raise InterruptedError("job cancelled")
                current_progress[0] = min(
                    1.0,
                    (initial_completed + value * pending_total)
                    / max(trace.total, 1),
                )
                publish_trace("Comparing complete OCR and speech evidence")

            def on_ai_event(event: dict[str, object]) -> None:
                if event.get("type") == "subset_completed":
                    completed_keys.update(
                        ("ocr", str(cue_id))
                        for cue_id in event.get("ocr_cue_ids", [])
                    )
                    completed_keys.update(
                        ("speech", str(cue_id))
                        for cue_id in event.get("speech_cue_ids", [])
                    )
                trace.consume(event)
                publish_trace()

            def save_fusion_checkpoint(partial: list[Cue]) -> None:
                combined = normalized_fusion_output([*resumed_cues, *partial])
                atomic_write_json(
                    checkpoint_path,
                    {
                        "signature": fusion_signature,
                        "completed_evidence": [list(row) for row in sorted(completed_keys)],
                        "cues": [cue.model_dump(mode="json") for cue in combined],
                    },
                )
                fusion_state.update(
                    "fusion",
                    fusion_signature,
                    status="running",
                    checkpoint=str(checkpoint_path),
                    completed_evidence=len(completed_keys),
                )

            if pending_ocr or pending_speech:
                fused_current = engine.fuse(
                    manifest,
                    pending_ocr,
                    pending_speech,
                    FusionOptions(
                        batch_size=request.options.batch_size,
                        context_cues=request.options.context_cues,
                        retries=request.options.retries,
                        retry_backoff_seconds=request.options.retry_backoff_seconds,
                    ),
                    fusion_progress,
                    stream_event=on_ai_event,
                    checkpoint=save_fusion_checkpoint,
                    full_ocr_reference=ocr_cues,
                    full_speech_reference=speech_cues,
                )
                fused = normalized_fusion_output([*resumed_cues, *fused_current])
            else:
                fused = normalized_fusion_output(resumed_cues)
            if not fused:
                raise TranslationError("fusion checkpoint did not contain output cues")
            store.replace_cues(project_id, fused)
            checkpoint_path.unlink(missing_ok=True)
            fusion_state.update(
                "fusion",
                fusion_signature,
                status="complete",
                artifact="source.csv",
                cue_count=len(fused),
            )
            return {
                "cue_count": len(fused),
                "source": "source.csv",
                "inputs": ["ocr.csv", "speech.csv"],
                "resumed": initial_completed > 0,
            }

        return jobs.submit(project_id, "fusion", runner)

    @app.post(
        "/api/projects/{project_id}/translation-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["translation"],
    )
    def start_translation_job(
        project_id: str,
        request: TranslationJobRequest,
        store: Storage = Depends(get_storage),
        jobs: JobManager = Depends(get_jobs),
    ) -> dict[str, object]:
        try:
            manifest = store.get_project(project_id)
            cues = store.list_cues(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        if not cues:
            raise HTTPException(status_code=409, detail="project has no source cues")

        translation_signature = task_signature(
            sources=[store.source_csv_path(project_id)],
            options={
                "provider": provider_task_identity(request.provider),
                "options": request.options.model_dump(mode="json"),
                "target_language": manifest.target_language,
                "title": manifest.title,
                "synopsis": manifest.synopsis,
                "genre_and_tone": manifest.genre_and_tone,
                "characters_context": manifest.characters_context,
                "glossary_context": manifest.glossary_context,
            },
        )
        translation_state = task_state(store, project_id)
        existing_translation = store.list_translated_cues(project_id)
        translated_ids = {cue.cue_id for cue in existing_translation}
        source_ids = {cue.cue_id for cue in cues}
        complete_translation = bool(
            existing_translation
            and translated_ids == source_ids
            and all(cue.target_text.strip() for cue in existing_translation)
        )
        translation_record = translation_state.get("translation")
        legacy_translation = bool(translation_record is None and complete_translation)
        reusable_translation = bool(
            complete_translation
            and (
                translation_state.matches(
                    "translation", translation_signature, "complete"
                )
                or legacy_translation
            )
        )

        def runner(progress, cancel_event):
            options = TranslationOptions(**request.options.model_dump())
            trace = AiTraceAccumulator("translation")
            trace.total = len(cues)
            if reusable_translation:
                trace.phase = "reused"
                trace.status = "Validated and reused the completed translation artifact"
                trace.completed = len(cues)
                translation_state.update(
                    "translation",
                    translation_signature,
                    status="complete",
                    artifact="translated.csv",
                    cue_count=len(cues),
                )
                progress(1.0, trace.status, {"ai_trace": trace.snapshot()})
                return {
                    "cue_count": len(cues),
                    "source": "translated.csv",
                    "reused": True,
                }

            checkpoint_path = ai_checkpoint_path(store, project_id, "translation")
            checkpoint_payload = read_json(checkpoint_path)
            translated_by_id: dict[str, Cue] = {}
            if (
                isinstance(checkpoint_payload, dict)
                and checkpoint_payload.get("signature") == translation_signature
            ):
                try:
                    translated_by_id = {
                        cue.cue_id: cue
                        for cue in (
                            Cue.model_validate(row)
                            for row in checkpoint_payload.get("cues", [])
                        )
                        if cue.cue_id in source_ids and cue.target_text.strip()
                    }
                except (TypeError, ValueError):
                    translated_by_id = {}
            pending_cues = [cue for cue in cues if cue.cue_id not in translated_by_id]
            initial_completed = len(translated_by_id)
            current_progress = [initial_completed / max(len(cues), 1)]

            def publish_trace(message: str | None = None) -> None:
                if cancel_event.is_set():
                    raise InterruptedError("job cancelled")
                trace.completed = len(translated_by_id)
                progress(
                    current_progress[0],
                    message or trace.status,
                    {"ai_trace": trace.snapshot()},
                )

            def translation_progress(value: float) -> None:
                if cancel_event.is_set():
                    raise InterruptedError("job cancelled")
                completed = initial_completed + round(value * len(pending_cues))
                current_progress[0] = min(1.0, completed / max(len(cues), 1))
                publish_trace(f"Translated {completed}/{len(cues)} cues")

            def on_ai_event(event: dict[str, object]) -> None:
                trace.consume(event)
                publish_trace()

            def save_translation_checkpoint(partial: list[Cue]) -> None:
                for cue in partial:
                    if cue.target_text.strip():
                        translated_by_id[cue.cue_id] = cue
                merged = [translated_by_id.get(cue.cue_id, cue) for cue in cues]
                atomic_write_json(
                    checkpoint_path,
                    {
                        "signature": translation_signature,
                        "cues": [cue.model_dump(mode="json") for cue in merged],
                    },
                )
                translation_state.update(
                    "translation",
                    translation_signature,
                    status="running",
                    checkpoint=str(checkpoint_path),
                    completed_cues=len(translated_by_id),
                )

            if pending_cues:
                translated_current = create_translator(request).translate(
                    manifest,
                    pending_cues,
                    options,
                    translation_progress,
                    stream_event=on_ai_event,
                    checkpoint=save_translation_checkpoint,
                    reference_cues=cues,
                )
                for cue in translated_current:
                    if cue.target_text.strip():
                        translated_by_id[cue.cue_id] = cue
            translated = [translated_by_id.get(cue.cue_id, cue) for cue in cues]
            if any(not cue.target_text.strip() for cue in translated):
                raise TranslationError("translation completed with unfinished cue rows")
            store.save_translated_cues(project_id, translated)
            checkpoint_path.unlink(missing_ok=True)
            translation_state.update(
                "translation",
                translation_signature,
                status="complete",
                artifact="translated.csv",
                cue_count=len(translated),
            )
            return {
                "cue_count": len(translated),
                "source": "translated.csv",
                "resumed": initial_completed > 0,
            }

        return jobs.submit(project_id, "translation", runner)

    @app.get(
        "/api/projects/{project_id}/cues",
        response_model=list[Cue],
        tags=["cues"],
    )
    def list_cues(
        project_id: str, store: Storage = Depends(get_storage)
    ) -> list[Cue]:
        try:
            return store.list_cues(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.put(
        "/api/projects/{project_id}/cues",
        response_model=list[Cue],
        tags=["cues"],
    )
    def replace_cues(
        project_id: str,
        batch: CueBatch,
        store: Storage = Depends(get_storage),
    ) -> list[Cue]:
        try:
            return store.replace_cues(project_id, batch.cues)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except DuplicateCueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/api/projects/{project_id}/cues",
        response_model=Cue,
        status_code=status.HTTP_201_CREATED,
        tags=["cues"],
    )
    def create_cue(
        project_id: str, cue: Cue, store: Storage = Depends(get_storage)
    ) -> Cue:
        try:
            return store.create_cue(project_id, cue)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except DuplicateCueError as exc:
            raise HTTPException(status_code=409, detail="cue already exists") from exc

    @app.put(
        "/api/projects/{project_id}/cues/{cue_id}",
        response_model=Cue,
        tags=["cues"],
    )
    def update_cue(
        project_id: str,
        cue_id: str,
        cue: Cue,
        store: Storage = Depends(get_storage),
    ) -> Cue:
        try:
            return store.update_cue(project_id, cue_id, cue)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except CueNotFoundError as exc:
            raise HTTPException(status_code=404, detail="cue not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch(
        "/api/projects/{project_id}/cues/{cue_id}/color",
        response_model=Cue,
        tags=["cues"],
    )
    def update_cue_color(
        project_id: str,
        cue_id: str,
        update: CueColorUpdate,
        store: Storage = Depends(get_storage),
    ) -> Cue:
        try:
            return store.update_cue_color(project_id, cue_id, update.speaker_color)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except CueNotFoundError as exc:
            raise HTTPException(status_code=404, detail="cue not found") from exc

    @app.patch(
        "/api/projects/{project_id}/speakers/color",
        response_model=list[Cue],
        tags=["cues"],
    )
    def update_speaker_color(
        project_id: str,
        update: SpeakerColorUpdate,
        store: Storage = Depends(get_storage),
    ) -> list[Cue]:
        try:
            return store.update_speaker_color(
                project_id,
                speaker_id=update.speaker_id,
                speaker_name=update.speaker_name,
                color=update.speaker_color,
            )
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except CueNotFoundError as exc:
            raise HTTPException(status_code=404, detail="speaker not found") from exc

    @app.delete(
        "/api/projects/{project_id}/cues/{cue_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["cues"],
    )
    def delete_cue(
        project_id: str, cue_id: str, store: Storage = Depends(get_storage)
    ) -> Response:
        try:
            store.delete_cue(project_id, cue_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except CueNotFoundError as exc:
            raise HTTPException(status_code=404, detail="cue not found") from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    web_candidates = [
        resource_root() / "web",
        application_root() / "web",
        application_root() / "apps" / "web" / "dist",
    ]
    web_directory = next((path for path in web_candidates if (path / "index.html").is_file()), None)
    if web_directory is not None:
        app.mount("/", StaticFiles(directory=web_directory, html=True), name="web")

    return app


app = create_app()
