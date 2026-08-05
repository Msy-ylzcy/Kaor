from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Cue(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    cue_id: str = Field(min_length=1, max_length=128)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    group_id: str = ""
    layer: int = Field(default=0, ge=0)
    track_id: str = "main"
    speaker_id: str = ""
    speaker_name: str = ""
    speaker_color: str = "#FFFFFF"
    source_kind: Literal["ocr", "manual", "speech", "imported"] = "ocr"
    source_text: str = ""
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    target_text: str = ""
    review_status: Literal[
        "pending", "ocr_ok", "needs_review", "translated", "approved"
    ] = "pending"

    @model_validator(mode="after")
    def validate_timing(self) -> "Cue":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self

    @field_validator("speaker_color")
    @classmethod
    def validate_color(cls, value: str) -> str:
        if len(value) != 7 or not value.startswith("#"):
            raise ValueError("speaker_color must use #RRGGBB format")
        try:
            int(value[1:], 16)
        except ValueError as exc:
            raise ValueError("speaker_color must use #RRGGBB format") from exc
        return value.upper()


class CharacterProfile(BaseModel):
    speaker_id: str
    name: str = ""
    description: str = ""
    color: str = "#FFFFFF"


class NormalizedRegion(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "NormalizedRegion":
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("normalized region must fit inside the frame")
        return self


class SubtitleLayout(BaseModel):
    font_family: str = "Noto Sans SC"
    font_size: int = Field(default=48, ge=12, le=160)
    outline: float = Field(default=2.4, ge=0, le=12)
    max_lines: int = Field(default=2, ge=1, le=8)
    max_chars: int = Field(default=24, ge=4, le=200)
    color_mode: Literal["speaker", "single"] = "speaker"
    avoid_faces: bool = True
    avoid_source: bool = True
    overlap_mode: Literal["layers", "split", "stack"] = "layers"


class ProjectManifest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    project_id: str
    title: str = Field(min_length=1, max_length=300)
    video_filename: str = ""
    video_path: str = ""
    duration_ms: int = Field(default=0, ge=0)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    fps: float = Field(default=0, ge=0)
    source_roi: NormalizedRegion = Field(
        default_factory=lambda: NormalizedRegion(x=0.05, y=0.62, width=0.9, height=0.32)
    )
    target_roi: NormalizedRegion = Field(
        default_factory=lambda: NormalizedRegion(x=0.08, y=0.08, width=0.84, height=0.22)
    )
    subtitle_layout: SubtitleLayout = Field(default_factory=SubtitleLayout)
    source_language: str = "auto"
    target_language: str = "zh-CN"
    synopsis: str = ""
    genre_and_tone: str = ""
    characters_context: str = ""
    glossary_context: str = ""
    character_profiles: list[CharacterProfile] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    video_filename: str = ""
    video_path: str = ""
    duration_ms: int = Field(default=0, ge=0)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    fps: float = Field(default=0, ge=0)
    source_roi: NormalizedRegion = Field(
        default_factory=lambda: NormalizedRegion(x=0.05, y=0.62, width=0.9, height=0.32)
    )
    target_roi: NormalizedRegion = Field(
        default_factory=lambda: NormalizedRegion(x=0.08, y=0.08, width=0.84, height=0.22)
    )
    subtitle_layout: SubtitleLayout = Field(default_factory=SubtitleLayout)
    source_language: str = "auto"
    target_language: str = "zh-CN"
    synopsis: str = ""
    genre_and_tone: str = ""
    characters_context: str = ""
    glossary_context: str = ""
    character_profiles: list[CharacterProfile] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)


class ProjectUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    video_filename: str | None = None
    video_path: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, ge=0)
    source_roi: NormalizedRegion | None = None
    target_roi: NormalizedRegion | None = None
    subtitle_layout: SubtitleLayout | None = None
    source_language: str | None = None
    target_language: str | None = None
    synopsis: str | None = None
    genre_and_tone: str | None = None
    characters_context: str | None = None
    glossary_context: str | None = None
    character_profiles: list[CharacterProfile] | None = None
    glossary: dict[str, str] | None = None


class CueBatch(BaseModel):
    cues: list[Cue]

    @field_validator("cues")
    @classmethod
    def cue_ids_must_be_unique(cls, value: list[Cue]) -> list[Cue]:
        cue_ids = [cue.cue_id for cue in value]
        if len(cue_ids) != len(set(cue_ids)):
            raise ValueError("cue_id values must be unique")
        return value


class CueColorUpdate(BaseModel):
    speaker_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")

    @field_validator("speaker_color")
    @classmethod
    def normalize_color(cls, value: str) -> str:
        return value.upper()


class SpeakerColorUpdate(CueColorUpdate):
    speaker_id: str = ""
    speaker_name: str = ""

    @model_validator(mode="after")
    def require_speaker_identity(self) -> "SpeakerColorUpdate":
        if not self.speaker_id.strip() and not self.speaker_name.strip():
            raise ValueError("speaker_id or speaker_name is required")
        return self


class AppConfig(BaseModel):
    api_base_url: str = ""
    api_model: str = ""
    api_reasoning_effort: str = ""
    api_path: str = "/chat/completions"
    custom_headers: dict[str, str] = Field(default_factory=dict)
    request_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    translation_batch_size: int = Field(default=100, ge=1, le=1000)
    static_directory: str = "static"


class TranslationProviderRequest(BaseModel):
    base_url: str = Field(min_length=1)
    api_key: str = ""
    model: str = Field(min_length=1)
    api_path: str = "/chat/completions"
    custom_headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=120, ge=1, le=3600)
    temperature: float = Field(default=0.2, ge=0, le=2)
    json_mode: bool = True
    reasoning_effort: str = Field(default="", max_length=32)


class TranslationModelsRequest(BaseModel):
    base_url: str = Field(min_length=1)
    api_key: str = ""
    custom_headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=120, ge=1, le=3600)


class TranslationModelInfo(BaseModel):
    id: str = Field(min_length=1)
    owned_by: str | None = None


class TranslationModelsResponse(BaseModel):
    models: list[TranslationModelInfo]


class TranslationOptionsRequest(BaseModel):
    max_lines: int = Field(default=2, ge=1, le=8)
    max_chars_per_line: int = Field(default=24, ge=4, le=200)
    batch_size: int = Field(default=80, ge=1, le=500)
    context_cues: int = Field(default=3, ge=0, le=20)
    retries: int = Field(default=2, ge=0, le=5)


class TranslateProjectRequest(BaseModel):
    provider: TranslationProviderRequest
    options: TranslationOptionsRequest = Field(default_factory=TranslationOptionsRequest)


class TranslationTestResponse(BaseModel):
    status: Literal["ok"] = "ok"
    response_preview: str


class TranslationProfile(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    path: str = "/chat/completions"
    custom_headers: str = "{}"
    timeout_seconds: int = Field(default=120, ge=1, le=3600)
    concurrency: int = Field(default=1, ge=1, le=16)
    reasoning_effort: str = Field(default="", max_length=32)
    send_title: bool = True
    send_story_context: bool = True
    send_character_profiles: bool = True
    send_glossary: bool = True


class LocalModelConfigureRequest(BaseModel):
    mode: Literal["managed", "external"] = "managed"
    base_url: str = ""
    api_path: str = "/chat/completions"
    model: str = ""
    executable_path: str = ""
    model_path: str = ""
    runtime_variant: Literal["auto", "cpu", "vulkan", "cuda"] = "auto"
    port: int = Field(default=18080, ge=1024, le=65535)
    context_size: int = Field(default=8192, ge=2048, le=131072)
    gpu_layers: int = Field(default=-1, ge=-1, le=999)
    threads: int = Field(default=max(1, (os.cpu_count() or 2) // 2), ge=1, le=256)
    auto_start: bool = True
    make_default: bool = True

    @model_validator(mode="after")
    def validate_local_model_source(self) -> "LocalModelConfigureRequest":
        if self.mode == "external":
            if not self.base_url.strip():
                raise ValueError("base_url is required for an external endpoint")
            if not self.model.strip():
                raise ValueError("model is required for an external endpoint")
        else:
            if not self.executable_path.strip():
                raise ValueError("executable_path is required for a managed endpoint")
            if not self.model_path.strip():
                raise ValueError("model_path is required for a managed endpoint")
        return self


class LocalModelDeployRequest(BaseModel):
    model_id: str = ""
    runtime_variant: Literal["auto", "cpu", "vulkan", "cuda"] = "auto"
    port: int = Field(default=18080, ge=1024, le=65535)
    context_size: int = Field(default=8192, ge=2048, le=131072)
    gpu_layers: int = Field(default=-1, ge=-1, le=999)
    threads: int = Field(default=max(1, (os.cpu_count() or 2) // 2), ge=1, le=256)
    auto_start: bool = True
    make_default: bool = True
    startup_timeout_seconds: int = Field(default=180, ge=10, le=900)


class TranslationJobRequest(TranslateProjectRequest):
    pass


class FusionOptionsRequest(BaseModel):
    batch_size: int = Field(default=80, ge=1, le=500)
    context_cues: int = Field(default=3, ge=0, le=20)
    retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=1.0, ge=0, le=30)


class FusionJobRequest(BaseModel):
    provider: TranslationProviderRequest
    options: FusionOptionsRequest = Field(default_factory=FusionOptionsRequest)


class RegionUpdate(BaseModel):
    source_roi: NormalizedRegion | None = None
    target_roi: NormalizedRegion | None = None

    @model_validator(mode="after")
    def at_least_one_region(self) -> "RegionUpdate":
        if self.source_roi is None and self.target_roi is None:
            raise ValueError("at least one region must be supplied")
        return self


class ProjectContextUpdate(BaseModel):
    synopsis: str = ""
    characters: str = ""
    glossary: str = ""
    translation_style: str = ""


class OcrJobRequest(BaseModel):
    language: str = "en"
    device: str = "auto"
    sample_fps: float = Field(default=4.0, ge=0.5, le=30)
    high_accuracy: bool = True
    prefer_embedded: bool = True
    filter_noise: bool = True
    batch_size: int = Field(default=0, ge=0, le=64)


class FrameOcrRequest(BaseModel):
    timestamp_ms: int = Field(ge=0)
    bbox: NormalizedRegion
    language: str = "en"
    device: str = "auto"
    high_accuracy: bool = True


class FrameOcrDetection(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    bbox: NormalizedRegion
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")


class FrameOcrResponse(BaseModel):
    timestamp_ms: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    text: str
    confidence: float = Field(ge=0, le=1)
    detections: list[FrameOcrDetection] = Field(default_factory=list)


class OcrDeviceOption(BaseModel):
    id: str
    label: str
    available: bool
    reason: str | None = None


class OcrCapabilitiesResponse(BaseModel):
    paddle_available: bool
    paddle_version: str | None = None
    paddleocr_available: bool
    paddleocr_version: str | None = None
    cuda_compiled: bool
    cuda_device_count: int
    default_device: str
    cpu_onednn_enabled: bool = False
    devices: list[OcrDeviceOption]
    error: str | None = None


class UvrModelInfo(BaseModel):
    id: str
    label: str
    filename: str
    available: bool
    runtime: str = "uvr5-local-core"
    load_mode: str = "local-or-auto-download"
    root_path: str | None = None
    path: str | None = None
    config_path: str | None = None
    size_bytes: int | None = None
    download_size_mb: int | None = None


class DiarizationModelInfo(BaseModel):
    id: str = "nemo-titanet-marblenet"
    label: str = "NeMo TitaNet + MarbleNet"
    available: bool
    runtime: str = "nemo-clustering"
    load_mode: str = "local-or-auto-download"
    root_path: str
    path: str | None = None
    config_path: str | None = None
    size_bytes: int | None = None


class AsrModelOption(BaseModel):
    id: str
    language: str
    language_label: str
    label: str
    engine: Literal["nemo", "funasr"]
    repository: str
    description: str
    recommended: bool = True
    download_size_mb: int = Field(ge=0)
    installed: bool
    local_path: str
    supports_word_timestamps: bool = True
    supports_speaker_labels: bool = False


class AudioCapabilitiesResponse(BaseModel):
    ffmpeg_available: bool
    torch_available: bool
    torch_version: str | None = None
    audio_separator_available: bool
    nemo_available: bool
    funasr_available: bool
    cuda_available: bool
    cuda_device_count: int = Field(ge=0)
    cuda_device_names: list[str] = Field(default_factory=list)
    default_device: str
    uvr_model: UvrModelInfo
    diarization_model: DiarizationModelInfo
    asr_models: list[AsrModelOption]
    errors: list[str] = Field(default_factory=list)


class AudioJobRequest(BaseModel):
    language: str = "ja"
    model_id: str = ""
    device: str = "auto"
    separate_vocals: bool = True
    diarization: bool = True
    forced_alignment: bool = True
    slicer_threshold_db: float = Field(default=-34.0, ge=-100, le=0)
    slicer_min_length_ms: int = Field(default=4_000, ge=100, le=600_000)
    slicer_min_interval_ms: int = Field(default=200, ge=10, le=60_000)
    slicer_hop_size_ms: int = Field(default=10, ge=1, le=1_000)
    slicer_max_sil_kept_ms: int = Field(default=500, ge=1, le=60_000)
    slicer_max_length_ms: int = Field(default=30_000, ge=1_000, le=600_000)
    asr_batch_size: int = Field(default=4, ge=1, le=64)

    @model_validator(mode="after")
    def validate_slicer_parameters(self) -> "AudioJobRequest":
        if self.slicer_min_length_ms < self.slicer_min_interval_ms:
            raise ValueError("slicer_min_length_ms must be at least slicer_min_interval_ms")
        if self.slicer_min_interval_ms < self.slicer_hop_size_ms:
            raise ValueError("slicer_min_interval_ms must be at least slicer_hop_size_ms")
        if self.slicer_max_sil_kept_ms < self.slicer_hop_size_ms:
            raise ValueError("slicer_max_sil_kept_ms must be at least slicer_hop_size_ms")
        if self.slicer_max_length_ms < self.slicer_min_length_ms:
            raise ValueError("slicer_max_length_ms must be at least slicer_min_length_ms")
        return self


class HybridJobRequest(AudioJobRequest):
    sample_fps: float = Field(default=4.0, ge=0.5, le=30)
    high_accuracy: bool = True
    prefer_embedded: bool = True
    filter_noise: bool = True
    batch_size: int = Field(default=0, ge=0, le=64)


class ExportJobRequest(BaseModel):
    preview: bool = False
    start_ms: int = Field(default=0, ge=0)
    preview_duration_ms: int = Field(default=10_000, ge=1000, le=120_000)
    video_encoder: str = "libx264"
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = "medium"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    data_directory: str
