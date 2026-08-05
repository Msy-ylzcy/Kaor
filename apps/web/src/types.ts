export type WorkflowStepId =
  | "import"
  | "roi"
  | "ocr"
  | "csv"
  | "translate"
  | "layout"
  | "export";

export type SourceKind = "ocr" | "speech" | "manual" | "imported";
export type ReviewStatus = "pending" | "ocr_ok" | "needs_review" | "translated" | "approved";
export type JobStatus =
  | "queued"
  | "running"
  | "paused"
  | "failed"
  | "cancelled"
  | "completed";

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Cue {
  cue_id: string;
  revision?: number;
  start_ms: number;
  end_ms: number;
  group_id: string | null;
  overlap_group_id?: string | null;
  layer: number;
  track_id?: string;
  track?: number;
  speaker_id: string;
  speaker_name: string;
  speaker_color: string;
  speaker_confidence?: number | null;
  speaker_evidence?: Array<"color" | "audio" | "position" | "manual">;
  source_kind: SourceKind;
  source_text: string;
  ocr_confidence: number | null;
  target_text: string;
  review_status: ReviewStatus;
  bbox: BBox | null;
  source_bbox?: BBox | null;
  warnings?: string[];
}

export interface ProjectContext {
  synopsis: string;
  characters: string;
  glossary: string;
  translation_style: string;
}

export interface Project {
  id: string;
  title: string;
  video_name: string;
  video_url: string | null;
  duration_ms: number;
  width: number;
  height: number;
  fps: number;
  source_language: string;
  target_language: string;
  source_roi: BBox;
  target_roi: BBox;
  context: ProjectContext;
  audio_ready?: boolean;
  audio_error?: string | null;
  updated_at?: string;
}

export interface Job {
  id: string;
  kind: "ocr" | "uvr" | "slicer" | "asr" | "audio" | "hybrid" | "fusion" | "translation" | "preview" | "export" | "analysis" | "local-model-deploy";
  status: JobStatus;
  stage: string;
  progress: number;
  message: string;
  error?: { code: string; detail: string } | null;
  result?: Record<string, unknown> | null;
  snapshot?: JobSnapshot | null;
  created_at?: string;
}

export interface OcrLiveText {
  text: string;
  color: string;
  confidence: number;
}

export interface OcrJobSnapshot {
  timestamp_ms: number;
  cues: Cue[];
  current: OcrLiveText[];
  metrics: Record<string, number | string>;
}

export interface AiTrace {
  phase?: string;
  stage?: string;
  status?: string;
  message?: string;
  reasoning?: string;
  reasoning_content?: string;
  content_preview?: string;
  output_preview?: string;
  batch?: number;
  batch_index?: number;
  total_batches?: number;
  completed?: number;
  total?: number;
  attempt?: number;
  events?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface JobSnapshot {
  timestamp_ms?: number;
  cues?: Cue[];
  current?: OcrLiveText[];
  metrics?: Record<string, number | string>;
  ai_trace?: AiTrace | null;
}

export interface FrameOcrDetection {
  text: string;
  confidence: number | null;
  bbox?: BBox | null;
}

export interface FrameOcrResult {
  text: string;
  confidence: number | null;
  detections?: FrameOcrDetection[];
}

export interface OcrDeviceOption {
  id: string;
  label: string;
  available: boolean;
  reason: string | null;
}

export interface OcrCapabilities {
  paddle_available: boolean;
  paddle_version: string | null;
  paddleocr_available: boolean;
  paddleocr_version: string | null;
  cuda_compiled: boolean;
  cuda_device_count: number;
  default_device: string;
  cpu_onednn_enabled: boolean;
  devices: OcrDeviceOption[];
  error: string | null;
}

export interface LocalAudioModelStatus {
  id: string;
  label: string;
  available: boolean;
  runtime: string;
  load_mode: string;
  root_path: string | null;
  path: string | null;
  config_path: string | null;
  size_bytes: number | null;
  download_size_mb: number | null;
}

export interface AsrModelOption {
  id: string;
  label: string;
  language: string;
  language_label: string;
  engine: string;
  installed: boolean;
  recommended: boolean;
  description: string;
  download_size_mb: number | null;
  supports_word_timestamps: boolean;
  supports_speaker_labels: boolean;
}

export interface AudioCapabilities {
  torch_available: boolean;
  audio_separator_available: boolean;
  ffmpeg_available: boolean;
  cuda_available: boolean;
  cuda_device_count: number;
  default_device: string;
  uvr_model: LocalAudioModelStatus;
  diarization_model: LocalAudioModelStatus;
  asr_models: AsrModelOption[];
  errors: string[];
}

export interface AudioRecognitionOptions {
  language: string;
  model_id: string;
  device: string;
  separate_vocals: boolean;
  diarization: boolean;
  forced_alignment: boolean;
  slicer_threshold_db: number;
  slicer_min_length_ms: number;
  slicer_max_length_ms: number;
  slicer_min_interval_ms: number;
  slicer_hop_size_ms: number;
  slicer_max_sil_kept_ms: number;
  asr_batch_size: number;
}

export interface HybridRecognitionOptions extends AudioRecognitionOptions {
  sample_fps: number;
  high_accuracy: boolean;
  prefer_embedded: boolean;
  filter_noise: boolean;
  batch_size: number;
}

export interface TranslationSettings {
  base_url: string;
  api_key: string;
  model: string;
  reasoning_effort: "" | "minimal" | "low" | "medium" | "high" | "xhigh";
  path: string;
  custom_headers: string;
  timeout_seconds: number;
  concurrency: number;
  send_title: boolean;
  send_story_context: boolean;
  send_character_profiles: boolean;
  send_glossary: boolean;
}

export interface TranslationModelOption {
  id: string;
  owned_by: string | null;
}

export type LocalRuntimeVariant = "auto" | "cpu" | "vulkan" | "cuda";

export interface LocalGpuAdapter {
  name: string;
  vendor: "amd" | "nvidia" | "intel" | "unknown";
  memory_bytes: number | null;
  driver_version: string;
}

export interface LocalHardwareProfile {
  system: string;
  architecture: string;
  cpu_name: string;
  logical_cpus: number;
  memory_bytes: number | null;
  gpus: LocalGpuAdapter[];
  build_profile: string;
}

export interface LocalModelRecommendation {
  runtime_variant: Exclude<LocalRuntimeVariant, "auto">;
  model_id: string;
  model_label: string;
  reason: string;
}

export interface LocalModelCatalogItem {
  id: string;
  label: string;
  filename: string;
  url: string;
  size_bytes: number;
  minimum_memory_bytes: number;
  description: string;
  sha256: string | null;
  revision: string;
  license_url: string;
  installed: boolean;
}

export interface LocalTranslationProvider {
  base_url: string;
  api_key: string;
  model: string;
  api_path: string;
  custom_headers: Record<string, string>;
  timeout_seconds: number;
  temperature: number;
  json_mode: boolean;
  reasoning_effort: string;
}

export interface LocalModelConfiguration {
  schema_version: number;
  mode: "managed" | "external";
  base_url: string;
  api_path: string;
  model: string;
  executable_path?: string;
  model_path?: string;
  runtime_variant?: Exclude<LocalRuntimeVariant, "auto">;
  port?: number;
  context_size?: number;
  gpu_layers?: number;
  threads?: number;
  auto_start?: boolean;
}

export interface LocalTranslationRuntimeStatus {
  state: "not_configured" | "ready" | "starting" | "failed" | "unreachable" | "stopped";
  ready: boolean;
  process_running: boolean;
  managed_process: boolean;
  configuration: LocalModelConfiguration | null;
  provider: LocalTranslationProvider | null;
  error: string | null;
  log_tail: string;
  recommendation: LocalModelRecommendation;
  catalog: LocalModelCatalogItem[];
  hardware?: LocalHardwareProfile;
  remote_profile_available: boolean;
}

export interface LocalModelCatalogResponse {
  hardware: LocalHardwareProfile;
  recommendation: LocalModelRecommendation;
  models: LocalModelCatalogItem[];
}

export interface DiagnosticLogSource {
  id: string;
  name: string;
  size_bytes: number;
  updated_at: string;
}

export interface DiagnosticRepairGuide {
  id: string;
  title: string;
  summary: string;
  patterns: string[];
  steps: string[];
  anchor: string;
}

export interface DiagnosticLogEntry {
  id: string;
  timestamp: string;
  level: "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL" | string;
  logger: string;
  message: string;
  source_id: string;
  source_name: string;
  guide_ids: string[];
}

export interface DiagnosticLogPayload {
  sources: DiagnosticLogSource[];
  entries: DiagnosticLogEntry[];
  guides: DiagnosticRepairGuide[];
}

export interface LayoutSettings {
  font_family: string;
  font_size: number;
  outline: number;
  max_lines: number;
  max_chars: number;
  color_mode: "speaker" | "single";
  avoid_faces: boolean;
  avoid_source: boolean;
  overlap_mode: "layers" | "split" | "stack";
}

export interface WorkspaceData {
  project: Project;
  cues: Cue[];
  ocr_cues?: Cue[];
  speech_cues?: Cue[];
  asr_cues?: Cue[];
  jobs: Job[];
  translation_settings: TranslationSettings;
  layout_settings: LayoutSettings;
}

export type RegionKind = "source" | "target";
export type DockView = "timeline" | "cues";
export type CueEvidenceView = "final" | "ocr" | "asr";
