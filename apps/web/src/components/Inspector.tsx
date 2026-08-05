import {
  AlertTriangle,
  AudioLines,
  Braces,
  CheckCircle2,
  ChevronRight,
  CircleStop,
  Combine,
  Cpu,
  Download,
  Eye,
  EyeOff,
  FileOutput,
  FileVideo2,
  FolderOpen,
  Gauge,
  HardDrive,
  KeyRound,
  Languages,
  Layers3,
  LoaderCircle,
  Palette,
  Pipette,
  Play,
  RefreshCw,
  Save,
  ScanLine,
  ScanText,
  Scissors,
  Server,
  ShieldCheck,
  Sparkles,
  Table2,
  TestTube2,
  Trash2,
  UsersRound,
  XCircle,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { LocalModelPanel } from "./LocalModelPanel";
import type {
  AudioRecognitionOptions,
  Cue,
  HybridRecognitionOptions,
  Job,
  LayoutSettings,
  AudioCapabilities,
  AsrModelOption,
  OcrCapabilities,
  Project,
  ProjectContext,
  RegionKind,
  TranslationSettings,
  TranslationModelOption,
  WorkflowStepId,
} from "../types";
import { confidenceLabel, formatTime } from "../utils";

interface InspectorProps {
  activeStep: WorkflowStepId;
  project: Project;
  cues: Cue[];
  jobs: Job[];
  selectedCue: Cue | null;
  isDemo: boolean;
  ocrCapabilities: OcrCapabilities | null;
  audioCapabilities: AudioCapabilities | null;
  asrModels: AsrModelOption[];
  translationSettings: TranslationSettings;
  layoutSettings: LayoutSettings;
  onImport: (file: File) => void;
  onResetWorkspace: () => Promise<void>;
  onDrawRegion: (kind: RegionKind) => void;
  onTranslationSettingsChange: (
    settings: TranslationSettings | ((current: TranslationSettings) => TranslationSettings),
  ) => void;
  onLayoutSettingsChange: (settings: LayoutSettings) => void;
  onContextChange: (context: ProjectContext) => void;
  onTargetLanguageChange: (language: string) => void;
  onRecognitionLanguageChange: (language: string) => void;
  onCueColorChange: (color: string) => void;
  onSpeakerColorChange: (color: string) => void;
  onStartEyedropper: () => void;
  onSaveTranslation: () => Promise<void>;
  onTestTranslation: () => Promise<boolean>;
  onFetchTranslationModels: () => Promise<TranslationModelOption[]>;
  onSaveLayout: () => Promise<void>;
  onStartJob: (kind: Job["kind"], payload?: Record<string, unknown>) => void;
  onCancelJob: (job: Job) => void;
}

const stepMeta: Record<WorkflowStepId, { eyebrow: string; title: string; description: string }> = {
  import: { eyebrow: "STEP 01", title: "媒体导入", description: "视频只在本机解析与缓存" },
  roi: { eyebrow: "STEP 02", title: "字幕区域", description: "框选识别区与译文安全区" },
  ocr: { eyebrow: "STEP 03", title: "高精度识别", description: "检测、多帧投票与低置信复核" },
  csv: { eyebrow: "STEP 04", title: "字幕校对", description: "时间、角色、原文与重叠关系" },
  translate: { eyebrow: "STEP 05", title: "上下文翻译", description: "在线接口、本地模型与中转站" },
  layout: { eyebrow: "STEP 06", title: "字幕排版", description: "颜色、图层与内容避让" },
  export: { eyebrow: "STEP 07", title: "导出交付", description: "CSV、ASS、软字幕或硬字幕" },
};

const targetLanguages = [
  ["zh-CN", "简体中文"],
  ["zh-TW", "繁體中文"],
  ["en", "English"],
  ["ja", "日本語"],
  ["ko", "한국어"],
  ["es", "Español"],
  ["fr", "Français"],
  ["de", "Deutsch"],
  ["ru", "Русский"],
  ["pt-BR", "Português (Brasil)"],
  ["vi", "Tiếng Việt"],
  ["th", "ไทย"],
  ["id", "Bahasa Indonesia"],
  ["ar", "العربية"],
] as const;

export function Inspector(props: InspectorProps) {
  const meta = stepMeta[props.activeStep];
  return (
    <aside className="inspector">
      <header className="inspector-header">
        <span>{meta.eyebrow}</span>
        <h2>{meta.title}</h2>
        <p>{meta.description}</p>
      </header>
      <div className="inspector-scroll">
        {props.activeStep === "import" && <ImportInspector {...props} />}
        {props.activeStep === "roi" && <RegionInspector {...props} />}
        {props.activeStep === "ocr" && <OcrInspector {...props} />}
        {props.activeStep === "csv" && <CsvInspector {...props} />}
        {props.activeStep === "translate" && <TranslationInspector {...props} />}
        {props.activeStep === "layout" && <LayoutInspector {...props} />}
        {props.activeStep === "export" && <ExportInspector {...props} />}
        <JobsSection projectId={props.project.id} jobs={props.jobs} onCancel={props.onCancelJob} />
      </div>
    </aside>
  );
}

function SectionTitle({ icon: Icon, children }: { icon: LucideIcon; children: React.ReactNode }) {
  return (
    <h3 className="section-title">
      <Icon size={15} />
      {children}
    </h3>
  );
}

function ImportInspector(props: InspectorProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [resetting, setResetting] = useState(false);
  const resetWorkspace = async () => {
    if (!window.confirm("重置当前工作区？字幕、任务记录与中间结果会被清除。")) return;
    setResetting(true);
    try {
      await props.onResetWorkspace();
    } finally {
      setResetting(false);
    }
  };
  return (
    <>
      <section className="inspector-section">
        <SectionTitle icon={FileVideo2}>当前媒体</SectionTitle>
        <dl className="property-list">
          <div><dt>文件</dt><dd title={props.project.video_name}>{props.project.video_name}</dd></div>
          <div><dt>画面</dt><dd>{props.project.width} × {props.project.height}</dd></div>
          <div><dt>帧率</dt><dd>{props.project.fps.toFixed(3)} fps</dd></div>
          <div><dt>时长</dt><dd>{formatTime(props.project.duration_ms, true)}</dd></div>
        </dl>
        <input
          ref={fileRef}
          type="file"
          className="visually-hidden"
          accept="video/*,.mkv,.mp4,.mov,.webm,.avi"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) props.onImport(file);
          }}
        />
        <button className="drop-target" type="button" onClick={() => fileRef.current?.click()}>
          <FolderOpen size={21} />
          <strong>选择或拖入视频</strong>
          <span>MP4 · MKV · MOV · WebM</span>
        </button>
        <button
          className="button secondary danger-button full-width workspace-reset-button"
          type="button"
          onClick={() => void resetWorkspace()}
          disabled={resetting}
        >
          {resetting ? <LoaderCircle className="spin" size={14} /> : <Trash2 size={14} />}
          {resetting ? "正在重置" : "重置当前工作区"}
        </button>
      </section>
      <section className="inspector-section privacy-note">
        <ShieldCheck size={17} />
        <div>
          <strong>本地媒体管线</strong>
          <span>视频、音频、帧缓存与 OCR 结果均留在本机。</span>
        </div>
      </section>
    </>
  );
}

function RegionInspector(props: InspectorProps) {
  return (
    <>
      <section className="inspector-section">
        <SectionTitle icon={ScanLine}>原字幕识别区</SectionTitle>
        <RegionValues project={props.project} kind="source" />
        <button className="button secondary full-width" type="button" onClick={() => props.onDrawRegion("source")}>
          <ScanLine size={15} />
          在画面中重新框选
        </button>
      </section>
      <section className="inspector-section">
        <SectionTitle icon={Palette}>译文安全区</SectionTitle>
        <RegionValues project={props.project} kind="target" />
        <button className="button secondary full-width" type="button" onClick={() => props.onDrawRegion("target")}>
          <Palette size={15} />
          调整译文位置
        </button>
      </section>
      <section className="inspector-section">
        <SectionTitle icon={Layers3}>并发对白</SectionTitle>
        <ToggleRow checked label="保留重叠时间段" detail="分别写入 layer 与 group_id" />
        <ToggleRow checked label="颜色轨道分离" detail="同屏不同颜色保持为独立 Cue" />
      </section>
    </>
  );
}

function RegionValues({ project, kind }: { project: Project; kind: RegionKind }) {
  const box = kind === "source" ? project.source_roi : project.target_roi;
  return (
    <div className="coordinate-grid">
      {(["x", "y", "width", "height"] as const).map((field) => (
        <label key={field}>
          <span>{field === "width" ? "W" : field === "height" ? "H" : field.toUpperCase()}</span>
          <input value={`${(box[field] * 100).toFixed(1)}%`} readOnly />
        </label>
      ))}
    </div>
  );
}

function OcrInspector(props: InspectorProps) {
  const [mode, setMode] = useState<"subtitle" | "audio" | "hybrid">("subtitle");
  const [sampleFps, setSampleFps] = useState(4);
  const [highAccuracy, setHighAccuracy] = useState(true);
  const [filterNoise, setFilterNoise] = useState(true);
  const [batchSize, setBatchSize] = useState(0);
  const [device, setDevice] = useState("auto");
  const [audioDevice, setAudioDevice] = useState("auto");
  const [forcedAlignment, setForcedAlignment] = useState(true);
  const [diarization, setDiarization] = useState(true);
  const [slicerThresholdDb, setSlicerThresholdDb] = useState(-34);
  const [slicerMinLengthMs, setSlicerMinLengthMs] = useState(4_000);
  const [slicerMaxLengthMs, setSlicerMaxLengthMs] = useState(30_000);
  const [slicerMinIntervalMs, setSlicerMinIntervalMs] = useState(200);
  const [slicerHopSizeMs, setSlicerHopSizeMs] = useState(10);
  const [slicerMaxSilKeptMs, setSlicerMaxSilKeptMs] = useState(500);
  const [asrBatchSize, setAsrBatchSize] = useState(4);
  const [language, setLanguage] = useState(
    props.project.source_language === "auto" ? "" : props.project.source_language,
  );
  const [modelId, setModelId] = useState("");
  const devices = props.ocrCapabilities?.devices ?? [
    { id: "auto", label: "自动", available: true, reason: null },
    { id: "cpu", label: "CPU", available: true, reason: null },
  ];
  const automaticDevice = props.ocrCapabilities?.default_device ?? "自动检测";
  const asrModels = props.asrModels;
  const languages = useMemo(() => {
    const unique = new Map<string, string>();
    for (const model of asrModels) {
      if (!unique.has(model.language)) unique.set(model.language, model.language_label);
    }
    return Array.from(unique, ([value, label]) => ({ value, label }));
  }, [asrModels]);
  const languageModels = useMemo(
    () => asrModels.filter((model) => model.language === language),
    [asrModels, language],
  );
  const selectedModel = languageModels.find((model) => model.id === modelId) ?? null;

  useEffect(() => {
    if (!languages.length || languages.some((item) => item.value === language)) return;
    const preferred = asrModels.find(
      (model) => model.recommended && model.language === props.project.source_language,
    ) ?? asrModels.find((model) => model.recommended) ?? asrModels[0];
    if (preferred) setLanguage(preferred.language);
  }, [asrModels, language, languages, props.project.source_language]);

  useEffect(() => {
    if (!languageModels.length) {
      setModelId("");
      return;
    }
    if (languageModels.some((model) => model.id === modelId)) return;
    const preferred = languageModels.find((model) => model.recommended) ?? languageModels[0];
    setModelId(preferred.id);
  }, [languageModels, modelId]);

  useEffect(() => {
    props.onRecognitionLanguageChange(language);
  }, [language]);

  const ocrPayload = {
    language,
    sample_fps: sampleFps,
    high_accuracy: highAccuracy,
    filter_noise: filterNoise,
    batch_size: batchSize,
    device,
  };
  const normalizedSlicerHopSizeMs = clampAudioInteger(slicerHopSizeMs, 1, 1_000);
  const normalizedSlicerMinIntervalMs = Math.max(
    normalizedSlicerHopSizeMs,
    clampAudioInteger(slicerMinIntervalMs, 10, 60_000),
  );
  const normalizedSlicerMinLengthMs = Math.max(
    normalizedSlicerMinIntervalMs,
    clampAudioInteger(slicerMinLengthMs, 100, 600_000),
  );
  const normalizedSlicerMaxLengthMs = Math.max(
    normalizedSlicerMinLengthMs,
    clampAudioInteger(slicerMaxLengthMs, 1_000, 600_000),
  );
  const normalizedSlicerMaxSilKeptMs = Math.max(
    normalizedSlicerHopSizeMs,
    clampAudioInteger(slicerMaxSilKeptMs, 1, 60_000),
  );
  const audioPayload: AudioRecognitionOptions = {
    language,
    model_id: modelId,
    device: audioDevice,
    separate_vocals: true,
    forced_alignment: forcedAlignment,
    diarization,
    slicer_threshold_db: clampAudioValue(slicerThresholdDb, -100, 0),
    slicer_min_length_ms: normalizedSlicerMinLengthMs,
    slicer_max_length_ms: normalizedSlicerMaxLengthMs,
    slicer_min_interval_ms: normalizedSlicerMinIntervalMs,
    slicer_hop_size_ms: normalizedSlicerHopSizeMs,
    slicer_max_sil_kept_ms: normalizedSlicerMaxSilKeptMs,
    asr_batch_size: clampAudioInteger(asrBatchSize, 1, 64),
  };
  const resetAudioParameters = () => {
    setSlicerThresholdDb(-34);
    setSlicerMinLengthMs(4_000);
    setSlicerMaxLengthMs(30_000);
    setSlicerMinIntervalMs(200);
    setSlicerHopSizeMs(10);
    setSlicerMaxSilKeptMs(500);
    setAsrBatchSize(4);
  };
  const slicerPayload = {
    slicer_threshold_db: audioPayload.slicer_threshold_db,
    slicer_min_length_ms: audioPayload.slicer_min_length_ms,
    slicer_max_length_ms: audioPayload.slicer_max_length_ms,
    slicer_min_interval_ms: audioPayload.slicer_min_interval_ms,
    slicer_hop_size_ms: audioPayload.slicer_hop_size_ms,
    slicer_max_sil_kept_ms: audioPayload.slicer_max_sil_kept_ms,
  };
  const startRecognition = () => {
    if (mode === "subtitle") {
      props.onStartJob("ocr", ocrPayload);
      return;
    }
    if (mode === "audio") {
      props.onStartJob("audio", { ...audioPayload });
      return;
    }
    const hybridPayload: HybridRecognitionOptions = {
      ...audioPayload,
      sample_fps: sampleFps,
      high_accuracy: highAccuracy,
      prefer_embedded: true,
      filter_noise: filterNoise,
      batch_size: batchSize,
    };
    props.onStartJob("hybrid", { ...hybridPayload });
  };
  const startAudioPipeline = () => props.onStartJob("audio", { ...audioPayload });

  return (
    <>
      <section className="inspector-section recognition-mode-section">
        <SectionTitle icon={Sparkles}>识别来源</SectionTitle>
        <div className="recognition-mode-control" role="tablist" aria-label="识别模式">
          <button type="button" className={mode === "subtitle" ? "is-active" : ""} onClick={() => setMode("subtitle")} role="tab" aria-selected={mode === "subtitle"}>
            <ScanText size={14} /><span>字幕</span>
          </button>
          <button type="button" className={mode === "audio" ? "is-active" : ""} onClick={() => setMode("audio")} role="tab" aria-selected={mode === "audio"}>
            <AudioLines size={14} /><span>音频</span>
          </button>
          <button type="button" className={mode === "hybrid" ? "is-active" : ""} onClick={() => setMode("hybrid")} role="tab" aria-selected={mode === "hybrid"}>
            <Combine size={14} /><span>混合</span>
          </button>
        </div>
      </section>

      {mode === "subtitle" && (
        <>
          <OcrPipeline highAccuracy={highAccuracy} filterNoise={filterNoise} />
          <OcrControls
            devices={devices}
            automaticDevice={automaticDevice}
            device={device}
            setDevice={setDevice}
            sampleFps={sampleFps}
            setSampleFps={setSampleFps}
            batchSize={batchSize}
            setBatchSize={setBatchSize}
            highAccuracy={highAccuracy}
            setHighAccuracy={setHighAccuracy}
            filterNoise={filterNoise}
            setFilterNoise={setFilterNoise}
          />
        </>
      )}

      {(mode === "audio" || mode === "hybrid") && (
        <>
          <section className="inspector-section audio-stage-section">
            <SectionTitle icon={AudioLines}>音频三阶段</SectionTitle>
            <div className="audio-stage-list">
              <div className="audio-stage-row">
                <span className="audio-stage-index">01</span>
                <p>
                  <strong>UVR5 人声分离</strong>
                  <small title={props.audioCapabilities?.uvr_model.path ?? undefined}>
                    {props.audioCapabilities?.uvr_model.available
                      ? `${props.audioCapabilities.uvr_model.label} · 本地就绪`
                      : props.audioCapabilities?.uvr_model.label ?? "BS-RoFormer"}
                  </small>
                </p>
                <button className="button secondary compact-button" type="button" aria-label="运行 UVR5 人声分离" onClick={() => props.onStartJob("uvr", { device: audioDevice })}>
                  <AudioLines size={12} />运行
                </button>
              </div>
              <div className="audio-stage-row">
                <span className="audio-stage-index">02</span>
                <p>
                  <strong>静音切分</strong>
                  <small>{audioPayload.slicer_threshold_db} dB · {audioPayload.slicer_min_length_ms}–{audioPayload.slicer_max_length_ms} ms</small>
                </p>
                <button className="button secondary compact-button" type="button" aria-label="执行静音切分" onClick={() => props.onStartJob("slicer", slicerPayload)}>
                  <Scissors size={12} />切分
                </button>
              </div>
              <div className="audio-stage-row">
                <span className="audio-stage-index">03</span>
                <p>
                  <strong>ASR 打标</strong>
                  <small>{selectedModel?.label ?? "选择语言专用模型"} · 批 {audioPayload.asr_batch_size}</small>
                </p>
                <button
                  className="button secondary compact-button"
                  type="button"
                  aria-label="开始 ASR 打标"
                  onClick={() => props.onStartJob("asr", { ...audioPayload })}
                  disabled={!language || !modelId}
                >
                  <ScanText size={12} />打标
                </button>
              </div>
            </div>
          </section>
          <section className="inspector-section">
            <SectionTitle icon={Languages}>三阶段参数</SectionTitle>
            <label className="field-label">
              <span>源语言</span>
              <select value={language} onChange={(event) => setLanguage(event.target.value)} disabled={!languages.length}>
                {!languages.length && <option value="">等待本地模型目录</option>}
                {languages.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
              </select>
            </label>
            <div className="field-row audio-parameter-row">
              <AudioNumberField label="静音阈值 · dB" value={slicerThresholdDb} min={-100} max={0} step={1} onChange={setSlicerThresholdDb} />
              <AudioNumberField label="ASR 批大小" value={asrBatchSize} min={1} max={64} step={1} integer onChange={setAsrBatchSize} />
            </div>
            <div className="field-row audio-parameter-row">
              <AudioNumberField
                label="最短片段 · ms"
                value={slicerMinLengthMs}
                min={audioPayload.slicer_min_interval_ms}
                max={600_000}
                step={100}
                integer
                onChange={setSlicerMinLengthMs}
                onCommit={(value) => setSlicerMaxLengthMs((current) => Math.max(current, value))}
              />
              <AudioNumberField
                label="最长片段 · ms"
                value={slicerMaxLengthMs}
                min={Math.max(1_000, audioPayload.slicer_min_length_ms)}
                max={600_000}
                step={1_000}
                integer
                onChange={setSlicerMaxLengthMs}
              />
            </div>
            <div className="field-row audio-parameter-row">
              <AudioNumberField
                label="最短静音 · ms"
                value={slicerMinIntervalMs}
                min={audioPayload.slicer_hop_size_ms}
                max={60_000}
                step={10}
                integer
                onChange={setSlicerMinIntervalMs}
                onCommit={(value) => {
                  setSlicerMinLengthMs((current) => Math.max(current, value));
                  setSlicerMaxLengthMs((current) => Math.max(current, value));
                }}
              />
              <AudioNumberField
                label="检测步长 · ms"
                value={slicerHopSizeMs}
                min={1}
                max={1_000}
                step={1}
                integer
                onChange={setSlicerHopSizeMs}
                onCommit={(value) => {
                  setSlicerMinIntervalMs((current) => Math.max(current, value));
                  setSlicerMinLengthMs((current) => Math.max(current, value));
                  setSlicerMaxLengthMs((current) => Math.max(current, value));
                  setSlicerMaxSilKeptMs((current) => Math.max(current, value));
                }}
              />
            </div>
            <div className="field-row audio-parameter-row">
              <AudioNumberField label="保留静音 · ms" value={slicerMaxSilKeptMs} min={audioPayload.slicer_hop_size_ms} max={60_000} step={10} integer onChange={setSlicerMaxSilKeptMs} />
            </div>
            <button className="button ghost compact-button audio-parameter-reset" type="button" onClick={resetAudioParameters} title="恢复音频切分与批处理默认值">
              <RefreshCw size={12} />恢复默认参数
            </button>
            <label className="field-label">
              <span>专用 ASR 模型</span>
              <select value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={!languageModels.length}>
                {!languageModels.length && <option value="">当前语言暂无模型</option>}
                {languageModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}{model.recommended ? " · 推荐" : ""}{model.installed ? " · 已下载" : ""}
                  </option>
                ))}
              </select>
              {selectedModel && <small className="field-help">{selectedModel.description}</small>}
            </label>
            <label className="field-label">
              <span>计算设备</span>
              <select value={audioDevice} onChange={(event) => setAudioDevice(event.target.value)}>
                <option value="auto">自动（当前 {props.audioCapabilities?.default_device ?? "检测中"}）</option>
                <option value="cuda" disabled={!props.audioCapabilities?.cuda_available}>NVIDIA CUDA{props.audioCapabilities?.cuda_device_count ? ` · ${props.audioCapabilities.cuda_device_count} 张` : ""}</option>
                <option value="cpu">CPU</option>
              </select>
            </label>
            <ToggleRow
              checked={forcedAlignment}
              onChange={setForcedAlignment}
              label="时间边界精修"
              detail={selectedModel?.supports_word_timestamps
                ? "结合模型时间戳与本地语音活动修正字幕起止边界"
                : "使用本地语音活动修正模型给出的粗略时间"}
            />
            <ToggleRow
              checked={diarization}
              onChange={setDiarization}
              label="说话人分离"
              detail={selectedModel?.supports_speaker_labels
                ? "保留模型输出的说话人标签"
                : props.audioCapabilities?.diarization_model.available
                  ? "使用本地 VAD 与声纹聚类区分人物"
                  : "首次启用时下载本地 VAD 与声纹模型"}
            />
            <AudioRuntimeStatus capabilities={props.audioCapabilities} model={selectedModel} />
          </section>
          {mode === "hybrid" && (
            <OcrControls
              compact
              hideDevice
              devices={devices}
              automaticDevice={automaticDevice}
              device={device}
              setDevice={setDevice}
              sampleFps={sampleFps}
              setSampleFps={setSampleFps}
              batchSize={batchSize}
              setBatchSize={setBatchSize}
              highAccuracy={highAccuracy}
              setHighAccuracy={setHighAccuracy}
              filterNoise={filterNoise}
              setFilterNoise={setFilterNoise}
            />
          )}
        </>
      )}

      <section className="inspector-section action-section">
        <button
          className="button primary full-width"
          type="button"
          onClick={startRecognition}
          disabled={mode !== "subtitle" && (!language || !modelId)}
        >
          <Play size={15} fill="currentColor" />
          {mode === "subtitle" ? "开始字幕识别" : mode === "audio" ? "一键完成音频三阶段" : "开始双源识别"}
        </button>
        {mode === "hybrid" && (
          <>
            <button className="button secondary full-width" type="button" onClick={startAudioPipeline} disabled={!language || !modelId}>
              <AudioLines size={15} />仅运行音频三阶段
            </button>
            <button className="button secondary full-width" type="button" onClick={() => props.onStartJob("fusion")}>
              <Combine size={15} />AI 融合校正
            </button>
          </>
        )}
        <span><Cpu size={13} /> 识别与人声分离仅使用本机算力</span>
      </section>
    </>
  );
}

function AudioNumberField({
  label,
  value,
  min,
  max,
  step,
  integer = false,
  onChange,
  onCommit,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  integer?: boolean;
  onChange: (value: number) => void;
  onCommit?: (value: number) => void;
}) {
  const normalize = (next: number) => {
    const bounded = clampAudioValue(next, min, max);
    const normalized = integer ? Math.round(bounded) : bounded;
    onChange(normalized);
    onCommit?.(normalized);
  };
  return (
    <label className="field-label audio-number-field">
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => {
          if (Number.isFinite(event.target.valueAsNumber)) onChange(event.target.valueAsNumber);
        }}
        onBlur={() => normalize(value)}
      />
    </label>
  );
}

function clampAudioValue(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}

function clampAudioInteger(value: number, min: number, max: number): number {
  return Math.round(clampAudioValue(value, min, max));
}

interface OcrControlsProps {
  compact?: boolean;
  hideDevice?: boolean;
  devices: OcrCapabilities["devices"];
  automaticDevice: string;
  device: string;
  setDevice: (value: string) => void;
  sampleFps: number;
  setSampleFps: (value: number) => void;
  batchSize: number;
  setBatchSize: (value: number) => void;
  highAccuracy: boolean;
  setHighAccuracy: (value: boolean) => void;
  filterNoise: boolean;
  setFilterNoise: (value: boolean) => void;
}

function OcrPipeline({ highAccuracy, filterNoise }: Pick<OcrControlsProps, "highAccuracy" | "filterNoise">) {
  return (
    <section className="inspector-section">
      <SectionTitle icon={Sparkles}>精度方案</SectionTitle>
      <div className="pipeline-list">
        <div><span>1</span><p><strong>视频帧采样</strong><small>按设定帧率读取字幕区域</small></p><CheckCircle2 size={14} /></div>
        <div><span>2</span><p><strong>PP-OCRv6 检测识别</strong><small>完全在本地运行</small></p><CheckCircle2 size={14} /></div>
        <div><span>3</span><p><strong>增强变体共识</strong><small>多路 OCR 结果投票</small></p>{highAccuracy ? <CheckCircle2 size={14} /> : <XCircle size={14} />}</div>
        <div><span>4</span><p><strong>噪声过滤与时间跟踪</strong><small>拦截单帧碎片并合并持续字幕</small></p>{filterNoise ? <CheckCircle2 size={14} /> : <XCircle size={14} />}</div>
        <div><span>5</span><p><strong>字幕颜色聚类</strong><small>按文字颜色区分说话人</small></p><CheckCircle2 size={14} /></div>
      </div>
    </section>
  );
}

function OcrControls(props: OcrControlsProps) {
  return (
    <section className="inspector-section">
      <SectionTitle icon={Gauge}>{props.compact ? "画面识别参数" : "识别参数"}</SectionTitle>
      {!props.hideDevice && (
        <label className="field-label">
          <span>计算设备</span>
          <select value={props.device} onChange={(event) => props.setDevice(event.target.value)}>
            {props.devices.map((option) => (
              <option key={option.id} value={option.id} disabled={!option.available} title={option.reason ?? undefined}>
                {option.id === "auto" ? `自动（当前 ${props.automaticDevice}）` : option.label}
                {!option.available ? " · 不可用" : ""}
              </option>
            ))}
          </select>
          {!props.compact && <small className="field-help">自动模式在 NVIDIA 版检测到 CUDA 时使用 GPU，否则使用 CPU。</small>}
        </label>
      )}
      <label className="slider-field">
        <span><b>采样帧率</b><output>{props.sampleFps} fps</output></span>
        <input type="range" min="0.5" max="12" step="0.5" value={props.sampleFps} onChange={(event) => props.setSampleFps(Number(event.target.value))} />
      </label>
      <label className="field-label">
        <span>帧批处理</span>
        <select value={props.batchSize} onChange={(event) => props.setBatchSize(Number(event.target.value))}>
          <option value={0}>自动（按显存）</option>
          <option value={6}>6 帧</option>
          <option value={12}>12 帧</option>
          <option value={16}>16 帧</option>
          <option value={24}>24 帧</option>
          <option value={32}>32 帧</option>
          <option value={40}>40 帧</option>
          <option value={48}>48 帧</option>
          <option value={64}>64 帧</option>
        </select>
        {!props.compact && <small className="field-help">自动模式按显存选择批量；16 GB 档使用 40 帧，出现显存不足时可手动降低。</small>}
      </label>
      <ToggleRow checked={props.highAccuracy} onChange={props.setHighAccuracy} label="高精度共识" detail="对增强后的同一画面执行多路 OCR 投票；速度较慢" />
      <ToggleRow checked={props.filterNoise} onChange={props.setFilterNoise} label="过滤疑似噪声" detail="移除低质量单帧文字和短计数碎片；关闭后保留原始结果" />
    </section>
  );
}

function AudioRuntimeStatus({ capabilities, model }: { capabilities: AudioCapabilities | null; model: AudioCapabilities["asr_models"][number] | null }) {
  if (!capabilities) {
    return <div className="audio-runtime-state is-warning"><LoaderCircle size={14} /><span><strong>运行时状态未取得</strong><small>本地服务连接后自动检测</small></span></div>;
  }
  const runtimeReady = capabilities.torch_available && capabilities.audio_separator_available && capabilities.ffmpeg_available && capabilities.uvr_model.available;
  return (
    <div className={`audio-runtime-state ${runtimeReady ? "is-ready" : "is-warning"}`}>
      {runtimeReady ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
      <span>
        <strong>{runtimeReady ? "音频运行时就绪" : "音频运行时待安装"}</strong>
        <small>
          {model?.installed
            ? "ASR 模型已下载"
            : model
              ? `首次任务自动下载${model.download_size_mb ? ` · 约 ${model.download_size_mb} MB` : ""}`
              : capabilities.errors[0] ?? "请选择语言专用模型"}
        </small>
      </span>
    </div>
  );
}

function CsvInspector(props: InspectorProps) {
  const [colorDraft, setColorDraft] = useState(props.selectedCue?.speaker_color ?? "#FFFFFF");
  const needsReview = props.cues.filter((cue) => cue.review_status === "needs_review");
  const approved = props.cues.filter((cue) => cue.review_status === "approved").length;
  const overlapGroups = new Set(
    props.cues.map((cue) => cue.group_id ?? cue.overlap_group_id).filter(Boolean),
  ).size;
  useEffect(() => {
    setColorDraft(props.selectedCue?.speaker_color ?? "#FFFFFF");
  }, [props.selectedCue?.cue_id, props.selectedCue?.speaker_color]);
  const commitColor = (scope: "cue" | "speaker") => {
    if (/^#[0-9a-f]{6}$/i.test(colorDraft)) {
      if (scope === "speaker") props.onSpeakerColorChange(colorDraft.toUpperCase());
      else props.onCueColorChange(colorDraft.toUpperCase());
    } else {
      setColorDraft(props.selectedCue?.speaker_color ?? "#FFFFFF");
    }
  };
  return (
    <>
      <section className="inspector-section">
        <SectionTitle icon={Table2}>数据完整性</SectionTitle>
        <div className="metric-grid">
          <div><strong>{props.cues.length}</strong><span>字幕条目</span></div>
          <div><strong>{approved}</strong><span>已确认</span></div>
          <div className={needsReview.length ? "metric-warning" : ""}><strong>{needsReview.length}</strong><span>需校对</span></div>
          <div><strong>{overlapGroups}</strong><span>并发组</span></div>
        </div>
      </section>
      {props.selectedCue && (
        <section className="inspector-section selected-cue-detail">
          <SectionTitle icon={UsersRound}>当前字幕</SectionTitle>
          <div className="cue-detail-heading">
            <span style={{ background: props.selectedCue.speaker_color }} />
            <strong>{props.selectedCue.speaker_name}</strong>
            <code>{props.selectedCue.cue_id}</code>
          </div>
          <p>{props.selectedCue.source_text}</p>
          <dl className="property-list compact-properties">
            <div><dt>识别置信</dt><dd>{confidenceLabel(props.selectedCue.ocr_confidence)}</dd></div>
            <div><dt>时间</dt><dd>{formatTime(props.selectedCue.start_ms, true)} – {formatTime(props.selectedCue.end_ms, true)}</dd></div>
            <div><dt>图层</dt><dd>Layer {props.selectedCue.layer + 1}</dd></div>
          </dl>
          <label className="field-label cue-color-field">
            <span>字幕颜色</span>
            <span className="cue-color-editor">
              <input
                type="color"
                value={/^#[0-9a-f]{6}$/i.test(colorDraft) ? colorDraft : props.selectedCue.speaker_color}
                onChange={(event) => setColorDraft(event.target.value.toUpperCase())}
                title="打开颜色选择器"
                aria-label="选择字幕颜色"
              />
              <input
                value={colorDraft}
                onChange={(event) => setColorDraft(event.target.value)}
                onBlur={() => {
                  if (!/^#[0-9a-f]{6}$/i.test(colorDraft)) {
                    setColorDraft(props.selectedCue?.speaker_color ?? "#FFFFFF");
                  }
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    commitColor("cue");
                    event.currentTarget.blur();
                  }
                }}
                maxLength={7}
                spellCheck={false}
                aria-label="当前字幕十六进制颜色"
              />
              <button
                type="button"
                onClick={props.onStartEyedropper}
                title="跳到对应帧，吸取并应用到当前字幕"
                aria-label="从视频帧吸取字幕颜色"
              >
                <Pipette size={15} />
              </button>
            </span>
          </label>
          <div className="cue-color-actions">
            <button className="button secondary" type="button" onClick={() => commitColor("cue")}>
              <Palette size={14} />仅当前字幕
            </button>
            <button
              className="button secondary"
              type="button"
              onClick={() => commitColor("speaker")}
              disabled={!props.selectedCue.speaker_id.trim() && !props.selectedCue.speaker_name.trim()}
              title="按人物编号匹配；没有编号时按人物名称匹配"
            >
              <UsersRound size={14} />该人物全部
            </button>
          </div>
        </section>
      )}
      <section className="inspector-section warning-list">
        <SectionTitle icon={AlertTriangle}>待处理</SectionTitle>
        {needsReview.length ? needsReview.slice(0, 4).map((cue) => (
          <button type="button" key={cue.cue_id}>
            <AlertTriangle size={14} />
            <span><strong>{cue.cue_id}</strong><small>{cue.source_text}</small></span>
            <ChevronRight size={14} />
          </button>
        )) : <p className="empty-note"><CheckCircle2 size={14} /> 全部字幕已通过校验</p>}
      </section>
    </>
  );
}

function TranslationInspector(props: InspectorProps) {
  const [tab, setTab] = useState<"api" | "local" | "context">("api");
  const [showKey, setShowKey] = useState(false);
  const [testState, setTestState] = useState<"idle" | "testing" | "ok" | "failed">("idle");
  const [saving, setSaving] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelOptions, setModelOptions] = useState<TranslationModelOption[]>([]);
  const isCommonTarget = targetLanguages.some(([value]) => value === props.project.target_language);
  const [customTarget, setCustomTarget] = useState(
    isCommonTarget ? "" : props.project.target_language,
  );
  const [customTargetMode, setCustomTargetMode] = useState(!isCommonTarget);
  const settings = props.translationSettings;
  const set = <K extends keyof TranslationSettings>(key: K, value: TranslationSettings[K]) =>
    props.onTranslationSettingsChange({ ...settings, [key]: value });

  const test = async () => {
    setTestState("testing");
    setTestState((await props.onTestTranslation()) ? "ok" : "failed");
  };
  const save = async () => {
    setSaving(true);
    await props.onSaveTranslation();
    setSaving(false);
  };
  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const models = await props.onFetchTranslationModels();
      setModelOptions(models);
      if (!settings.model && models[0]) set("model", models[0].id);
    } catch {
      // The parent reports the provider error in the workspace toast.
    } finally {
      setLoadingModels(false);
    }
  };

  return (
    <>
      <section className="inspector-section">
        <SectionTitle icon={Languages}>目标语言</SectionTitle>
        <label className="field-label">
          <span>翻译成</span>
          <select
            value={customTargetMode ? "__custom__" : props.project.target_language}
            onChange={(event) => {
              if (event.target.value === "__custom__") {
                setCustomTargetMode(true);
                return;
              }
              setCustomTargetMode(false);
              props.onTargetLanguageChange(event.target.value);
            }}
          >
            {targetLanguages.map(([value, label]) => (
              <option value={value} key={value}>{label} · {value}</option>
            ))}
            <option value="__custom__">自定义语言</option>
          </select>
        </label>
        {customTargetMode && (
          <label className="field-label">
            <span>语言名称或代码</span>
            <input
              value={customTarget}
              placeholder="Italiano / it-IT"
              onChange={(event) => setCustomTarget(event.target.value)}
              onBlur={() => {
                const value = customTarget.trim();
                if (value) props.onTargetLanguageChange(value);
              }}
            />
          </label>
        )}
      </section>
      <div className="inspector-tabs">
        <button type="button" className={tab === "api" ? "is-active" : ""} onClick={() => setTab("api")}><Server size={14} />接口</button>
        <button type="button" className={tab === "local" ? "is-active" : ""} onClick={() => setTab("local")}><Cpu size={14} />本地模型</button>
        <button type="button" className={tab === "context" ? "is-active" : ""} onClick={() => setTab("context")}><Braces size={14} />上下文</button>
      </div>
      {tab === "local" ? (
        <LocalModelPanel
          projectId={props.project.id}
          settings={settings}
          ocrCapabilities={props.ocrCapabilities}
          audioCapabilities={props.audioCapabilities}
          onSettingsChange={props.onTranslationSettingsChange}
        />
      ) : tab === "api" ? (
        <>
          <section className="inspector-section">
            <SectionTitle icon={Server}>OpenAI 兼容接口</SectionTitle>
            <label className="field-label"><span>Base URL</span><input value={settings.base_url} onChange={(event) => set("base_url", event.target.value)} placeholder="https://relay.example.com/v1" spellCheck={false} /></label>
            <label className="field-label"><span>API Key</span><span className="input-with-action"><KeyRound size={14} /><input type={showKey ? "text" : "password"} value={settings.api_key} onChange={(event) => set("api_key", event.target.value)} placeholder="sk-..." autoComplete="off" /><button type="button" onClick={() => setShowKey(!showKey)} title={showKey ? "隐藏密钥" : "显示密钥"}>{showKey ? <EyeOff size={14} /> : <Eye size={14} />}</button></span></label>
            <label className="field-label">
              <span>模型</span>
              {modelOptions.length ? (
                <select value={settings.model} onChange={(event) => set("model", event.target.value)}>
                  {settings.model && !modelOptions.some((model) => model.id === settings.model) && (
                    <option value={settings.model}>{settings.model}</option>
                  )}
                  {modelOptions.map((model) => (
                    <option value={model.id} key={model.id}>
                      {model.owned_by ? `${model.id} · ${model.owned_by}` : model.id}
                    </option>
                  ))}
                </select>
              ) : (
                <input value={settings.model} onChange={(event) => set("model", event.target.value)} />
              )}
            </label>
            <button className="button secondary full-width model-fetch-button" type="button" onClick={() => void fetchModels()} disabled={loadingModels}>
              <RefreshCw className={loadingModels ? "spin" : ""} size={15} />
              {loadingModels ? "正在读取模型" : "从上游获取模型"}
            </button>
            <div className="field-row">
              <label className="field-label">
                <span>思考强度</span>
                <select value={settings.reasoning_effort} onChange={(event) => set("reasoning_effort", event.target.value as TranslationSettings["reasoning_effort"])}>
                  <option value="">由模型决定</option>
                  <option value="minimal">最低 · minimal</option>
                  <option value="low">低 · low</option>
                  <option value="medium">中 · medium</option>
                  <option value="high">高 · high</option>
                  <option value="xhigh">极高 · xhigh</option>
                </select>
              </label>
              <label className="field-label"><span>请求路径</span><input value={settings.path} onChange={(event) => set("path", event.target.value)} /></label>
            </div>
            <label className="field-label"><span>自定义 Headers · JSON</span><textarea className="code-input" value={settings.custom_headers} onChange={(event) => set("custom_headers", event.target.value)} spellCheck={false} /></label>
            <div className="field-row"><label className="field-label"><span>超时 · 秒</span><input type="number" min="10" max="600" value={settings.timeout_seconds} onChange={(event) => set("timeout_seconds", Number(event.target.value))} /></label><label className="field-label"><span>并发</span><input type="number" min="1" max="8" value={settings.concurrency} onChange={(event) => set("concurrency", Number(event.target.value))} /></label></div>
          </section>
          <section className="inspector-section button-row">
            <button className="button secondary" type="button" onClick={test} disabled={testState === "testing"}>
              {testState === "testing" ? <LoaderCircle className="spin" size={15} /> : testState === "ok" ? <CheckCircle2 size={15} /> : testState === "failed" ? <XCircle size={15} /> : <TestTube2 size={15} />}
              {testState === "ok" ? "连接正常" : testState === "failed" ? "连接失败" : "测试连接"}
            </button>
            <button className="button primary" type="button" onClick={save} disabled={saving}>
              {saving ? <LoaderCircle className="spin" size={15} /> : <Save size={15} />}
              保存
            </button>
          </section>
        </>
      ) : (
        <ContextEditor project={props.project} onChange={props.onContextChange} settings={settings} onSettingsChange={props.onTranslationSettingsChange} />
      )}
      <section className="inspector-section upload-summary">
        <SectionTitle icon={Languages}>发送给翻译模型</SectionTitle>
        <ul>
          <li className={settings.send_title ? "included" : ""}>视频标题</li>
          <li className={settings.send_story_context ? "included" : ""}>剧情简介</li>
          <li className={settings.send_character_profiles ? "included" : ""}>角色资料</li>
          <li className={settings.send_glossary ? "included" : ""}>术语表</li>
          <li className="included">字幕 CSV 文本</li>
        </ul>
        <p><ShieldCheck size={13} /> 不发送视频、音频或图像帧</p>
      </section>
      <section className="inspector-section action-section">
        <button className="button primary full-width" type="button" onClick={() => props.onStartJob("translation", { cue_ids: props.cues.map((cue) => cue.cue_id) })}>
          <Languages size={15} /> 翻译 {props.cues.length} 条字幕
        </button>
      </section>
    </>
  );
}

function ContextEditor({ project, onChange, settings, onSettingsChange }: { project: Project; onChange: (context: ProjectContext) => void; settings: TranslationSettings; onSettingsChange: (settings: TranslationSettings) => void }) {
  const context = project.context;
  return (
    <section className="inspector-section">
      <SectionTitle icon={Braces}>故事与角色上下文</SectionTitle>
      <label className="field-label"><span>剧情简介</span><textarea value={context.synopsis} onChange={(event) => onChange({ ...context, synopsis: event.target.value })} /></label>
      <label className="field-label"><span>角色资料</span><textarea value={context.characters} onChange={(event) => onChange({ ...context, characters: event.target.value })} /></label>
      <label className="field-label"><span>术语表</span><textarea value={context.glossary} onChange={(event) => onChange({ ...context, glossary: event.target.value })} /></label>
      <label className="field-label"><span>翻译风格</span><textarea value={context.translation_style} onChange={(event) => onChange({ ...context, translation_style: event.target.value })} /></label>
      <ToggleRow checked={settings.send_title} onChange={(checked) => onSettingsChange({ ...settings, send_title: checked })} label="发送视频标题" />
      <ToggleRow checked={settings.send_story_context} onChange={(checked) => onSettingsChange({ ...settings, send_story_context: checked })} label="发送剧情简介" />
      <ToggleRow checked={settings.send_character_profiles} onChange={(checked) => onSettingsChange({ ...settings, send_character_profiles: checked })} label="发送角色资料" />
      <ToggleRow checked={settings.send_glossary} onChange={(checked) => onSettingsChange({ ...settings, send_glossary: checked })} label="发送术语表" />
    </section>
  );
}

function LayoutInspector(props: InspectorProps) {
  const settings = props.layoutSettings;
  const set = <K extends keyof LayoutSettings>(key: K, value: LayoutSettings[K]) =>
    props.onLayoutSettingsChange({ ...settings, [key]: value });
  return (
    <>
      <section className="inspector-section">
        <SectionTitle icon={Palette}>字幕样式</SectionTitle>
        <label className="field-label"><span>字体</span><input value={settings.font_family} onChange={(event) => set("font_family", event.target.value)} /></label>
        <div className="field-row"><label className="field-label"><span>字号</span><input type="number" min="16" max="96" value={settings.font_size} onChange={(event) => set("font_size", Number(event.target.value))} /></label><label className="field-label"><span>描边</span><input type="number" min="0" max="10" value={settings.outline} onChange={(event) => set("outline", Number(event.target.value))} /></label></div>
        <label className="field-label"><span>颜色</span><select value={settings.color_mode} onChange={(event) => set("color_mode", event.target.value as LayoutSettings["color_mode"])}><option value="speaker">跟随说话人颜色</option><option value="single">统一译文颜色</option></select></label>
      </section>
      <section className="inspector-section">
        <SectionTitle icon={Layers3}>同时说话排版</SectionTitle>
        <div className="option-stack">
          <OptionRadio checked={settings.overlap_mode === "layers"} title="独立图层" detail="同一区域按 ASS Layer 上下错位" onClick={() => set("overlap_mode", "layers")} />
          <OptionRadio checked={settings.overlap_mode === "split"} title="左右分区" detail="根据原字幕位置分配左右区域" onClick={() => set("overlap_mode", "split")} />
          <OptionRadio checked={settings.overlap_mode === "stack"} title="顺序堆叠" detail="按说话人颜色逐行展示" onClick={() => set("overlap_mode", "stack")} />
        </div>
      </section>
      <section className="inspector-section">
        <SectionTitle icon={ShieldCheck}>内容避让</SectionTitle>
        <ToggleRow checked={settings.avoid_source} onChange={(checked) => set("avoid_source", checked)} label="避开原字幕区域" />
        <ToggleRow checked={settings.avoid_faces} onChange={(checked) => set("avoid_faces", checked)} label="避开人脸与主体" />
        <div className="field-row"><label className="field-label"><span>最大行数</span><input type="number" min="1" max="4" value={settings.max_lines} onChange={(event) => set("max_lines", Number(event.target.value))} /></label><label className="field-label"><span>每行字符</span><input type="number" min="8" max="60" value={settings.max_chars} onChange={(event) => set("max_chars", Number(event.target.value))} /></label></div>
      </section>
      <section className="inspector-section button-row">
        <button className="button secondary" type="button" onClick={() => props.onDrawRegion("target")}><Palette size={15} />调整位置</button>
        <button className="button primary" type="button" onClick={() => void props.onSaveLayout()}><Save size={15} />保存样式</button>
      </section>
    </>
  );
}

function ExportInspector(props: InspectorProps) {
  const [format, setFormat] = useState("hard-sub");
  return (
    <>
      <section className="inspector-section">
        <SectionTitle icon={FileOutput}>输出格式</SectionTitle>
        <div className="option-stack">
          <OptionRadio checked={format === "hard-sub"} title="硬字幕视频" detail="FFmpeg + libass 内嵌译文" onClick={() => setFormat("hard-sub")} />
          <OptionRadio checked={format === "soft-sub"} title="软字幕封装" detail="保留视频码流，字幕可开关" onClick={() => setFormat("soft-sub")} />
          <OptionRadio checked={format === "ass"} title="ASS + CSV" detail="交付字幕文件与结构化数据" onClick={() => setFormat("ass")} />
        </div>
      </section>
      <section className="inspector-section">
        <SectionTitle icon={HardDrive}>输出位置</SectionTitle>
        <label className="field-label"><span>文件名</span><input defaultValue={`${props.project.title}-zh-CN.mp4`} /></label>
        <label className="field-label"><span>目录</span><span className="input-with-action"><FolderOpen size={14} /><input defaultValue="exports" /><button type="button" title="选择目录"><FolderOpen size={14} /></button></span></label>
        <ToggleRow checked label="同时导出 source.csv" />
        <ToggleRow checked label="保留可复现任务日志" />
      </section>
      <section className="inspector-section export-summary">
        <div><span>预计输出</span><strong>842 MB</strong></div>
        <div><span>编码器</span><strong>H.264 · CRF 18</strong></div>
        <div><span>字幕层</span><strong>{Math.max(1, ...props.cues.map((cue) => cue.layer + 1))} layers</strong></div>
      </section>
      <section className="inspector-section action-section">
        <button className="button primary full-width" type="button" onClick={() => props.onStartJob("export", { format })}><FileOutput size={15} />开始导出</button>
      </section>
    </>
  );
}

function ToggleRow({ checked, onChange, label, detail }: { checked: boolean; onChange?: (checked: boolean) => void; label: string; detail?: string }) {
  return (
    <label className="toggle-row">
      <span><strong>{label}</strong>{detail && <small>{detail}</small>}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange?.(event.target.checked)} />
      <i aria-hidden="true" />
    </label>
  );
}

function OptionRadio({ checked, title, detail, onClick }: { checked: boolean; title: string; detail: string; onClick: () => void }) {
  return (
    <button type="button" className={`option-radio ${checked ? "is-active" : ""}`} onClick={onClick}>
      <i>{checked && <span />}</i>
      <p><strong>{title}</strong><small>{detail}</small></p>
    </button>
  );
}

function ocrMetricsSummary(job: Job): string | null {
  const metrics = job.snapshot?.metrics;
  if ((job.kind !== "ocr" && job.kind !== "hybrid") || !metrics) return null;
  const parts: string[] = [];
  if (typeof metrics.processed_frames === "number") {
    const total = typeof metrics.sampled_frames_total === "number"
      ? metrics.sampled_frames_total
      : null;
    parts.push(`采样 ${metrics.processed_frames}${total !== null ? `/${total}` : ""}`);
  }
  if (typeof metrics.device === "string") parts.push(metrics.device.toUpperCase());
  if (typeof metrics.batch_size === "number") parts.push(`批 ${metrics.batch_size}`);
  if (typeof metrics.ocr_frames === "number") parts.push(`OCR ${metrics.ocr_frames}`);
  if (typeof metrics.reused_frames === "number") parts.push(`复用 ${metrics.reused_frames}`);
  if (typeof metrics.stabilized_detections === "number" && metrics.stabilized_detections > 0) {
    parts.push(`渐入修正 ${metrics.stabilized_detections}`);
  }
  if (typeof metrics.effective_sample_fps === "number") {
    parts.push(`${metrics.effective_sample_fps.toFixed(1)} 帧/秒`);
  }
  return parts.length ? parts.join(" · ") : null;
}

const jobKindLabels: Record<Job["kind"], string> = {
  ocr: "OCR 字幕识别",
  uvr: "UVR5 人声分离",
  slicer: "静音切分",
  asr: "ASR 打标",
  audio: "音频三阶段",
  hybrid: "双源识别",
  fusion: "AI 融合校正",
  translation: "AI 翻译",
  preview: "预览渲染",
  export: "视频导出",
  analysis: "本地分析",
  "local-model-deploy": "本地模型部署",
};

const jobStatusLabels: Record<Job["status"], string> = {
  queued: "等待中",
  running: "运行中",
  paused: "已暂停",
  failed: "失败",
  cancelled: "已取消",
  completed: "已完成",
};

function completedResultSummary(job: Job): string | null {
  if (job.status !== "completed" || !job.result) return null;
  const count = (key: string) => {
    const value = job.result?.[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };
  if (job.kind === "hybrid") {
    const ocrCount = count("ocr_cue_count");
    const speechCount = count("speech_cue_count");
    if (ocrCount !== null || speechCount !== null) {
      return `已生成 OCR ${ocrCount ?? 0} 条 · 语音 ${speechCount ?? 0} 条`;
    }
  }
  if (job.kind === "slicer") {
    const segmentCount = count("segment_count");
    if (segmentCount !== null) return `已生成 ${segmentCount} 个语音片段`;
  }
  const cueCount = count("cue_count");
  if (cueCount === null) return null;
  if (job.kind === "ocr") return `已生成 OCR ${cueCount} 条`;
  if (job.kind === "audio" || job.kind === "asr") return `已生成语音 ${cueCount} 条`;
  return `已生成 ${cueCount} 条字幕`;
}

function evidenceArtifacts(job: Job): Array<"ocr" | "speech"> {
  if (job.status !== "completed") return [];
  const declared = Array.isArray(job.result?.artifacts) ? job.result.artifacts : [];
  const artifacts = declared.filter((value): value is string => typeof value === "string");
  const result: Array<"ocr" | "speech"> = [];
  if (job.kind === "ocr" || job.kind === "hybrid" || artifacts.includes("ocr.csv")) result.push("ocr");
  if (job.kind === "asr" || job.kind === "audio" || job.kind === "hybrid" || artifacts.includes("speech.csv")) result.push("speech");
  return result;
}

function JobsSection({ projectId, jobs, onCancel }: { projectId: string; jobs: Job[]; onCancel: (job: Job) => void }) {
  const visible = useMemo(() => {
    const sorted = jobs
      .map((job, index) => ({ job, index }))
      .sort((left, right) => {
        const leftTime = left.job.created_at ? Date.parse(left.job.created_at) : Number.NaN;
        const rightTime = right.job.created_at ? Date.parse(right.job.created_at) : Number.NaN;
        if (Number.isFinite(leftTime) && Number.isFinite(rightTime)) return rightTime - leftTime;
        return right.index - left.index;
      })
      .map(({ job }) => job);
    const active = sorted.filter((job) => job.status === "running" || job.status === "queued");
    const finished = sorted.filter((job) => job.status !== "running" && job.status !== "queued");
    return [...active, ...finished.slice(0, Math.max(0, 5 - active.length))];
  }, [jobs]);
  if (!visible.length) return null;
  return (
    <section className="inspector-section jobs-section">
      <SectionTitle icon={Cpu}>本地任务</SectionTitle>
      {visible.map((job) => {
        const percent = Math.round(job.progress * 100);
        const running = job.status === "running" || job.status === "queued";
        const metrics = ocrMetricsSummary(job);
        const resultSummary = completedResultSummary(job);
        const artifacts = evidenceArtifacts(job);
        return (
          <div className="job-row" key={job.id}>
            <span className={`job-state ${job.status}`} role="img" aria-label={`${jobKindLabels[job.kind]}${jobStatusLabels[job.status]}`}>
              {running ? <LoaderCircle className="spin" size={14} /> : job.status === "completed" ? <CheckCircle2 size={14} /> : job.status === "failed" ? <XCircle size={14} /> : <CircleStop size={14} />}
            </span>
            <div>
              <p><strong>{jobKindLabels[job.kind]}</strong><span>{percent}%</span></p>
              <div
                className="job-progress"
                role="progressbar"
                aria-label={`${jobKindLabels[job.kind]}进度`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={percent}
              ><i style={{ width: `${percent}%` }} /></div>
              <small>{job.message}</small>
              {metrics && <small className="job-metrics">{metrics}</small>}
              {resultSummary && <small className="job-result-summary">{resultSummary}</small>}
              {artifacts.length > 0 && (
                <div className="job-artifacts" aria-label="识别证据表">
                  {artifacts.map((artifact) => (
                    <a
                      key={artifact}
                      href={api.projectCsvUrl(projectId, artifact)}
                      download={`${artifact}.csv`}
                      title={`下载${artifact === "ocr" ? "OCR" : "语音"}证据表`}
                    >
                      <Download size={12} />
                      {artifact === "ocr" ? "OCR CSV" : "语音 CSV"}
                    </a>
                  ))}
                </div>
              )}
            </div>
            {running && <button type="button" title={`取消${jobKindLabels[job.kind]}`} aria-label={`取消${jobKindLabels[job.kind]}任务`} onClick={() => onCancel(job)}><CircleStop size={14} /></button>}
          </div>
        );
      })}
    </section>
  );
}
