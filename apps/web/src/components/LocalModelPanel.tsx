import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Download,
  Gauge,
  HardDrive,
  LoaderCircle,
  Play,
  Plug,
  Power,
  RefreshCw,
  RotateCcw,
  Server,
  Square,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type {
  AudioCapabilities,
  Job,
  LocalModelCatalogItem,
  LocalTranslationRuntimeStatus,
  LocalRuntimeVariant,
  LocalTranslationProvider,
  OcrCapabilities,
  TranslationSettings,
} from "../types";

type LocalMode = "catalog" | "gguf" | "external";

interface LocalModelPanelProps {
  projectId: string;
  settings: TranslationSettings;
  ocrCapabilities: OcrCapabilities | null;
  audioCapabilities: AudioCapabilities | null;
  onSettingsChange: (
    settings: TranslationSettings | ((current: TranslationSettings) => TranslationSettings),
  ) => void;
}

const formatBytes = (value: number | null | undefined) => {
  if (!value) return "未知";
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.round(value / 1024 ** 2)} MB`;
};

const stateLabels: Record<LocalTranslationRuntimeStatus["state"], string> = {
  not_configured: "未配置",
  ready: "运行中",
  starting: "正在启动",
  failed: "启动失败",
  unreachable: "连接中断",
  stopped: "已停止",
};

export function LocalModelPanel({
  projectId,
  settings,
  ocrCapabilities,
  audioCapabilities,
  onSettingsChange,
}: LocalModelPanelProps) {
  const [mode, setMode] = useState<LocalMode>("catalog");
  const [status, setStatus] = useState<LocalTranslationRuntimeStatus | null>(null);
  const [models, setModels] = useState<LocalModelCatalogItem[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [runtime, setRuntime] = useState<LocalRuntimeVariant>("auto");
  const [contextSize, setContextSize] = useState(8192);
  const [deployJob, setDeployJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [externalUrl, setExternalUrl] = useState("http://127.0.0.1:11434/v1");
  const [externalModel, setExternalModel] = useState("");
  const [externalPath, setExternalPath] = useState("/chat/completions");
  const [serverPath, setServerPath] = useState("");
  const [ggufPath, setGgufPath] = useState("");

  const applyProvider = useCallback((provider: LocalTranslationProvider) => {
    onSettingsChange((current) => ({
      ...current,
      base_url: provider.base_url,
      api_key: "",
      model: provider.model,
      path: provider.api_path,
      custom_headers: JSON.stringify(provider.custom_headers ?? {}, null, 2),
      timeout_seconds: provider.timeout_seconds,
      reasoning_effort: "",
      concurrency: 1,
    }));
  }, [onSettingsChange]);

  const refresh = useCallback(async (refreshHardware = false) => {
    setError("");
    const [statusResult, catalogResult, jobsResult] = await Promise.allSettled([
      api.getLocalModelStatus(refreshHardware),
      api.getLocalModelCatalog(),
      api.listJobs("local-model-runtime"),
    ]);
    if (statusResult.status === "fulfilled") setStatus(statusResult.value);
    if (catalogResult.status === "fulfilled") {
      setModels(catalogResult.value.models);
      setSelectedModel((current) => current || catalogResult.value.recommendation.model_id);
    }
    if (jobsResult.status === "fulfilled") {
      const active = jobsResult.value.find((job) =>
        job.kind === "local-model-deploy" && ["queued", "running"].includes(job.status),
      );
      if (active) {
        setDeployJob(active);
        setBusy(true);
      }
    }
    const rejected = [statusResult, catalogResult, jobsResult].filter((result) => result.status === "rejected");
    if (rejected.length === 3) {
      const reason = rejected[0].reason;
      setError(reason instanceof Error ? reason.message : "本地模型状态读取失败");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const deployJobId = deployJob?.id ?? null;
  useEffect(() => {
    if (!deployJobId) return;
    let disposed = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const job = await api.getJob(deployJobId);
        if (disposed) return;
        if (job.status === "completed") {
          const nextStatus = await api.getLocalModelStatus();
          if (disposed) return;
          setStatus(nextStatus);
          if (nextStatus.provider) applyProvider(nextStatus.provider);
          setDeployJob(job);
          setBusy(false);
        } else if (job.status === "failed" || job.status === "cancelled") {
          setDeployJob(job);
          setError(job.error?.detail ?? job.message);
          setBusy(false);
        } else {
          setDeployJob(job);
          timer = window.setTimeout(() => void poll(), 700);
        }
      } catch (reason) {
        if (disposed) return;
        setError(reason instanceof Error ? reason.message : "部署进度读取失败");
        timer = window.setTimeout(() => void poll(), 1500);
      }
    };
    void poll();
    return () => {
      disposed = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [applyProvider, deployJobId]);

  const recommended = status?.recommendation;
  const selected = useMemo(
    () => models.find((model) => model.id === selectedModel) ?? null,
    [models, selectedModel],
  );
  const active = Boolean(
    status?.provider && settings.base_url.replace(/\/$/, "") === status.provider.base_url,
  );
  const activeDeployment = Boolean(deployJob && ["queued", "running"].includes(deployJob.status));
  const ocrDevice = ocrCapabilities?.default_device?.toUpperCase() || "--";
  const audioDevice = audioCapabilities?.default_device?.toUpperCase() || "--";
  const translationDevice = (
    status?.configuration?.runtime_variant
    ?? recommended?.runtime_variant
    ?? "auto"
  ).toUpperCase();

  const deploy = async () => {
    setBusy(true);
    setError("");
    try {
      const job = await api.deployLocalModel({
        model_id: selectedModel,
        runtime_variant: runtime,
        context_size: contextSize,
        auto_start: true,
        make_default: true,
      });
      setDeployJob(job);
    } catch (reason) {
      setBusy(false);
      setError(reason instanceof Error ? reason.message : "一键部署启动失败");
    }
  };

  const waitUntilReady = async (
    initial: LocalTranslationRuntimeStatus,
    timeoutMs = 180_000,
  ): Promise<LocalTranslationRuntimeStatus> => {
    let current = initial;
    const deadline = Date.now() + timeoutMs;
    while (!current.ready && Date.now() < deadline) {
      if (current.state === "failed") {
        throw new Error(current.error || current.log_tail || "本地模型启动失败");
      }
      await new Promise((resolve) => window.setTimeout(resolve, 700));
      current = await api.getLocalModelStatus();
      setStatus(current);
    }
    if (!current.ready) throw new Error("本地模型在 180 秒内没有完成启动，请查看运行日志");
    return current;
  };

  const configure = async (kind: "gguf" | "external") => {
    setBusy(true);
    setError("");
    try {
      const response = kind === "external"
        ? await api.configureLocalModel({
            mode: "external",
            base_url: externalUrl,
            api_path: externalPath,
            model: externalModel,
            make_default: false,
          })
        : await api.configureLocalModel({
            mode: "managed",
            executable_path: serverPath,
            model_path: ggufPath,
            model: ggufPath.split(/[\\/]/).pop()?.replace(/\.gguf$/i, "") ?? "kaor-local",
            runtime_variant: runtime,
            port: 18080,
            context_size: contextSize,
            gpu_layers: runtime === "cpu" ? 0 : -1,
            auto_start: true,
            make_default: false,
          });
      if (kind === "gguf") {
        const ready = await waitUntilReady(await api.startLocalModel());
        setStatus(ready);
      } else {
        if (!response.status.ready) {
          throw new Error("本地兼容接口不可达，请先确认服务和 /v1/models 端点");
        }
        setStatus(response.status);
      }
      const activated = await api.activateLocalModel();
      applyProvider(activated.provider);
      setStatus(await api.getLocalModelStatus());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "本地模型配置失败");
      void refresh();
    } finally {
      setBusy(false);
    }
  };

  const activate = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await api.activateLocalModel();
      applyProvider(response.provider);
      setStatus(await api.getLocalModelStatus());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "切换本地模型失败");
    } finally {
      setBusy(false);
    }
  };

  const restoreRemote = async () => {
    setBusy(true);
    setError("");
    try {
      await api.deactivateLocalModel();
      const profile = await api.getTranslationSettings(projectId);
      onSettingsChange((current) => ({
        ...current,
        base_url: profile.base_url,
        api_key: profile.api_key,
        model: profile.model,
        path: profile.path,
        custom_headers: profile.custom_headers,
        timeout_seconds: profile.timeout_seconds,
        reasoning_effort: profile.reasoning_effort,
        concurrency: profile.concurrency,
      }));
      setStatus(await api.getLocalModelStatus());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "远程配置恢复失败");
    } finally {
      setBusy(false);
    }
  };

  const toggleServer = async () => {
    setBusy(true);
    setError("");
    try {
      if (status?.process_running) {
        setStatus(await api.stopLocalModel());
      } else {
        setStatus(await waitUntilReady(await api.startLocalModel()));
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "本地服务操作失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <section className="inspector-section local-runtime-overview">
        <div className="local-runtime-title">
          <span className={`runtime-state-dot is-${status?.state ?? "not_configured"}`} />
          <span>
            <strong>{stateLabels[status?.state ?? "not_configured"]}</strong>
            <small>{status?.configuration?.model ?? "本地翻译模型"}</small>
          </span>
          <button className="icon-button compact" type="button" onClick={() => void refresh(true)} title="重新检测硬件" aria-label="重新检测硬件">
            <RefreshCw size={14} />
          </button>
        </div>
        <div className="runtime-stage-grid">
          <span><Cpu size={13} /><small>OCR 实际设备</small><strong>{ocrDevice}</strong></span>
          <span><Gauge size={13} /><small>音频实际设备</small><strong>{audioDevice}</strong></span>
          <span><Server size={13} /><small>本地翻译后端</small><strong>{translationDevice}</strong></span>
        </div>
        {status?.hardware?.gpus[0] && (
          <p className="local-device-line" title={status.hardware.gpus[0].driver_version}>
            {status.hardware.gpus[0].name} · {formatBytes(status.hardware.gpus[0].memory_bytes)}
          </p>
        )}
      </section>

      <div className="segmented-control local-mode-switch" role="tablist" aria-label="本地模型接入方式">
        <button type="button" role="tab" aria-selected={mode === "catalog"} disabled={activeDeployment} className={mode === "catalog" ? "is-active" : ""} onClick={() => setMode("catalog")}><Download size={14} /><span>一键部署</span></button>
        <button type="button" role="tab" aria-selected={mode === "gguf"} disabled={activeDeployment} className={mode === "gguf" ? "is-active" : ""} onClick={() => setMode("gguf")}><HardDrive size={14} /><span>本地 GGUF</span></button>
        <button type="button" role="tab" aria-selected={mode === "external"} disabled={activeDeployment} className={mode === "external" ? "is-active" : ""} onClick={() => setMode("external")}><Plug size={14} /><span>已有服务</span></button>
      </div>

      {mode === "catalog" && (
        <section className="inspector-section">
          <div className="recommended-model-line">
            <CheckCircle2 size={15} />
            <span><small>硬件推荐</small><strong>{recommended?.model_label ?? "检测中"}</strong></span>
          </div>
          <label className="field-label">
            <span>模型</span>
            <select value={selectedModel} disabled={activeDeployment} onChange={(event) => setSelectedModel(event.target.value)}>
              {models.map((model) => (
                <option value={model.id} key={model.id}>
                  {model.label} · {formatBytes(model.size_bytes)}{model.installed ? " · 已下载" : ""}
                </option>
              ))}
            </select>
          </label>
          <div className="field-row">
            <label className="field-label"><span>运行后端</span><select value={runtime} disabled={activeDeployment} onChange={(event) => setRuntime(event.target.value as LocalRuntimeVariant)}><option value="auto">自动推荐</option><option value="cpu">CPU</option><option value="vulkan">Vulkan</option><option value="cuda">CUDA</option></select></label>
            <label className="field-label"><span>上下文</span><select value={contextSize} disabled={activeDeployment} onChange={(event) => setContextSize(Number(event.target.value))}><option value={8192}>8K</option><option value={16384}>16K</option><option value={32768}>32K</option></select></label>
          </div>
          {selected && <p className="model-integrity-line"><HardDrive size={13} />{selected.revision.slice(0, 8)} · SHA-256 固定校验</p>}
          <button className="button primary full-width" type="button" onClick={() => void deploy()} disabled={busy || activeDeployment || !selectedModel}>
            {deployJob && ["queued", "running"].includes(deployJob.status) ? <LoaderCircle className="spin" size={15} /> : <Download size={15} />}
            {deployJob && ["queued", "running"].includes(deployJob.status) ? deployJob.message : selected?.installed ? "校验并启动" : "下载、校验并配置"}
          </button>
          {deployJob && ["queued", "running"].includes(deployJob.status) && (
            <div className="local-deploy-progress" role="progressbar" aria-label={deployJob.message} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(deployJob.progress * 100)}><span style={{ width: `${Math.round(deployJob.progress * 100)}%` }} /><strong>{Math.round(deployJob.progress * 100)}%</strong></div>
          )}
        </section>
      )}

      {mode === "gguf" && (
        <section className="inspector-section">
          <label className="field-label"><span>llama-server.exe</span><input value={serverPath} disabled={activeDeployment} onChange={(event) => setServerPath(event.target.value)} spellCheck={false} /></label>
          <label className="field-label"><span>GGUF 模型路径</span><input value={ggufPath} disabled={activeDeployment} onChange={(event) => setGgufPath(event.target.value)} spellCheck={false} /></label>
          <div className="field-row">
            <label className="field-label"><span>运行后端</span><select value={runtime} disabled={activeDeployment} onChange={(event) => setRuntime(event.target.value as LocalRuntimeVariant)}><option value="auto">自动推荐</option><option value="cpu">CPU</option><option value="vulkan">Vulkan</option><option value="cuda">CUDA</option></select></label>
            <label className="field-label"><span>上下文</span><select value={contextSize} disabled={activeDeployment} onChange={(event) => setContextSize(Number(event.target.value))}><option value={8192}>8K</option><option value={16384}>16K</option><option value={32768}>32K</option></select></label>
          </div>
          <button className="button primary full-width" type="button" onClick={() => void configure("gguf")} disabled={busy || activeDeployment || !serverPath.trim() || !ggufPath.trim()}>{busy ? <LoaderCircle className="spin" size={15} /> : <Play size={15} />}配置并启动</button>
        </section>
      )}

      {mode === "external" && (
        <section className="inspector-section">
          <label className="field-label"><span>Base URL</span><input value={externalUrl} disabled={activeDeployment} onChange={(event) => setExternalUrl(event.target.value)} spellCheck={false} /></label>
          <label className="field-label"><span>模型 ID</span><input value={externalModel} disabled={activeDeployment} onChange={(event) => setExternalModel(event.target.value)} placeholder="qwen3:8b" spellCheck={false} /></label>
          <label className="field-label"><span>请求路径</span><input value={externalPath} disabled={activeDeployment} onChange={(event) => setExternalPath(event.target.value)} spellCheck={false} /></label>
          <button className="button primary full-width" type="button" onClick={() => void configure("external")} disabled={busy || activeDeployment || !externalUrl.trim() || !externalModel.trim()}>{busy ? <LoaderCircle className="spin" size={15} /> : <Plug size={15} />}接入本地服务</button>
        </section>
      )}

      {error && <section className="inline-error"><AlertTriangle size={15} /><span>{error}</span></section>}
      {status?.error && !error && <section className="inline-error"><AlertTriangle size={15} /><span>{status.error}</span></section>}

      {status?.provider && (
        <section className="inspector-section button-row local-runtime-actions">
          {status.configuration?.mode === "managed" && (
            <button className="button secondary" type="button" onClick={() => void toggleServer()} disabled={busy || activeDeployment}>
              {status.process_running ? <Square size={14} /> : <Power size={14} />}{status.process_running ? "停止服务" : "启动服务"}
            </button>
          )}
          {(!active || status.remote_profile_available) && (
            <button className="button secondary" type="button" onClick={() => void (active ? restoreRemote() : activate())} disabled={busy || activeDeployment || (!active && !status.ready)}>
              {active ? <RotateCcw size={14} /> : <Plug size={14} />}{active ? "恢复在线 API" : "设为翻译模型"}
            </button>
          )}
        </section>
      )}
    </>
  );
}
