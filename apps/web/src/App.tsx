import {
  AlertCircle,
  CheckCircle2,
  CloudOff,
  Maximize2,
  Minimize2,
  PanelBottom,
  ScanText,
  Table2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { AiTraceWindow } from "./components/AiTraceWindow";
import { CueGrid } from "./components/CueGrid";
import { Inspector } from "./components/Inspector";
import { LogConsole } from "./components/LogConsole";
import { StepRail } from "./components/StepRail";
import { Timeline } from "./components/Timeline";
import { TopBar } from "./components/TopBar";
import { VideoStage } from "./components/VideoStage";
import { demoWorkspace } from "./demo";
import type {
  AudioCapabilities,
  AsrModelOption,
  BBox,
  Cue,
  CueEvidenceView,
  DockView,
  Job,
  OcrCapabilities,
  OcrJobSnapshot,
  Project,
  ProjectContext,
  RegionKind,
  TranslationModelOption,
  WorkflowStepId,
  WorkspaceData,
} from "./types";
import { clamp } from "./utils";

type ConnectionState = "connecting" | "live" | "demo";
type DrawMode = "inspect" | "eyedropper" | "cue_ocr" | RegionKind;
type Toast = { id: number; kind: "success" | "warning" | "error"; message: string };
type FrameOcrCandidate = {
  cueId: string;
  bbox: BBox;
  text: string;
  confidence: number | null;
};

const sortCues = (cues: Cue[]) => [...cues].sort(
  (left, right) => left.start_ms - right.start_ms
    || left.layer - right.layer
    || left.cue_id.localeCompare(right.cue_id),
);

const DEFAULT_DOCK_HEIGHT = 320;
const MIN_DOCK_HEIGHT = 220;
const MIN_VIDEO_HEIGHT = 300;
const CONNECTION_STRIP_HEIGHT = 25;

export default function App() {
  const [workspace, setWorkspace] = useState<WorkspaceData>(demoWorkspace);
  const [ocrCapabilities, setOcrCapabilities] = useState<OcrCapabilities | null>(null);
  const [audioCapabilities, setAudioCapabilities] = useState<AudioCapabilities | null>(null);
  const [asrModels, setAsrModels] = useState<AsrModelOption[]>([]);
  const [recognitionLanguage, setRecognitionLanguage] = useState(
    demoWorkspace.project.source_language === "auto" ? "" : demoWorkspace.project.source_language,
  );
  const [liveOcrSnapshot, setLiveOcrSnapshot] = useState<OcrJobSnapshot | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [activeStep, setActiveStep] = useState<WorkflowStepId>("roi");
  const [dockView, setDockView] = useState<DockView>("timeline");
  const [pageView, setPageView] = useState<"workbench" | "logs">("workbench");
  const [cueEvidenceView, setCueEvidenceView] = useState<CueEvidenceView>("final");
  const [dockHeight, setDockHeight] = useState(DEFAULT_DOCK_HEIGHT);
  const [dockExpanded, setDockExpanded] = useState(false);
  const [drawMode, setDrawMode] = useState<DrawMode>("inspect");
  const [selectedCueId, setSelectedCueId] = useState<string | null>(demoWorkspace.cues[0]?.cue_id ?? null);
  const [currentTime, setCurrentTime] = useState(17_400);
  const [playing, setPlaying] = useState(false);
  const [localVideoUrl, setLocalVideoUrl] = useState<string | null>(null);
  const [frameOcrPending, setFrameOcrPending] = useState(false);
  const [frameOcrCandidate, setFrameOcrCandidate] = useState<FrameOcrCandidate | null>(null);
  const [aiTraceMinimized, setAiTraceMinimized] = useState(false);
  const [aiTraceJobId, setAiTraceJobId] = useState<string | null>(null);
  const [closedAiTraceJobId, setClosedAiTraceJobId] = useState<string | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(() =>
    window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark",
  );
  const [toasts, setToasts] = useState<Toast[]>([]);
  const cueSaveTimers = useRef(new Map<string, number>());
  const toastCounter = useRef(0);
  const editorWorkspaceRef = useRef<HTMLElement | null>(null);
  const dockHeightRef = useRef(DEFAULT_DOCK_HEIGHT);
  const restoredDockHeightRef = useRef(DEFAULT_DOCK_HEIGHT);
  const lastAiTraceJobIdRef = useRef<string | null>(null);

  const project = workspace.project;
  const selectedCue = workspace.cues.find((cue) => cue.cue_id === selectedCueId) ?? null;
  const activeJob = workspace.jobs.find((job) => job.status === "running" || job.status === "queued");
  const cueEvidenceCues = cueEvidenceView === "ocr"
    ? workspace.ocr_cues ?? []
    : cueEvidenceView === "asr"
      ? workspace.speech_cues ?? workspace.asr_cues ?? []
      : workspace.cues;
  const activeAiTraceJob = useMemo(() => {
    const candidates = workspace.jobs.filter((job) => job.kind === "fusion" || job.kind === "translation");
    return [...candidates].reverse().find((job) => job.status === "running" || job.status === "queued") ?? null;
  }, [workspace.jobs]);
  const aiTraceJob = workspace.jobs.find((job) => job.id === aiTraceJobId) ?? activeAiTraceJob;
  const playbackUrl = localVideoUrl ?? project.video_url;
  const resolveRecognitionLanguage = (requested?: unknown) => {
    const candidates = [requested, recognitionLanguage, project.source_language];
    for (const candidate of candidates) {
      const language = String(candidate ?? "").trim();
      if (language && language !== "auto") return language;
    }
    return asrModels.find((model) => model.recommended)?.language
      ?? asrModels[0]?.language
      ?? "en";
  };

  useEffect(() => {
    setRecognitionLanguage(project.source_language === "auto" ? "" : project.source_language);
  }, [project.id, project.source_language]);

  useEffect(() => {
    if (!activeAiTraceJob || activeAiTraceJob.id === lastAiTraceJobIdRef.current) return;
    lastAiTraceJobIdRef.current = activeAiTraceJob.id;
    setAiTraceJobId(activeAiTraceJob.id);
    setClosedAiTraceJobId(null);
    setAiTraceMinimized(false);
  }, [activeAiTraceJob]);

  const notify = useCallback((kind: Toast["kind"], message: string) => {
    const id = ++toastCounter.current;
    setToasts((current) => [...current, { id, kind, message }]);
    window.setTimeout(() => setToasts((current) => current.filter((toast) => toast.id !== id)), 3600);
  }, []);

  useEffect(() => {
    let active = true;
    api
      .loadWorkspace()
      .then((data) => {
        if (!active) return;
        setWorkspace(data);
        setConnection("live");
        setSelectedCueId(data.cues[0]?.cue_id ?? null);
        setCurrentTime(data.cues[0]?.start_ms ?? 0);
        void api.getAudioModels().then((models) => {
          if (active) setAsrModels(models);
        }).catch(() => {
          // The full capability response below also carries the model catalog.
        });
        void api.getOcrCapabilities().then((capabilities) => {
          if (active) setOcrCapabilities(capabilities);
        }).catch(() => {
          if (active) setOcrCapabilities(null);
        });
        void api.getAudioCapabilities().then((capabilities) => {
          if (!active) return;
          setAudioCapabilities(capabilities);
          setAsrModels(capabilities.asr_models);
        }).catch(() => {
          if (active) setAudioCapabilities(null);
        });
      })
      .catch(() => {
        if (!active) return;
        setConnection("demo");
        notify("warning", "本地服务未连接，当前使用可交互演示数据");
      });
    return () => {
      active = false;
      cueSaveTimers.current.forEach((timer) => window.clearTimeout(timer));
      if (localVideoUrl) URL.revokeObjectURL(localVideoUrl);
    };
  }, [notify]);

  useEffect(() => {
    if (connection !== "live" || !activeJob) return;
    let disposed = false;
    let refreshing = false;
    const poll = async () => {
      if (refreshing) return;
      refreshing = true;
      try {
        const jobs = await api.listJobs(project.id);
        if (disposed) return;
        const current = jobs.find((job) => job.id === activeJob.id);
        if (current && ["completed", "failed", "cancelled"].includes(current.status)) {
          const refreshed = await api.loadWorkspace();
          if (disposed) return;
          setWorkspace(refreshed);
          if (["uvr", "slicer", "asr", "audio", "hybrid"].includes(current.kind)) {
            void api.getAudioCapabilities().then((capabilities) => {
              if (!disposed) {
                setAudioCapabilities(capabilities);
                setAsrModels(capabilities.asr_models);
              }
            }).catch(() => {
              if (!disposed) setAudioCapabilities(null);
            });
          }
          if (current.kind === "ocr") setLiveOcrSnapshot(null);
          if (current.status === "completed") notify("success", `${current.kind} 任务完成`);
          if (current.status === "failed") notify("error", current.error?.detail ?? "任务失败");
          return;
        }
        setWorkspace((value) => ({
          ...value,
          jobs,
          cues:
            current?.kind === "ocr" && current.snapshot?.cues
              ? current.snapshot.cues
              : value.cues,
        }));
        if (current?.kind === "ocr" && isOcrSnapshot(current.snapshot)) {
          setLiveOcrSnapshot(current.snapshot);
        }
      } catch {
        // A transient polling failure should not discard the current workspace.
      } finally {
        refreshing = false;
      }
    };
    void poll();
    const timer = window.setInterval(poll, 400);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [activeJob?.id, connection, notify, project.id]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const getDockMaximum = useCallback(() => {
    const workspaceHeight = editorWorkspaceRef.current?.clientHeight ?? 760;
    return Math.max(MIN_DOCK_HEIGHT, workspaceHeight - CONNECTION_STRIP_HEIGHT - MIN_VIDEO_HEIGHT);
  }, []);

  const applyDockHeight = useCallback((height: number) => {
    const nextHeight = clamp(height, MIN_DOCK_HEIGHT, getDockMaximum());
    dockHeightRef.current = nextHeight;
    setDockHeight(nextHeight);
    return nextHeight;
  }, [getDockMaximum]);

  const expandDock = useCallback(() => {
    if (!dockExpanded) restoredDockHeightRef.current = dockHeightRef.current;
    applyDockHeight(getDockMaximum());
    setDockExpanded(true);
  }, [applyDockHeight, dockExpanded, getDockMaximum]);

  const toggleDockExpanded = useCallback(() => {
    if (dockExpanded) {
      applyDockHeight(restoredDockHeightRef.current);
      setDockExpanded(false);
      return;
    }
    expandDock();
  }, [applyDockHeight, dockExpanded, expandDock]);

  const resizeDockBy = useCallback((delta: number) => {
    const nextHeight = applyDockHeight(dockHeightRef.current + delta);
    restoredDockHeightRef.current = nextHeight;
    setDockExpanded(false);
  }, [applyDockHeight]);

  const beginDockResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = dockHeightRef.current;
    setDockExpanded(false);
    document.body.classList.add("is-resizing-dock");

    const onPointerMove = (pointerEvent: PointerEvent) => {
      const nextHeight = applyDockHeight(startHeight + startY - pointerEvent.clientY);
      restoredDockHeightRef.current = nextHeight;
    };
    const stopResize = () => {
      document.body.classList.remove("is-resizing-dock");
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", stopResize);
      window.removeEventListener("pointercancel", stopResize);
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", stopResize);
    window.addEventListener("pointercancel", stopResize);
  }, [applyDockHeight]);

  useEffect(() => {
    const onResize = () => {
      if (dockExpanded) applyDockHeight(getDockMaximum());
      else applyDockHeight(dockHeightRef.current);
    };
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [applyDockHeight, dockExpanded, getDockMaximum]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (pageView !== "workbench") return;
      const element = event.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT", "BUTTON", "A"].includes(element.tagName) || element.isContentEditable) return;
      if (event.code === "Space") {
        event.preventDefault();
        setPlaying((value) => !value);
      }
      if (event.code === "ArrowLeft") {
        event.preventDefault();
        setCurrentTime((time) => {
          if (event.shiftKey) return clamp(time - 1000, 0, project.duration_ms);
          const frameMs = 1000 / Math.max(project.fps, 1);
          return clamp((Math.round(time / frameMs) - 1) * frameMs, 0, project.duration_ms);
        });
      }
      if (event.code === "ArrowRight") {
        event.preventDefault();
        setCurrentTime((time) => {
          if (event.shiftKey) return clamp(time + 1000, 0, project.duration_ms);
          const frameMs = 1000 / Math.max(project.fps, 1);
          return clamp((Math.round(time / frameMs) + 1) * frameMs, 0, project.duration_ms);
        });
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [pageView, project.duration_ms, project.fps]);

  const completedSteps = useMemo(() => {
    const completed = new Set<WorkflowStepId>();
    if (project.video_name) completed.add("import");
    if (project.source_roi.width > 0 && project.target_roi.width > 0) completed.add("roi");
    if (workspace.cues.length > 0) completed.add("ocr");
    if (workspace.cues.length > 0 && workspace.cues.every((cue) => cue.review_status !== "pending")) completed.add("csv");
    if (workspace.cues.length > 0 && workspace.cues.every((cue) => Boolean(cue.target_text))) completed.add("translate");
    if (project.target_roi.width > 0) completed.add("layout");
    if (workspace.jobs.some((job) => job.kind === "export" && job.status === "completed")) completed.add("export");
    return completed;
  }, [project.source_roi.width, project.target_roi.width, project.video_name, workspace.cues, workspace.jobs]);

  const selectStep = (step: WorkflowStepId) => {
    setActiveStep(step);
    if (step === "csv" || step === "translate") setDockView("cues");
    if (step === "csv") expandDock();
    if (step === "roi" || step === "layout") setDockView("timeline");
    if (step === "roi") setDrawMode("source");
    if (step === "layout") setDrawMode("target");
  };

  const updateProject = (updater: (project: Project) => Project) => {
    setWorkspace((current) => ({ ...current, project: updater(current.project) }));
  };

  const updateTargetLanguage = (target_language: string) => {
    updateProject((current) => ({ ...current, target_language }));
    if (connection === "live") {
      void api.updateTargetLanguage(project.id, target_language).catch((error) =>
        notify("error", `目标语言保存失败：${error instanceof Error ? error.message : "未知错误"}`),
      );
    }
  };

  const updateTranslationSettings = useCallback((update: WorkspaceData["translation_settings"] | ((current: WorkspaceData["translation_settings"]) => WorkspaceData["translation_settings"])) => {
    setWorkspace((current) => ({
      ...current,
      translation_settings: typeof update === "function"
        ? update(current.translation_settings)
        : update,
    }));
  }, []);

  const importVideo = async (file: File) => {
    setPlaying(false);
    if (localVideoUrl) URL.revokeObjectURL(localVideoUrl);
    const objectUrl = URL.createObjectURL(file);
    setLocalVideoUrl(objectUrl);
    try {
      const metadata = await readVideoMetadata(objectUrl);
      updateProject((current) => ({
        ...current,
        title: file.name.replace(/\.[^.]+$/, ""),
        video_name: file.name,
        duration_ms: metadata.duration_ms || current.duration_ms,
        width: metadata.width || current.width,
        height: metadata.height || current.height,
      }));
      setCurrentTime(0);
    } catch {
      updateProject((current) => ({ ...current, title: file.name, video_name: file.name }));
    }

    if (connection === "live") {
      try {
        const imported = await api.importVideo(file);
        updateProject(() => imported);
        if (imported.audio_error) {
          notify("warning", `视频已导入，但音轨提取失败：${imported.audio_error}`);
        } else {
          notify("success", "视频与本地音轨已准备完成");
        }
      } catch (error) {
        notify("error", `本地服务导入失败：${error instanceof Error ? error.message : "未知错误"}`);
      }
    } else {
      notify("success", "视频已在浏览器本地打开");
    }
    setActiveStep("roi");
    setCueEvidenceView("final");
    setDrawMode("source");
  };

  const resetWorkspace = async () => {
    setPlaying(false);
    cueSaveTimers.current.forEach((timer) => window.clearTimeout(timer));
    cueSaveTimers.current.clear();
    if (connection !== "live") {
      setWorkspace((current) => ({ ...current, cues: [], jobs: [] }));
      setSelectedCueId(null);
      setCurrentTime(0);
      setLiveOcrSnapshot(null);
      setCueEvidenceView("final");
      setDockView("timeline");
      setDrawMode("source");
      setActiveStep("roi");
      notify("success", "当前工作区已重置");
      return;
    }
    try {
      const reset = await api.resetProject(project.id);
      const refreshed = await api.loadWorkspace();
      setWorkspace(refreshed);
      setSelectedCueId(refreshed.cues[0]?.cue_id ?? null);
      setCurrentTime(refreshed.cues[0]?.start_ms ?? 0);
      setLiveOcrSnapshot(null);
      setCueEvidenceView("final");
      setDockView("timeline");
      setDrawMode("source");
      setActiveStep("roi");
      if (reset.audio_error) {
        notify("warning", `工作区已清空，但音轨重新提取失败：${reset.audio_error}`);
      } else {
        notify("success", "工作区已重置，片源与新音轨已保留");
      }
    } catch (error) {
      notify("error", `重置失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const updateRegion = (kind: RegionKind, bbox: BBox) => {
    updateProject((current) => ({ ...current, [`${kind}_roi`]: bbox }));
    notify("success", kind === "source" ? "原字幕识别区已更新" : "译文安全区已更新");
    if (connection === "live") {
      void api.updateRegion(project.id, kind, bbox).catch((error) =>
        notify("error", `区域保存失败：${error instanceof Error ? error.message : "未知错误"}`),
      );
    }
  };

  const updateCue = (updatedCue: Cue) => {
    setWorkspace((current) => ({
      ...current,
      cues: current.cues.map((cue) => (cue.cue_id === updatedCue.cue_id ? updatedCue : cue)),
    }));
    if (connection !== "live") return;
    const existing = cueSaveTimers.current.get(updatedCue.cue_id);
    if (existing) window.clearTimeout(existing);
    cueSaveTimers.current.set(
      updatedCue.cue_id,
      window.setTimeout(() => {
        void api.updateCue(project.id, updatedCue).catch((error) =>
          notify("error", `字幕 ${updatedCue.cue_id} 保存失败：${error instanceof Error ? error.message : "未知错误"}`),
        );
        cueSaveTimers.current.delete(updatedCue.cue_id);
      }, 650),
    );
  };

  const addCue = async () => {
    const usedIds = new Set(workspace.cues.map((cue) => cue.cue_id));
    let sequence = 1;
    while (usedIds.has(`MANUAL${String(sequence).padStart(6, "0")}`)) sequence += 1;
    const cueId = `MANUAL${String(sequence).padStart(6, "0")}`;
    const boundedDuration = Math.max(project.duration_ms, 1);
    const startMs = Math.round(clamp(currentTime, 0, Math.max(0, boundedDuration - 1)));
    const endMs = Math.max(startMs + 1, Math.min(boundedDuration, startMs + 2_000));
    const cue: Cue = {
      cue_id: cueId,
      start_ms: startMs,
      end_ms: endMs,
      group_id: "",
      layer: selectedCue?.layer ?? 0,
      track_id: "manual",
      speaker_id: selectedCue?.speaker_id ?? "",
      speaker_name: selectedCue?.speaker_name ?? "",
      speaker_color: selectedCue?.speaker_color ?? "#FFFFFF",
      source_kind: "manual",
      source_text: "",
      ocr_confidence: null,
      target_text: "",
      review_status: "pending",
      bbox: null,
    };
    const previous = workspace.cues;
    setWorkspace((current) => ({ ...current, cues: sortCues([...current.cues, cue]) }));
    setSelectedCueId(cueId);
    setDockView("cues");
    if (connection !== "live") {
      notify("success", `已新增字幕 ${cueId}`);
      return;
    }
    try {
      const saved = await api.createCue(project.id, cue);
      setWorkspace((current) => ({
        ...current,
        cues: sortCues(current.cues.map((item) => item.cue_id === cueId ? { ...item, ...saved } : item)),
      }));
      notify("success", `已新增字幕 ${cueId}`);
    } catch (error) {
      setWorkspace((current) => ({ ...current, cues: previous }));
      setSelectedCueId(previous[0]?.cue_id ?? null);
      notify("error", `新增字幕失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const deleteSelectedCue = async () => {
    if (!selectedCue) return;
    if (!window.confirm(`删除字幕 ${selectedCue.cue_id}？`)) return;
    const removing = selectedCue;
    const previous = workspace.cues;
    const index = previous.findIndex((cue) => cue.cue_id === removing.cue_id);
    const remaining = previous.filter((cue) => cue.cue_id !== removing.cue_id);
    const nextSelected = remaining[Math.min(Math.max(index, 0), remaining.length - 1)] ?? null;
    const pending = cueSaveTimers.current.get(removing.cue_id);
    if (pending) window.clearTimeout(pending);
    cueSaveTimers.current.delete(removing.cue_id);
    setWorkspace((current) => ({ ...current, cues: remaining }));
    setSelectedCueId(nextSelected?.cue_id ?? null);
    if (connection !== "live") {
      notify("success", `已删除字幕 ${removing.cue_id}`);
      return;
    }
    try {
      await api.deleteCue(project.id, removing.cue_id);
      notify("success", `已删除字幕 ${removing.cue_id}`);
    } catch (error) {
      setWorkspace((current) => ({ ...current, cues: previous }));
      setSelectedCueId(removing.cue_id);
      notify("error", `删除字幕失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const updateCueColor = async (color: string) => {
    if (!selectedCue) return;
    const normalized = color.toUpperCase();
    const previousColor = selectedCue.speaker_color;
    const updatedCue: Cue = {
      ...selectedCue,
      speaker_color: normalized,
      speaker_evidence: Array.from(
        new Set([...(selectedCue.speaker_evidence ?? []), "manual" as const]),
      ),
    };
    setWorkspace((current) => ({
      ...current,
      cues: current.cues.map((cue) =>
        cue.cue_id === updatedCue.cue_id ? updatedCue : cue,
      ),
    }));
    setDrawMode("inspect");
    if (connection !== "live") {
      notify("success", `字幕 ${updatedCue.cue_id} 颜色已更新为 ${normalized}`);
      return;
    }
    try {
      const saved = await api.updateCueColor(project.id, updatedCue.cue_id, normalized);
      setWorkspace((current) => ({
        ...current,
        cues: current.cues.map((cue) =>
          cue.cue_id === saved.cue_id
            ? { ...cue, speaker_color: saved.speaker_color }
            : cue,
        ),
      }));
      notify("success", `字幕 ${updatedCue.cue_id} 颜色已更新为 ${normalized}`);
    } catch (error) {
      setWorkspace((current) => ({
        ...current,
        cues: current.cues.map((cue) =>
          cue.cue_id === updatedCue.cue_id && cue.speaker_color === normalized
            ? { ...cue, speaker_color: previousColor }
            : cue,
        ),
      }));
      notify("error", `颜色保存失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const updateSpeakerColor = async (color: string) => {
    if (!selectedCue) return;
    const speakerId = selectedCue.speaker_id.trim();
    const speakerName = selectedCue.speaker_name.trim();
    if (!speakerId && !speakerName) {
      notify("warning", "当前字幕没有人物编号或人物名称，请先填写说话人");
      return;
    }
    const normalized = color.toUpperCase();
    const matches = (cue: Cue) => speakerId
      ? cue.speaker_id === speakerId
      : cue.speaker_name.trim().toLocaleLowerCase() === speakerName.toLocaleLowerCase();
    const previousColors = new Map(
      workspace.cues.filter(matches).map((cue) => [cue.cue_id, cue.speaker_color]),
    );
    setWorkspace((current) => ({
      ...current,
      cues: current.cues.map((cue) => matches(cue) ? { ...cue, speaker_color: normalized } : cue),
    }));
    if (connection !== "live") {
      notify("success", `已更新 ${previousColors.size} 条该人物字幕的颜色`);
      return;
    }
    try {
      const saved = await api.updateSpeakerColor(project.id, speakerId, speakerName, normalized);
      const savedIds = new Set(saved.map((cue) => cue.cue_id));
      setWorkspace((current) => ({
        ...current,
        cues: current.cues.map((cue) => savedIds.has(cue.cue_id) ? { ...cue, speaker_color: normalized } : cue),
      }));
      notify("success", `已更新 ${saved.length} 条该人物字幕的颜色`);
    } catch (error) {
      setWorkspace((current) => ({
        ...current,
        cues: current.cues.map((cue) => {
          const previousColor = previousColors.get(cue.cue_id);
          return previousColor ? { ...cue, speaker_color: previousColor } : cue;
        }),
      }));
      notify("error", `人物颜色保存失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const startEyedropper = () => {
    if (!selectedCue) {
      notify("warning", "请先在字幕表中选择一条字幕");
      return;
    }
    if (!playbackUrl) {
      notify("warning", "当前项目还没有可取色的视频画面");
      return;
    }
    const cueMiddle = selectedCue.start_ms + (selectedCue.end_ms - selectedCue.start_ms) / 2;
    setPlaying(false);
    setCurrentTime(clamp(cueMiddle, 0, project.duration_ms));
    setDrawMode("eyedropper");
    notify("success", "已跳到所选字幕中段，点击视频中的字幕颜色");
  };

  const startCueFrameOcr = () => {
    if (cueEvidenceView !== "final" || !selectedCue) {
      notify("warning", "请先在最终字幕表中选择一条字幕");
      return;
    }
    if (connection !== "live") {
      notify("warning", "单帧 OCR 需要已连接的本地服务");
      return;
    }
    if (!playbackUrl) {
      notify("warning", "当前项目还没有可识别的视频画面");
      return;
    }
    setPlaying(false);
    setDrawMode("cue_ocr");
    notify("success", `请在当前帧框选 ${selectedCue.cue_id} 的文字区域`);
  };

  const recognizeCueFrame = async (bbox: BBox) => {
    const cue = selectedCue;
    setDrawMode("inspect");
    if (!cue || connection !== "live") return;
    setFrameOcrPending(true);
    try {
      const result = await api.recognizeFrame(project.id, {
        timestamp_ms: Math.round(currentTime),
        bbox,
        language: resolveRecognitionLanguage(),
        device: ocrCapabilities?.default_device ?? "auto",
        high_accuracy: true,
      });
      const text = String(result.text ?? "").trim()
        || (result.detections ?? []).map((item) => item.text.trim()).filter(Boolean).join(" ");
      if (!text) {
        notify("warning", "该区域没有识别到文字，请换一帧或扩大框选范围");
        return;
      }
      const rawConfidence = result.confidence == null ? Number.NaN : Number(result.confidence);
      const confidence = Number.isFinite(rawConfidence) ? clamp(rawConfidence, 0, 1) : null;
      setFrameOcrCandidate({ cueId: cue.cue_id, bbox, text, confidence });
    } catch (error) {
      notify("error", `单帧 OCR 失败：${error instanceof Error ? error.message : "未知错误"}`);
    } finally {
      setFrameOcrPending(false);
    }
  };

  const confirmFrameOcrOverwrite = () => {
    if (!frameOcrCandidate) return;
    const cue = workspace.cues.find((item) => item.cue_id === frameOcrCandidate.cueId);
    if (!cue) {
      setFrameOcrCandidate(null);
      notify("error", "原字幕已被删除，识别结果未写入");
      return;
    }
    updateCue({
      ...cue,
      source_text: frameOcrCandidate.text.trim(),
      ocr_confidence: frameOcrCandidate.confidence,
      review_status: "pending",
    });
    setFrameOcrCandidate(null);
    setSelectedCueId(cue.cue_id);
    notify("success", `已用单帧 OCR 结果覆盖 ${cue.cue_id} 的原文`);
  };

  const startDemoJob = (kind: Job["kind"]): Job => ({
    id: `demo-${kind}-${Date.now()}`,
    kind,
    status: "running",
    stage: kind === "ocr" ? "multi_frame_detection" : kind === "translation" ? "contextual_translation" : kind === "export" ? "ffmpeg_render" : "preparing",
    progress: 0.02,
    message: kind === "ocr" ? "正在检测稳定字幕帧" : kind === "translation" ? "正在构建剧情上下文" : "正在准备本地导出",
  });

  const simulateJob = (job: Job) => {
    const timer = window.setInterval(() => {
      let completed = false;
      setWorkspace((current) => ({
        ...current,
        jobs: current.jobs.map((item) => {
          if (item.id !== job.id || item.status !== "running") return item;
          const next = Math.min(1, item.progress + 0.08 + Math.random() * 0.08);
          completed = next >= 1;
          return {
            ...item,
            progress: next,
            status: completed ? "completed" : "running",
            message: completed ? "任务完成" : item.message,
          };
        }),
      }));
      if (completed) {
        window.clearInterval(timer);
        notify("success", `${job.kind} 任务已完成`);
      }
    }, 650);
  };

  const startJob = async (kind: Job["kind"], payload: Record<string, unknown> = {}) => {
    if (connection !== "live") {
      const job = startDemoJob(kind);
      setWorkspace((current) => ({ ...current, jobs: [...current.jobs, job] }));
      simulateJob(job);
      return;
    }
    try {
      let requestPayload = payload;
      if (kind === "ocr") {
        setLiveOcrSnapshot(null);
        setDockView("cues");
        requestPayload = {
          language: resolveRecognitionLanguage(payload.language),
          device: String(payload.device ?? "auto"),
          sample_fps: Number(payload.sample_fps ?? 4),
          high_accuracy: Boolean(payload.high_accuracy ?? true),
          filter_noise: Boolean(payload.filter_noise ?? true),
          batch_size: Number(payload.batch_size ?? 0),
          prefer_embedded: true,
        };
      } else if (kind === "uvr") {
        requestPayload = {
          device: String(payload.device ?? "auto"),
        };
      } else if (kind === "slicer") {
        requestPayload = normalizedSlicerSettings(payload);
      } else if (kind === "asr" || kind === "audio" || kind === "hybrid") {
        setDockView("cues");
        requestPayload = {
          language: resolveRecognitionLanguage(payload.language),
          model_id: String(payload.model_id ?? ""),
          device: String(payload.device ?? "auto"),
          ...(kind !== "asr" ? { separate_vocals: Boolean(payload.separate_vocals ?? true) } : {}),
          forced_alignment: Boolean(payload.forced_alignment ?? true),
          diarization: Boolean(payload.diarization ?? true),
          ...normalizedSlicerSettings(payload),
          asr_batch_size: boundedInteger(payload.asr_batch_size, 4, 1, 64),
          ...(kind === "hybrid"
            ? {
                sample_fps: Number(payload.sample_fps ?? 4),
                high_accuracy: Boolean(payload.high_accuracy ?? true),
                prefer_embedded: Boolean(payload.prefer_embedded ?? true),
                filter_noise: Boolean(payload.filter_noise ?? true),
                batch_size: Number(payload.batch_size ?? 0),
              }
            : {}),
        };
      } else if (kind === "translation" || kind === "fusion") {
        let customHeaders: Record<string, string> = {};
        try {
          customHeaders = JSON.parse(workspace.translation_settings.custom_headers || "{}");
        } catch {
          notify("error", "自定义 Headers 不是合法 JSON");
          return;
        }
        requestPayload = {
          provider: {
            base_url: workspace.translation_settings.base_url,
            api_key: workspace.translation_settings.api_key,
            model: workspace.translation_settings.model,
            reasoning_effort: workspace.translation_settings.reasoning_effort,
            api_path: workspace.translation_settings.path,
            custom_headers: customHeaders,
            timeout_seconds: workspace.translation_settings.timeout_seconds,
          },
          options: {
            ...(kind === "translation"
              ? {
                  max_lines: workspace.layout_settings.max_lines,
                  max_chars_per_line: workspace.layout_settings.max_chars,
                  batch_size: 80,
                  context_cues: 3,
                  retries: 2,
                }
              : {
                  batch_size: 80,
                  context_cues: 3,
                  retries: 2,
                }),
          },
        };
      } else if (kind === "export") {
        requestPayload = { preview: false, crf: 18, preset: "medium" };
      } else if (kind === "preview") {
        requestPayload = { preview: true, start_ms: Math.round(currentTime), preview_duration_ms: 10_000 };
      }
      const job = await api.startJob(project.id, kind, requestPayload);
      setWorkspace((current) => ({ ...current, jobs: [...current.jobs, job] }));
      notify("success", `${kind} 任务已加入队列`);
    } catch (error) {
      notify("error", `任务启动失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const cancelJob = (job: Job) => {
    setWorkspace((current) => ({
      ...current,
      jobs: current.jobs.map((item) => item.id === job.id ? { ...item, status: "cancelled", message: "任务已取消" } : item),
    }));
    if (connection === "live") void api.cancelJob(job.id).catch(() => notify("error", "取消任务失败"));
  };

  const saveTranslation = async () => {
    if (connection !== "live") {
      await wait(350);
      notify("success", "翻译接口配置已保留在当前会话");
      return;
    }
    try {
      await api.saveProjectContext(project.id, project.context);
      const settings = await api.saveTranslationSettings(project.id, workspace.translation_settings);
      setWorkspace((current) => ({ ...current, translation_settings: settings }));
      notify("success", "翻译接口配置已保存到本地凭据存储");
    } catch (error) {
      notify("error", `保存失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const testTranslation = async (): Promise<boolean> => {
    if (connection !== "live") {
      await wait(700);
      return true;
    }
    try {
      const result = await api.testTranslation(workspace.translation_settings);
      if (result.ok) notify("success", result.latency_ms ? `连接正常 · ${result.latency_ms} ms` : "连接正常");
      return result.ok;
    } catch (error) {
      notify("error", `连接失败：${error instanceof Error ? error.message : "未知错误"}`);
      return false;
    }
  };

  const fetchTranslationModels = async (): Promise<TranslationModelOption[]> => {
    if (connection !== "live") {
      await wait(250);
      return [
        { id: workspace.translation_settings.model || "demo-model", owned_by: "demo" },
      ];
    }
    try {
      const models = await api.fetchTranslationModels(workspace.translation_settings);
      notify("success", `已从上游获取 ${models.length} 个模型`);
      return models;
    } catch (error) {
      notify("error", `获取模型失败：${error instanceof Error ? error.message : "未知错误"}`);
      throw error;
    }
  };

  const saveLayout = async () => {
    if (connection !== "live") {
      await wait(250);
      notify("success", "字幕样式已保留在当前项目");
      return;
    }
    try {
      const settings = await api.saveLayoutSettings(project.id, workspace.layout_settings);
      setWorkspace((current) => ({ ...current, layout_settings: settings }));
      notify("success", "字幕样式已保存");
    } catch (error) {
      notify("error", `样式保存失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  return (
    <div className="app-shell">
      <TopBar
        project={project}
        isDemo={connection !== "live"}
        theme={theme}
        activeJob={activeJob}
        onThemeToggle={() => setTheme((value) => value === "dark" ? "light" : "dark")}
        onImport={(file) => {
          setPageView("workbench");
          void importVideo(file);
        }}
        onExport={() => {
          setPageView("workbench");
          selectStep("export");
        }}
        pageView={pageView}
        onLogsToggle={() => {
          setPlaying(false);
          setPageView((value) => value === "logs" ? "workbench" : "logs");
        }}
      />

      <div className={`workspace-shell ${pageView === "logs" ? "is-page-hidden" : ""}`}>
        <StepRail active={activeStep} completed={completedSteps} onSelect={selectStep} />

        <main
          className="editor-workspace"
          ref={editorWorkspaceRef}
          style={{ gridTemplateRows: `25px minmax(${MIN_VIDEO_HEIGHT}px, 1fr) ${dockHeight}px` }}
        >
          <div className="connection-strip" data-state={connection}>
            {connection === "connecting" ? <><span className="pulse-dot" />正在连接本地服务</> : connection === "demo" ? <><CloudOff size={13} />演示模式 · 接口恢复后将自动使用本地项目</> : <><CheckCircle2 size={13} />本地服务已连接 · 媒体不会上传</>}
          </div>

          <VideoStage
            project={project}
            cues={workspace.cues}
            liveOcrSnapshot={liveOcrSnapshot}
            currentTime={currentTime}
            playing={playing}
            selectedCue={selectedCue}
            sourceUrl={playbackUrl}
            drawMode={drawMode}
            onDrawModeChange={setDrawMode}
            onTimeChange={setCurrentTime}
            onPlayingChange={setPlaying}
            onRegionChange={updateRegion}
            onFrameOcrRegion={(bbox) => void recognizeCueFrame(bbox)}
            onColorPick={(color) => void updateCueColor(color)}
          />

          <section className="bottom-dock">
            <div
              className="dock-resize-handle"
              role="separator"
              aria-label="调整字幕工作区高度"
              aria-orientation="horizontal"
              aria-valuemin={MIN_DOCK_HEIGHT}
              aria-valuemax={Math.round(getDockMaximum())}
              aria-valuenow={Math.round(dockHeight)}
              tabIndex={0}
              onPointerDown={beginDockResize}
              onKeyDown={(event) => {
                if (event.key === "ArrowUp") {
                  event.preventDefault();
                  resizeDockBy(event.shiftKey ? 80 : 24);
                }
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  resizeDockBy(event.shiftKey ? -80 : -24);
                }
              }}
              title="拖动调整字幕工作区高度"
            ><span /></div>
            <header className="dock-header">
              <div className="dock-tabs">
                <button type="button" className={dockView === "timeline" ? "is-active" : ""} onClick={() => setDockView("timeline")}>
                  <PanelBottom size={14} />时间轴
                </button>
                <button type="button" className={dockView === "cues" ? "is-active" : ""} onClick={() => setDockView("cues")}>
                  <Table2 size={14} />字幕表
                  <span>{workspace.cues.length}</span>
                </button>
              </div>
              <div className="dock-header-end">
                {selectedCue && (
                  <div className="dock-selection">
                    <i style={{ background: selectedCue.speaker_color }} />
                    <strong>{selectedCue.speaker_name}</strong>
                    <code>{selectedCue.cue_id}</code>
                    {(selectedCue.group_id ?? selectedCue.overlap_group_id) && <span>并发组 {selectedCue.group_id ?? selectedCue.overlap_group_id}</span>}
                  </div>
                )}
                <button
                  className="icon-button compact dock-expand-button"
                  type="button"
                  onClick={toggleDockExpanded}
                  aria-label={dockExpanded ? "恢复字幕工作区高度" : "展开字幕工作区"}
                  title={dockExpanded ? "恢复字幕工作区高度" : "展开字幕工作区"}
                >
                  {dockExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                </button>
              </div>
            </header>
            {dockView === "timeline" ? (
              <Timeline
                project={project}
                cues={workspace.cues}
                currentTime={currentTime}
                selectedCueId={selectedCueId}
                onSeek={setCurrentTime}
                onSelectCue={(cue) => setSelectedCueId(cue.cue_id)}
              />
            ) : (
              <CueGrid
                key={cueEvidenceView}
                cues={cueEvidenceCues}
                view={cueEvidenceView}
                selectedCueId={selectedCueId}
                frameOcrPending={frameOcrPending}
                onViewChange={(view) => {
                  setCueEvidenceView(view);
                  if (view !== "final" && drawMode === "cue_ocr") setDrawMode("inspect");
                }}
                onSelectCue={(cue) => {
                  if (cueEvidenceView === "final") setSelectedCueId(cue.cue_id);
                  setCurrentTime(cue.start_ms);
                }}
                onUpdateCue={updateCue}
                onAddCue={() => void addCue()}
                onDeleteCue={() => void deleteSelectedCue()}
                onFrameOcr={startCueFrameOcr}
              />
            )}
          </section>
        </main>

        <Inspector
          activeStep={activeStep}
          project={project}
          cues={workspace.cues}
          jobs={workspace.jobs}
          selectedCue={selectedCue}
          isDemo={connection !== "live"}
          ocrCapabilities={ocrCapabilities}
          audioCapabilities={audioCapabilities}
          asrModels={asrModels}
          translationSettings={workspace.translation_settings}
          layoutSettings={workspace.layout_settings}
          onImport={importVideo}
          onResetWorkspace={resetWorkspace}
          onDrawRegion={(kind) => {
            setDrawMode(kind);
            setActiveStep(kind === "source" ? "roi" : "layout");
          }}
          onTranslationSettingsChange={updateTranslationSettings}
          onLayoutSettingsChange={(layout_settings) => setWorkspace((current) => ({ ...current, layout_settings }))}
          onContextChange={(context: ProjectContext) => updateProject((current) => ({ ...current, context }))}
          onTargetLanguageChange={updateTargetLanguage}
          onRecognitionLanguageChange={setRecognitionLanguage}
          onCueColorChange={(color) => void updateCueColor(color)}
          onSpeakerColorChange={(color) => void updateSpeakerColor(color)}
          onStartEyedropper={startEyedropper}
          onSaveTranslation={saveTranslation}
          onTestTranslation={testTranslation}
          onFetchTranslationModels={fetchTranslationModels}
          onSaveLayout={saveLayout}
          onStartJob={startJob}
          onCancelJob={cancelJob}
        />
      </div>
      {pageView === "logs" && <LogConsole isLive={connection === "live"} />}

      {aiTraceJob && closedAiTraceJobId !== aiTraceJob.id && (
        <AiTraceWindow
          job={aiTraceJob}
          minimized={aiTraceMinimized}
          onMinimizedChange={setAiTraceMinimized}
          onClose={() => setClosedAiTraceJobId(aiTraceJob.id)}
        />
      )}

      {frameOcrCandidate && (
        <div className="modal-backdrop" role="presentation">
          <section className="frame-ocr-confirm" role="dialog" aria-modal="true" aria-labelledby="frame-ocr-title">
            <header>
              <ScanText size={17} />
              <div>
                <strong id="frame-ocr-title">确认覆盖字幕原文</strong>
                <span>{frameOcrCandidate.cueId} · 当前帧局部识别</span>
              </div>
              <button type="button" onClick={() => setFrameOcrCandidate(null)} aria-label="关闭确认窗口" title="关闭">
                <X size={15} />
              </button>
            </header>
            <div className="frame-ocr-result-meta">
              <span>识别置信度</span>
              <strong>{frameOcrCandidate.confidence == null ? "--" : `${(frameOcrCandidate.confidence * 100).toFixed(1)}%`}</strong>
              <code>
                x {frameOcrCandidate.bbox.x.toFixed(3)} · y {frameOcrCandidate.bbox.y.toFixed(3)} · w {frameOcrCandidate.bbox.width.toFixed(3)} · h {frameOcrCandidate.bbox.height.toFixed(3)}
              </code>
            </div>
            <label>
              <span>OCR 结果</span>
              <textarea
                autoFocus
                value={frameOcrCandidate.text}
                onChange={(event) => setFrameOcrCandidate((current) => current ? { ...current, text: event.target.value } : current)}
              />
            </label>
            <footer>
              <button className="button secondary" type="button" onClick={() => setFrameOcrCandidate(null)}>取消</button>
              <button className="button primary" type="button" disabled={!frameOcrCandidate.text.trim()} onClick={confirmFrameOcrOverwrite}>确认覆盖原文</button>
            </footer>
          </section>
        </div>
      )}

      <div className="toast-stack" aria-live="polite">
        {toasts.map((toast) => (
          <div className={`toast ${toast.kind}`} key={toast.id}>
            {toast.kind === "success" ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
            <span>{toast.message}</span>
            <button type="button" onClick={() => setToasts((current) => current.filter((item) => item.id !== toast.id))} aria-label="关闭通知"><X size={14} /></button>
          </div>
        ))}
      </div>
    </div>
  );
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isOcrSnapshot(snapshot: Job["snapshot"]): snapshot is OcrJobSnapshot {
  return Boolean(
    snapshot
    && typeof snapshot.timestamp_ms === "number"
    && Array.isArray(snapshot.cues)
    && Array.isArray(snapshot.current)
    && snapshot.metrics,
  );
}

function boundedNumber(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  return Math.min(max, Math.max(min, Number.isFinite(parsed) ? parsed : fallback));
}

function boundedInteger(value: unknown, fallback: number, min: number, max: number): number {
  return Math.round(boundedNumber(value, fallback, min, max));
}

function normalizedSlicerSettings(payload: Record<string, unknown>) {
  const slicerHopSizeMs = boundedInteger(payload.slicer_hop_size_ms, 10, 1, 1_000);
  const slicerMinIntervalMs = Math.max(
    slicerHopSizeMs,
    boundedInteger(payload.slicer_min_interval_ms, 200, 10, 60_000),
  );
  return {
    slicer_threshold_db: boundedNumber(payload.slicer_threshold_db, -34, -100, 0),
    slicer_min_length_ms: Math.max(
      slicerMinIntervalMs,
      boundedInteger(payload.slicer_min_length_ms, 4_000, 100, 600_000),
    ),
    slicer_max_length_ms: Math.max(
      slicerMinIntervalMs,
      boundedInteger(payload.slicer_min_length_ms, 4_000, 100, 600_000),
      boundedInteger(payload.slicer_max_length_ms, 30_000, 1_000, 600_000),
    ),
    slicer_min_interval_ms: slicerMinIntervalMs,
    slicer_hop_size_ms: slicerHopSizeMs,
    slicer_max_sil_kept_ms: Math.max(
      slicerHopSizeMs,
      boundedInteger(payload.slicer_max_sil_kept_ms, 500, 1, 60_000),
    ),
  };
}

function readVideoMetadata(url: string): Promise<{ duration_ms: number; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.onloadedmetadata = () => {
      resolve({
        duration_ms: Number.isFinite(video.duration) ? Math.round(video.duration * 1000) : 0,
        width: video.videoWidth,
        height: video.videoHeight,
      });
    };
    video.onerror = () => reject(new Error("metadata unavailable"));
    video.src = url;
  });
}
