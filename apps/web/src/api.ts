import type {
  AudioCapabilities,
  AsrModelOption,
  BBox,
  Cue,
  FrameOcrResult,
  Job,
  LayoutSettings,
  DiagnosticLogPayload,
  LocalModelCatalogResponse,
  LocalTranslationRuntimeStatus,
  LocalRuntimeVariant,
  LocalTranslationProvider,
  OcrCapabilities,
  Project,
  TranslationModelOption,
  TranslationSettings,
  WorkspaceData,
} from "./types";

const API_ROOT = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string; message?: string };
      detail = body.detail ?? body.message ?? detail;
    } catch {
      // Keep the HTTP status when the server does not return JSON.
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function requestFirst<T>(attempts: Array<() => Promise<T>>): Promise<T> {
  let lastError: unknown;
  for (const attempt of attempts) {
    try {
      return await attempt();
    } catch (error) {
      lastError = error;
      if (!(error instanceof ApiError) || error.status !== 404) throw error;
    }
  }
  throw lastError;
}

export const api = {
  projectCsvUrl(projectId: string, name: "ocr" | "speech"): string {
    return `${API_ROOT}/projects/${encodeURIComponent(projectId)}/${name}.csv`;
  },

  getOcrCapabilities(): Promise<OcrCapabilities> {
    return request<OcrCapabilities>("/ocr/capabilities");
  },

  getAudioCapabilities(): Promise<AudioCapabilities> {
    return request<AudioCapabilities>("/audio/capabilities");
  },

  getAudioModels(): Promise<AsrModelOption[]> {
    return request<AsrModelOption[]>("/audio/models");
  },

  getLocalModelStatus(refreshHardware = false): Promise<LocalTranslationRuntimeStatus> {
    return request<LocalTranslationRuntimeStatus>(
      `/local-models/status${refreshHardware ? "?refresh_hardware=true" : ""}`,
    );
  },

  getLocalModelCatalog(): Promise<LocalModelCatalogResponse> {
    return request<LocalModelCatalogResponse>("/local-models/catalog");
  },

  deployLocalModel(payload: {
    model_id?: string;
    runtime_variant?: LocalRuntimeVariant;
    port?: number;
    context_size?: number;
    gpu_layers?: number;
    threads?: number;
    auto_start?: boolean;
    make_default?: boolean;
  }): Promise<Job> {
    return request<Job>("/local-models/deploy-jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  configureLocalModel(payload: Record<string, unknown>): Promise<{
    provider: LocalTranslationProvider;
    status: LocalTranslationRuntimeStatus;
  }> {
    return request("/local-models/configuration", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  },

  startLocalModel(): Promise<LocalTranslationRuntimeStatus> {
    return request<LocalTranslationRuntimeStatus>("/local-models/start", { method: "POST", body: "{}" });
  },

  stopLocalModel(): Promise<LocalTranslationRuntimeStatus> {
    return request<LocalTranslationRuntimeStatus>("/local-models/stop", { method: "POST", body: "{}" });
  },

  activateLocalModel(): Promise<{ provider: LocalTranslationProvider; active: true }> {
    return request("/local-models/activate", { method: "POST", body: "{}" });
  },

  deactivateLocalModel(): Promise<void> {
    return request<unknown>("/local-models/deactivate", { method: "POST", body: "{}" })
      .then(() => undefined);
  },

  getDiagnosticLogs(options: {
    source?: string;
    tail?: number;
    query?: string;
    levels?: string[];
  } = {}): Promise<DiagnosticLogPayload> {
    const params = new URLSearchParams();
    if (options.source && options.source !== "all") params.set("source", options.source);
    if (options.tail) params.set("tail", String(options.tail));
    if (options.query) params.set("query", options.query);
    if (options.levels?.length) params.set("level", options.levels.join(","));
    const suffix = params.size ? `?${params.toString()}` : "";
    return request<DiagnosticLogPayload>(`/diagnostics/logs${suffix}`);
  },

  diagnosticExportUrl(): string {
    return `${API_ROOT}/diagnostics/export`;
  },

  troubleshootingGuideUrl(anchor: string): string {
    return `${API_ROOT}/diagnostics/troubleshooting#${encodeURIComponent(anchor)}`;
  },

  async loadWorkspace(): Promise<WorkspaceData> {
    try {
      return await request<WorkspaceData>("/workspace");
    } catch (error) {
      if (!(error instanceof ApiError) || error.status !== 404) throw error;
    }

    const project = await requestFirst<Project>([
      () => request<Project>("/projects/current"),
      () => request<Project>("/project"),
    ]);
    const [cues, jobs, translationSettings, layoutSettings] = await Promise.all([
      requestFirst<Cue[]>([
        () => request<Cue[]>(`/projects/${project.id}/cues`),
        () => request<Cue[]>("/cues"),
      ]),
      request<Job[]>("/jobs").catch(() => []),
      request<TranslationSettings>(`/projects/${project.id}/translation-profile`).catch(
        () => request<TranslationSettings>("/settings/translation"),
      ),
      request<LayoutSettings>(`/projects/${project.id}/subtitle-layout`).catch(() =>
        request<LayoutSettings>("/settings/layout"),
      ),
    ]);

    return {
      project,
      cues,
      jobs,
      translation_settings: translationSettings,
      layout_settings: layoutSettings,
    };
  },

  importVideo(file: File): Promise<Project> {
    return request<Project>("/media/register", {
      method: "POST",
      body: file,
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Kaor-Filename": encodeURIComponent(file.name),
      },
    });
  },

  resetProject(projectId: string): Promise<Project> {
    return request<Project>(`/projects/${encodeURIComponent(projectId)}/reset`, {
      method: "POST",
      body: "{}",
    });
  },

  updateRegion(projectId: string, kind: "source" | "target", bbox: BBox): Promise<Project> {
    return requestFirst<Project>([
      () =>
        request<Project>(`/projects/${projectId}/regions`, {
          method: "PUT",
          body: JSON.stringify({ [`${kind}_roi`]: bbox }),
        }),
      () =>
        request<Project>("/projects/current/regions", {
          method: "PUT",
          body: JSON.stringify({ kind, bbox }),
        }),
    ]);
  },

  recognizeFrame(
    projectId: string,
    payload: {
      timestamp_ms: number;
      bbox: BBox;
      language: string;
      device: string;
      high_accuracy: boolean;
    },
  ): Promise<FrameOcrResult> {
    return requestFirst<FrameOcrResult>([
      () =>
        request<FrameOcrResult>(`/projects/${encodeURIComponent(projectId)}/frame-ocr`, {
          method: "POST",
          body: JSON.stringify(payload),
        }),
      () =>
        request<FrameOcrResult>(`/projects/${encodeURIComponent(projectId)}/ocr/frame`, {
          method: "POST",
          body: JSON.stringify(payload),
        }),
    ]);
  },

  updateCue(projectId: string, cue: Cue): Promise<Cue> {
    return requestFirst<Cue>([
      () =>
        request<Cue>(`/projects/${projectId}/cues/${encodeURIComponent(cue.cue_id)}`, {
          method: "PUT",
          body: JSON.stringify(cue),
        }),
      () =>
        request<Cue>(`/cues/${encodeURIComponent(cue.cue_id)}`, {
          method: "PATCH",
          body: JSON.stringify(cue),
        }),
    ]);
  },

  createCue(projectId: string, cue: Cue): Promise<Cue> {
    return request<Cue>(`/projects/${projectId}/cues`, {
      method: "POST",
      body: JSON.stringify(cue),
    });
  },

  deleteCue(projectId: string, cueId: string): Promise<void> {
    return request<void>(
      `/projects/${projectId}/cues/${encodeURIComponent(cueId)}`,
      { method: "DELETE" },
    );
  },

  updateCueColor(projectId: string, cueId: string, speakerColor: string): Promise<Cue> {
    return request<Cue>(
      `/projects/${projectId}/cues/${encodeURIComponent(cueId)}/color`,
      {
        method: "PATCH",
        body: JSON.stringify({ speaker_color: speakerColor }),
      },
    );
  },

  updateSpeakerColor(
    projectId: string,
    speakerId: string,
    speakerName: string,
    speakerColor: string,
  ): Promise<Cue[]> {
    return request<Cue[]>(`/projects/${projectId}/speakers/color`, {
      method: "PATCH",
      body: JSON.stringify({
        speaker_id: speakerId,
        speaker_name: speakerName,
        speaker_color: speakerColor,
      }),
    });
  },

  updateTargetLanguage(projectId: string, targetLanguage: string): Promise<void> {
    return request<unknown>(`/projects/${projectId}`, {
      method: "PATCH",
      body: JSON.stringify({ target_language: targetLanguage }),
    }).then(() => undefined);
  },

  saveTranslationSettings(
    projectId: string,
    settings: TranslationSettings,
  ): Promise<TranslationSettings> {
    return requestFirst<TranslationSettings>([
      () =>
        request<TranslationSettings>(`/projects/${projectId}/translation-profile`, {
          method: "PUT",
          body: JSON.stringify(settings),
        }),
      () =>
        request<TranslationSettings>("/settings/translation", {
          method: "PUT",
          body: JSON.stringify(settings),
        }),
    ]);
  },

  getTranslationSettings(projectId: string): Promise<TranslationSettings> {
    return request<TranslationSettings>(`/projects/${projectId}/translation-profile`);
  },

  saveLayoutSettings(projectId: string, settings: LayoutSettings): Promise<LayoutSettings> {
    return requestFirst<LayoutSettings>([
      () =>
        request<LayoutSettings>(`/projects/${projectId}/subtitle-layout`, {
          method: "PUT",
          body: JSON.stringify(settings),
        }),
      () =>
        request<LayoutSettings>("/settings/layout", {
          method: "PUT",
          body: JSON.stringify(settings),
        }),
    ]);
  },

  testTranslation(settings: TranslationSettings): Promise<{ ok: boolean; latency_ms?: number }> {
    const startedAt = performance.now();
    return request<{ status: "ok"; response_preview: string }>("/translation/test", {
      method: "POST",
      body: JSON.stringify({
        provider: {
          base_url: settings.base_url,
          api_key: settings.api_key,
          model: settings.model,
          reasoning_effort: settings.reasoning_effort,
          api_path: settings.path,
          custom_headers: JSON.parse(settings.custom_headers || "{}"),
          timeout_seconds: settings.timeout_seconds,
        },
        options: {},
      }),
    }).then(() => ({ ok: true, latency_ms: Math.round(performance.now() - startedAt) }));
  },

  fetchTranslationModels(settings: TranslationSettings): Promise<TranslationModelOption[]> {
    return request<{ models: TranslationModelOption[] }>("/translation/models", {
      method: "POST",
      body: JSON.stringify({
        base_url: settings.base_url,
        api_key: settings.api_key,
        custom_headers: JSON.parse(settings.custom_headers || "{}"),
        timeout_seconds: settings.timeout_seconds,
      }),
    }).then((response) => response.models);
  },

  saveProjectContext(projectId: string, context: Project["context"]): Promise<Project["context"]> {
    return request<Project["context"]>(`/projects/${projectId}/context`, {
      method: "PUT",
      body: JSON.stringify(context),
    });
  },

  listJobs(projectId?: string): Promise<Job[]> {
    const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
    return request<Job[]>(`/jobs${query}`);
  },

  getJob(jobId: string): Promise<Job> {
    return request<Job>(`/jobs/${encodeURIComponent(jobId)}`);
  },

  startJob(
    projectId: string,
    kind: Job["kind"],
    payload: Record<string, unknown> = {},
  ): Promise<Job> {
    return requestFirst<Job>([
      () =>
        request<Job>(`/projects/${projectId}/${kind}-jobs`, {
          method: "POST",
          body: JSON.stringify(payload),
        }),
      () =>
        request<Job>("/jobs", {
          method: "POST",
          body: JSON.stringify({ project_id: projectId, kind, ...payload }),
        }),
    ]);
  },

  cancelJob(jobId: string): Promise<Job> {
    return request<Job>(`/jobs/${jobId}/cancel`, { method: "POST", body: "{}" });
  },
};
