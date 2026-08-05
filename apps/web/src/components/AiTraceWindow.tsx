import { BrainCircuit, ChevronDown, ChevronUp, X } from "lucide-react";
import type { AiTrace, Job } from "../types";

interface AiTraceWindowProps {
  job: Job;
  minimized: boolean;
  onMinimizedChange: (minimized: boolean) => void;
  onClose: () => void;
}

const textValue = (value: unknown): string => typeof value === "string" ? value.trim() : "";

function traceText(trace: AiTrace | null | undefined, ...fields: string[]): string {
  if (!trace) return "";
  for (const field of fields) {
    const value = textValue(trace[field]);
    if (value) return value;
  }
  return "";
}

export function AiTraceWindow({
  job,
  minimized,
  onMinimizedChange,
  onClose,
}: AiTraceWindowProps) {
  const trace = job.snapshot?.ai_trace;
  const reasoning = traceText(trace, "reasoning_content", "reasoning", "thinking");
  const outputPreview = traceText(trace, "content_preview", "output_preview", "response_preview");
  const phase = traceText(trace, "phase", "stage", "status") || job.stage || "等待上游响应";
  const batch = Number(trace?.batch ?? trace?.batch_index);
  const totalBatches = Number(trace?.total_batches);
  const completed = Number(trace?.completed);
  const total = Number(trace?.total);
  const batchLabel = Number.isFinite(batch) && batch > 0
    ? `批次 ${batch}${Number.isFinite(totalBatches) && totalBatches > 0 ? ` / ${totalBatches}` : ""}`
    : Number.isFinite(completed) && Number.isFinite(total) && total > 0
      ? `${completed} / ${total} 条`
    : null;
  const progress = Math.max(0, Math.min(1, job.progress));
  const statusLabel = job.status === "completed"
    ? "已完成"
    : job.status === "failed"
      ? "失败"
      : job.status === "queued"
        ? "排队中"
        : "生成中";

  if (minimized) {
    return (
      <aside className="ai-trace-window is-minimized" aria-label="AI 实时进度">
        <BrainCircuit size={15} />
        <div className="ai-trace-minimized-copy">
          <strong>{job.kind === "fusion" ? "AI 校对" : "AI 翻译"}</strong>
          <span>{phase}</span>
        </div>
        <span className="ai-trace-percent">{Math.round(progress * 100)}%</span>
        <button type="button" onClick={() => onMinimizedChange(false)} title="展开 AI 过程" aria-label="展开 AI 过程">
          <ChevronUp size={15} />
        </button>
        <button type="button" onClick={onClose} title="关闭 AI 过程" aria-label="关闭 AI 过程">
          <X size={15} />
        </button>
        <i style={{ width: `${progress * 100}%` }} />
      </aside>
    );
  }

  return (
    <aside className="ai-trace-window" aria-label="AI 实时过程">
      <header>
        <BrainCircuit size={16} />
        <div>
          <strong>{job.kind === "fusion" ? "AI 校对过程" : "AI 翻译过程"}</strong>
          <span>{statusLabel}{batchLabel ? ` · ${batchLabel}` : ""}</span>
        </div>
        <span className="ai-trace-percent">{Math.round(progress * 100)}%</span>
        <button type="button" onClick={() => onMinimizedChange(true)} title="收起 AI 过程" aria-label="收起 AI 过程">
          <ChevronDown size={15} />
        </button>
        <button type="button" onClick={onClose} title="关闭 AI 过程" aria-label="关闭 AI 过程">
          <X size={15} />
        </button>
      </header>

      <div className="ai-trace-progress" aria-hidden="true">
        <i style={{ width: `${progress * 100}%` }} />
      </div>

      <div className="ai-trace-body" aria-live="polite">
        <div className="ai-trace-phase">
          <span>当前阶段</span>
          <strong>{phase}</strong>
        </div>
        <section>
          <span>模型思考流</span>
          <pre>{reasoning || (job.status === "running" ? "等待上游模型返回思考流..." : "上游模型未返回独立思考流。")}</pre>
        </section>
        {outputPreview && (
          <section>
            <span>输出预览</span>
            <pre>{outputPreview}</pre>
          </section>
        )}
      </div>

      <footer>{traceText(trace, "message") || job.message}</footer>
    </aside>
  );
}
