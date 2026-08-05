import { CheckCircle2, Download, Plus, ScanText, Search, Trash2, TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { Cue, CueEvidenceView } from "../types";
import { confidenceLabel, formatTime, parseTime } from "../utils";

interface CueGridProps {
  cues: Cue[];
  view: CueEvidenceView;
  selectedCueId: string | null;
  frameOcrPending: boolean;
  onViewChange: (view: CueEvidenceView) => void;
  onSelectCue: (cue: Cue) => void;
  onUpdateCue: (cue: Cue) => void;
  onAddCue: () => void;
  onDeleteCue: () => void;
  onFrameOcr: () => void;
}

function escapeCsv(value: unknown): string {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

const viewLabels: Record<CueEvidenceView, string> = {
  final: "最终字幕",
  ocr: "OCR 原始",
  asr: "ASR 原始",
};

export function CueGrid({
  cues,
  view,
  selectedCueId,
  frameOcrPending,
  onViewChange,
  onSelectCue,
  onUpdateCue,
  onAddCue,
  onDeleteCue,
  onFrameOcr,
}: CueGridProps) {
  const [query, setQuery] = useState("");
  const [evidenceSelectedCueId, setEvidenceSelectedCueId] = useState<string | null>(null);
  const readOnly = view !== "final";

  useEffect(() => {
    setEvidenceSelectedCueId(null);
  }, [view]);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return cues;
    return cues.filter((cue) =>
      [cue.cue_id, cue.speaker_name, cue.source_text, cue.target_text]
        .join(" ")
        .toLowerCase()
        .includes(normalized),
    );
  }, [cues, query]);

  const downloadCsv = () => {
    const fields: Array<keyof Cue> = [
      "cue_id",
      "start_ms",
      "end_ms",
      "group_id",
      "layer",
      "track_id",
      "speaker_id",
      "speaker_name",
      "speaker_color",
      "source_kind",
      "source_text",
      "ocr_confidence",
      "review_status",
    ];
    const rows = [
      fields.join(","),
      ...cues.map((cue) =>
        fields
          .map((field) => escapeCsv(cue[field]))
          .join(","),
      ),
    ];
    const blob = new Blob(["\ufeff", rows.join("\r\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = view === "ocr" ? "ocr.csv" : view === "asr" ? "speech.csv" : "source.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="cue-grid-panel">
      <div className="grid-toolbar">
        <div className="cue-source-tabs" role="tablist" aria-label="字幕证据来源">
          {(["final", "ocr", "asr"] as const).map((option) => (
            <button
              key={option}
              type="button"
              role="tab"
              aria-selected={view === option}
              className={view === option ? "is-active" : ""}
              onClick={() => onViewChange(option)}
            >
              {option === "final" ? "最终" : option.toUpperCase()}
            </button>
          ))}
        </div>
        <label className="search-box">
          <Search size={14} />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索台词、角色或编号"
          />
        </label>
        <span className="grid-count" title={viewLabels[view]}>{viewLabels[view]} · {filtered.length} / {cues.length} 条</span>
        <button
          className="button ghost compact-button frame-ocr-button"
          type="button"
          onClick={onFrameOcr}
          disabled={readOnly || !selectedCueId || frameOcrPending}
          title="在当前视频帧框选区域，只识别并覆盖所选字幕的原文"
        >
          <ScanText size={14} />
          {frameOcrPending ? "识别中" : "框选帧 OCR"}
        </button>
        <button className="button ghost compact-button" type="button" onClick={onAddCue} disabled={readOnly} title="在当前播放时间新建字幕">
          <Plus size={14} />
          新增字幕
        </button>
        <button className="button ghost compact-button danger-button" type="button" onClick={onDeleteCue} disabled={readOnly || !selectedCueId} title="删除当前选中的字幕">
          <Trash2 size={14} />
          删除字幕
        </button>
        <button className="button ghost compact-button" type="button" onClick={downloadCsv}>
          <Download size={14} />
          导出 {view === "final" ? "字幕" : view.toUpperCase()} CSV
        </button>
      </div>

      <div className="cue-table-wrap">
        <table className="cue-table">
          <thead>
            <tr>
              <th className="cell-index">#</th>
              <th className="cell-time">开始</th>
              <th className="cell-time">结束</th>
              <th className="cell-speaker">说话人</th>
              <th className="cell-source">原文</th>
              <th className="cell-confidence">识别</th>
              <th className="cell-target">译文</th>
              <th className="cell-status">状态</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr className="cue-empty-row">
                <td colSpan={8}>
                  {query ? "没有匹配的字幕" : `${viewLabels[view]}尚无可用结果`}
                </td>
              </tr>
            )}
            {filtered.map((cue) => (
              <tr
                key={cue.cue_id}
                className={(readOnly ? evidenceSelectedCueId : selectedCueId) === cue.cue_id ? "is-selected" : ""}
                onClick={() => {
                  if (readOnly) setEvidenceSelectedCueId(cue.cue_id);
                  onSelectCue(cue);
                }}
              >
                <td className="cell-index">
                  <span>{cue.cue_id}</span>
                  {(cue.group_id ?? cue.overlap_group_id) && <i title="与其他对白同时发生">并</i>}
                </td>
                <td className="cell-time">
                  <input
                    defaultValue={formatTime(cue.start_ms, true)}
                    readOnly={readOnly}
                    onBlur={(event) => {
                      if (readOnly) return;
                      const next = parseTime(event.target.value);
                      if (next != null && next < cue.end_ms) onUpdateCue({ ...cue, start_ms: next });
                      else event.target.value = formatTime(cue.start_ms, true);
                    }}
                    aria-label={`${cue.cue_id} 开始时间`}
                  />
                </td>
                <td className="cell-time">
                  <input
                    defaultValue={formatTime(cue.end_ms, true)}
                    readOnly={readOnly}
                    onBlur={(event) => {
                      if (readOnly) return;
                      const next = parseTime(event.target.value);
                      if (next != null && next > cue.start_ms) onUpdateCue({ ...cue, end_ms: next });
                      else event.target.value = formatTime(cue.end_ms, true);
                    }}
                    aria-label={`${cue.cue_id} 结束时间`}
                  />
                </td>
                <td className="cell-speaker">
                  <span className="speaker-dot" style={{ background: cue.speaker_color }} />
                  <input
                    value={cue.speaker_name}
                    readOnly={readOnly}
                    onChange={(event) => onUpdateCue({ ...cue, speaker_name: event.target.value })}
                    aria-label={`${cue.cue_id} 说话人`}
                  />
                </td>
                <td className="cell-source">
                  <textarea
                    value={cue.source_text}
                    readOnly={readOnly}
                    onChange={(event) =>
                      onUpdateCue({ ...cue, source_text: event.target.value, review_status: "pending" })
                    }
                    aria-label={`${cue.cue_id} 原文`}
                  />
                </td>
                <td className="cell-confidence">
                  <span className={(cue.ocr_confidence ?? 0) < 0.9 ? "confidence-low" : ""}>
                    {(cue.ocr_confidence ?? 0) < 0.9 ? <TriangleAlert size={12} /> : <CheckCircle2 size={12} />}
                    {confidenceLabel(cue.ocr_confidence)}
                  </span>
                </td>
                <td className="cell-target">
                  <textarea
                    value={cue.target_text}
                    readOnly={readOnly}
                    onChange={(event) =>
                      onUpdateCue({ ...cue, target_text: event.target.value, review_status: "pending" })
                    }
                    aria-label={`${cue.cue_id} 译文`}
                  />
                </td>
                <td className="cell-status">
                  <select
                    value={cue.review_status}
                    disabled={readOnly}
                    onChange={(event) =>
                      onUpdateCue({ ...cue, review_status: event.target.value as Cue["review_status"] })
                    }
                    aria-label={`${cue.cue_id} 校对状态`}
                  >
                    <option value="pending">待处理</option>
                    <option value="needs_review">需校对</option>
                    <option value="approved">已确认</option>
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
