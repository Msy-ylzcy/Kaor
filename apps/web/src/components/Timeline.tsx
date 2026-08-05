import { Layers3, Minus, Plus } from "lucide-react";
import { useMemo, useState } from "react";
import type { Cue, Project } from "../types";
import { clamp, formatTime } from "../utils";

interface TimelineProps {
  project: Project;
  cues: Cue[];
  currentTime: number;
  selectedCueId: string | null;
  onSeek: (timeMs: number) => void;
  onSelectCue: (cue: Cue) => void;
}

export function Timeline({
  project,
  cues,
  currentTime,
  selectedCueId,
  onSeek,
  onSelectCue,
}: TimelineProps) {
  const [zoom, setZoom] = useState(1);
  const fullDuration = Math.max(project.duration_ms, 1);
  const windowDuration = fullDuration / zoom;
  const page = Math.floor(currentTime / windowDuration);
  const windowStart = clamp(page * windowDuration, 0, Math.max(0, fullDuration - windowDuration));
  const windowEnd = windowStart + windowDuration;
  const visibleCues = useMemo(
    () => cues.filter((cue) => cue.end_ms >= windowStart && cue.start_ms <= windowEnd),
    [cues, windowEnd, windowStart],
  );
  const laneCount = Math.max(2, ...visibleCues.map((cue) => cue.layer + 1));
  const playhead = clamp(((currentTime - windowStart) / windowDuration) * 100, 0, 100);
  const ticks = Array.from({ length: 9 }, (_, index) => windowStart + (windowDuration * index) / 8);

  return (
    <div className="timeline-panel">
      <div className="timeline-toolbar">
        <div className="timeline-legend">
          {[...new Map(cues.map((cue) => [cue.speaker_id, cue])).values()].slice(0, 5).map((cue) => (
            <span key={cue.speaker_id}>
              <i style={{ background: cue.speaker_color }} />
              {cue.speaker_name || cue.speaker_id}
            </span>
          ))}
        </div>
        <div className="timeline-zoom">
          <Minus size={13} />
          <input
            type="range"
            min="1"
            max="8"
            step="1"
            value={zoom}
            onChange={(event) => setZoom(Number(event.target.value))}
            aria-label="时间轴缩放"
          />
          <Plus size={13} />
          <span>{zoom}×</span>
        </div>
      </div>

      <div
        className="timeline-canvas"
        style={{ "--lane-count": laneCount } as React.CSSProperties}
        onPointerDown={(event) => {
          if ((event.target as HTMLElement).closest(".cue-block")) return;
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
          onSeek(windowStart + ratio * windowDuration);
        }}
      >
        <div className="timeline-ruler">
          {ticks.map((tick, index) => (
            <span key={`${tick}-${index}`} style={{ left: `${(index / 8) * 100}%` }}>
              {formatTime(tick)}
            </span>
          ))}
        </div>
        <div className="timeline-lanes">
          {Array.from({ length: laneCount }, (_, lane) => (
            <div className="timeline-lane" key={lane}>
              <span className="lane-label">L{lane + 1}</span>
            </div>
          ))}
          {visibleCues.map((cue) => {
            const start = clamp(cue.start_ms, windowStart, windowEnd);
            const end = clamp(cue.end_ms, windowStart, windowEnd);
            const left = ((start - windowStart) / windowDuration) * 100;
            const width = Math.max(((end - start) / windowDuration) * 100, 0.7);
            const overlaps = Boolean(cue.group_id ?? cue.overlap_group_id);
            return (
              <button
                key={cue.cue_id}
                type="button"
                className={`cue-block ${selectedCueId === cue.cue_id ? "is-selected" : ""} ${
                  cue.review_status === "needs_review" ? "needs-review" : ""
                }`}
                style={
                  {
                    left: `${left}%`,
                    width: `${width}%`,
                    top: `calc(28px + ${cue.layer} * var(--lane-height))`,
                    "--speaker-color": cue.speaker_color,
                  } as React.CSSProperties
                }
                onClick={(event) => {
                  event.stopPropagation();
                  onSelectCue(cue);
                  onSeek(cue.start_ms);
                }}
                title={`${cue.speaker_name}: ${cue.source_text}`}
              >
                {overlaps && <Layers3 size={11} aria-label="重叠对白" />}
                <span>{cue.speaker_name || cue.speaker_id}</span>
                <strong>{cue.target_text || cue.source_text}</strong>
              </button>
            );
          })}
        </div>
        <div className="timeline-playhead" style={{ left: `${playhead}%` }}>
          <span />
        </div>
      </div>
    </div>
  );
}
