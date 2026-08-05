import type { BBox, Cue } from "./types";

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function formatTime(ms: number, precise = false): string {
  const safe = Math.max(0, ms);
  const hours = Math.floor(safe / 3_600_000);
  const minutes = Math.floor((safe % 3_600_000) / 60_000);
  const seconds = Math.floor((safe % 60_000) / 1000);
  const millis = Math.floor(safe % 1000);
  const head = hours > 0 ? `${String(hours).padStart(2, "0")}:` : "";
  const base = `${head}${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  return precise ? `${base}.${String(millis).padStart(3, "0")}` : base;
}

export function parseTime(value: string): number | null {
  const match = value.trim().match(/^(?:(\d{1,2}):)?(\d{1,2}):(\d{1,2})(?:[.,](\d{1,3}))?$/);
  if (!match) return null;
  const [, hours = "0", minutes, seconds, millis = "0"] = match;
  return (
    Number(hours) * 3_600_000 +
    Number(minutes) * 60_000 +
    Number(seconds) * 1000 +
    Number(millis.padEnd(3, "0"))
  );
}

export function normalizeBox(start: { x: number; y: number }, end: { x: number; y: number }): BBox {
  const x = clamp(Math.min(start.x, end.x), 0, 1);
  const y = clamp(Math.min(start.y, end.y), 0, 1);
  return {
    x,
    y,
    width: clamp(Math.abs(end.x - start.x), 0.01, 1 - x),
    height: clamp(Math.abs(end.y - start.y), 0.01, 1 - y),
  };
}

export function cuesAtTime(cues: Cue[], timeMs: number): Cue[] {
  return cues.filter((cue) => cue.start_ms <= timeMs && cue.end_ms >= timeMs);
}

export function confidenceLabel(confidence: number | null): string {
  if (confidence == null) return "--";
  return `${Math.round(confidence * 100)}%`;
}
