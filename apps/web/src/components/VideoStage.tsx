import {
  Captions,
  Crosshair,
  Maximize2,
  MousePointer2,
  Pause,
  Play,
  ScanLine,
  StepBack,
  StepForward,
  Volume2,
  ZoomIn,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { BBox, Cue, OcrJobSnapshot, Project, RegionKind } from "../types";
import { clamp, cuesAtTime, formatTime, normalizeBox } from "../utils";

type DrawMode = "inspect" | "eyedropper" | "cue_ocr" | RegionKind;
type BoxDrawMode = RegionKind | "cue_ocr";
type Point = { x: number; y: number };
type ResizeHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
type RegionInteraction = {
  mode: BoxDrawMode;
  action: "draw" | "move" | "resize";
  start: Point;
  initialBox: BBox;
  handle?: ResizeHandle;
};

const MIN_REGION_SIZE = 0.02;
const HANDLE_SIZE_PX = 9;
const HANDLE_HIT_RADIUS_PX = 10;
const EDGE_HIT_RADIUS_PX = 5;

const RESIZE_HANDLES: Array<{ handle: ResizeHandle; x: 0 | 0.5 | 1; y: 0 | 0.5 | 1 }> = [
  { handle: "nw", x: 0, y: 0 },
  { handle: "ne", x: 1, y: 0 },
  { handle: "se", x: 1, y: 1 },
  { handle: "sw", x: 0, y: 1 },
  { handle: "n", x: 0.5, y: 0 },
  { handle: "e", x: 1, y: 0.5 },
  { handle: "s", x: 0.5, y: 1 },
  { handle: "w", x: 0, y: 0.5 },
];

const HANDLE_CURSORS: Record<ResizeHandle, string> = {
  nw: "nwse-resize",
  n: "ns-resize",
  ne: "nesw-resize",
  e: "ew-resize",
  se: "nwse-resize",
  s: "ns-resize",
  sw: "nesw-resize",
  w: "ew-resize",
};

const isBoxDrawMode = (mode: DrawMode): mode is BoxDrawMode =>
  mode === "source" || mode === "target" || mode === "cue_ocr";

const regionHandlePoints = (box: BBox) =>
  RESIZE_HANDLES.map(({ handle, x, y }) => ({
    handle,
    x: box.x + box.width * x,
    y: box.y + box.height * y,
  }));

const pointIsInsideBox = (point: Point, box: BBox) =>
  point.x >= box.x &&
  point.x <= box.x + box.width &&
  point.y >= box.y &&
  point.y <= box.y + box.height;

const hitTestResizeHandle = (box: BBox, point: Point, rect: DOMRect): ResizeHandle | null => {
  const pointerX = point.x * rect.width;
  const pointerY = point.y * rect.height;
  const halfHandle = HANDLE_SIZE_PX / 2;

  for (const handlePoint of regionHandlePoints(box)) {
    const handleX = clamp(handlePoint.x * rect.width, halfHandle, rect.width - halfHandle);
    const handleY = clamp(handlePoint.y * rect.height, halfHandle, rect.height - halfHandle);
    if (
      Math.abs(pointerX - handleX) <= HANDLE_HIT_RADIUS_PX &&
      Math.abs(pointerY - handleY) <= HANDLE_HIT_RADIUS_PX
    ) {
      return handlePoint.handle;
    }
  }

  const left = box.x * rect.width;
  const right = (box.x + box.width) * rect.width;
  const top = box.y * rect.height;
  const bottom = (box.y + box.height) * rect.height;
  const withinHorizontalEdge = pointerX >= left && pointerX <= right;
  const withinVerticalEdge = pointerY >= top && pointerY <= bottom;

  if (withinHorizontalEdge && Math.abs(pointerY - top) <= EDGE_HIT_RADIUS_PX) return "n";
  if (withinHorizontalEdge && Math.abs(pointerY - bottom) <= EDGE_HIT_RADIUS_PX) return "s";
  if (withinVerticalEdge && Math.abs(pointerX - left) <= EDGE_HIT_RADIUS_PX) return "w";
  if (withinVerticalEdge && Math.abs(pointerX - right) <= EDGE_HIT_RADIUS_PX) return "e";
  return null;
};

const cursorForRegionPoint = (box: BBox, point: Point, rect: DOMRect) => {
  const handle = hitTestResizeHandle(box, point, rect);
  if (handle) return HANDLE_CURSORS[handle];
  return pointIsInsideBox(point, box) ? "move" : "crosshair";
};

const boxFromDrag = (start: Point, current: Point): BBox => {
  const box = normalizeBox(start, current);
  if (box.width < MIN_REGION_SIZE) {
    box.x = current.x < start.x
      ? Math.max(0, start.x - MIN_REGION_SIZE)
      : Math.min(start.x, 1 - MIN_REGION_SIZE);
    box.width = MIN_REGION_SIZE;
  }
  if (box.height < MIN_REGION_SIZE) {
    box.y = current.y < start.y
      ? Math.max(0, start.y - MIN_REGION_SIZE)
      : Math.min(start.y, 1 - MIN_REGION_SIZE);
    box.height = MIN_REGION_SIZE;
  }
  box.x = clamp(box.x, 0, 1 - box.width);
  box.y = clamp(box.y, 0, 1 - box.height);
  return box;
};

const moveBox = (initialBox: BBox, start: Point, current: Point): BBox => {
  const width = clamp(initialBox.width, MIN_REGION_SIZE, 1);
  const height = clamp(initialBox.height, MIN_REGION_SIZE, 1);
  return {
    x: clamp(initialBox.x + current.x - start.x, 0, 1 - width),
    y: clamp(initialBox.y + current.y - start.y, 0, 1 - height),
    width,
    height,
  };
};

const resizeBox = (initialBox: BBox, current: Point, handle: ResizeHandle): BBox => {
  let left = clamp(initialBox.x, 0, 1 - MIN_REGION_SIZE);
  let right = clamp(initialBox.x + initialBox.width, left + MIN_REGION_SIZE, 1);
  let top = clamp(initialBox.y, 0, 1 - MIN_REGION_SIZE);
  let bottom = clamp(initialBox.y + initialBox.height, top + MIN_REGION_SIZE, 1);

  if (handle.includes("w")) left = clamp(current.x, 0, right - MIN_REGION_SIZE);
  if (handle.includes("e")) right = clamp(current.x, left + MIN_REGION_SIZE, 1);
  if (handle.includes("n")) top = clamp(current.y, 0, bottom - MIN_REGION_SIZE);
  if (handle.includes("s")) bottom = clamp(current.y, top + MIN_REGION_SIZE, 1);

  return { x: left, y: top, width: right - left, height: bottom - top };
};

const boxForInteraction = (interaction: RegionInteraction, current: Point): BBox => {
  if (interaction.action === "draw") return boxFromDrag(interaction.start, current);
  if (interaction.action === "move") {
    return moveBox(interaction.initialBox, interaction.start, current);
  }
  return resizeBox(interaction.initialBox, current, interaction.handle ?? "se");
};

interface VideoStageProps {
  project: Project;
  cues: Cue[];
  liveOcrSnapshot: OcrJobSnapshot | null;
  currentTime: number;
  playing: boolean;
  selectedCue: Cue | null;
  sourceUrl: string | null;
  drawMode: DrawMode;
  onDrawModeChange: (mode: DrawMode) => void;
  onTimeChange: (timeMs: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onRegionChange: (kind: RegionKind, box: BBox) => void;
  onFrameOcrRegion: (box: BBox) => void;
  onColorPick: (colorHex: string) => void;
}

export function VideoStage({
  project,
  cues,
  liveOcrSnapshot,
  currentTime,
  playing,
  selectedCue,
  sourceUrl,
  drawMode,
  onDrawModeChange,
  onTimeChange,
  onPlayingChange,
  onRegionChange,
  onFrameOcrRegion,
  onColorPick,
}: VideoStageProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const colorSamplerRef = useRef<HTMLCanvasElement | null>(null);
  const canvasWrapRef = useRef<HTMLDivElement>(null);
  const regionInteractionRef = useRef<RegionInteraction | null>(null);
  const [draftBox, setDraftBox] = useState<BBox | null>(null);
  const [regionCursor, setRegionCursor] = useState("crosshair");
  const [zoom, setZoom] = useState(1);
  const activeCues = useMemo(() => cuesAtTime(cues, currentTime), [cues, currentTime]);

  useEffect(() => {
    regionInteractionRef.current = null;
    setDraftBox(null);
    setRegionCursor("crosshair");
  }, [drawMode]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !sourceUrl) return;
    const frameMs = 1000 / Math.max(project.fps, 1);
    if (Math.abs(video.currentTime * 1000 - currentTime) > Math.max(0.5, frameMs * 0.05)) {
      video.currentTime = currentTime / 1000;
    }
  }, [currentTime, project.fps, sourceUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !sourceUrl) return;
    if (playing) {
      void video.play().catch(() => onPlayingChange(false));
    } else {
      video.pause();
    }
  }, [onPlayingChange, playing, sourceUrl]);

  useEffect(() => {
    if (sourceUrl || !playing) return;
    let frame = 0;
    let previous = performance.now();
    const tick = (now: number) => {
      const elapsed = now - previous;
      previous = now;
      const next = currentTime + elapsed;
      onTimeChange(next >= project.duration_ms ? 0 : next);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [currentTime, onTimeChange, playing, project.duration_ms, sourceUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = canvasWrapRef.current;
    if (!canvas || !wrap) return;

    const draw = () => {
      const rect = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      if (drawMode === "eyedropper") return;

      const paintBox = (box: BBox, color: string, label: string, dashed = false) => {
        const x = box.x * rect.width;
        const y = box.y * rect.height;
        const width = box.width * rect.width;
        const height = box.height * rect.height;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = `${color}1f`;
        ctx.lineWidth = 1.5;
        if (dashed) ctx.setLineDash([7, 5]);
        ctx.fillRect(x, y, width, height);
        ctx.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);
        ctx.setLineDash([]);
        ctx.font = "600 11px Inter, system-ui, sans-serif";
        const labelWidth = ctx.measureText(label).width + 14;
        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - 22), labelWidth, 20);
        ctx.fillStyle = "#101514";
        ctx.fillText(label, x + 7, Math.max(14, y - 8));
        ctx.restore();
      };

      const paintRegionHandles = (box: BBox, color: string) => {
        const halfHandle = HANDLE_SIZE_PX / 2;
        ctx.save();
        ctx.fillStyle = "#f7faf9";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.shadowColor = "rgb(0 0 0 / 0.58)";
        ctx.shadowBlur = 3;
        for (const point of regionHandlePoints(box)) {
          const centerX = clamp(point.x * rect.width, halfHandle, rect.width - halfHandle);
          const centerY = clamp(point.y * rect.height, halfHandle, rect.height - halfHandle);
          ctx.fillRect(
            centerX - halfHandle,
            centerY - halfHandle,
            HANDLE_SIZE_PX,
            HANDLE_SIZE_PX,
          );
          ctx.strokeRect(
            centerX - halfHandle,
            centerY - halfHandle,
            HANDLE_SIZE_PX,
            HANDLE_SIZE_PX,
          );
        }
        ctx.restore();
      };

      const sourceBox = drawMode === "source" && draftBox ? draftBox : project.source_roi;
      const targetBox = drawMode === "target" && draftBox ? draftBox : project.target_roi;
      paintBox(sourceBox, "#f0c75e", "原字幕区域", drawMode !== "source");
      paintBox(targetBox, "#66c7b5", "译文区域", drawMode !== "target");

      const cueBox = selectedCue?.bbox ?? selectedCue?.source_bbox;
      if (cueBox) {
        paintBox(cueBox, selectedCue?.speaker_color || "#ffffff", selectedCue?.cue_id ?? "cue", true);
      }
      if (drawMode === "cue_ocr" && draftBox) {
        paintBox(draftBox, "#66c7b5", "单帧 OCR");
      }
      if (drawMode === "source") paintRegionHandles(sourceBox, "#f0c75e");
      if (drawMode === "target") paintRegionHandles(targetBox, "#66c7b5");
    };

    const observer = new ResizeObserver(draw);
    observer.observe(wrap);
    draw();
    return () => observer.disconnect();
  }, [draftBox, drawMode, project.source_roi, project.target_roi, selectedCue, zoom]);

  const pointerPosition = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: clamp((event.clientX - rect.left) / rect.width, 0, 1),
      y: clamp((event.clientY - rect.top) / rect.height, 0, 1),
    };
  };

  const pickVideoColor = (clientX: number, clientY: number) => {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    if (video.videoWidth <= 0 || video.videoHeight <= 0) return;

    const rect = video.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    // The video uses object-fit: contain, so remove its letterbox area before
    // mapping the displayed pointer position back to a decoded-frame pixel.
    const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
    const renderedWidth = video.videoWidth * scale;
    const renderedHeight = video.videoHeight * scale;
    const renderedLeft = rect.left + (rect.width - renderedWidth) / 2;
    const renderedTop = rect.top + (rect.height - renderedHeight) / 2;
    const renderedX = clientX - renderedLeft;
    const renderedY = clientY - renderedTop;

    if (
      renderedX < 0 ||
      renderedY < 0 ||
      renderedX >= renderedWidth ||
      renderedY >= renderedHeight
    ) {
      return;
    }

    const sourceX = Math.min(
      video.videoWidth - 1,
      Math.floor((renderedX / renderedWidth) * video.videoWidth),
    );
    const sourceY = Math.min(
      video.videoHeight - 1,
      Math.floor((renderedY / renderedHeight) * video.videoHeight),
    );
    const sampler = colorSamplerRef.current ?? document.createElement("canvas");
    colorSamplerRef.current = sampler;
    sampler.width = 1;
    sampler.height = 1;
    const context = sampler.getContext("2d", { willReadFrequently: true });
    if (!context) return;

    try {
      context.imageSmoothingEnabled = false;
      context.clearRect(0, 0, 1, 1);
      context.drawImage(video, sourceX, sourceY, 1, 1, 0, 0, 1, 1);
      const [red, green, blue] = context.getImageData(0, 0, 1, 1).data;
      const hex = [red, green, blue]
        .map((channel) => channel.toString(16).padStart(2, "0"))
        .join("")
        .toUpperCase();
      onColorPick(`#${hex}`);
    } catch {
      // A cross-origin source can make the sampling canvas unreadable.
    }
  };

  const seekByFrames = (frames: number) => {
    const frameMs = 1000 / Math.max(project.fps, 1);
    const targetFrame = Math.round(currentTime / frameMs) + frames;
    onTimeChange(clamp(targetFrame * frameMs, 0, project.duration_ms));
  };

  return (
    <section className="video-stage" aria-label="视频预览与字幕区域框选">
      <div className="stage-toolbar">
        <div className="segmented-control" aria-label="画布工具">
          <button
            type="button"
            className={drawMode === "inspect" ? "is-active" : ""}
            onClick={() => onDrawModeChange("inspect")}
            title="选择与检查"
          >
            <MousePointer2 size={15} />
            <span>检查</span>
          </button>
          <button
            type="button"
            className={drawMode === "source" ? "is-active source-mode" : ""}
            onClick={() => onDrawModeChange("source")}
            title="拖动框选原字幕识别区域"
          >
            <ScanLine size={15} />
            <span>原字幕框</span>
          </button>
          <button
            type="button"
            className={drawMode === "target" ? "is-active" : ""}
            onClick={() => onDrawModeChange("target")}
            title="拖动框选译文显示区域"
          >
            <Captions size={15} />
            <span>译文框</span>
          </button>
        </div>

        <div className="stage-toolbar-spacer" />
        <Crosshair size={15} className="muted-icon" aria-hidden="true" />
        <span className="stage-coordinate">
          {project.width}×{project.height}
        </span>
        <label className="zoom-control" title="预览缩放">
          <ZoomIn size={15} />
          <input
            type="range"
            min="1"
            max="1.8"
            step="0.1"
            value={zoom}
            onChange={(event) => setZoom(Number(event.target.value))}
          />
          <span>{Math.round(zoom * 100)}%</span>
        </label>
        <button className="icon-button compact" type="button" title="全屏预览" aria-label="全屏预览">
          <Maximize2 size={16} />
        </button>
      </div>

      <div className="media-viewport">
        <div
          ref={canvasWrapRef}
          className="media-content"
          style={{
            transform: `scale(${zoom})`,
            aspectRatio: `${project.width || 16} / ${project.height || 9}`,
          }}
        >
          {sourceUrl ? (
            <video
              ref={videoRef}
              src={sourceUrl}
              preload="metadata"
              onTimeUpdate={(event) => onTimeChange(event.currentTarget.currentTime * 1000)}
              onEnded={() => onPlayingChange(false)}
            />
          ) : (
            <DemoFrame activeCues={activeCues} />
          )}

          {drawMode !== "eyedropper" && (
            <div
              className="subtitle-preview"
              style={{
                left: `${project.target_roi.x * 100}%`,
                top: `${project.target_roi.y * 100}%`,
                width: `${project.target_roi.width * 100}%`,
                height: `${project.target_roi.height * 100}%`,
              }}
            >
              {activeCues.map((cue) => (
                <span key={cue.cue_id} style={{ color: cue.speaker_color }}>
                  {cue.target_text || cue.source_text}
                </span>
              ))}
            </div>
          )}

          {drawMode !== "eyedropper" && liveOcrSnapshot && liveOcrSnapshot.current.length > 0 && (
            <div
              className="ocr-live-preview"
              style={{
                left: `${project.source_roi.x * 100}%`,
                top: `${project.source_roi.y * 100}%`,
                width: `${project.source_roi.width * 100}%`,
                height: `${project.source_roi.height * 100}%`,
              }}
              aria-live="polite"
            >
              {liveOcrSnapshot.current.map((item, index) => (
                <span key={`${liveOcrSnapshot.timestamp_ms}-${index}`} style={{ color: item.color }}>
                  {item.text}
                </span>
              ))}
            </div>
          )}

          <canvas
            ref={canvasRef}
            className={`roi-canvas ${isBoxDrawMode(drawMode) ? "is-drawing" : ""} ${
              drawMode === "eyedropper" ? "is-eyedropper" : ""
            }`}
            style={{ cursor: isBoxDrawMode(drawMode) ? regionCursor : undefined }}
            role={drawMode === "eyedropper" ? "button" : undefined}
            tabIndex={drawMode === "eyedropper" ? 0 : -1}
            aria-label={drawMode === "eyedropper" ? "从当前视频帧吸取字幕颜色" : drawMode === "cue_ocr" ? "框选当前帧中要识别的文字" : undefined}
            title={drawMode === "eyedropper" ? "点击视频画面吸取字幕颜色" : drawMode === "cue_ocr" ? "拖动框选文字，松开后执行单帧 OCR" : undefined}
            onPointerDown={(event) => {
              if (drawMode === "eyedropper") {
                pickVideoColor(event.clientX, event.clientY);
                return;
              }
              if (!isBoxDrawMode(drawMode)) return;
              event.preventDefault();
              const point = pointerPosition(event);
              const activeBox = drawMode === "source"
                ? project.source_roi
                : drawMode === "target"
                  ? project.target_roi
                  : { x: point.x, y: point.y, width: MIN_REGION_SIZE, height: MIN_REGION_SIZE };
              const handle = drawMode === "cue_ocr"
                ? null
                : hitTestResizeHandle(
                    activeBox,
                    point,
                    event.currentTarget.getBoundingClientRect(),
                  );
              const action: RegionInteraction["action"] = drawMode === "cue_ocr"
                ? "draw"
                : handle
                  ? "resize"
                  : pointIsInsideBox(point, activeBox)
                    ? "move"
                    : "draw";
              const interaction: RegionInteraction = {
                mode: drawMode,
                action,
                start: point,
                initialBox: { ...activeBox },
                handle: handle ?? undefined,
              };
              regionInteractionRef.current = interaction;
              event.currentTarget.setPointerCapture(event.pointerId);
              setDraftBox(boxForInteraction(interaction, point));
              setRegionCursor(
                handle ? HANDLE_CURSORS[handle] : action === "move" ? "grabbing" : "crosshair",
              );
            }}
            onPointerMove={(event) => {
              if (!isBoxDrawMode(drawMode)) return;
              const point = pointerPosition(event);
              const interaction = regionInteractionRef.current;
              if (interaction) {
                setDraftBox(boxForInteraction(interaction, point));
                return;
              }
              if (drawMode === "cue_ocr") {
                setRegionCursor("crosshair");
                return;
              }
              const activeBox = drawMode === "source" ? project.source_roi : project.target_roi;
              const nextCursor = cursorForRegionPoint(
                activeBox,
                point,
                event.currentTarget.getBoundingClientRect(),
              );
              setRegionCursor((current) => (current === nextCursor ? current : nextCursor));
            }}
            onPointerUp={(event) => {
              const interaction = regionInteractionRef.current;
              if (!interaction) return;
              const point = pointerPosition(event);
              const box = boxForInteraction(interaction, point);
              regionInteractionRef.current = null;
              setDraftBox(null);
              setRegionCursor(interaction.mode === "cue_ocr"
                ? "crosshair"
                : cursorForRegionPoint(box, point, event.currentTarget.getBoundingClientRect()));
              if (interaction.mode === "cue_ocr") onFrameOcrRegion(box);
              else onRegionChange(interaction.mode, box);
            }}
            onKeyDown={(event) => {
              if (drawMode !== "eyedropper" || (event.key !== "Enter" && event.key !== " ")) {
                return;
              }
              event.preventDefault();
              const video = videoRef.current;
              if (!video) return;
              const rect = video.getBoundingClientRect();
              pickVideoColor(rect.left + rect.width / 2, rect.top + rect.height / 2);
            }}
            onPointerLeave={() => {
              if (!regionInteractionRef.current) setRegionCursor("crosshair");
            }}
            onPointerCancel={() => {
              regionInteractionRef.current = null;
              setDraftBox(null);
              setRegionCursor("crosshair");
            }}
            onLostPointerCapture={() => {
              if (!regionInteractionRef.current) return;
              regionInteractionRef.current = null;
              setDraftBox(null);
              setRegionCursor("crosshair");
            }}
          />
        </div>
      </div>

      <div className="transport-bar">
        <button
          className="icon-button compact"
          type="button"
          title="上一帧"
          aria-label="上一帧"
          onClick={() => seekByFrames(-1)}
        >
          <StepBack size={16} />
        </button>
        <button
          className="play-button"
          type="button"
          title={playing ? "暂停" : "播放"}
          aria-label={playing ? "暂停" : "播放"}
          onClick={() => onPlayingChange(!playing)}
        >
          {playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />}
        </button>
        <button
          className="icon-button compact"
          type="button"
          title="下一帧"
          aria-label="下一帧"
          onClick={() => seekByFrames(1)}
        >
          <StepForward size={16} />
        </button>
        <span className="transport-time">{formatTime(currentTime, true)}</span>
        <input
          className="seek-slider"
          type="range"
          min="0"
          max={project.duration_ms}
          step="1"
          value={Math.min(currentTime, project.duration_ms)}
          onChange={(event) => onTimeChange(Number(event.target.value))}
          aria-label="视频时间"
        />
        <span className="transport-time muted">{formatTime(project.duration_ms, true)}</span>
        <button className="icon-button compact" type="button" title="音量" aria-label="音量">
          <Volume2 size={16} />
        </button>
      </div>
    </section>
  );
}

function DemoFrame({ activeCues }: { activeCues: Cue[] }) {
  return (
    <div className="demo-frame" aria-label="演示视频画面">
      <div className="demo-sky" />
      <div className="demo-moon" />
      <div className="demo-building building-a" />
      <div className="demo-building building-b" />
      <div className="demo-building building-c" />
      <div className="demo-platform" />
      <div className="demo-figure figure-a" />
      <div className="demo-figure figure-b" />
      <div className="source-subtitle-simulation">
        {activeCues.map((cue) => (
          <span key={cue.cue_id} style={{ color: cue.speaker_color }}>
            {cue.source_text}
          </span>
        ))}
      </div>
      <span className="demo-frame-label">LOCAL PREVIEW · FRAME 01842</span>
    </div>
  );
}
