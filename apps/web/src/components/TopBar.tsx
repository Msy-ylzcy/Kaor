import {
  Captions,
  Download,
  HardDrive,
  Moon,
  TerminalSquare,
  Sun,
  Upload,
} from "lucide-react";
import { useRef } from "react";
import type { Job, Project } from "../types";

interface TopBarProps {
  project: Project;
  isDemo: boolean;
  theme: "dark" | "light";
  activeJob?: Job;
  onThemeToggle: () => void;
  onImport: (file: File) => void;
  onExport: () => void;
  pageView: "workbench" | "logs";
  onLogsToggle: () => void;
}

export function TopBar({
  project,
  isDemo,
  theme,
  activeJob,
  onThemeToggle,
  onImport,
  onExport,
  pageView,
  onLogsToggle,
}: TopBarProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const jobProgress = activeJob ? Math.round(activeJob.progress * 100) : 0;

  return (
    <header className="topbar">
      <div className="brand-lockup">
        <span className="brand-mark" aria-hidden="true">
          <Captions size={20} strokeWidth={2.2} />
        </span>
        <div className="brand-copy">
          <strong>Kaor</strong>
          <span>Subtitle Workbench</span>
        </div>
      </div>

      <div className="project-summary">
        <div className="project-title-row">
          <strong title={project.title}>{project.title}</strong>
          <span className={`runtime-badge ${isDemo ? "is-demo" : ""}`}>
            <HardDrive size={12} />
            {isDemo ? "演示数据" : "本地项目"}
          </span>
        </div>
        <span className="project-meta" title={project.video_name}>
          {project.video_name} · {project.width}×{project.height} · {project.fps.toFixed(2)} fps
        </span>
      </div>

      <div className="topbar-spacer" />

      {activeJob && (
        <div className="top-job" title={activeJob.message}>
          <div className="top-job-copy">
            <span>{activeJob.stage.replaceAll("_", " ")}</span>
            <strong>{jobProgress}%</strong>
          </div>
          <div className="micro-progress" aria-label={`任务进度 ${jobProgress}%`}>
            <span style={{ width: `${jobProgress}%` }} />
          </div>
        </div>
      )}

      <button
        className={`icon-button ${pageView === "logs" ? "is-active" : ""}`}
        type="button"
        onClick={onLogsToggle}
        title={pageView === "logs" ? "返回工作区" : "查看运行日志"}
        aria-label={pageView === "logs" ? "返回工作区" : "查看运行日志"}
      >
        <TerminalSquare size={18} />
      </button>

      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="video/*,.mkv,.mp4,.mov,.webm,.avi"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onImport(file);
          event.currentTarget.value = "";
        }}
      />
      <button
        className="icon-button"
        type="button"
        onClick={onThemeToggle}
        title={theme === "dark" ? "切换浅色主题" : "切换深色主题"}
        aria-label={theme === "dark" ? "切换浅色主题" : "切换深色主题"}
      >
        {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
      </button>
      <button className="button secondary" type="button" onClick={() => inputRef.current?.click()}>
        <Upload size={16} />
        <span className="button-label">导入视频</span>
      </button>
      <button className="button primary" type="button" onClick={onExport}>
        <Download size={16} />
        <span className="button-label">导出</span>
      </button>
    </header>
  );
}
