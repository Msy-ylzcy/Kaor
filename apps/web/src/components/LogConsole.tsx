import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ClipboardList,
  Copy,
  Download,
  ExternalLink,
  FileText,
  Filter,
  Info,
  RefreshCw,
  Search,
  TerminalSquare,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type { DiagnosticLogEntry, DiagnosticLogPayload, DiagnosticRepairGuide } from "../types";

interface LogConsoleProps {
  isLive: boolean;
}

const levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] as const;

const levelIcon = (level: string) => {
  if (level === "ERROR" || level === "CRITICAL") return <XCircle size={14} />;
  if (level === "WARNING") return <AlertTriangle size={14} />;
  if (level === "DEBUG") return <TerminalSquare size={14} />;
  return <Info size={14} />;
};

const formatTimestamp = (value: string) => {
  const date = new Date(value.replace(" ", "T").replace(/,(\d{3})(?=Z|[+-]\d{2}:?\d{2}$)/, ".$1"));
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
};

export function LogConsole({ isLive }: LogConsoleProps) {
  const [payload, setPayload] = useState<DiagnosticLogPayload>({ sources: [], entries: [], guides: [] });
  const [source, setSource] = useState("all");
  const [query, setQuery] = useState("");
  const [activeLevels, setActiveLevels] = useState<string[]>(["CRITICAL", "ERROR", "WARNING", "INFO"]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedGuide, setSelectedGuide] = useState<string | null>(null);
  const [expandedEntry, setExpandedEntry] = useState<string | null>(null);
  const [copiedEntry, setCopiedEntry] = useState<string | null>(null);
  const requestSequence = useRef(0);

  const load = useCallback(async (quiet = false) => {
    if (!isLive) return;
    const sequence = ++requestSequence.current;
    if (!quiet) setLoading(true);
    try {
      const next = await api.getDiagnosticLogs({
        source,
        tail: 1200,
        query: query.trim(),
        levels: activeLevels,
      });
      if (sequence === requestSequence.current) {
        setPayload(next);
        setError("");
      }
    } catch (reason) {
      if (sequence === requestSequence.current) {
        setError(reason instanceof Error ? reason.message : "日志读取失败");
      }
    } finally {
      if (!quiet) setLoading(false);
    }
  }, [activeLevels, isLive, query, source]);

  useEffect(() => {
    if (!isLive) return;
    let disposed = false;
    let timer: number | undefined;
    let first = true;
    const poll = async () => {
      await load(!first);
      first = false;
      if (!disposed) timer = window.setTimeout(() => void poll(), 1500);
    };
    timer = window.setTimeout(() => void poll(), query.trim() ? 300 : 0);
    return () => {
      disposed = true;
      requestSequence.current += 1;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [isLive, load, query]);

  const visibleGuides = useMemo(() => {
    const ids = new Set(payload.entries.flatMap((entry) => entry.guide_ids));
    return payload.guides.filter((guide) => ids.has(guide.id));
  }, [payload.entries, payload.guides]);

  const toggleLevel = (level: string) => {
    setActiveLevels((current) => {
      if (current.includes(level) && current.length === 1) return current;
      return current.includes(level)
        ? current.filter((item) => item !== level)
        : [...current, level];
    });
  };

  const selectEntry = (entry: DiagnosticLogEntry) => {
    setExpandedEntry((current) => current === entry.id ? null : entry.id);
    const first = entry.guide_ids[0];
    if (first) setSelectedGuide(first);
  };

  const copyEntry = async (entry: DiagnosticLogEntry) => {
    await navigator.clipboard.writeText(
      `${entry.timestamp} | ${entry.level} | ${entry.logger} | ${entry.message}`,
    );
    setCopiedEntry(entry.id);
    window.setTimeout(() => setCopiedEntry((current) => current === entry.id ? null : current), 1200);
  };

  return (
    <main className="log-console" aria-label="运行日志">
      <header className="log-console-header">
        <div className="log-console-title">
          <span className="log-console-mark"><TerminalSquare size={18} /></span>
          <div>
            <strong>运行输出</strong>
            <span>{isLive ? "本地实时日志" : "演示日志"}</span>
          </div>
          <span className={`log-live-dot ${isLive ? "is-live" : ""}`} title={isLive ? "实时刷新中" : "服务未连接"} />
        </div>
        <div className="log-console-actions">
          <button className="button secondary" type="button" onClick={() => void load()} disabled={!isLive || loading}>
            <RefreshCw className={loading ? "spin" : ""} size={15} />刷新
          </button>
          <a className="button secondary" href={isLive ? api.diagnosticExportUrl() : undefined} aria-disabled={!isLive}>
            <Download size={15} />导出诊断包
          </a>
        </div>
      </header>

      <section className="log-toolbar">
        <label className="log-search">
          <Search size={15} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void load(); }} placeholder="搜索日志内容" />
        </label>
        <label className="log-source-select">
          <FileText size={14} />
          <select value={source} onChange={(event) => setSource(event.target.value)}>
            <option value="all">全部日志源</option>
            {payload.sources.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}
          </select>
          <ChevronDown size={14} />
        </label>
        <div className="log-level-filter" role="group" aria-label="日志级别">
          <Filter size={14} />
          {levels.map((level) => (
            <button type="button" key={level} aria-pressed={activeLevels.includes(level)} className={activeLevels.includes(level) ? `is-active level-${level.toLowerCase()}` : ""} onClick={() => toggleLevel(level)}>{level}</button>
          ))}
        </div>
        <span className="log-count">{payload.entries.length} 条</span>
      </section>

      {error && <div className="log-inline-error"><AlertTriangle size={15} />{error}</div>}
      {!isLive && <div className="log-inline-note"><Info size={15} />连接本地服务后显示真实运行输出</div>}

      <div className="log-console-grid">
        <section className="log-list">
          {payload.entries.length ? payload.entries.map((entry) => (
            <div className="log-entry-row" key={entry.id}>
              <button type="button" className={`log-entry level-${entry.level.toLowerCase()}`} aria-expanded={expandedEntry === entry.id} onClick={() => selectEntry(entry)}>
                <span className="log-entry-level">{levelIcon(entry.level)}<b>{entry.level}</b></span>
                <time>{formatTimestamp(entry.timestamp)}</time>
                <span className="log-entry-source">{entry.source_name}</span>
                <span className="log-entry-message" title={entry.message}>{entry.message}</span>
                {entry.guide_ids.length > 0 && <span className="log-entry-guide"><ClipboardList size={13} />{entry.guide_ids.length}</span>}
              </button>
              {expandedEntry === entry.id && (
                <div className="log-entry-detail">
                  <div><code>{entry.logger}</code><button type="button" onClick={() => void copyEntry(entry)} title="复制完整日志" aria-label="复制完整日志"><Copy size={13} />{copiedEntry === entry.id ? "已复制" : "复制"}</button></div>
                  <pre>{entry.message}</pre>
                </div>
              )}
            </div>
          )) : (
            <div className="log-empty"><CheckCircle2 size={20} /><strong>暂无匹配日志</strong><span>任务运行后，输出会自动出现在这里</span></div>
          )}
        </section>

        <aside className="repair-panel">
          <header><ClipboardList size={16} /><strong>维修指南</strong><span>{visibleGuides.length || payload.guides.length}</span></header>
          <div className="repair-list">
            {(visibleGuides.length ? visibleGuides : payload.guides).map((guide) => (
              <RepairGuideCard key={guide.id} guide={guide} active={selectedGuide === guide.id} onSelect={() => setSelectedGuide(guide.id)} />
            ))}
          </div>
        </aside>
      </div>
    </main>
  );
}

function RepairGuideCard({ guide, active, onSelect }: { guide: DiagnosticRepairGuide; active: boolean; onSelect: () => void }) {
  return (
    <article className={`repair-card ${active ? "is-active" : ""}`}>
      <button type="button" className="repair-card-header" aria-expanded={active} onClick={onSelect}>
        <span><AlertTriangle size={14} /><strong>{guide.title}</strong></span>
        <ChevronDown size={14} className={active ? "repair-chevron is-open" : "repair-chevron"} />
      </button>
      {active && (
        <div className="repair-card-body">
          <p>{guide.summary}</p>
          <ol>{guide.steps.map((step) => <li key={step}>{step}</li>)}</ol>
          <a className="repair-doc-link" href={api.troubleshootingGuideUrl(guide.anchor)} target="_blank" rel="noreferrer">
            <ExternalLink size={12} />打开完整排障文档
          </a>
        </div>
      )}
    </article>
  );
}
