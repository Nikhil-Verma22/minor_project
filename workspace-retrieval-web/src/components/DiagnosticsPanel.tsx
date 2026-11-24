/**
 * DiagnosticsPanel Component.
 * Collapsible performance telemetry, model specifications, latency breakdown,
 * local memory counters, VLM readiness, and persistent Session Audit Logs.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  ChevronDown,
  ChevronUp,
  Cpu,
  HardDrive,
  Timer,
  Sparkles,
  ScrollText,
  Download,
  Trash2,
  Search,
  Filter,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Info,
  Clock,
  Layers,
  FileText,
  X,
  Copy,
} from 'lucide-react';
import { SGGInferenceResult } from '../models/sgg/SGGEngine';
import { SGG_VARIANTS } from '../config/modelConfig';
import { sessionLogger } from '../logging/sessionLogger';
import { LogRepository } from '../logging/logRepository';
import { SessionLogEvent, SessionSummary, SessionLogCategory, SessionLogLevel } from '../logging/logTypes';
import { exportLogsAsJson, exportLogsAsCsv } from '../logging/logExport';

interface DiagnosticsPanelProps {
  sggResult: SGGInferenceResult | null;
  instanceCount: number;
  frameCount: number;
}

export const DiagnosticsPanel: React.FC<DiagnosticsPanelProps> = ({
  sggResult,
  instanceCount,
  frameCount,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'telemetry' | 'logs'>('telemetry');
  const [hasWebGpu, setHasWebGpu] = useState<boolean | null>(null);

  // Session Logs State
  const [logs, setLogs] = useState<SessionLogEvent[]>([]);
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<SessionLogCategory | 'all'>('all');
  const [selectedLevel, setSelectedLevel] = useState<SessionLogLevel | 'all'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [inspectedEvent, setInspectedEvent] = useState<SessionLogEvent | null>(null);
  const [copiedJson, setCopiedJson] = useState(false);
  const [showConfirmClear, setShowConfirmClear] = useState(false);

  useEffect(() => {
    if ('gpu' in navigator && (navigator as any).gpu) {
      (navigator as any).gpu
        .requestAdapter()
        .then((adapter: any) => {
          setHasWebGpu(!!adapter);
        })
        .catch(() => setHasWebGpu(false));
    } else {
      setHasWebGpu(false);
    }
  }, []);

  const loadLogs = useCallback(async () => {
    const currentSessionId = sessionLogger.getSessionId();
    const fetchedLogs = await LogRepository.getLogs({
      sessionId: currentSessionId,
      category: selectedCategory,
      level: selectedLevel,
      searchTerm: searchTerm.trim() || undefined,
      limit: 200,
    });
    const fetchedSummary = await LogRepository.getSessionSummary(currentSessionId);
    setLogs(fetchedLogs);
    setSummary(fetchedSummary);
  }, [selectedCategory, selectedLevel, searchTerm]);

  useEffect(() => {
    if (isOpen && activeTab === 'logs') {
      loadLogs();
      const unsubscribe = sessionLogger.subscribe(() => {
        loadLogs();
      });
      return unsubscribe;
    }
  }, [isOpen, activeTab, loadLogs]);

  const handleClearLogs = async () => {
    await LogRepository.clearLogs(sessionLogger.getSessionId());
    setShowConfirmClear(false);
    await loadLogs();
  };

  const copyEventJson = () => {
    if (!inspectedEvent) return;
    navigator.clipboard.writeText(JSON.stringify(inspectedEvent, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  return (
    <div className="diagnostics-container">
      {/* Event Details Inspection Modal */}
      {inspectedEvent && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-fadeIn"
          onClick={() => setInspectedEvent(null)}
        >
          <div
            className="bg-zinc-900 border border-zinc-700 rounded-xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800 bg-zinc-950/70">
              <div className="flex items-center gap-2">
                <FileText size={16} className="text-cyan-400" />
                <span className="font-semibold text-sm text-zinc-100">
                  Event Details: {inspectedEvent.eventType}
                </span>
                <span className="font-mono text-[10px] text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded">
                  {inspectedEvent.logId}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className="px-2.5 py-1 text-xs text-zinc-300 hover:text-white rounded bg-zinc-800 hover:bg-zinc-700 flex items-center gap-1.5 transition"
                  onClick={copyEventJson}
                >
                  {copiedJson ? <CheckCircle size={12} className="text-emerald-400" /> : <Copy size={12} />}
                  {copiedJson ? 'Copied' : 'Copy JSON'}
                </button>
                <button
                  type="button"
                  className="p-1 text-zinc-400 hover:text-white transition"
                  onClick={() => setInspectedEvent(null)}
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            <div className="p-5 overflow-y-auto space-y-3 font-mono text-xs">
              <div className="grid grid-cols-2 gap-2 text-[11px] bg-zinc-950/50 p-3 rounded-lg border border-zinc-800/80">
                <div>
                  <span className="text-zinc-500">Timestamp: </span>
                  <span className="text-zinc-300">{inspectedEvent.timestamp}</span>
                </div>
                <div>
                  <span className="text-zinc-500">Category: </span>
                  <span className="text-cyan-400 font-semibold">{inspectedEvent.category}</span>
                </div>
                <div>
                  <span className="text-zinc-500">Level: </span>
                  <span
                    className={`font-semibold ${
                      inspectedEvent.level === 'error'
                        ? 'text-rose-400'
                        : inspectedEvent.level === 'warn'
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {inspectedEvent.level.toUpperCase()}
                  </span>
                </div>
                {inspectedEvent.frameId && (
                  <div>
                    <span className="text-zinc-500">Frame ID: </span>
                    <span className="text-zinc-200">{inspectedEvent.frameId}</span>
                  </div>
                )}
                {inspectedEvent.queryId && (
                  <div>
                    <span className="text-zinc-500">Query ID: </span>
                    <span className="text-zinc-200">{inspectedEvent.queryId}</span>
                  </div>
                )}
                {inspectedEvent.durationMs !== undefined && (
                  <div>
                    <span className="text-zinc-500">Duration: </span>
                    <span className="text-amber-400 font-bold">{inspectedEvent.durationMs} ms</span>
                  </div>
                )}
              </div>

              <div>
                <span className="text-zinc-400 block mb-1 text-[11px]">Message:</span>
                <p className="font-sans text-xs text-zinc-100 bg-zinc-800/40 p-2.5 rounded border border-zinc-800">
                  {inspectedEvent.message}
                </p>
              </div>

              {inspectedEvent.details && Object.keys(inspectedEvent.details).length > 0 && (
                <div>
                  <span className="text-zinc-400 block mb-1 text-[11px]">Structured Payload:</span>
                  <pre className="bg-zinc-950 p-3 rounded-lg border border-zinc-800/80 overflow-x-auto text-cyan-300 text-[11px] leading-relaxed">
                    {JSON.stringify(inspectedEvent.details, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal for Clearing Logs */}
      {showConfirmClear && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={() => setShowConfirmClear(false)}
        >
          <div
            className="bg-zinc-900 border border-zinc-700 p-5 rounded-lg max-w-sm w-full shadow-2xl space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 text-rose-400 font-semibold text-sm">
              <Trash2 size={16} />
              <h4>Clear Session Logs?</h4>
            </div>
            <p className="text-xs text-zinc-300 leading-relaxed">
              This will permanently delete audit logs for the current session.
              Workspace memory, observation frames, and detected objects will remain untouched.
            </p>
            <div className="flex justify-end gap-2 pt-2 border-t border-zinc-800">
              <button
                type="button"
                className="px-3 py-1.5 text-xs text-zinc-300 hover:text-white rounded bg-zinc-800 hover:bg-zinc-700 transition"
                onClick={() => setShowConfirmClear(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="px-3 py-1.5 text-xs text-white bg-rose-600 hover:bg-rose-500 rounded font-medium transition"
                onClick={handleClearLogs}
              >
                Clear Logs
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Collapsible Header Toggle */}
      <div className="diagnostics-toggle-btn flex items-center justify-between w-full">
        <button
          type="button"
          className="flex items-center gap-3 flex-1 text-left"
          onClick={() => setIsOpen(!isOpen)}
          title="Toggle system diagnostics and performance telemetry"
        >
          <Activity size={15} className="text-cyan-400" />
          <span className="font-semibold text-white">System Diagnostics &amp; Telemetry</span>
          {sggResult ? (
            <span className="text-[11px] text-zinc-400">
              &bull; Last Run: <strong className="text-emerald-400">{sggResult.timings.totalMs} ms</strong> ({sggResult.executionProvider.toUpperCase()})
            </span>
          ) : (
            <span className="text-[11px] text-zinc-500">&bull; Standby</span>
          )}
        </button>

        {isOpen && (
          <div className="flex items-center gap-2 mr-3">
            <button
              type="button"
              className={`px-2.5 py-1 text-xs rounded transition flex items-center gap-1.5 ${
                activeTab === 'telemetry'
                  ? 'bg-cyan-950 text-cyan-300 border border-cyan-800'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
              onClick={() => setActiveTab('telemetry')}
            >
              <Cpu size={12} /> Telemetry
            </button>
            <button
              type="button"
              className={`px-2.5 py-1 text-xs rounded transition flex items-center gap-1.5 ${
                activeTab === 'logs'
                  ? 'bg-cyan-950 text-cyan-300 border border-cyan-800'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
              onClick={() => setActiveTab('logs')}
            >
              <ScrollText size={12} /> Session Logs {summary ? `(${summary.totalEvents})` : ''}
            </button>
          </div>
        )}

        <button
          type="button"
          className="p-1 text-zinc-400 hover:text-white"
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </button>
      </div>

      {/* Expanded Content */}
      {isOpen && (
        <div className="diagnostics-body p-4 bg-zinc-950/90 border-t border-zinc-800">
          {activeTab === 'telemetry' ? (
            /* Tab 1: Telemetry Grid */
            <div className="diagnostics-grid">
              {/* Group A — SGG Engine */}
              <div className="diag-card">
                <div className="diag-card-header">
                  <Cpu size={14} className="text-cyan-400" />
                  <span>REACT++ SGG Engine</span>
                </div>
                <div className="space-y-1">
                  <div className="diag-metric-row">
                    <span>Checkpoint</span>
                    <span className="diag-metric-value">react_pp_yolo12n.onnx</span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Model Size</span>
                    <span className="diag-metric-value">
                      {(SGG_VARIANTS.yolo12n.sizeBytes / (1024 * 1024)).toFixed(1)} MB
                    </span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Provider</span>
                    <span className="diag-metric-value text-emerald-400">
                      {sggResult?.executionProvider || 'WASM SIMD'}
                    </span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Input Shape</span>
                    <span className="diag-metric-value">[1, 3, 640, 640]</span>
                  </div>
                  <div className="diag-metric-row">
                    <span>WebGPU Hardware</span>
                    <span className="diag-metric-value">
                      {hasWebGpu ? 'Available' : 'Not Detected'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Group B — Latency Telemetry */}
              <div className="diag-card">
                <div className="diag-card-header">
                  <Timer size={14} className="text-amber-400" />
                  <span>Latency Telemetry</span>
                </div>
                {sggResult ? (
                  <div className="space-y-1">
                    <div className="diag-metric-row">
                      <span>Preprocess</span>
                      <span className="diag-metric-value">{sggResult.timings.preprocessMs} ms</span>
                    </div>
                    <div className="diag-metric-row">
                      <span>Model Forward</span>
                      <span className="diag-metric-value text-emerald-400">
                        {sggResult.timings.inferenceMs} ms
                      </span>
                    </div>
                    <div className="diag-metric-row">
                      <span>Postprocess</span>
                      <span className="diag-metric-value">{sggResult.timings.postprocessMs} ms</span>
                    </div>
                    <div className="diag-metric-row pt-1 border-t border-zinc-800">
                      <span className="font-semibold text-zinc-300">Total Latency</span>
                      <span className="diag-metric-value text-cyan-400 font-bold">
                        {sggResult.timings.totalMs} ms
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="text-zinc-500 italic text-xs py-2">
                    Run scene understanding to record latency metrics.
                  </div>
                )}
              </div>

              {/* Group C — Local Memory Storage */}
              <div className="diag-card">
                <div className="diag-card-header">
                  <HardDrive size={14} className="text-indigo-400" />
                  <span>Workspace Memory</span>
                </div>
                <div className="space-y-1">
                  <div className="diag-metric-row">
                    <span>Database</span>
                    <span className="diag-metric-value">workspace_retrieval_db</span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Engine</span>
                    <span className="diag-metric-value">IndexedDB (v2)</span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Physical Instances</span>
                    <span className="diag-metric-value text-indigo-400 font-semibold">
                      {instanceCount}
                    </span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Stored Frames</span>
                    <span className="diag-metric-value text-indigo-400 font-semibold">
                      {frameCount}
                    </span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Privacy Boundary</span>
                    <span className="diag-metric-value text-emerald-400">Strictly Local</span>
                  </div>
                </div>
              </div>

              {/* Group D — SmolVLM Fallback */}
              <div className="diag-card">
                <div className="diag-card-header">
                  <Sparkles size={14} className="text-amber-400" />
                  <span>SmolVLM Fallback</span>
                </div>
                <div className="space-y-1">
                  <div className="diag-metric-row">
                    <span>Model</span>
                    <span className="diag-metric-value">SmolVLM-500M-Instruct</span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Strategy</span>
                    <span className="diag-metric-value">Lazy Load on Ambiguity</span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Coordinates</span>
                    <span className="diag-metric-value text-emerald-400">Zero-Coordinate Safe</span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Acceleration</span>
                    <span className="diag-metric-value">ExecuTorch / WebGPU</span>
                  </div>
                  <div className="diag-metric-row">
                    <span>Status</span>
                    <span className="diag-metric-value text-amber-400">Standby Ready</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Tab 2: Session Audit Logs */
            <div className="space-y-3">
              {/* Summary Stats Row */}
              {summary && (
                <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-xs">
                  <div className="bg-zinc-900 border border-zinc-800 p-2.5 rounded">
                    <span className="text-[10px] text-zinc-500 block uppercase">Session</span>
                    <span className="font-mono text-cyan-400 text-[11px] truncate block" title={summary.sessionId}>
                      {summary.sessionId}
                    </span>
                  </div>
                  <div className="bg-zinc-900 border border-zinc-800 p-2.5 rounded">
                    <span className="text-[10px] text-zinc-500 block uppercase">Total Events</span>
                    <span className="font-bold text-white text-sm">{summary.totalEvents}</span>
                  </div>
                  <div className="bg-zinc-900 border border-zinc-800 p-2.5 rounded">
                    <span className="text-[10px] text-zinc-500 block uppercase">Uploads</span>
                    <span className="font-bold text-emerald-400 text-sm">{summary.uploadCount}</span>
                  </div>
                  <div className="bg-zinc-900 border border-zinc-800 p-2.5 rounded">
                    <span className="text-[10px] text-zinc-500 block uppercase">Inferences</span>
                    <span className="font-bold text-cyan-400 text-sm">{summary.inferenceCount}</span>
                  </div>
                  <div className="bg-zinc-900 border border-zinc-800 p-2.5 rounded">
                    <span className="text-[10px] text-zinc-500 block uppercase">Queries</span>
                    <span className="font-bold text-indigo-400 text-sm">{summary.queryCount}</span>
                  </div>
                  <div className="bg-zinc-900 border border-zinc-800 p-2.5 rounded">
                    <span className="text-[10px] text-zinc-500 block uppercase">Errors / Warns</span>
                    <span className={`font-bold text-sm ${summary.errorCount > 0 ? 'text-rose-400' : 'text-zinc-400'}`}>
                      {summary.errorCount} / {summary.warnCount}
                    </span>
                  </div>
                </div>
              )}

              {/* Toolbar: Search, Filters & Export Actions */}
              <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-zinc-800/80">
                <div className="flex items-center gap-2 flex-1 min-w-[200px]">
                  <div className="relative flex-1 max-w-xs">
                    <Search size={13} className="absolute left-2.5 top-2.5 text-zinc-500" />
                    <input
                      type="text"
                      placeholder="Search logs by keyword, ID..."
                      className="w-full bg-zinc-900 border border-zinc-700/80 rounded pl-8 pr-2.5 py-1 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-cyan-500"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                    />
                  </div>

                  {/* Level Filter */}
                  <select
                    className="bg-zinc-900 border border-zinc-700/80 rounded px-2 py-1 text-xs text-zinc-300 focus:outline-none focus:border-cyan-500"
                    value={selectedLevel}
                    onChange={(e) => setSelectedLevel(e.target.value as any)}
                  >
                    <option value="all">All Levels</option>
                    <option value="info">Info</option>
                    <option value="warn">Warn</option>
                    <option value="error">Error</option>
                    <option value="debug">Debug</option>
                  </select>

                  {/* Category Filter */}
                  <select
                    className="bg-zinc-900 border border-zinc-700/80 rounded px-2 py-1 text-xs text-zinc-300 focus:outline-none focus:border-cyan-500"
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value as any)}
                  >
                    <option value="all">All Categories</option>
                    <option value="upload">Upload</option>
                    <option value="inference">Inference</option>
                    <option value="frame">Frame</option>
                    <option value="detection">Detection</option>
                    <option value="identity">Identity</option>
                    <option value="scene_graph">Scene Graph</option>
                    <option value="query">Query</option>
                    <option value="vlm">VLM</option>
                    <option value="deletion">Deletion</option>
                    <option value="memory">Memory</option>
                  </select>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className="px-2.5 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded flex items-center gap-1.5 transition"
                    onClick={() => exportLogsAsJson(sessionLogger.getSessionId())}
                    title="Export session logs as structured JSON file"
                  >
                    <Download size={12} /> Export JSON
                  </button>
                  <button
                    type="button"
                    className="px-2.5 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded flex items-center gap-1.5 transition"
                    onClick={() => exportLogsAsCsv(sessionLogger.getSessionId())}
                    title="Export session logs as CSV table"
                  >
                    <FileText size={12} /> CSV
                  </button>
                  <button
                    type="button"
                    className="px-2.5 py-1 text-xs bg-rose-950/60 hover:bg-rose-900 text-rose-300 border border-rose-800/60 rounded flex items-center gap-1.5 transition"
                    onClick={() => setShowConfirmClear(true)}
                    title="Clear current session logs"
                  >
                    <Trash2 size={12} /> Clear
                  </button>
                </div>
              </div>

              {/* Event Table */}
              <div className="overflow-x-auto max-h-72 border border-zinc-800 rounded-lg">
                {logs.length === 0 ? (
                  <div className="text-center py-10 text-xs text-zinc-500 italic">
                    No log events recorded matching the active filters.
                  </div>
                ) : (
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-zinc-900 text-zinc-400 text-[11px] uppercase tracking-wider sticky top-0 border-b border-zinc-800">
                      <tr>
                        <th className="py-2 px-3">Time</th>
                        <th className="py-2 px-3">Level</th>
                        <th className="py-2 px-3">Category</th>
                        <th className="py-2 px-3">Event Type</th>
                        <th className="py-2 px-3">Scope / Frame</th>
                        <th className="py-2 px-3">Message</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-900 bg-zinc-950">
                      {logs.map((log) => {
                        const timeStr = log.timestamp.split('T')[1]?.split('.')[0] || log.timestamp;
                        return (
                          <tr
                            key={log.logId}
                            className="hover:bg-zinc-900/60 cursor-pointer transition"
                            onClick={() => setInspectedEvent(log)}
                            title="Click to inspect structured JSON event details"
                          >
                            <td className="py-1.5 px-3 text-zinc-400 whitespace-nowrap">{timeStr}</td>
                            <td className="py-1.5 px-3 whitespace-nowrap">
                              <span
                                className={`text-[10px] px-1.5 py-0.5 rounded font-semibold uppercase ${
                                  log.level === 'error'
                                    ? 'bg-rose-950 text-rose-400 border border-rose-800'
                                    : log.level === 'warn'
                                    ? 'bg-amber-950 text-amber-400 border border-amber-800'
                                    : log.level === 'debug'
                                    ? 'bg-zinc-800 text-zinc-400'
                                    : 'bg-cyan-950 text-cyan-300 border border-cyan-800/60'
                                }`}
                              >
                                {log.level}
                              </span>
                            </td>
                            <td className="py-1.5 px-3 text-cyan-400/90 whitespace-nowrap capitalize">
                              {log.category}
                            </td>
                            <td className="py-1.5 px-3 text-zinc-200 font-semibold whitespace-nowrap">
                              {log.eventType}
                            </td>
                            <td className="py-1.5 px-3 text-zinc-400 whitespace-nowrap">
                              {log.frameId ||
                                (log.selectedFrameIds ? `${log.selectedFrameIds.length} frames` : '—')}
                            </td>
                            <td className="py-1.5 px-3 text-zinc-300 max-w-md truncate font-sans">
                              {log.message}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
