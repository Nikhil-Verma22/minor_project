/**
 * DiagnosticsPanel Component.
 * Collapsible performance telemetry, model specifications, latency breakdown,
 * local memory counters, and VLM readiness.
 * Collapsed by default to avoid dominating the workspace experience.
 */

import React, { useState, useEffect } from 'react';
import { Activity, ChevronDown, ChevronUp, Cpu, HardDrive, Timer, Sparkles } from 'lucide-react';
import { SGGInferenceResult } from '../models/sgg/SGGEngine';
import { SGG_VARIANTS } from '../config/modelConfig';

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
  // Collapsed by default per design specification
  const [isOpen, setIsOpen] = useState(false);
  const [hasWebGpu, setHasWebGpu] = useState<boolean | null>(null);

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

  return (
    <div className="diagnostics-container">
      {/* Collapsible Header Toggle */}
      <button
        type="button"
        className="diagnostics-toggle-btn"
        onClick={() => setIsOpen(!isOpen)}
        title="Toggle system diagnostics and performance telemetry"
      >
        <div className="flex items-center gap-2">
          <Activity size={15} className="text-cyan-400" />
          <span className="font-semibold text-white">System Diagnostics &amp; Telemetry</span>
          {sggResult ? (
            <span className="text-[11px] text-zinc-400">
              &bull; Last Run: <strong className="text-emerald-400">{sggResult.timings.totalMs} ms</strong> ({sggResult.executionProvider.toUpperCase()})
            </span>
          ) : (
            <span className="text-[11px] text-zinc-500">&bull; Standby</span>
          )}
        </div>
        <div className="flex items-center gap-1.5 text-xs text-zinc-400">
          <span>{isOpen ? 'Hide Details' : 'Show Details'}</span>
          {isOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </div>
      </button>

      {/* Expanded Telemetry Grid (4 Groups) */}
      {isOpen && (
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
                <span className="diag-metric-value">IndexedDB (v1)</span>
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
                <span>Footprint</span>
                <span className="diag-metric-value">&lt; 2.0 GB Memory</span>
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
      )}
    </div>
  );
};
