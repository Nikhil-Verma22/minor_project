/**
 * DiagnosticsPanel Component.
 * Real-time performance telemetry, execution providers, and memory counts.
 */

import React, { useState, useEffect } from 'react';
import { Activity, ChevronDown, ChevronUp, Cpu, HardDrive, Timer } from 'lucide-react';
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
  const [isOpen, setIsOpen] = useState(false);
  const [hasWebGpu, setHasWebGpu] = useState<boolean | null>(null);

  useEffect(() => {
    if ('gpu' in navigator && (navigator as any).gpu) {
      (navigator as any).gpu.requestAdapter().then((adapter: any) => {
        setHasWebGpu(!!adapter);
      }).catch(() => setHasWebGpu(false));
    } else {
      setHasWebGpu(false);
    }
  }, []);

  return (
    <div className="panel-card mt-4 border-zinc-800">
      <button
        type="button"
        className="w-full flex items-center justify-between text-left text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center gap-2 font-medium">
          <Activity size={15} className="text-cyan-400" />
          <span>System Diagnostics & Performance Telemetry</span>
          {sggResult && (
            <span className="text-[11px] text-zinc-400">
              (Last Inference: {sggResult.timings.totalMs} ms &bull; {sggResult.executionProvider.toUpperCase()})
            </span>
          )}
        </div>
        {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {isOpen && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4 pt-3 border-t border-zinc-800/80 text-xs">
          {/* 1. SGG Model Telemetry */}
          <div className="diag-box">
            <div className="flex items-center gap-1.5 text-zinc-300 font-semibold mb-2">
              <Cpu size={14} className="text-cyan-400" /> REACT++ SGG Engine
            </div>
            <div className="space-y-1 text-zinc-400">
              <div>Checkpoint: <span className="text-zinc-200 font-mono">react_pp_yolo12n.onnx</span></div>
              <div>Size: <span className="text-zinc-200 font-mono">{(SGG_VARIANTS.yolo12n.sizeBytes / (1024 * 1024)).toFixed(1)} MB</span></div>
              <div>Provider: <span className="text-emerald-400 font-mono">{sggResult?.executionProvider || 'WASM SIMD'}</span></div>
              <div>Input Tensor: <span className="text-zinc-200 font-mono">[1, 3, 640, 640]</span></div>
              <div>WebGPU Hardware: <span className="text-zinc-200 font-mono">{hasWebGpu ? 'Available' : 'Not Detected'}</span></div>
            </div>
          </div>

          {/* 2. Latency Breakdown */}
          <div className="diag-box">
            <div className="flex items-center gap-1.5 text-zinc-300 font-semibold mb-2">
              <Timer size={14} className="text-amber-400" /> Latency Timers
            </div>
            {sggResult ? (
              <div className="space-y-1 text-zinc-400">
                <div>Letterbox Preprocess: <span className="text-zinc-200 font-mono">{sggResult.timings.preprocessMs} ms</span></div>
                <div>Model Forward Pass: <span className="text-emerald-400 font-mono font-semibold">{sggResult.timings.inferenceMs} ms</span></div>
                <div>Un-pad Postprocess: <span className="text-zinc-200 font-mono">{sggResult.timings.postprocessMs} ms</span></div>
                <div className="pt-1 border-t border-zinc-800">
                  Total Pipeline: <span className="text-cyan-400 font-mono font-bold">{sggResult.timings.totalMs} ms</span>
                </div>
              </div>
            ) : (
              <div className="text-zinc-500 italic">No inference run recorded yet.</div>
            )}
          </div>

          {/* 3. Memory & Storage */}
          <div className="diag-box">
            <div className="flex items-center gap-1.5 text-zinc-300 font-semibold mb-2">
              <HardDrive size={14} className="text-indigo-400" /> Local Memory Storage
            </div>
            <div className="space-y-1 text-zinc-400">
              <div>Database: <span className="text-zinc-200 font-mono">workspace_retrieval_db (v1)</span></div>
              <div>Engine: <span className="text-zinc-200 font-mono">IndexedDB (idb)</span></div>
              <div>Object Instances: <span className="text-indigo-400 font-mono font-semibold">{instanceCount}</span></div>
              <div>Recorded Frames: <span className="text-indigo-400 font-mono font-semibold">{frameCount}</span></div>
              <div>Privacy Policy: <span className="text-emerald-400">Strictly Local (0 external APIs)</span></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
