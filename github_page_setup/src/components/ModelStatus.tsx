/**
 * ModelStatus Component.
 * Compact, unified status chips for REACT++, IndexedDB Memory, and SmolVLM-500M.
 */

import React from 'react';
import { Cpu, Database, Sparkles, CheckCircle2, Clock } from 'lucide-react';
import { ModelStatus as SggStatus } from '../models/sgg/SGGEngine';
import { VLMStatus } from '../models/vlm/VLMEngine';
import { VLM_CONFIG } from '../config/modelConfig';

interface ModelStatusProps {
  sggStatus: SggStatus;
  vlmStatus: VLMStatus;
  instanceCount: number;
  frameCount: number;
  stageMessage?: string;
}

export const ModelStatus: React.FC<ModelStatusProps> = ({
  sggStatus,
  vlmStatus,
  instanceCount,
  frameCount,
  stageMessage,
}) => {
  const getProviderText = (provider: string) => {
    switch (provider) {
      case 'webgpu':
        return 'WebGPU';
      case 'wasm':
        return 'WASM SIMD';
      case 'local_service':
        return 'Local Svc';
      default:
        return 'Loading';
    }
  };

  return (
    <div className="header-status-group">
      {/* SGG Status */}
      <div className="compact-status-chip" title="REACT++ YOLO12n Scene Graph Engine">
        <Cpu size={14} className="text-cyan-400" />
        <div className="flex flex-col">
          <span className="chip-label">REACT++</span>
          <span className="chip-value">
            {sggStatus.loaded ? (
              <span className="text-emerald-400 flex items-center gap-1">
                <CheckCircle2 size={11} /> {getProviderText(sggStatus.provider)}
              </span>
            ) : sggStatus.provider === 'unavailable' && sggStatus.error ? (
              <span className="text-rose-400" title={sggStatus.error}>Failed</span>
            ) : (
              <span className="text-cyan-300" title={stageMessage || 'Initializing REACT++...'}>
                {stageMessage && stageMessage.includes('%')
                  ? stageMessage.match(/\d+%/)?.[0] || 'Loading...'
                  : 'Init...'}
              </span>
            )}
          </span>
        </div>
      </div>

      {/* Memory Status */}
      <div className="compact-status-chip" title="Persistent Workspace Memory (IndexedDB)">
        <Database size={14} className="text-indigo-400" />
        <div className="flex flex-col">
          <span className="chip-label">Memory</span>
          <span className="chip-value">
            <span className="text-indigo-300">
              {instanceCount} obj{instanceCount === 1 ? '' : 's'} &bull; {frameCount} frame{frameCount === 1 ? '' : 's'}
            </span>
          </span>
        </div>
      </div>

      {/* VLM Fallback Status */}
      <div
        className="compact-status-chip"
        title={VLM_CONFIG.enabled ? 'SmolVLM-500M Visual Fallback Engine' : VLM_CONFIG.unavailableReason}
      >
        <Sparkles size={14} className={VLM_CONFIG.enabled ? 'text-amber-400' : 'text-zinc-500'} />
        <div className="flex flex-col">
          <span className="chip-label">SmolVLM</span>
          <span className="chip-value">
            {!VLM_CONFIG.enabled ? (
              <span className="text-zinc-400">Unavailable</span>
            ) : vlmStatus.loaded ? (
              <span className="text-cyan-400">WebGPU</span>
            ) : (
              <span className="text-amber-400/90 flex items-center gap-1">
                <Clock size={10} /> Standby
              </span>
            )}
          </span>
        </div>
      </div>
    </div>
  );
};
