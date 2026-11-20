/**
 * ModelStatus Component.
 * Displays live status chips for REACT++, IndexedDB Memory, and SmolVLM-500M.
 */

import React from 'react';
import { Cpu, Database, Sparkles, CheckCircle2, Clock } from 'lucide-react';
import { ModelStatus as SggStatus } from '../models/sgg/SGGEngine';
import { VLMStatus } from '../models/vlm/VLMEngine';

interface ModelStatusProps {
  sggStatus: SggStatus;
  vlmStatus: VLMStatus;
  instanceCount: number;
  frameCount: number;
}

export const ModelStatus: React.FC<ModelStatusProps> = ({
  sggStatus,
  vlmStatus,
  instanceCount,
  frameCount,
}) => {
  const getProviderBadge = (provider: string) => {
    switch (provider) {
      case 'webgpu':
        return <span className="status-badge status-webgpu">WebGPU Primary</span>;
      case 'wasm':
        return <span className="status-badge status-wasm">WASM SIMD CPU</span>;
      case 'local_service':
        return <span className="status-badge status-local">Local Python Service</span>;
      default:
        return <span className="status-badge status-unavailable">Unavailable</span>;
    }
  };

  return (
    <div className="status-bar-container">
      <div className="status-chip">
        <Cpu className="status-icon text-cyan-400" size={18} />
        <div>
          <div className="status-title">REACT++ SGG</div>
          <div className="status-subtitle flex items-center gap-1.5">
            {sggStatus.loaded ? (
              <>
                <CheckCircle2 size={13} className="text-emerald-400" />
                {getProviderBadge(sggStatus.provider)}
              </>
            ) : (
              <span className="text-zinc-400">Loading checkpoint...</span>
            )}
          </div>
        </div>
      </div>

      <div className="status-chip">
        <Database className="status-icon text-indigo-400" size={18} />
        <div>
          <div className="status-title">Workspace Memory</div>
          <div className="status-subtitle">
            <span className="text-emerald-400 font-semibold">{instanceCount}</span> objects,{' '}
            <span className="text-emerald-400 font-semibold">{frameCount}</span> frames (IndexedDB)
          </div>
        </div>
      </div>

      <div className="status-chip">
        <Sparkles className="status-icon text-amber-400" size={18} />
        <div>
          <div className="status-title">SmolVLM-500M Fallback</div>
          <div className="status-subtitle">
            {vlmStatus.loaded ? (
              <span className="status-badge status-webgpu">Active (WebGPU)</span>
            ) : (
              <span className="status-badge status-standby flex items-center gap-1">
                <Clock size={11} /> Lazy Standby (&lt;2GB)
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
