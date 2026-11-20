/**
 * AnswerEvidence Component.
 * Displays provenance and evidence supporting an NLQ answer.
 */

import React from 'react';
import { ShieldCheck, Database, Sparkles, Clock, MapPin } from 'lucide-react';
import { AnswerEvidence as EvidenceType } from '../nlq/queryTypes';

interface AnswerEvidenceProps {
  evidence: EvidenceType;
}

export const AnswerEvidence: React.FC<AnswerEvidenceProps> = ({ evidence }) => {
  const getSourceBadge = (source: string) => {
    switch (source) {
      case 'scene_graph':
        return (
          <span className="badge-source badge-source-sgg flex items-center gap-1">
            <ShieldCheck size={13} /> Scene Graph (Ground Truth)
          </span>
        );
      case 'workspace_memory':
        return (
          <span className="badge-source badge-source-memory flex items-center gap-1">
            <Database size={13} /> Workspace Memory (IndexedDB)
          </span>
        );
      case 'vlm':
        return (
          <span className="badge-source badge-source-vlm flex items-center gap-1">
            <Sparkles size={13} /> SmolVLM-500M Fallback (Provisional)
          </span>
        );
      default:
        return null;
    }
  };

  return (
    <div className="evidence-card">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {getSourceBadge(evidence.source)}
          {evidence.provisional && (
            <span className="badge-provisional">Unconfirmed Provisional Fact</span>
          )}
        </div>
        {evidence.confidence !== undefined && (
          <span className="text-xs text-zinc-400 font-mono">
            Confidence: {(evidence.confidence * 100).toFixed(1)}%
          </span>
        )}
      </div>

      <div className="text-xs text-zinc-300 mb-2 leading-relaxed">
        {evidence.explanation}
      </div>

      <div className="flex flex-wrap items-center gap-3 text-[11px] text-zinc-400 pt-2 border-t border-zinc-800/60">
        {evidence.frameIds.length > 0 && (
          <span className="flex items-center gap-1">
            <MapPin size={11} className="text-cyan-400" />
            Frames: <span className="text-zinc-200 font-mono">{evidence.frameIds.join(', ')}</span>
          </span>
        )}

        {evidence.timestamps.length > 0 && (
          <span className="flex items-center gap-1">
            <Clock size={11} className="text-indigo-400" />
            Timestamp: <span className="text-zinc-200 font-mono">{evidence.timestamps[0]}</span>
          </span>
        )}

        {evidence.objectInstanceIds.length > 0 && (
          <span className="flex items-center gap-1">
            Objects: <span className="text-zinc-200 font-mono">{evidence.objectInstanceIds.join(', ')}</span>
          </span>
        )}
      </div>
    </div>
  );
};
