/**
 * QueryPanel Component.
 * Prominent Natural Language Query composer spanning the workspace.
 * Features a comfortable search input, suggested query pills, live execution state,
 * and a rich answer card with source attribution.
 */

import React, { useState } from 'react';
import { MessageSquare, Send, Sparkles, Loader2, Search, ArrowRight } from 'lucide-react';
import { NLQResult } from '../nlq/queryTypes';
import { AnswerEvidence } from './AnswerEvidence';
import { PipelineStage } from '../pipeline/pipelineTypes';

interface QueryPanelProps {
  onAskQuery: (query: string) => void;
  queryResult: NLQResult | null;
  pipelineStage: PipelineStage;
  isEnabled: boolean;
}

const SUGGESTED_QUERIES = [
  'Where is the laptop?',
  'What is beside the laptop?',
  'Where was my phone last seen?',
  'What was next to the notebook in the previous observation?',
  'Did the screwdriver move?',
  'Which object is on the workspace surface?',
];

export const QueryPanel: React.FC<QueryPanelProps> = ({
  onAskQuery,
  queryResult,
  pipelineStage,
  isEnabled,
}) => {
  const [queryText, setQueryText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (queryText.trim() && isEnabled) {
      onAskQuery(queryText.trim());
    }
  };

  const isQuerying = [
    'query_parsing',
    'graph_memory_reasoning',
    'vlm_loading',
    'vlm_reasoning',
  ].includes(pipelineStage);

  const getSourceBadge = (source: string) => {
    switch (source) {
      case 'scene_graph':
        return (
          <span className="badge-source badge-source-sgg">
            Scene Graph (Ground Truth)
          </span>
        );
      case 'workspace_memory':
        return (
          <span className="badge-source badge-source-memory">
            Workspace Memory Fact
          </span>
        );
      case 'vlm_visual_reasoning':
        return (
          <span className="badge-source badge-source-vlm">
            SmolVLM Visual Fallback
          </span>
        );
      default:
        return <span className="badge-source">{source}</span>;
    }
  };

  return (
    <div className="nlq-panel">
      {/* Header */}
      <div className="nlq-header">
        <h3 className="nlq-title">
          <MessageSquare size={18} className="text-cyan-400" />
          Ask About Your Workspace
        </h3>
        <span className="text-xs text-zinc-400">
          Deterministic 2D Scene Graph &bull; Temporal Memory &bull; SmolVLM Fallback
        </span>
      </div>

      {/* Modern Wide Search Composer */}
      <form onSubmit={handleSubmit} className="nlq-composer-form">
        <div className="nlq-input-wrapper">
          <Search size={16} className="nlq-input-icon" />
          <input
            type="text"
            className="nlq-input"
            placeholder={
              isEnabled
                ? 'Ask a question about object locations, spatial relationships, or movement...'
                : 'Upload an observation image to enable queries'
            }
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            disabled={!isEnabled || isQuerying}
          />
        </div>

        <button
          type="submit"
          className="nlq-ask-btn"
          disabled={!isEnabled || !queryText.trim() || isQuerying}
        >
          {isQuerying ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              <span>Reasoning...</span>
            </>
          ) : (
            <>
              <Send size={15} />
              <span>Ask</span>
            </>
          )}
        </button>
      </form>

      {/* Suggested Query Chips */}
      <div className="flex items-center gap-2 flex-wrap text-xs text-zinc-400 mt-2">
        <span className="text-zinc-500 font-medium">Suggested:</span>
        <div className="nlq-suggestions">
          {SUGGESTED_QUERIES.map((q) => (
            <button
              key={q}
              type="button"
              className="nlq-suggestion-chip"
              onClick={() => {
                setQueryText(q);
                if (isEnabled) onAskQuery(q);
              }}
              disabled={!isEnabled || isQuerying}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Query Result Presentation */}
      {queryResult && (
        <div className="nlq-answer-card">
          <div className="flex items-center justify-between gap-4 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
              Answer
            </span>
            {getSourceBadge(queryResult.evidence.source)}
          </div>

          <div className="nlq-answer-text">
            {queryResult.answer}
          </div>

          {/* Structured Evidence Readout */}
          <AnswerEvidence evidence={queryResult.evidence} />

          {/* VLM Fallback Rationale (if applicable) */}
          {queryResult.requiresVLM && queryResult.vlmReason && (
            <div className="mt-3 p-2.5 rounded bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 flex items-center gap-2">
              <Sparkles size={14} className="text-amber-400 flex-shrink-0" />
              <span>{queryResult.vlmReason}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
