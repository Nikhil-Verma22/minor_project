/**
 * QueryPanel Component.
 * Natural Language Query console with example chips, step indicators, and answer display.
 */

import React, { useState } from 'react';
import { MessageSquare, Send, Sparkles, Loader2 } from 'lucide-react';
import { NLQResult } from '../nlq/queryTypes';
import { AnswerEvidence } from './AnswerEvidence';
import { PipelineStage } from '../pipeline/pipelineTypes';

interface QueryPanelProps {
  onAskQuery: (query: string) => void;
  queryResult: NLQResult | null;
  pipelineStage: PipelineStage;
  isEnabled: boolean;
}

const EXAMPLE_QUERIES = [
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

  return (
    <div className="panel-card">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <MessageSquare size={18} className="text-cyan-400" />
          <h2 className="panel-title">Natural Language Query (NLQ)</h2>
        </div>
      </div>

      {/* Example Chips */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {EXAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            type="button"
            className="chip-example"
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

      {/* Query Input */}
      <form onSubmit={handleSubmit} className="flex items-center gap-2 mb-3">
        <input
          type="text"
          className="text-input flex-1"
          placeholder={
            isEnabled
              ? 'Ask a question about objects, locations, or relations...'
              : 'Upload an observation image to enable queries'
          }
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          disabled={!isEnabled || isQuerying}
        />
        <button
          type="submit"
          className="btn-primary flex items-center gap-1.5 px-4"
          disabled={!isEnabled || !queryText.trim() || isQuerying}
        >
          {isQuerying ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Reasoning...
            </>
          ) : (
            <>
              <Send size={15} />
              Ask
            </>
          )}
        </button>
      </form>

      {/* Reasoning Steps Sequence Indicator */}
      {isQuerying && (
        <div className="reasoning-steps-bar mb-3">
          <div className={`step-item ${pipelineStage === 'query_parsing' ? 'step-active' : 'step-done'}`}>
            1. Parsing Query
          </div>
          <div
            className={`step-item ${
              pipelineStage === 'graph_memory_reasoning'
                ? 'step-active'
                : pipelineStage === 'vlm_loading' || pipelineStage === 'vlm_reasoning'
                ? 'step-done'
                : 'step-pending'
            }`}
          >
            2. Searching Scene Graph & Memory
          </div>
          {(pipelineStage === 'vlm_loading' || pipelineStage === 'vlm_reasoning') && (
            <div className="step-item step-vlm step-active flex items-center gap-1">
              <Sparkles size={12} /> 3. SmolVLM Fallback
            </div>
          )}
        </div>
      )}

      {/* Answer Output */}
      {queryResult && (
        <div className="answer-container">
          <div className="answer-text mb-3">{queryResult.answer}</div>
          <AnswerEvidence evidence={queryResult.evidence} />
        </div>
      )}
    </div>
  );
};
