/**
 * WorkspaceAssistant Component.
 * Intelligent, evidence-based workspace conversational agent.
 *
 * Implements:
 * - Natural-language query parsing with pronoun resolution.
 * - Categorized suggestions (Find Objects, Spatial Relations, History, Object Changes).
 * - Multi-turn conversation thread with clearable state (preserving IndexedDB memory).
 * - Exact source badges (Scene Graph, Workspace Memory, Visual Reasoning).
 * - Human-calibrated certainty levels.
 * - Collapsible "How did I answer?" verifiable reasoning trace.
 * - Interactive visual linking to image viewer and historical observations.
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Bot,
  Send,
  Sparkles,
  Loader2,
  Search,
  ArrowRight,
  Database,
  ShieldCheck,
  Eye,
  ChevronDown,
  ChevronUp,
  Trash2,
  Clock,
  Layers,
  HelpCircle,
  Compass,
  History,
  Activity,
  CheckCircle2,
  Network,
} from 'lucide-react';
import {
  NLQResult,
  QueryScope,
  ConversationMessage,
  ConversationContext,
  CertaintyLevel,
} from '../nlq/queryTypes';
import { PipelineStage } from '../pipeline/pipelineTypes';

interface WorkspaceAssistantProps {
  onAskQuery: (
    query: string,
    context?: ConversationContext,
    scope?: QueryScope,
    forceVLM?: boolean,
    forceSGG?: boolean
  ) => void;
  queryResult: NLQResult | null;
  pipelineStage: PipelineStage;
  isEnabled: boolean;
  onSelectEntity?: (id: string | null) => void;
  onSelectHistoricalFrame?: (frameId: string) => void;
  availableFramesCount?: number;
}

interface SuggestionCategory {
  title: string;
  icon: React.ReactNode;
  questions: string[];
}

const SUGGESTION_CATEGORIES: SuggestionCategory[] = [
  {
    title: 'Find Objects',
    icon: <Search size={13} className="text-cyan-400" />,
    questions: [
      'Where is the laptop?',
      'Where is the blue pen?',
      'Which objects are on the workspace surface?',
    ],
  },
  {
    title: 'Spatial Relationships',
    icon: <Compass size={13} className="text-emerald-400" />,
    questions: [
      'What is beside the laptop?',
      'What is next to the notebook?',
      'Where is the person sitting?',
    ],
  },
  {
    title: 'Workspace History',
    icon: <History size={13} className="text-indigo-400" />,
    questions: [
      'Where was my phone last seen?',
      'What was next to the notebook previously?',
      'Which objects were visible in the previous observation?',
    ],
  },
  {
    title: 'Scene Graph Only (/sgg)',
    icon: <Network size={13} className="text-cyan-400" />,
    questions: [
      '/sgg where is the laptop',
      '/sgg what is beside the laptop',
      '/sgg which objects are on the workspace surface',
    ],
  },
  {
    title: 'Direct Visual AI (VLM)',
    icon: <Sparkles size={13} className="text-amber-400" />,
    questions: [
      '/visual cloth color on the table',
      '/visual what is the person wearing',
      '/visual describe items on the desk',
    ],
  },
];

export const WorkspaceAssistant: React.FC<WorkspaceAssistantProps> = ({
  onAskQuery,
  queryResult,
  pipelineStage,
  isEnabled,
  onSelectEntity,
  onSelectHistoricalFrame,
  availableFramesCount = 0,
}) => {
  const [queryText, setQueryText] = useState('');
  const [scope, setScope] = useState<QueryScope>('current');
  const [showAllSuggestions, setShowAllSuggestions] = useState(false);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [conversationContext, setConversationContext] = useState<ConversationContext>({});
  const [expandedTraceIds, setExpandedTraceIds] = useState<Set<string>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // When a new queryResult arrives, append to thread and update conversation context
  useEffect(() => {
    if (!queryResult) return;

    const newAssistantMsg: ConversationMessage = {
      id: `assistant_${Date.now()}`,
      sender: 'assistant',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      text: queryResult.answer,
      scope: queryResult.queryPlan.scope,
      result: queryResult,
    };

    setMessages((prev) => {
      // Avoid duplicate consecutive identical answers
      const last = prev[prev.length - 1];
      if (last && last.sender === 'assistant' && last.text === queryResult.answer) {
        return prev;
      }
      return [...prev, newAssistantMsg];
    });

    // Update conversation context for pronoun & follow-up resolution
    const targetObj = queryResult.queryPlan.targetObject;
    const refObj = queryResult.queryPlan.referenceObject;
    const firstEntityId = queryResult.evidence.objectInstanceIds[0];

    setConversationContext({
      lastTargetEntity: targetObj
        ? {
            instanceId: targetObj.instanceId || firstEntityId,
            category: targetObj.category,
            displayName: targetObj.mention,
            mention: targetObj.mention,
          }
        : firstEntityId
        ? {
            instanceId: firstEntityId,
            displayName: firstEntityId.split('_')[0],
            mention: firstEntityId.split('_')[0],
          }
        : undefined,
      lastReferenceEntity: refObj
        ? {
            instanceId: refObj.instanceId,
            category: refObj.category,
            displayName: refObj.mention,
            mention: refObj.mention,
          }
        : undefined,
      lastFrameId: queryResult.evidence.frameIds[0],
      lastAnswer: queryResult.answer,
      lastSource: queryResult.evidence.source,
      lastObjectIds: queryResult.evidence.objectInstanceIds,
      scope,
    });
  }, [queryResult]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (
    e?: React.FormEvent,
    customQuery?: string,
    forceVLM?: boolean,
    forceSGG?: boolean
  ) => {
    if (e) e.preventDefault();
    const finalQuery = (customQuery || queryText).trim();
    if (!finalQuery || !isEnabled) return;

    const isDirectSgg = forceSGG || scope === 'sgg' || /^\s*\/(sgg|graph)\b/i.test(finalQuery);
    const isDirectVlm =
      !isDirectSgg &&
      (forceVLM || scope === 'vlm' || /^\s*\/(visual|vlm|inspect|image)\b/i.test(finalQuery));

    // Add user message to conversation thread
    const userMsg: ConversationMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      text: finalQuery,
      scope: isDirectSgg ? 'sgg' : (isDirectVlm ? 'vlm' : scope),
    };

    setMessages((prev) => [...prev, userMsg]);
    setQueryText('');

    // Trigger pipeline query execution with active conversation context
    onAskQuery(
      finalQuery,
      conversationContext,
      isDirectSgg ? 'sgg' : (isDirectVlm ? 'vlm' : scope),
      isDirectVlm,
      isDirectSgg
    );
  };

  const handleVerifyWithVLM = (msg: ConversationMessage) => {
    let originalQuery = msg.result?.queryPlan.rawQuery;
    if (!originalQuery) {
      const idx = messages.findIndex((m) => m.id === msg.id);
      if (idx > 0 && messages[idx - 1].sender === 'user') {
        originalQuery = messages[idx - 1].text;
      }
    }
    const clean = originalQuery
      ? originalQuery.replace(/^\s*\/(sgg|graph|visual|vlm|inspect|image)\s+/i, '').trim()
      : msg.text;
    handleSubmit(undefined, `/visual ${clean}`, true, false);
  };

  const handleCheckWithSGG = (msg: ConversationMessage) => {
    let originalQuery = msg.result?.queryPlan.rawQuery;
    if (!originalQuery) {
      const idx = messages.findIndex((m) => m.id === msg.id);
      if (idx > 0 && messages[idx - 1].sender === 'user') {
        originalQuery = messages[idx - 1].text;
      }
    }
    const clean = originalQuery
      ? originalQuery.replace(/^\s*\/(sgg|graph|visual|vlm|inspect|image)\s+/i, '').trim()
      : msg.text;
    handleSubmit(undefined, `/sgg ${clean}`, false, true);
  };

  const handleClearConversation = () => {
    if (messages.length === 0) return;
    if (window.confirm('Clear current assistant conversation thread?\n(Note: Your persistent IndexedDB workspace memory and observations will NOT be deleted.)')) {
      setMessages([]);
      setConversationContext({});
      setExpandedTraceIds(new Set());
    }
  };

  const toggleTrace = (msgId: string) => {
    setExpandedTraceIds((prev) => {
      const next = new Set(prev);
      if (next.has(msgId)) next.delete(msgId);
      else next.add(msgId);
      return next;
    });
  };

  const isQuerying = [
    'query_parsing',
    'graph_memory_reasoning',
    'vlm_loading',
    'vlm_reasoning',
  ].includes(pipelineStage);

  // Status message for query progress
  const getProgressStateLabel = () => {
    switch (pipelineStage) {
      case 'query_parsing':
        return 'Understanding query...';
      case 'graph_memory_reasoning':
        return 'Searching scene graph & memory...';
      case 'vlm_loading':
        return 'Visual reasoning required...';
      case 'vlm_reasoning':
        return 'Inspecting image with SmolVLM...';
      default:
        return 'Processing answer...';
    }
  };

  // Source attribution badge rendering
  const renderSourceBadge = (source: 'scene_graph' | 'workspace_memory' | 'vlm') => {
    switch (source) {
      case 'scene_graph':
        return (
          <span className="assistant-badge assistant-badge-sg flex items-center gap-1.5" title="Derived directly from REACT++ 2D Scene Graph">
            <ShieldCheck size={12} className="text-emerald-400" />
            <span>Scene Graph</span>
          </span>
        );
      case 'workspace_memory':
        return (
          <span className="assistant-badge assistant-badge-mem flex items-center gap-1.5" title="Retrieved from IndexedDB temporal observation memory">
            <Database size={12} className="text-indigo-400" />
            <span>Workspace Memory</span>
          </span>
        );
      case 'vlm':
        return (
          <span className="assistant-badge assistant-badge-vlm flex items-center gap-1.5" title="Evaluated via local SmolVLM-500M visual language model">
            <Sparkles size={12} className="text-amber-400" />
            <span>Visual Reasoning</span>
          </span>
        );
    }
  };

  // Certainty label rendering
  const renderCertaintyLabel = (certainty: CertaintyLevel = 'confirmed') => {
    switch (certainty) {
      case 'confirmed':
        return (
          <span className="text-[11px] font-medium text-emerald-400 flex items-center gap-1">
            <CheckCircle2 size={11} /> Confirmed stored information
          </span>
        );
      case 'high_visual':
        return (
          <span className="text-[11px] font-medium text-cyan-400 flex items-center gap-1">
            <Eye size={11} /> High-confidence visual interpretation
          </span>
        );
      case 'provisional':
        return (
          <span className="text-[11px] font-medium text-amber-400 flex items-center gap-1">
            <Sparkles size={11} /> Provisional inference
          </span>
        );
      case 'insufficient':
        return (
          <span className="text-[11px] font-medium text-zinc-400 flex items-center gap-1">
            <HelpCircle size={11} /> Insufficient evidence
          </span>
        );
    }
  };

  return (
    <div className="workspace-assistant-card">
      {/* 1. Header with Title, Scope Selector, and Clear Action */}
      <div className="assistant-top-bar">
        <div className="assistant-title-group">
          <div className="flex items-center gap-2">
            <div className="assistant-avatar">
              <Bot size={18} className="text-cyan-400" />
            </div>
            <div>
              <h3 className="assistant-title">Workspace Assistant</h3>
              <p className="assistant-subtitle">
                Ask about objects, locations, relationships and workspace history.
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Query Scope Selector */}
          <div className="scope-selector-group" title="Select retrieval scope">
            <span className="scope-label">Scope:</span>
            <div className="scope-segmented">
              <button
                type="button"
                className={`scope-chip ${scope === 'current' ? 'active' : ''}`}
                onClick={() => setScope('current')}
              >
                Current Workspace
              </button>
              <button
                type="button"
                className={`scope-chip ${scope === 'history' ? 'active' : ''}`}
                onClick={() => setScope('history')}
                title={availableFramesCount < 2 ? 'Only 1 observation currently recorded' : undefined}
              >
                Observation History
              </button>
              <button
                type="button"
                className={`scope-chip ${scope === 'all' ? 'active' : ''}`}
                onClick={() => setScope('all')}
              >
                All Available Memory
              </button>
              <button
                type="button"
                className={`scope-chip ${scope === 'sgg' ? 'active sgg-active' : ''}`}
                onClick={() => setScope('sgg')}
                title="Strictly query REACT++ 2D Scene Graph ground truth facts only (no VLM)"
              >
                <Network size={11} className="text-cyan-400" />
                <span>Scene Graph (SGG)</span>
              </button>
              <button
                type="button"
                className={`scope-chip ${scope === 'vlm' ? 'active vlm-active' : ''}`}
                onClick={() => setScope('vlm')}
                title="Directly query the RGB image with SmolVLM-500M visual model"
              >
                <Sparkles size={11} className="text-amber-400" />
                <span>Visual AI (VLM)</span>
              </button>
            </div>
          </div>

          {/* Clear Thread Button */}
          {messages.length > 0 && (
            <button
              type="button"
              className="btn-clear-conversation"
              onClick={handleClearConversation}
              title="Clear conversation thread (preserves IndexedDB workspace memory)"
            >
              <Trash2 size={13} />
              <span>Clear</span>
            </button>
          )}
        </div>
      </div>

      {/* 2. Conversation Thread Container */}
      <div className="assistant-thread-container">
        {messages.length === 0 ? (
          <div className="assistant-welcome-state">
            <Bot size={28} className="text-zinc-600 mb-2" />
            <p className="text-xs text-zinc-400 font-medium">Ready to answer questions about your workspace.</p>
            <p className="text-[11px] text-zinc-500 mt-1 max-w-md">
              Ask about detected items, spatial relations (&ldquo;beside the laptop&rdquo;), historical states, or object movements.
            </p>
          </div>
        ) : (
          <div className="assistant-messages-list">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`assistant-msg-wrapper ${msg.sender === 'user' ? 'msg-user' : 'msg-assistant'}`}
              >
                {/* User Message Bubble */}
                {msg.sender === 'user' ? (
                  <div className="user-bubble">
                    <div className="flex items-center justify-between gap-4 mb-1">
                      <span className="text-[11px] text-zinc-400 font-medium">You</span>
                      <div className="flex items-center gap-2">
                        {msg.scope && msg.scope !== 'current' && (
                          <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">
                            {msg.scope}
                          </span>
                        )}
                        <span className="text-[10px] text-zinc-500">{msg.timestamp}</span>
                      </div>
                    </div>
                    <p className="text-sm text-zinc-100">{msg.text}</p>
                  </div>
                ) : (
                  /* Assistant Structured Answer Card */
                  <div className="assistant-answer-box">
                    <div className="assistant-answer-header">
                      <div className="flex items-center gap-2">
                        <span className="assistant-label-header">Answer</span>
                        {msg.result && renderSourceBadge(msg.result.evidence.source)}
                      </div>
                      <div className="flex items-center gap-2">
                        {msg.result && renderCertaintyLabel(msg.result.certainty)}
                        <span className="text-[10px] text-zinc-500">{msg.timestamp}</span>
                      </div>
                    </div>

                    {/* Main Natural Language Response */}
                    <div className="assistant-main-text">
                      {msg.text}
                    </div>

                    {/* Ambiguous candidates list if applicable */}
                    {msg.result?.ambiguousOptions && msg.result.ambiguousOptions.length > 0 && (
                      <div className="ambiguous-candidates-panel">
                        <span className="text-[11px] text-zinc-400 font-medium mb-1 block">
                          Matching Candidate Objects:
                        </span>
                        <div className="flex flex-wrap gap-2">
                          {msg.result.ambiguousOptions.map((opt) => (
                            <button
                              key={opt.instanceId}
                              type="button"
                              className="candidate-choice-chip"
                              onClick={() => {
                                onSelectEntity?.(opt.instanceId);
                                handleSubmit(undefined, `Where is ${opt.instanceId}?`);
                              }}
                            >
                              <span className="font-semibold">{opt.displayName}</span>
                              <span className="font-mono text-[10px] opacity-75">({opt.instanceId})</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Evidence & Visual Linking Details */}
                    {msg.result && (
                      <div className="assistant-evidence-bar">
                        <div className="flex items-center flex-wrap gap-2 text-[11px] text-zinc-400">
                          {msg.result.evidence.frameIds.length > 0 && (
                            <span className="evidence-pill">
                              <Clock size={11} className="text-indigo-400" />
                              <span>Frame:</span>
                              <span className="font-mono text-zinc-200">
                                {msg.result.evidence.frameIds.join(', ')}
                              </span>
                              {onSelectHistoricalFrame && msg.result.evidence.frameIds[0] && (
                                <button
                                  type="button"
                                  className="evidence-action-link"
                                  onClick={() => onSelectHistoricalFrame(msg.result!.evidence.frameIds[0])}
                                  title="View this historical frame observation"
                                >
                                  View Frame
                                </button>
                              )}
                            </span>
                          )}

                          {msg.result.evidence.objectInstanceIds.length > 0 && (
                            <span className="evidence-pill">
                              <Layers size={11} className="text-cyan-400" />
                              <span>Objects:</span>
                              <div className="flex items-center gap-1">
                                {msg.result.evidence.objectInstanceIds.map((id) => (
                                  <button
                                    key={id}
                                    type="button"
                                    className="object-tag-btn"
                                    onClick={() => onSelectEntity?.(id)}
                                    title="Click to focus object in image viewer"
                                  >
                                    {id}
                                  </button>
                                ))}
                              </div>
                            </span>
                          )}

                          {msg.result.evidence.targetBbox && onSelectEntity && (
                            <button
                              type="button"
                              className="btn-inspect-image"
                              onClick={() => {
                                if (msg.result!.evidence.objectInstanceIds[0]) {
                                  onSelectEntity(msg.result!.evidence.objectInstanceIds[0]);
                                }
                              }}
                              title="Highlight in current image viewer"
                            >
                              <Eye size={12} />
                              <span>Inspect in Image</span>
                            </button>
                          )}

                          {/* Option to manually prompt VLM if user is unsatisfied with SGG answer */}
                          {msg.result.evidence.source !== 'vlm' && (
                            <button
                              type="button"
                              className="btn-prompt-vlm"
                              onClick={() => handleVerifyWithVLM(msg)}
                              title="Not satisfied with Scene Graph answer? Inspect visually with SmolVLM"
                            >
                              <Sparkles size={12} className="text-amber-400" />
                              <span>Verify visually with VLM</span>
                            </button>
                          )}

                          {/* Option to check Scene Graph facts if answered by VLM */}
                          {msg.result.evidence.source === 'vlm' && (
                            <button
                              type="button"
                              className="btn-prompt-sgg"
                              onClick={() => handleCheckWithSGG(msg)}
                              title="Check strictly verified facts in the Scene Graph only (/sgg)"
                            >
                              <Network size={12} className="text-cyan-400" />
                              <span>Check SGG Ground Truth</span>
                            </button>
                          )}
                        </div>

                        {/* Collapsible "How did I answer?" Reasoning Trace */}
                        <div className="reasoning-trace-section">
                          <button
                            type="button"
                            className="btn-toggle-trace"
                            onClick={() => toggleTrace(msg.id)}
                          >
                            <span>How did I answer?</span>
                            {expandedTraceIds.has(msg.id) ? (
                              <ChevronUp size={12} />
                            ) : (
                              <ChevronDown size={12} />
                            )}
                          </button>

                          {expandedTraceIds.has(msg.id) && (
                            <div className="reasoning-trace-box">
                              <div className="font-mono text-[10px] text-cyan-400 mb-1">
                                Execution Reasoning Trace:
                              </div>
                              <ol className="trace-list">
                                {msg.result.reasoningTrace.map((step, idx) => (
                                  <li key={idx} className="trace-step-item">
                                    <span className="step-num">{idx + 1}.</span>
                                    <span>{step}</span>
                                  </li>
                                ))}
                              </ol>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Contextual Follow-Up Suggestions */}
                    {msg.result?.followUpSuggestions && msg.result.followUpSuggestions.length > 0 && (
                      <div className="assistant-followups">
                        <span className="text-[10px] uppercase font-semibold tracking-wider text-zinc-500">
                          Follow-up:
                        </span>
                        <div className="flex flex-wrap gap-1.5 mt-1">
                          {msg.result.followUpSuggestions.map((fu) => (
                            <button
                              key={fu}
                              type="button"
                              className="followup-chip"
                              onClick={() => handleSubmit(undefined, fu)}
                              disabled={isQuerying}
                            >
                              <span>{fu}</span>
                              <ArrowRight size={10} className="text-zinc-500" />
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* Live Processing State Bubble */}
            {isQuerying && (
              <div className="assistant-msg-wrapper msg-assistant">
                <div className="assistant-answer-box p-3 border-cyan-500/30 bg-cyan-950/10 flex items-center gap-2.5">
                  <Loader2 size={15} className="animate-spin text-cyan-400 flex-shrink-0" />
                  <span className="text-xs text-cyan-300 font-medium">
                    {getProgressStateLabel()}
                  </span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* 3. Suggested Queries (Categorized & Expandable) */}
      <div className="assistant-suggestions-bar">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] font-medium text-zinc-400 flex items-center gap-1.5">
            <Sparkles size={12} className="text-amber-400" />
            <span>Suggested Questions:</span>
          </span>
          <button
            type="button"
            className="text-[11px] text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1"
            onClick={() => setShowAllSuggestions(!showAllSuggestions)}
          >
            <span>{showAllSuggestions ? 'Hide categories' : 'More suggestions'}</span>
            {showAllSuggestions ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
        </div>

        {/* Compact View: Top 4 Quick Questions */}
        {!showAllSuggestions ? (
          <div className="flex flex-wrap gap-1.5">
            {SUGGESTION_CATEGORIES.map((cat) => cat.questions[0]).map((q) => (
              <button
                key={q}
                type="button"
                className="nlq-suggestion-chip"
                onClick={() => handleSubmit(undefined, q)}
                disabled={!isEnabled || isQuerying}
              >
                {q}
              </button>
            ))}
          </div>
        ) : (
          /* Expanded View: All 4 Structured Categories */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2.5 mt-2 pt-2 border-t border-zinc-800/60">
            {SUGGESTION_CATEGORIES.map((cat) => (
              <div key={cat.title} className="suggestion-category-col">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-zinc-300 mb-1.5">
                  {cat.icon}
                  <span>{cat.title}</span>
                </div>
                <div className="flex flex-col gap-1">
                  {cat.questions.map((q) => (
                    <button
                      key={q}
                      type="button"
                      className="category-question-chip"
                      onClick={() => {
                        setShowAllSuggestions(false);
                        handleSubmit(undefined, q);
                      }}
                      disabled={!isEnabled || isQuerying}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 4. Query Composer Input Form */}
      <form onSubmit={(e) => handleSubmit(e)} className="assistant-composer">
        <div className="assistant-input-wrapper">
          <Search size={16} className="assistant-input-icon text-zinc-500" />
          {(scope === 'sgg' || /^\s*\/(sgg|graph)\b/i.test(queryText)) && (
            <span className="sgg-mode-badge" title="Direct Scene Graph ground truth execution (no VLM)">
              <Network size={11} className="text-cyan-400" />
              <span>SGG</span>
            </span>
          )}
          {(scope === 'vlm' || /^\s*\/(visual|vlm|inspect)\b/i.test(queryText)) && (
            <span className="vlm-mode-badge" title="Direct SmolVLM-500M visual execution">
              <Sparkles size={11} className="text-amber-400" />
              <span>VLM</span>
            </span>
          )}
          <input
            type="text"
            className="assistant-input"
            placeholder={
              isEnabled
                ? 'Ask about workspace (type /sgg for graph only, /visual for VLM)...'
                : 'Upload an observation image to enable queries'
            }
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            disabled={!isEnabled || isQuerying}
          />
          {isEnabled && (
            <div className="flex items-center gap-1 mr-1 flex-shrink-0">
              <button
                type="button"
                className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-zinc-800 text-cyan-400 hover:bg-zinc-700 transition-colors"
                onClick={() =>
                  setQueryText((prev) =>
                    prev ? `/sgg ${prev.replace(/^\s*\/(sgg|graph|visual|vlm)\s*/i, '')}` : '/sgg '
                  )
                }
                title="Prefix query with /sgg to answer strictly from Scene Graph facts"
              >
                /sgg
              </button>
              <button
                type="button"
                className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-zinc-800 text-amber-400 hover:bg-zinc-700 transition-colors"
                onClick={() =>
                  setQueryText((prev) =>
                    prev ? `/visual ${prev.replace(/^\s*\/(sgg|graph|visual|vlm)\s*/i, '')}` : '/visual '
                  )
                }
                title="Prefix query with /visual to prompt VLM directly"
              >
                /visual
              </button>
            </div>
          )}
        </div>

        <button
          type="submit"
          className="assistant-ask-btn"
          disabled={!isEnabled || !queryText.trim() || isQuerying}
        >
          {isQuerying ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              <span>Reasoning...</span>
            </>
          ) : (
            <>
              <Send size={14} />
              <span>Ask</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
};
