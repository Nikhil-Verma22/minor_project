/**
 * Workspace Assistant & NLQ Query Types and Contracts.
 * Supports structured reasoning, conversation context, scope filtering,
 * evidence tracking, and verifiable explanation traces.
 */

import { CanonicalPredicate } from '../config/predicates';
import { PersistentObjectInstance } from '../memory/memoryTypes';

export type QueryIntent =
  | 'object_location'
  | 'object_attribute'
  | 'spatial_relationship'
  | 'historical_relationship'
  | 'last_seen_location'
  | 'object_change'
  | 'ambiguous_visual'
  | 'unsupported_occluded'
  | 'describe_scene'
  | 'unknown';

export type QueryScope = 'current' | 'history' | 'all' | 'vlm' | 'sgg';

export type CertaintyLevel =
  | 'confirmed'       // Confirmed stored information in scene graph or memory
  | 'high_visual'     // High-confidence visual interpretation
  | 'provisional'     // Provisional inference from VLM fallback
  | 'insufficient';   // Insufficient evidence or unobserved area

export interface ConversationContext {
  lastTargetEntity?: {
    instanceId?: string;
    category?: string;
    displayName?: string;
    mention?: string;
    attributes?: Record<string, string | null>;
  };
  lastReferenceEntity?: {
    instanceId?: string;
    category?: string;
    displayName?: string;
    mention?: string;
  };
  lastFrameId?: string;
  lastObservationId?: string;
  lastAnswer?: string;
  lastSource?: 'scene_graph' | 'workspace_memory' | 'vlm';
  lastObjectIds?: string[];
  scope?: QueryScope;
}

export interface QueryPlan {
  rawQuery: string;
  intent: QueryIntent;
  targetObject?: {
    category?: string;
    instanceId?: string;
    attributes?: Record<string, string | null>;
    mention: string;
    requestedAttribute?: string; // e.g. "color", "type", "material"
  };
  referenceObject?: {
    category?: string;
    instanceId?: string;
    mention: string;
  };
  predicate?: CanonicalPredicate;
  scope: QueryScope;
  timeScope: 'current' | 'last_seen' | 'previous_observation' | 'specific_timestamp' | 'all_history';
  timestamp?: string;
  needsVisualReasoning: boolean;
  forceVLM?: boolean;
  forceSGG?: boolean;
  unresolvedSlots: string[];
}

export interface AnswerEvidence {
  source: 'scene_graph' | 'workspace_memory' | 'vlm';
  workspaceId: string;
  frameIds: string[];
  timestamps: string[];
  objectInstanceIds: string[];
  relationshipIds: string[];
  confidence?: number;
  explanation: string;
  provisional: boolean;
  targetBbox?: [number, number, number, number] | null;
}

export interface AmbiguousOption {
  instanceId: string;
  displayName: string;
  categoryId: string;
  bbox?: [number, number, number, number] | null;
  frameId?: string;
}

export interface NLQResult {
  queryPlan: QueryPlan;
  answer: string;
  evidence: AnswerEvidence;
  requiresVLM: boolean;
  vlmReason?: string;
  certainty: CertaintyLevel;
  reasoningTrace: string[];
  followUpSuggestions?: string[];
  ambiguousOptions?: AmbiguousOption[];
}

export interface ConversationMessage {
  id: string;
  sender: 'user' | 'assistant';
  timestamp: string;
  text: string;
  scope?: QueryScope;
  result?: NLQResult;
}
