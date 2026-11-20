/**
 * NLQ Query Types and Contracts.
 */

import { CanonicalPredicate } from '../config/predicates';
import { PersistentObjectInstance } from '../memory/memoryTypes';

export type QueryIntent =
  | 'locate_current_object'
  | 'locate_last_seen_object'
  | 'find_current_relationship'
  | 'find_historical_relationship'
  | 'compare_object_state'
  | 'list_scene_changes'
  | 'describe_scene'
  | 'unknown';

export type TimeScope =
  | 'current'
  | 'last_seen'
  | 'previous_observation'
  | 'specific_timestamp'
  | 'all_history';

export interface QueryPlan {
  rawQuery: string;
  intent: QueryIntent;
  targetObject?: {
    category?: string;
    instanceId?: string;
    attributes?: Record<string, string | null>;
    mention: string;
  };
  referenceObject?: {
    category?: string;
    instanceId?: string;
    mention: string;
  };
  predicate?: CanonicalPredicate;
  timeScope: TimeScope;
  timestamp?: string;
  needsVisualReasoning: boolean;
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
}

export interface NLQResult {
  queryPlan: QueryPlan;
  answer: string;
  evidence: AnswerEvidence;
  requiresVLM: boolean;
  vlmReason?: string;
  ambiguousOptions?: PersistentObjectInstance[];
}
