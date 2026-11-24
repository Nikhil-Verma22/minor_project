/**
 * VLM Response Schema & Candidate Types.
 */

import { CanonicalPredicate } from '../../config/predicates';

export interface VLMCandidateRelation {
  subject_id: string;
  predicate_text: string;
  object_id: string;
  canonical_predicate?: CanonicalPredicate;
  confidence: number;
  evidence: string;
}

export interface VLMStructuredResponse {
  answer: string;
  candidate_relations: VLMCandidateRelation[];
  unresolved: boolean;
  limitations: string[];
}
