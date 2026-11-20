/**
 * Exactly six canonical relationship predicates approved for the project ontology.
 * Directed triplet: <subject, predicate, object>
 */

export type CanonicalPredicate =
  | 'on'
  | 'under'
  | 'next_to'
  | 'in_front_of'
  | 'behind'
  | 'connected_to';

export interface PredicateDefinition {
  canonical: CanonicalPredicate;
  displayName: string;
  description: string;
  synonyms: string[];
}

export const CANONICAL_PREDICATES: PredicateDefinition[] = [
  {
    canonical: 'on',
    displayName: 'on',
    description: 'Subject is directly supported by or resting on object',
    synonyms: ['resting on', 'placed on', 'sitting on', 'atop', 'standing on'],
  },
  {
    canonical: 'under',
    displayName: 'under',
    description: 'Subject is beneath or covered by object',
    synonyms: ['below', 'beneath', 'underneath'],
  },
  {
    canonical: 'next_to',
    displayName: 'next to',
    description: 'Subject is horizontally beside or adjacent to object',
    synonyms: ['beside', 'adjacent to', 'by', 'near', 'alongside', 'standing next to', 'sitting next to', 'lying next to'],
  },
  {
    canonical: 'in_front_of',
    displayName: 'in front of',
    description: 'Subject is closer to the viewpoint/camera than object along depth axis',
    synonyms: ['in front of', 'ahead of', 'before'],
  },
  {
    canonical: 'behind',
    displayName: 'behind',
    description: 'Subject is further from the viewpoint/camera than object along depth axis',
    synonyms: ['behind', 'in back of', 'rear of'],
  },
  {
    canonical: 'connected_to',
    displayName: 'connected to',
    description: 'Subject is physically or electronically connected/attached to object',
    synonyms: ['plugged into', 'attached to', 'wired to', 'hooked to'],
  },
];

export const VALID_PREDICATES_SET = new Set<string>(
  CANONICAL_PREDICATES.map((p) => p.canonical)
);
