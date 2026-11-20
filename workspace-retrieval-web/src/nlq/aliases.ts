/**
 * Aliases, Synonyms, and Token Maps for NLQ Parsing.
 */

import { WORKSPACE_CATEGORIES } from '../config/categories';
import { CanonicalPredicate } from '../config/predicates';

export interface EntityAliasMatch {
  categoryId: string;
  matchedText: string;
}

export const PREDICATE_SYNONYMS: Record<string, CanonicalPredicate> = {
  // on
  'on': 'on',
  'upon': 'on',
  'atop': 'on',
  'resting on': 'on',
  'placed on': 'on',
  'sitting on': 'on',
  'standing on': 'on',

  // under
  'under': 'under',
  'below': 'under',
  'beneath': 'under',
  'underneath': 'under',

  // next_to
  'next to': 'next_to',
  'beside': 'next_to',
  'by': 'next_to',
  'near': 'next_to',
  'adjacent to': 'next_to',
  'alongside': 'next_to',
  'close to': 'next_to',

  // in_front_of
  'in front of': 'in_front_of',
  'in front': 'in_front_of',
  'ahead of': 'in_front_of',

  // behind
  'behind': 'behind',
  'in back of': 'behind',
  'rear of': 'behind',

  // connected_to
  'connected to': 'connected_to',
  'plugged into': 'connected_to',
  'attached to': 'connected_to',
  'wired to': 'connected_to',
  'hooked to': 'connected_to',
};

export const TIME_SCOPE_PATTERNS = [
  { pattern: /\b(last seen|last observed|where was|previous location)\b/i, scope: 'last_seen' as const },
  { pattern: /\b(previous observation|earlier frame|before|past|previously)\b/i, scope: 'previous_observation' as const },
  { pattern: /\b(history|timeline|all along|over time)\b/i, scope: 'all_history' as const },
  { pattern: /\b(now|currently|current|present|at present)\b/i, scope: 'current' as const },
];

export const ATTRIBUTE_COLORS = [
  'blue',
  'black',
  'white',
  'red',
  'green',
  'yellow',
  'silver',
  'grey',
  'gray',
  'orange',
  'purple',
];
