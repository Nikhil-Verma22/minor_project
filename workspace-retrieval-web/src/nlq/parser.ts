/**
 * NLQ Query Parser.
 * Rule-based slot-filling parser converting natural language questions into typed QueryPlans.
 */

import { QueryPlan, QueryIntent, TimeScope } from './queryTypes';
import { PREDICATE_SYNONYMS, TIME_SCOPE_PATTERNS } from './aliases';
import { CanonicalPredicate } from '../config/predicates';
import { EntityResolver } from './entityResolver';
import { PersistentObjectInstance } from '../memory/memoryTypes';

export class QueryParser {
  private resolver = new EntityResolver();

  parseQuery(rawQuery: string, activeInstances: PersistentObjectInstance[] = []): QueryPlan {
    const text = rawQuery.trim();
    const lower = text.toLowerCase();

    // 1. Detect Time Scope
    let timeScope: TimeScope = 'current';
    for (const tsp of TIME_SCOPE_PATTERNS) {
      if (tsp.pattern.test(lower)) {
        timeScope = tsp.scope;
        break;
      }
    }

    // 2. Detect Predicate
    let detectedPredicate: CanonicalPredicate | undefined;
    for (const [phrase, canonical] of Object.entries(PREDICATE_SYNONYMS)) {
      const regex = new RegExp(`\\b${phrase}\\b`, 'i');
      if (regex.test(lower)) {
        detectedPredicate = canonical;
        break;
      }
    }

    // 3. Detect Intent
    let intent: QueryIntent = 'unknown';

    // State comparison: "Did X move?", "Has X moved?"
    if (/\b(did|has|have)\b.*\b(move|moved|change position)\b/i.test(lower)) {
      intent = 'compare_object_state';
    }
    // Location queries: "Where is/was X?"
    else if (/^\s*where\s+(is|was|are|were)\b/i.test(lower)) {
      if (timeScope === 'last_seen' || timeScope === 'previous_observation') {
        intent = 'locate_last_seen_object';
      } else {
        intent = 'locate_current_object';
      }
    }
    // Relational queries: "What is next to X?", "Which object is on the surface?"
    else if (/\b(what|which|who)\b.*\b(is|was|are|were)\b/i.test(lower) && detectedPredicate) {
      if (timeScope === 'previous_observation') {
        intent = 'find_historical_relationship';
      } else {
        intent = 'find_current_relationship';
      }
    }
    // Scene description
    else if (/\b(describe|what is in|list objects|overview)\b/i.test(lower)) {
      intent = 'describe_scene';
    } else if (detectedPredicate) {
      intent = 'find_current_relationship';
    } else {
      intent = 'locate_current_object';
    }

    // 4. Extract Target and Reference Objects
    let targetMention = '';
    let referenceMention = '';

    if (intent === 'compare_object_state') {
      const match = lower.match(/\b(did|has)\s+(the|my|a|an)?\s*([a-z0-9_\s-]+?)\s+(move|moved)\b/i);
      if (match) targetMention = match[3].trim();
    } else if (intent === 'locate_current_object' || intent === 'locate_last_seen_object') {
      const match = lower.match(/\bwhere\s+(is|was)\s+(the|my|a|an)?\s*([a-z0-9_\s-]+?)(\s+last seen|\s+now|\?|$)/i);
      if (match) targetMention = match[3].trim();
    } else if (intent === 'find_current_relationship' || intent === 'find_historical_relationship') {
      // E.g. "What is beside the laptop?" -> target: "?", reference: "laptop", predicate: "next_to"
      // E.g. "Is the pen on the desk?" -> target: "pen", reference: "desk", predicate: "on"
      for (const [phrase] of Object.entries(PREDICATE_SYNONYMS)) {
        if (new RegExp(`\\b${phrase}\\b`, 'i').test(lower)) {
          const parts = lower.split(new RegExp(`\\b${phrase}\\b`, 'i'));
          if (parts.length >= 2) {
            targetMention = parts[0].replace(/^(what|which|who|is|was|are)\s+(is|was|are)?\s*(the|my|a|an)?/i, '').trim();
            referenceMention = parts[1].replace(/^(the|my|a|an)?/i, '').replace(/(\?|\.|\s+in the previous observation.*)$/i, '').trim();
          }
          break;
        }
      }
    }

    const targetResolved = targetMention ? this.resolver.resolveMention(targetMention, activeInstances) : undefined;
    const refResolved = referenceMention ? this.resolver.resolveMention(referenceMention, activeInstances) : undefined;

    return {
      rawQuery: text,
      intent,
      targetObject: targetResolved
        ? {
            category: targetResolved.category,
            instanceId: targetResolved.instanceId,
            attributes: targetResolved.attributes,
            mention: targetMention,
          }
        : undefined,
      referenceObject: refResolved
        ? {
            category: refResolved.category,
            instanceId: refResolved.instanceId,
            mention: referenceMention,
          }
        : undefined,
      predicate: detectedPredicate,
      timeScope,
      needsVisualReasoning: false,
      unresolvedSlots: [],
    };
  }
}
