/**
 * NLQ Query Parser.
 * Rule-based slot-filling parser converting natural language questions into typed QueryPlans.
 * Supports conversational pronoun resolution, attribute extraction, and occluded-area detection.
 */

import { QueryPlan, QueryIntent, QueryScope, ConversationContext } from './queryTypes';
import { PREDICATE_SYNONYMS, TIME_SCOPE_PATTERNS, OCCLUDED_AREA_PATTERNS } from './aliases';
import { CanonicalPredicate } from '../config/predicates';
import { EntityResolver } from './entityResolver';
import { PersistentObjectInstance } from '../memory/memoryTypes';

export class QueryParser {
  private resolver = new EntityResolver();

  parseQuery(
    rawQuery: string,
    activeInstances: PersistentObjectInstance[] = [],
    context?: ConversationContext,
    scope: QueryScope = 'current'
  ): QueryPlan {
    const text = rawQuery.trim();
    const lower = text.toLowerCase();

    // 1. Check for Occluded / Non-visible Storage Query (e.g. "behind closed drawer")
    for (const pat of OCCLUDED_AREA_PATTERNS) {
      if (pat.test(lower)) {
        return {
          rawQuery: text,
          intent: 'unsupported_occluded',
          scope,
          timeScope: 'current',
          needsVisualReasoning: false,
          unresolvedSlots: [],
        };
      }
    }

    // 2. Detect Time Scope
    let timeScope: QueryPlan['timeScope'] = scope === 'history' ? 'last_seen' : 'current';
    for (const tsp of TIME_SCOPE_PATTERNS) {
      if (tsp.pattern.test(lower)) {
        timeScope = tsp.scope;
        break;
      }
    }

    // 3. Detect Predicate
    let detectedPredicate: CanonicalPredicate | undefined;
    for (const [phrase, canonical] of Object.entries(PREDICATE_SYNONYMS)) {
      const regex = new RegExp(`\\b${phrase}\\b`, 'i');
      if (regex.test(lower)) {
        detectedPredicate = canonical;
        break;
      }
    }

    // 4. Detect Intent
    let intent: QueryIntent = 'unknown';
    let requestedAttribute: string | undefined;

    // A. Attribute lookup: "What color is the table?", "What is the colour of the laptop?"
    if (/\b(what|which)\s+(is\s+the\s+)?(color|colour|type|material|kind)\s+(of\s+)?/i.test(lower) ||
        /\b(color|colour)\s+(is|of)\b/i.test(lower)) {
      intent = 'object_attribute';
      requestedAttribute = /\b(color|colour)\b/i.test(lower) ? 'color' : 'type';
    }
    // B. Object movement / state change: "Did the screwdriver move?", "Which objects changed their locations?"
    else if (/\b(did|has|have)\b.*\b(move|moved|change position|reappear)\b/i.test(lower) ||
             /\b(which|what)\s+objects\s+(changed|moved|reappeared)\b/i.test(lower)) {
      intent = 'object_change';
    }
    // C. Location queries: "Where is/was X?"
    else if (/^\s*where\s+(is|was|are|were)\b/i.test(lower)) {
      if (timeScope === 'last_seen' || timeScope === 'previous_observation' || scope === 'history' || /\byesterday\b/i.test(lower)) {
        intent = 'last_seen_location';
      } else {
        intent = 'object_location';
      }
    }
    // D. Relational queries: "What is next to X?", "Which object is on the surface?", "Which objects are connected?"
    else if (/\b(what|which|who)\b.*\b(is|was|are|were|connected)\b/i.test(lower) && detectedPredicate) {
      if (timeScope === 'previous_observation' || scope === 'history') {
        intent = 'historical_relationship';
      } else {
        intent = 'spatial_relationship';
      }
    }
    // E. Scene description / overview
    else if (/\b(describe|what is in|list objects|overview)\b/i.test(lower)) {
      intent = 'describe_scene';
    } else if (detectedPredicate) {
      intent = (timeScope === 'previous_observation' || scope === 'history')
        ? 'historical_relationship'
        : 'spatial_relationship';
    } else {
      intent = 'object_location';
    }

    // 5. Extract Target and Reference Objects with Pronoun Resolution
    let targetMention = '';
    let referenceMention = '';

    // Pronoun replacement helper
    const hasPronoun = (str: string) => /\b(it|that object|the same one|that one|its|there)\b/i.test(str);

    if (intent === 'object_attribute') {
      const match = lower.match(/(?:color|colour|type|material|kind)\s+(?:is|of)\s+(?:the\s+|my\s+|a\s+|an\s+)?([a-z0-9_\s-]+?)(?:\?|$)/i);
      if (match) targetMention = match[1].trim();
    } else if (intent === 'object_change') {
      const match = lower.match(/\b(did|has)\s+(the|my|a|an)?\s*([a-z0-9_\s-]+?)\s+(move|moved)\b/i);
      if (match) targetMention = match[3].trim();
    } else if (intent === 'object_location' || intent === 'last_seen_location') {
      const match = lower.match(/\bwhere\s+(is|was)\s+(the|my|a|an)?\s*([a-z0-9_\s-]+?)(\s+last seen|\s+now|\s+yesterday|\?|$)/i);
      if (match) targetMention = match[3].trim();
    } else if (intent === 'spatial_relationship' || intent === 'historical_relationship') {
      // E.g. "What is beside the laptop?" -> target: "?", reference: "laptop", predicate: "next_to"
      // E.g. "What was next to it?" -> target: "?", reference: "it", predicate: "next_to"
      for (const [phrase] of Object.entries(PREDICATE_SYNONYMS)) {
        if (new RegExp(`\\b${phrase}\\b`, 'i').test(lower)) {
          const parts = lower.split(new RegExp(`\\b${phrase}\\b`, 'i'));
          if (parts.length >= 2) {
            targetMention = parts[0].replace(/^(what|which|who|is|was|are)\s+(is|was|are)?\s*(the|my|a|an|objects\s+are)?/i, '').trim();
            referenceMention = parts[1].replace(/^(the|my|a|an)?/i, '').replace(/(\?|\.|\s+in the previous observation.*|\s+previously.*)$/i, '').trim();
          }
          break;
        }
      }
    }

    // Resolve Pronoun to Conversation Context if applicable
    if (context && context.lastTargetEntity) {
      if (hasPronoun(targetMention)) {
        targetMention = context.lastTargetEntity.mention || context.lastTargetEntity.displayName || targetMention;
      }
      if (hasPronoun(referenceMention)) {
        referenceMention = context.lastTargetEntity.mention || context.lastTargetEntity.displayName || referenceMention;
      }
      // If query is just "What is next to it?" and reference is "it"
      if (!referenceMention && hasPronoun(lower)) {
        referenceMention = context.lastTargetEntity.mention || context.lastTargetEntity.displayName || 'object';
      }
    }

    let targetResolved = targetMention ? this.resolver.resolveMention(targetMention, activeInstances) : undefined;
    let refResolved = referenceMention ? this.resolver.resolveMention(referenceMention, activeInstances) : undefined;

    // Direct Context Entity Inheritance for Pronoun References
    if (context?.lastTargetEntity) {
      if (hasPronoun(targetMention) || targetMention === context.lastTargetEntity.mention) {
        targetResolved = {
          category: context.lastTargetEntity.category || targetResolved?.category,
          instanceId: context.lastTargetEntity.instanceId || targetResolved?.instanceId,
          mention: context.lastTargetEntity.displayName || targetMention,
          attributes: context.lastTargetEntity.attributes || targetResolved?.attributes || {},
          isAmbiguous: false,
        };
      }
      if (hasPronoun(referenceMention) || referenceMention === context.lastTargetEntity.mention) {
        refResolved = {
          category: context.lastTargetEntity.category || refResolved?.category,
          instanceId: context.lastTargetEntity.instanceId || refResolved?.instanceId,
          mention: context.lastTargetEntity.displayName || referenceMention,
          attributes: context.lastTargetEntity.attributes || refResolved?.attributes || {},
          isAmbiguous: false,
        };
      }
    }

    return {
      rawQuery: text,
      intent,
      scope,
      targetObject: targetResolved
        ? {
            category: targetResolved.category,
            instanceId: targetResolved.instanceId,
            attributes: targetResolved.attributes,
            mention: targetMention,
            requestedAttribute,
          }
        : targetMention
        ? {
            mention: targetMention,
            requestedAttribute,
          }
        : undefined,
      referenceObject: refResolved
        ? {
            category: refResolved.category,
            instanceId: refResolved.instanceId,
            mention: referenceMention,
          }
        : referenceMention
        ? {
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
