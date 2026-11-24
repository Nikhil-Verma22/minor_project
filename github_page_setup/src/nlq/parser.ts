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
    const isDirectSgg = /^\s*\/(sgg|graph)\b/i.test(text) || scope === 'sgg';
    const isDirectVlm = /^\s*\/(visual|vlm|inspect|image)\b/i.test(text) || scope === 'vlm';
    const cleanText = text
      .replace(/^\s*\/(sgg|graph)\s+/i, '')
      .replace(/^\s*\/(visual|vlm|inspect|image)\s+/i, '')
      .trim();
    const lower = (cleanText || text).toLowerCase();

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

    // Detect if asking about sitting / support: "Where is the person sitting?", "What is the person sitting on?"
    const sittingMatch = lower.match(/\bwhere\s+(?:is|are)\s+(?:the\s+)?([a-z0-9_\s-]+?)\s+(?:sitting|seated|placed|resting|standing)\b/i) ||
      lower.match(/\bwhat\s+(?:is|are)\s+(?:the\s+)?([a-z0-9_\s-]+?)\s+(?:sitting|seated|resting)\s+on\b/i);

    // A. Attribute lookup: "What color is the table?", "What is the color of the shirt?", "cloth color on the table"
    const isAskingColor = /\b(color|colour|colored|coloured|hue)\b/i.test(lower);
    if (
      /\b(what|which)\s+(is\s+the\s+)?(color|colour|colored|type|material|kind)\s+(of\s+)?/i.test(lower) ||
      /\b(color|colour|colored)\s+(is|of)\b/i.test(lower) ||
      isAskingColor ||
      /\b(what\s+is\s+the\s+person\s+wearing|what\s+color\s+shirt)\b/i.test(lower)
    ) {
      intent = 'object_attribute';
      if (/\b(shirt|clothes|clothing|pants|top|t-shirt|wearing)\b/i.test(lower)) {
        requestedAttribute = 'shirt_color';
      } else {
        requestedAttribute = isAskingColor ? 'color' : 'type';
      }
    }
    // B. Relational location query: "Where is the person sitting?"
    else if (sittingMatch) {
      intent = 'spatial_relationship';
      detectedPredicate = 'on';
    }
    // C. Object movement / state change: "Did the screwdriver move?", "What changed between these observations?"
    else if (
      /\b(did|has|have)\b.*\b(move|moved|change position|reappear)\b/i.test(lower) ||
      /\b(which|what)\s+objects\s+(changed|moved|reappeared)\b/i.test(lower) ||
      /\bwhat\s+(?:has\s+)?changed(?:\s+between\s+.*)?\b/i.test(lower) ||
      /\b(compare|differences|changes)\b/i.test(lower)
    ) {
      intent = 'object_change';
    }
    // D. Standard location queries: "Where is/was X?"
    else if (/^\s*where\s+(is|was|are|were)\b/i.test(lower)) {
      if (timeScope === 'last_seen' || timeScope === 'previous_observation' || scope === 'history' || /\byesterday\b/i.test(lower)) {
        intent = 'last_seen_location';
      } else {
        intent = 'object_location';
      }
    }
    // E. Relational queries: "What is next to X?", "Which object is on the surface?", "Which objects are connected?"
    else if (/\b(what|which|who)\b.*\b(is|was|are|were|connected)\b/i.test(lower) && detectedPredicate) {
      if (timeScope === 'previous_observation' || scope === 'history') {
        intent = 'historical_relationship';
      } else {
        intent = 'spatial_relationship';
      }
    }
    // F. Scene description / overview
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

    // Check for explicit instance ID in query (e.g. "person_001", "chair_002", "det_2_0")
    const explicitIdMatch = lower.match(/\b([a-z]+(?:_[a-z]+)*_[0-9]+)\b/i) || lower.match(/\b(det_[0-9]+(?:_[0-9]+)?)\b/i);

    if (intent === 'object_attribute') {
      if (requestedAttribute === 'shirt_color') {
        // e.g. "what is the color of the shirt of person_001" or "what is the color of the shirt"
        if (explicitIdMatch) {
          targetMention = explicitIdMatch[1];
        } else if (lower.includes('person')) {
          targetMention = 'person';
        } else {
          targetMention = 'shirt';
        }
      } else {
        // Strip conversational prefix like "what is the", "what are", "tell me the", "what's the"
        const cleanForAttr = lower
          .replace(/^(?:what\s+is\s+(?:the\s+|a\s+|my\s+|an\s+)?|what's\s+(?:the\s+|a\s+|my\s+|an\s+)?|what\s+are\s+(?:the\s+)?|tell\s+me\s+(?:the\s+)?)/i, '')
          .trim();

        // 1. "what color is the [target]"
        const matchWhatColorIs = lower.match(/\bwhat\s+(?:color|colour)\s+(?:is|was|are|were)\s+(?:the\s+|my\s+|a\s+|an\s+)?([a-z0-9_\s-]+?)(?:\s+(?:on|in|at)\s+.*|\?|$)/i);
        // 2. "color/type of (the) [target]"
        const matchOf = lower.match(/(?:color|colour|colored|type|material|kind)\s+(?:is|of|for)?\s+(?:the\s+|my\s+|a\s+|an\s+)?([a-z0-9_\s-]+?)(?:\s+(?:on|in|at)\s+.*|\?|$)/i);
        // 3. "what is the [target] color/colored"
        const matchWhatIsTargetColor = lower.match(/what\s+(?:is|was)\s+(?:the\s+|a\s+|an\s+|my\s+)?([a-z0-9_\s-]+?)\s+(?:color|colour|colored|coloured)\b/i);
        // 4. "[target] color" (e.g. "pot color", "plant color", "chair color")
        const matchPrefix = cleanForAttr.match(/(?:the\s+|my\s+|a\s+|an\s+)?([a-z0-9_\s-]+?)\s+(?:color|colour|colored|coloured|type|material|kind)\b/i);

        if (matchWhatColorIs) {
          targetMention = matchWhatColorIs[1].trim();
        } else if (matchOf) {
          targetMention = matchOf[1].trim();
        } else if (matchWhatIsTargetColor) {
          targetMention = matchWhatIsTargetColor[1].trim();
        } else if (matchPrefix) {
          targetMention = matchPrefix[1].trim();
        } else if (explicitIdMatch) {
          targetMention = explicitIdMatch[1];
        }

        // Clean targetMention of any stray trailing prepositions or qualifiers
        targetMention = targetMention.replace(/\s+(on|in|at|near|beside)\s+.*$/i, '').trim();

        // Extract reference if spatial qualifier present (e.g. "cloth color on the table")
        if (/\bon\s+(?:the\s+|a\s+|my\s+)?([a-z0-9_\s-]+?)(?:\?|$)/i.test(lower)) {
          const matchRef = lower.match(/\bon\s+(?:the\s+|a\s+|my\s+)?([a-z0-9_\s-]+?)(?:\?|$)/i);
          if (matchRef && matchRef[1].trim() !== targetMention) {
            referenceMention = matchRef[1].trim();
          }
        }
      }
    } else if (sittingMatch) {
      targetMention = sittingMatch[1].trim();
    } else if (intent === 'object_change') {
      const match = lower.match(/\b(did|has)\s+(the|my|a|an)?\s*([a-z0-9_\s-]+?)\s+(move|moved)\b/i);
      if (match) {
        targetMention = match[3].trim();
      } else if (explicitIdMatch) {
        targetMention = explicitIdMatch[1];
      }
    } else if (intent === 'object_location' || intent === 'last_seen_location') {
      if (explicitIdMatch) {
        targetMention = explicitIdMatch[1];
      } else {
        const match = lower.match(/\bwhere\s+(is|was)\s+(the|my|a|an)?\s*([a-z0-9_\s-]+?)(\s+last seen|\s+now|\s+yesterday|\?|$)/i);
        if (match) targetMention = match[3].trim();
      }
    } else if (intent === 'spatial_relationship' || intent === 'historical_relationship') {
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

    // Direct resolution if targetMention or referenceMention is an explicit instance ID
    let targetResolved: any = undefined;
    if (explicitIdMatch && targetMention === explicitIdMatch[1]) {
      const id = explicitIdMatch[1];
      const baseCat = id.replace(/_[0-9]+$/, '');
      targetResolved = {
        category: baseCat,
        instanceId: id,
        mention: id,
        attributes: {},
        isAmbiguous: false,
      };
    } else if (targetMention) {
      targetResolved = this.resolver.resolveMention(targetMention, activeInstances);
    }

    let refResolved: any = undefined;
    if (referenceMention) {
      const refExplicitMatch = referenceMention.match(/\b([a-z]+(?:_[a-z]+)*_[0-9]+)\b/i) || referenceMention.match(/\b(det_[0-9]+(?:_[0-9]+)?)\b/i);
      if (refExplicitMatch) {
        const id = refExplicitMatch[1];
        const baseCat = id.replace(/_[0-9]+$/, '');
        refResolved = {
          category: baseCat,
          instanceId: id,
          mention: id,
          attributes: {},
          isAmbiguous: false,
        };
      } else {
        refResolved = this.resolver.resolveMention(referenceMention, activeInstances);
      }
    }

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
      scope: isDirectSgg ? 'sgg' : (isDirectVlm ? 'vlm' : scope),
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
      forceVLM: isDirectVlm,
      forceSGG: isDirectSgg,
      needsVisualReasoning: isDirectVlm,
      unresolvedSlots: [],
    };
  }
}
