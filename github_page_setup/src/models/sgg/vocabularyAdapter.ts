/**
 * Vocabulary Adapter: Maps OpenPSG model labels to the 39 workspace categories
 * and 6 canonical predicates.
 *
 * Rules:
 * 1. Only map when a safe, defensible mapping exists.
 * 2. If unmapped, retain raw model label, set projectCategory = null, mapped = false.
 * 3. Never invent categories or predicates.
 */

import { CanonicalPredicate } from '../../config/predicates';

export interface CategoryMappingResult {
  mapped: boolean;
  projectCategory: string | null;
  modelLabel: string;
  confidence: number;
}

export interface PredicateMappingResult {
  mapped: boolean;
  canonicalPredicate: CanonicalPredicate | null;
  predicateModelLabel: string;
}

/**
 * Direct and safe synonym mapping from OpenPSG 133 classes to the 39 workspace ontology.
 */
const PSG_TO_WORKSPACE_MAP: Record<string, { category: string; confidence: number }> = {
  // Direct matches
  'laptop': { category: 'laptop', confidence: 1.0 },
  'cell phone': { category: 'smartphone', confidence: 1.0 },
  'keyboard': { category: 'keyboard', confidence: 1.0 },
  'mouse': { category: 'mouse', confidence: 1.0 },
  'book': { category: 'book', confidence: 1.0 },
  'paper': { category: 'loose_paper', confidence: 1.0 },
  'bottle': { category: 'water_bottle', confidence: 0.9 },
  'scissors': { category: 'scissors', confidence: 1.0 },
  'tv': { category: 'monitor', confidence: 0.85 },
  'remote': { category: 'calculator', confidence: 0.5 }, // Candidate marker if handheld tool
};

/**
 * Mapping OpenPSG 57 predicates to the 6 canonical predicates.
 */
const PSG_TO_CANONICAL_PREDICATES: Record<string, CanonicalPredicate> = {
  'on': 'on',
  'sitting on': 'on',
  'standing on': 'on',
  'lying on': 'on',
  'under': 'under',
  'beside': 'next_to',
  'standing next to': 'next_to',
  'sitting next to': 'next_to',
  'lying next to': 'next_to',
  'in front of': 'in_front_of',
  'behind': 'behind',
  'connected to': 'connected_to',
  'attached to': 'connected_to',
  'over': 'on',
};

export class VocabularyAdapter {
  /**
   * Map an OpenPSG object label to project category.
   */
  static mapObjectCategory(modelLabel: string): CategoryMappingResult {
    const normalized = modelLabel.trim().toLowerCase();
    const match = PSG_TO_WORKSPACE_MAP[normalized];

    if (match) {
      return {
        mapped: true,
        projectCategory: match.category,
        modelLabel,
        confidence: match.confidence,
      };
    }

    return {
      mapped: false,
      projectCategory: null,
      modelLabel,
      confidence: 0,
    };
  }

  /**
   * Map an OpenPSG predicate label to canonical predicate.
   */
  static mapPredicate(predicateModelLabel: string): PredicateMappingResult {
    const normalized = predicateModelLabel.trim().toLowerCase();
    const canonical = PSG_TO_CANONICAL_PREDICATES[normalized];

    if (canonical) {
      return {
        mapped: true,
        canonicalPredicate: canonical,
        predicateModelLabel,
      };
    }

    return {
      mapped: false,
      canonicalPredicate: null,
      predicateModelLabel,
    };
  }
}
