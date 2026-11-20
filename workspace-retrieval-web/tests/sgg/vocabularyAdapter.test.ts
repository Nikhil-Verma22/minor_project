/**
 * Tests for VocabularyAdapter.
 * Verifies mapping from OpenPSG to the 39 workspace categories and 6 canonical predicates.
 */

import { VocabularyAdapter } from '../../src/models/sgg/vocabularyAdapter';

export function testVocabularyAdapter() {
  console.log('--- Testing VocabularyAdapter ---');

  // 1. Direct object category matches
  const laptop = VocabularyAdapter.mapObjectCategory('laptop');
  if (laptop.mapped !== true || laptop.projectCategory !== 'laptop') {
    throw new Error(`Expected laptop -> laptop, got ${JSON.stringify(laptop)}`);
  }

  const phone = VocabularyAdapter.mapObjectCategory('cell phone');
  if (phone.mapped !== true || phone.projectCategory !== 'smartphone') {
    throw new Error(`Expected 'cell phone' -> 'smartphone', got ${JSON.stringify(phone)}`);
  }

  const paper = VocabularyAdapter.mapObjectCategory('paper');
  if (paper.mapped !== true || paper.projectCategory !== 'loose_paper') {
    throw new Error(`Expected 'paper' -> 'loose_paper', got ${JSON.stringify(paper)}`);
  }

  // 2. Out-of-vocabulary classes should remain unmapped and preserve modelLabel
  const zebra = VocabularyAdapter.mapObjectCategory('zebra');
  if (zebra.mapped !== false || zebra.projectCategory !== null || zebra.modelLabel !== 'zebra') {
    throw new Error(`Expected zebra -> unmapped, got ${JSON.stringify(zebra)}`);
  }

  // 3. Predicate mappings to the 6 canonical predicates
  const onPred = VocabularyAdapter.mapPredicate('sitting on');
  if (onPred.mapped !== true || onPred.canonicalPredicate !== 'on') {
    throw new Error(`Expected 'sitting on' -> 'on', got ${JSON.stringify(onPred)}`);
  }

  const nextToPred = VocabularyAdapter.mapPredicate('beside');
  if (nextToPred.mapped !== true || nextToPred.canonicalPredicate !== 'next_to') {
    throw new Error(`Expected 'beside' -> 'next_to', got ${JSON.stringify(nextToPred)}`);
  }

  const connectedPred = VocabularyAdapter.mapPredicate('attached to');
  if (connectedPred.mapped !== true || connectedPred.canonicalPredicate !== 'connected_to') {
    throw new Error(`Expected 'attached to' -> 'connected_to', got ${JSON.stringify(connectedPred)}`);
  }

  // 4. Non-canonical predicate should remain unmapped
  const eatPred = VocabularyAdapter.mapPredicate('eating');
  if (eatPred.mapped !== false || eatPred.canonicalPredicate !== null) {
    throw new Error(`Expected 'eating' -> unmapped, got ${JSON.stringify(eatPred)}`);
  }

  console.log('✓ VocabularyAdapter tests passed!');
}
