/**
 * Tests for NLQ Parser.
 * Verifies intent extraction, predicate synonym mapping, and time scope resolution.
 */

import { QueryParser } from '../../src/nlq/parser';

export function testQueryParser() {
  console.log('--- Testing QueryParser ---');

  const parser = new QueryParser();

  // 1. Current location
  const p1 = parser.parseQuery('Where is the laptop?');
  if (p1.intent !== 'locate_current_object' || p1.targetObject?.category !== 'laptop') {
    throw new Error(`Failed on 'Where is the laptop?': ${JSON.stringify(p1)}`);
  }

  // 2. Last seen query
  const p2 = parser.parseQuery('Where was my phone last seen?');
  if (p2.intent !== 'locate_last_seen_object' || p2.timeScope !== 'last_seen') {
    throw new Error(`Failed on 'Where was my phone last seen?': ${JSON.stringify(p2)}`);
  }

  // 3. Current relation with synonym "beside" -> "next_to"
  const p3 = parser.parseQuery('What is beside the laptop?');
  if (p3.intent !== 'find_current_relationship' || p3.predicate !== 'next_to') {
    throw new Error(`Failed on 'What is beside the laptop?': ${JSON.stringify(p3)}`);
  }

  // 4. Historical relation in previous observation
  const p4 = parser.parseQuery('What was next to the notebook in the previous observation?');
  if (p4.intent !== 'find_historical_relationship' || p4.timeScope !== 'previous_observation') {
    throw new Error(`Failed on historical query: ${JSON.stringify(p4)}`);
  }

  // 5. State comparison
  const p5 = parser.parseQuery('Did the screwdriver move?');
  if (p5.intent !== 'compare_object_state') {
    throw new Error(`Failed on 'Did the screwdriver move?': ${JSON.stringify(p5)}`);
  }

  console.log('✓ QueryParser tests passed!');
}
