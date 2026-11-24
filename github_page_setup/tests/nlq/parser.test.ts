/**
 * Tests for NLQ Parser & Workspace Assistant Query Intent Classification.
 * Verifies intent extraction, predicate synonym mapping, pronoun resolution,
 * attribute extraction, and occluded storage area detection.
 */

import { QueryParser } from '../../src/nlq/parser';
import { ConversationContext } from '../../src/nlq/queryTypes';

export function testQueryParser() {
  console.log('--- Testing QueryParser ---');

  const parser = new QueryParser();

  // 1. Object location query
  const p1 = parser.parseQuery('Where is the laptop?');
  if (p1.intent !== 'object_location' || p1.targetObject?.category !== 'laptop') {
    throw new Error(`Failed on 'Where is the laptop?': ${JSON.stringify(p1)}`);
  }

  // 2. Last seen query
  const p2 = parser.parseQuery('Where was my phone last seen?');
  if (p2.intent !== 'last_seen_location' || p2.timeScope !== 'last_seen') {
    throw new Error(`Failed on 'Where was my phone last seen?': ${JSON.stringify(p2)}`);
  }

  // 3. Current relation with synonym "beside" -> "next_to"
  const p3 = parser.parseQuery('What is beside the laptop?');
  if (p3.intent !== 'spatial_relationship' || p3.predicate !== 'next_to') {
    throw new Error(`Failed on 'What is beside the laptop?': ${JSON.stringify(p3)}`);
  }

  // 4. Historical relation in previous observation
  const p4 = parser.parseQuery('What was next to the notebook in the previous observation?');
  if (p4.intent !== 'historical_relationship' || p4.timeScope !== 'previous_observation') {
    throw new Error(`Failed on historical query: ${JSON.stringify(p4)}`);
  }

  // 5. Object movement / state change
  const p5 = parser.parseQuery('Did the screwdriver move?');
  if (p5.intent !== 'object_change') {
    throw new Error(`Failed on 'Did the screwdriver move?': ${JSON.stringify(p5)}`);
  }

  // 6. Attribute query: "What color is the table?"
  const p6 = parser.parseQuery('What color is the table?');
  if (p6.intent !== 'object_attribute' || p6.targetObject?.requestedAttribute !== 'color') {
    throw new Error(`Failed on 'What color is the table?': ${JSON.stringify(p6)}`);
  }

  // 7. Unsupported occluded space: "What is behind the closed drawer?"
  const p7 = parser.parseQuery('What is behind the closed drawer?');
  if (p7.intent !== 'unsupported_occluded') {
    throw new Error(`Failed on occluded query: ${JSON.stringify(p7)}`);
  }

  // 8. Conversational Follow-up Pronoun Resolution: "What is next to it?"
  const mockContext: ConversationContext = {
    lastTargetEntity: {
      instanceId: 'phone_001',
      category: 'cell_phone',
      displayName: 'phone',
      mention: 'phone',
    },
  };
  const p8 = parser.parseQuery('What is next to it?', [], mockContext);
  const validPhoneCat = p8.referenceObject?.category === 'cell_phone' || p8.referenceObject?.category === 'smartphone';
  if (p8.intent !== 'spatial_relationship' || p8.predicate !== 'next_to' || !validPhoneCat) {
    throw new Error(`Failed on follow-up pronoun resolution: ${JSON.stringify(p8)}`);
  }

  // 9. Cloth color on the table: attribute lookup
  const p9 = parser.parseQuery('cloth color on the table');
  if (p9.intent !== 'object_attribute' || p9.targetObject?.requestedAttribute !== 'color' || p9.targetObject?.mention !== 'cloth') {
    throw new Error(`Failed on 'cloth color on the table': ${JSON.stringify(p9)}`);
  }

  // 10. Manual /visual slash command for direct VLM
  const p10 = parser.parseQuery('/visual cloth color on the table');
  if (!p10.forceVLM) {
    throw new Error(`Failed on '/visual cloth color on the table': expected forceVLM to be true: ${JSON.stringify(p10)}`);
  }

  // 11. Pot color query
  const p11 = parser.parseQuery('what is the pot color');
  if (p11.intent !== 'object_attribute' || p11.targetObject?.requestedAttribute !== 'color' || p11.targetObject?.mention !== 'pot') {
    throw new Error(`Failed on 'what is the pot color': ${JSON.stringify(p11)}`);
  }

  // 12. Pot of plant colored
  const p12 = parser.parseQuery('what is the pot of plant colored');
  if (p12.intent !== 'object_attribute' || p12.targetObject?.requestedAttribute !== 'color' || p12.targetObject?.mention !== 'pot of plant') {
    throw new Error(`Failed on 'what is the pot of plant colored': ${JSON.stringify(p12)}`);
  }

  // 13. Color of chair (must resolve to chair, NOT charger)
  const p13 = parser.parseQuery('what is the color of chair');
  if (p13.intent !== 'object_attribute' || p13.targetObject?.requestedAttribute !== 'color' || p13.targetObject?.category === 'charger' || p13.targetObject?.mention !== 'chair') {
    throw new Error(`Failed on 'what is the color of chair' (resolved to charger): ${JSON.stringify(p13)}`);
  }

  // 14. Manual /sgg slash command for strictly Scene Graph only
  const p14 = parser.parseQuery('/sgg where is laptop');
  if (!p14.forceSGG || p14.scope !== 'sgg') {
    throw new Error(`Failed on '/sgg where is laptop': expected forceSGG: true, scope: 'sgg': ${JSON.stringify(p14)}`);
  }

  console.log('✓ QueryParser tests passed!');
}
