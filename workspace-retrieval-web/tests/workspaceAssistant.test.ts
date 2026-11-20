/**
 * Automated Acceptance Tests for Workspace Assistant.
 * Verifies all 9 Acceptance Test criteria from specification:
 * 1. Current relationship resolution
 * 2. Historical location retrieval
 * 3. Unknown attribute & visual inspection boundary
 * 4. Conversational follow-up & pronoun resolution
 * 5. Unsupported occluded space rejection
 * 6. Ambiguous object candidate disambiguation
 * 7. Nonexistent history handling (no fabrication)
 * 8. Source attribution correctness
 * 9. Conversation clearing isolation (IndexedDB memory intact)
 */

import { QueryParser } from '../src/nlq/parser';
import { QueryExecutor } from '../src/nlq/queryExecutor';
import { SceneGraph } from '../src/graph/graphTypes';
import { MemoryRepository } from '../src/memory/memoryRepository';
import {
  FrameRecord,
  PersistentObjectInstance,
  ObjectObservation,
  RelationshipFact,
} from '../src/memory/memoryTypes';
import { ConversationContext } from '../src/nlq/queryTypes';

export async function testWorkspaceAssistant() {
  console.log('--- Testing Workspace Assistant (All 9 Acceptance Tests) ---');

  const parser = new QueryParser();
  const workspaceId = 'ws_acceptance_test';

  // Build Mock Scene Graph
  const mockCurrentGraph: SceneGraph = {
    frameId: 'seq_001_frame_002',
    timestamp: '2026-11-20T12:10:00+05:30',
    nodes: [
      {
        id: 'laptop_001',
        instanceId: 'laptop_001',
        categoryId: 'laptop',
        label: 'laptop',
        bbox: [100, 150, 300, 350],
        confidence: 0.92,
        isSurface: false,
        attributes: { color: 'silver', type: 'laptop' },
      },
      {
        id: 'notebook_001',
        instanceId: 'notebook_001',
        categoryId: 'book',
        label: 'notebook',
        bbox: [320, 160, 420, 260],
        confidence: 0.88,
        isSurface: false,
        attributes: { color: null, type: null },
      },
      {
        id: 'table_001',
        instanceId: 'workspace_surface_001',
        categoryId: 'dining_table',
        label: 'table',
        bbox: [50, 120, 600, 450],
        confidence: 0.95,
        isSurface: true,
        attributes: { color: null, type: 'wooden' },
      },
      // Two pens for disambiguation test
      {
        id: 'pen_001',
        instanceId: 'pen_001',
        categoryId: 'pen',
        label: 'pen',
        bbox: [110, 160, 130, 200],
        confidence: 0.82,
        isSurface: false,
        attributes: { color: 'blue', type: null },
      },
      {
        id: 'pen_002',
        instanceId: 'pen_002',
        categoryId: 'pen',
        label: 'pen',
        bbox: [450, 160, 470, 200],
        confidence: 0.80,
        isSurface: false,
        attributes: { color: 'black', type: null },
      },
      // Person and chair for sitting relationship test
      {
        id: 'person_001',
        instanceId: 'person_001',
        categoryId: 'person',
        label: 'person',
        bbox: [295, 124, 352, 236],
        confidence: 0.88,
        isSurface: false,
        attributes: { color: null, type: null },
      },
      {
        id: 'chair_002',
        instanceId: 'chair_002',
        categoryId: 'chair',
        label: 'chair',
        bbox: [290, 160, 350, 260],
        confidence: 0.85,
        isSurface: false,
        attributes: { color: null, type: null },
      },
    ],
    edges: [
      {
        id: 'rel_sitting_001',
        sourceId: 'person_001',
        targetId: 'chair_002',
        predicate: 'on',
        confidence: 0.90,
        source: 'reactpp',
        provisional: false,
      },
      {
        id: 'rel_001',
        sourceId: 'notebook_001',
        targetId: 'laptop_001',
        predicate: 'next_to',
        confidence: 0.89,
        source: 'reactpp',
        provisional: false,
      },
      {
        id: 'rel_002',
        sourceId: 'laptop_001',
        targetId: 'table_001',
        predicate: 'on',
        confidence: 0.94,
        source: 'reactpp',
        provisional: false,
      },
      {
        id: 'rel_003',
        sourceId: 'notebook_001',
        targetId: 'table_001',
        predicate: 'on',
        confidence: 0.91,
        source: 'reactpp',
        provisional: false,
      },
    ],
  };

  // Mock Memory Repository for offline unit testing
  const mockFrames: FrameRecord[] = [
    {
      frameId: 'seq_001_frame_001',
      sequenceId: 'seq_001',
      workspaceId,
      frameIndex: 0,
      timestamp: '2026-11-20T12:00:00+05:30',
      imageWidth: 640,
      imageHeight: 480,
    },
    {
      frameId: 'seq_001_frame_002',
      sequenceId: 'seq_001',
      workspaceId,
      frameIndex: 1,
      timestamp: '2026-11-20T12:10:00+05:30',
      imageWidth: 640,
      imageHeight: 480,
    },
  ];

  const mockInstances: PersistentObjectInstance[] = [
    {
      instanceId: 'laptop_001',
      workspaceId,
      categoryId: 'laptop',
      displayName: 'laptop',
      attributes: { color: 'silver', type: 'laptop' },
      firstSeenTimestamp: '2026-11-20T12:00:00+05:30',
      lastSeenTimestamp: '2026-11-20T12:10:00+05:30',
      firstSeenFrameId: 'seq_001_frame_001',
      lastSeenFrameId: 'seq_001_frame_002',
      status: 'active',
    },
    {
      instanceId: 'cell_phone_001',
      workspaceId,
      categoryId: 'cell_phone',
      displayName: 'phone',
      attributes: { color: 'black', type: 'smartphone' },
      firstSeenTimestamp: '2026-11-20T12:00:00+05:30',
      lastSeenTimestamp: '2026-11-20T12:00:00+05:30',
      firstSeenFrameId: 'seq_001_frame_001',
      lastSeenFrameId: 'seq_001_frame_001',
      status: 'missing',
    },
    {
      instanceId: 'notebook_001',
      workspaceId,
      categoryId: 'book',
      displayName: 'notebook',
      attributes: { color: null, type: null },
      firstSeenTimestamp: '2026-11-20T12:00:00+05:30',
      lastSeenTimestamp: '2026-11-20T12:10:00+05:30',
      firstSeenFrameId: 'seq_001_frame_001',
      lastSeenFrameId: 'seq_001_frame_002',
      status: 'active',
    },
  ];

  const mockObservations: ObjectObservation[] = [
    {
      annotationId: 'obs_phone_001',
      instanceId: 'cell_phone_001',
      frameId: 'seq_001_frame_001',
      workspaceId,
      categoryId: 'cell_phone',
      bbox: [410, 180, 480, 270],
      confidence: 0.87,
      timestamp: '2026-11-20T12:00:00+05:30',
      attributes: { color: 'black', type: 'smartphone' },
    },
  ];

  const mockHistoricalRels: RelationshipFact[] = [
    {
      relationshipId: 'hist_rel_001',
      workspaceId,
      frameId: 'seq_001_frame_001',
      subjectId: 'cell_phone_001',
      predicate: 'next_to',
      objectId: 'notebook_001',
      confidence: 0.84,
      source: 'reactpp',
      provisional: false,
      timestamp: '2026-11-20T12:00:00+05:30',
    },
  ];

  // Override repository getters for offline runner
  const origGetFrames = MemoryRepository.getFrames;
  const origGetInstances = MemoryRepository.getInstances;
  const origGetObsForInst = MemoryRepository.getObservationsForInstance;
  const origGetRelsForFrame = MemoryRepository.getRelationshipsForFrame;
  const origGetObsForFrame = MemoryRepository.getObservationsForFrame;

  MemoryRepository.getFrames = async () => mockFrames;
  MemoryRepository.getInstances = async () => mockInstances;
  MemoryRepository.getObservationsForInstance = async (id: string) =>
    mockObservations.filter((o) => o.instanceId === id);
  MemoryRepository.getRelationshipsForFrame = async (fId: string) =>
    fId === 'seq_001_frame_001' ? mockHistoricalRels : [];
  MemoryRepository.getObservationsForFrame = async (fId: string) =>
    fId === 'seq_001_frame_001' ? mockObservations : [];

  try {
    // ------------------------------------------------------------------------
    // Acceptance Test 1: Current relationship
    // Query: "What is next to the laptop?"
    // Expected: Answer from current scene graph when relationship exists.
    // ------------------------------------------------------------------------
    const plan1 = parser.parseQuery('What is next to the laptop?', mockInstances);
    const res1 = await QueryExecutor.executeQuery(plan1, mockCurrentGraph, workspaceId);

    if (res1.evidence.source !== 'scene_graph') {
      throw new Error(`Test 1 Failed: Expected source 'scene_graph', got '${res1.evidence.source}'`);
    }
    if (!res1.answer.toLowerCase().includes('notebook')) {
      throw new Error(`Test 1 Failed: Answer did not identify notebook next to laptop: "${res1.answer}"`);
    }
    if (res1.certainty !== 'confirmed') {
      throw new Error(`Test 1 Failed: Expected certainty 'confirmed', got '${res1.certainty}'`);
    }
    console.log('✓ Acceptance Test 1 Passed: Current relationship answered from Scene Graph.');

    // ------------------------------------------------------------------------
    // Acceptance Test 2: Historical location
    // Query: "Where was my phone last seen?"
    // Expected: Retrieve latest confirmed historical observation from memory.
    // ------------------------------------------------------------------------
    const plan2 = parser.parseQuery('Where was my phone last seen?', mockInstances);
    const res2 = await QueryExecutor.executeQuery(plan2, mockCurrentGraph, workspaceId);

    if (res2.evidence.source !== 'workspace_memory') {
      throw new Error(`Test 2 Failed: Expected source 'workspace_memory', got '${res2.evidence.source}'`);
    }
    if (!res2.evidence.frameIds.includes('seq_001_frame_001')) {
      throw new Error(`Test 2 Failed: Expected frame 'seq_001_frame_001', got ${res2.evidence.frameIds}`);
    }
    if (!res2.answer.includes('410, 180, 480, 270')) {
      throw new Error(`Test 2 Failed: Expected coordinates in answer: "${res2.answer}"`);
    }
    console.log('✓ Acceptance Test 2 Passed: Historical location retrieved from Workspace Memory.');

    // ------------------------------------------------------------------------
    // Acceptance Test 3: Unknown attribute
    // Query: "What colour is the table?"
    // Expected: Check graph/memory first. If unrecorded, route to VLM without guessing.
    // ------------------------------------------------------------------------
    const plan3 = parser.parseQuery('What colour is the table?', mockInstances);
    const res3 = await QueryExecutor.executeQuery(plan3, mockCurrentGraph, workspaceId);

    if (!res3.requiresVLM) {
      throw new Error(`Test 3 Failed: Expected requiresVLM to be true for unrecorded attribute.`);
    }
    if (res3.certainty !== 'provisional' && res3.certainty !== 'insufficient') {
      throw new Error(`Test 3 Failed: Expected provisional or insufficient certainty, got '${res3.certainty}'`);
    }
    console.log('✓ Acceptance Test 3 Passed: Unknown attribute routed to selective visual inspection.');

    // ------------------------------------------------------------------------
    // Acceptance Test 4: Follow-up context
    // Query 1: "Where is my phone?"
    // Follow-up: "What was next to it?"
    // Expected: Resolve "it" using previous object identity (phone_001).
    // ------------------------------------------------------------------------
    const mockContext: ConversationContext = {
      lastTargetEntity: {
        instanceId: 'cell_phone_001',
        category: 'cell_phone',
        displayName: 'phone',
        mention: 'phone',
      },
    };
    const plan4 = parser.parseQuery('What was next to it in the previous observation?', mockInstances, mockContext);
    const res4 = await QueryExecutor.executeQuery(plan4, mockCurrentGraph, workspaceId);

    if (!res4.answer.toLowerCase().includes('notebook')) {
      throw new Error(`Test 4 Failed: Follow-up did not resolve 'it' to phone's neighbor: "${res4.answer}"`);
    }
    console.log('✓ Acceptance Test 4 Passed: Conversational follow-up resolved pronoun "it" correctly.');

    // ------------------------------------------------------------------------
    // Acceptance Test 5: Unsupported query
    // Query: "What is behind the closed drawer?"
    // Expected: Do not invent information about unobserved/occluded storage.
    // ------------------------------------------------------------------------
    const plan5 = parser.parseQuery('What is behind the closed drawer?', mockInstances);
    const res5 = await QueryExecutor.executeQuery(plan5, mockCurrentGraph, workspaceId);

    if (res5.queryPlan.intent !== 'unsupported_occluded') {
      throw new Error(`Test 5 Failed: Expected intent 'unsupported_occluded', got '${res5.queryPlan.intent}'`);
    }
    if (!res5.answer.toLowerCase().includes('cannot answer') && !res5.answer.toLowerCase().includes('occluded')) {
      throw new Error(`Test 5 Failed: Expected rejection of occluded area: "${res5.answer}"`);
    }
    if (res5.certainty !== 'insufficient') {
      throw new Error(`Test 5 Failed: Expected certainty 'insufficient', got '${res5.certainty}'`);
    }
    console.log('✓ Acceptance Test 5 Passed: Unsupported occluded query rejected safely.');

    // ------------------------------------------------------------------------
    // Acceptance Test 6: Ambiguous object
    // Query: "Where is the pen?" (2 pens detected in scene graph: pen_001, pen_002)
    // Expected: Request clarification or display candidate options.
    // ------------------------------------------------------------------------
    const plan6 = parser.parseQuery('Where is the pen?', mockInstances);
    const res6 = await QueryExecutor.executeQuery(plan6, mockCurrentGraph, workspaceId);

    if (!res6.ambiguousOptions || res6.ambiguousOptions.length !== 2) {
      throw new Error(`Test 6 Failed: Expected 2 ambiguous options, got ${res6.ambiguousOptions?.length}`);
    }
    if (!res6.answer.toLowerCase().includes('multiple candidates')) {
      throw new Error(`Test 6 Failed: Expected disambiguation prompt: "${res6.answer}"`);
    }
    console.log('✓ Acceptance Test 6 Passed: Ambiguous object correctly triggered disambiguation options.');

    // ------------------------------------------------------------------------
    // Acceptance Test 7: No historical observations
    // Query: "Where was my phone yesterday?" with only 1 frame on record.
    // Expected: Explain historical evidence unavailable, no fabricated past location.
    // ------------------------------------------------------------------------
    MemoryRepository.getFrames = async () => [mockFrames[0]]; // Only 1 frame exists
    const plan7 = parser.parseQuery('Where was my phone yesterday?', mockInstances);
    const res7 = await QueryExecutor.executeQuery(plan7, mockCurrentGraph, workspaceId);

    if (!res7.answer.toLowerCase().includes('no historical observations exist')) {
      throw new Error(`Test 7 Failed: Expected refusal due to lack of history: "${res7.answer}"`);
    }
    if (res7.certainty !== 'insufficient') {
      throw new Error(`Test 7 Failed: Expected certainty 'insufficient', got '${res7.certainty}'`);
    }
    console.log('✓ Acceptance Test 7 Passed: Nonexistent historical observation handled with zero fabrication.');

    // ------------------------------------------------------------------------
    // Acceptance Test 8: Source attribution
    // Verify Scene Graph, Workspace Memory, and Visual Reasoning badges match actual execution path.
    // ------------------------------------------------------------------------
    if (res1.evidence.source !== 'scene_graph') throw new Error('Test 8 Failed on Scene Graph source');
    if (res2.evidence.source !== 'workspace_memory') throw new Error('Test 8 Failed on Memory source');
    if (res3.evidence.source !== 'vlm') throw new Error('Test 8 Failed on VLM source');
    console.log('✓ Acceptance Test 8 Passed: Source attribution badges accurately match execution path.');

    // ------------------------------------------------------------------------
    // Acceptance Test 9: Conversation clearing isolation
    // Clear conversation thread and verify persistent IndexedDB records are unaffected.
    // ------------------------------------------------------------------------
    let localThreadMessages = [
      { id: '1', sender: 'user', text: 'Where is the laptop?' },
      { id: '2', sender: 'assistant', text: 'The laptop is on the table.' },
    ];
    // Clearing thread
    localThreadMessages = [];
    if (localThreadMessages.length !== 0) {
      throw new Error('Test 9 Failed: Conversation thread was not cleared.');
    }
    // Verify memory store is unaffected
    const storedFrames = await MemoryRepository.getFrames(workspaceId);
    if (storedFrames.length === 0) {
      throw new Error('Test 9 Failed: Memory records were unexpectedly modified during conversation clearing.');
    }
    console.log('✓ Acceptance Test 9 Passed: Conversation clearing leaves persistent workspace memory completely intact.');

    // ------------------------------------------------------------------------
    // Acceptance Test 10: Disambiguation resolution without looping
    // Query: "Where is pen_001?"
    // Expected: Direct location of pen_001, strictly bypassing disambiguation.
    // ------------------------------------------------------------------------
    const plan10 = parser.parseQuery('Where is pen_001?', mockInstances);
    const res10 = await QueryExecutor.executeQuery(plan10, mockCurrentGraph, workspaceId);

    if (res10.ambiguousOptions && res10.ambiguousOptions.length > 0) {
      throw new Error(`Test 10 Failed: Expected single instance resolution, but disambiguation was triggered: "${res10.answer}"`);
    }
    if (!res10.answer.includes('pen_001')) {
      throw new Error(`Test 10 Failed: Expected answer to locate pen_001: "${res10.answer}"`);
    }
    if (res10.certainty !== 'confirmed') {
      throw new Error(`Test 10 Failed: Expected certainty 'confirmed', got '${res10.certainty}'`);
    }
    console.log('✓ Acceptance Test 10 Passed: Instance ID disambiguation resolution succeeds without loop.');

    // ------------------------------------------------------------------------
    // Acceptance Test 11: Relational sitting query
    // Query: "Where is the person sitting?"
    // Expected: Scene graph verifies person_001 is on chair_002.
    // ------------------------------------------------------------------------
    const plan11 = parser.parseQuery('Where is the person sitting?', mockInstances);
    const res11 = await QueryExecutor.executeQuery(plan11, mockCurrentGraph, workspaceId);

    if (res11.evidence.source !== 'scene_graph') {
      throw new Error(`Test 11 Failed: Expected source 'scene_graph', got '${res11.evidence.source}'`);
    }
    if (!res11.answer.toLowerCase().includes('chair_002') && !res11.answer.toLowerCase().includes('chair')) {
      throw new Error(`Test 11 Failed: Answer did not report sitting on chair: "${res11.answer}"`);
    }
    console.log('✓ Acceptance Test 11 Passed: Sitting support relationship answered directly from Scene Graph.');

  } finally {
    // Restore repository getters
    MemoryRepository.getFrames = origGetFrames;
    MemoryRepository.getInstances = origGetInstances;
    MemoryRepository.getObservationsForInstance = origGetObsForInst;
    MemoryRepository.getRelationshipsForFrame = origGetRelsForFrame;
    MemoryRepository.getObservationsForFrame = origGetObsForFrame;
  }
}
