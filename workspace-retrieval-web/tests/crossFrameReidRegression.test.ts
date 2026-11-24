/**
 * Cross-Frame Physical Object Re-Identification Regression Test.
 *
 * Verifies that physical object identity is strictly preserved across sequential workspace observations:
 * - The brown ceramic mug retains persistent identity `cup_001` as it moves:
 *   Frame 007/008 (on laptop) -> Frame 009 (on book) -> Frame 010 (on desk).
 * - Clear glass container stays `cup_002` and pink pencil holder stays `cup_003`.
 * - No identity theft occurs (no phantom `cup_004` minted).
 * - Missed detection (`book_001` unobserved in Frame 009) is NOT marked as removed,
 *   and seamlessly re-associates when observed again in Frame 010.
 * - Scene graph edges contain zero self-loops, zero duplicate triplets, and gate low-confidence edges.
 */

import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import { fileURLToPath } from 'url';
import { IdentityAssociation } from '../src/memory/identityAssociation';
import { SGGObjectDetection, SGGRelation } from '../src/models/sgg/SGGEngine';
import { PersistentObjectInstance, ObjectObservation, RelationshipFact } from '../src/memory/memoryTypes';
import { decodePngSync, extractVisualDescriptorFromImageData } from '../src/utils/visualDescriptor';
import { GraphBuilder } from '../src/graph/graphBuilder';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export async function testCrossFrameReidRegression() {
  console.log('--- Testing Cross-Frame Physical Object Identity & Re-ID Regression ---');

  // 1. Load Fixture Memory State
  const fixturePath = path.resolve(__dirname, 'fixtures/workspace-memory-fixture.json');
  if (!fs.existsSync(fixturePath)) {
    throw new Error(`Regression fixture missing at ${fixturePath}`);
  }
  const fixtureData = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  const stores = fixtureData.stores;

  // Extract baseline state up through Frame 008 (excluding the bug-created cup_004)
  const baselineInstances: PersistentObjectInstance[] = stores.objectInstances.filter(
    (i: any) => i.instanceId !== 'cup_004'
  );

  // Decode baseline Frame 007 and Frame 008 images to seed visual descriptors
  const img007Path = path.resolve(__dirname, '../docs/demo_assets/Gemini_Generated_Image_vlxqd7vlxqd7vlxq.png');
  const img007Decoded = fs.existsSync(img007Path) ? decodePngSync(fs.readFileSync(img007Path), zlib) : null;

  const img008Path = path.resolve(__dirname, '../docs/demo_assets/Gemini_Generated_Image_9yjgvp9yjgvp9yjg.png');
  const img008Decoded = fs.existsSync(img008Path) ? decodePngSync(fs.readFileSync(img008Path), zlib) : null;

  const latestObsMap = new Map<string, ObjectObservation>();

  // Seed Frame 007 observations (where cup_003 was observed)
  const obs007: ObjectObservation[] = stores.objectObservations.filter(
    (o: any) => o.frameId === 'seq_001_frame_007'
  );
  for (const obs of obs007) {
    if (img007Decoded && obs.bbox) {
      obs.visualDescriptor = extractVisualDescriptorFromImageData(
        img007Decoded.data,
        img007Decoded.width,
        img007Decoded.height,
        obs.bbox
      );
    }
    latestObsMap.set(obs.instanceId, obs);
  }

  // Seed Frame 008 observations (overwriting with latest state)
  const obs008: ObjectObservation[] = stores.objectObservations.filter(
    (o: any) => o.frameId === 'seq_001_frame_008'
  );
  for (const obs of obs008) {
    if (img008Decoded && obs.bbox) {
      obs.visualDescriptor = extractVisualDescriptorFromImageData(
        img008Decoded.data,
        img008Decoded.width,
        img008Decoded.height,
        obs.bbox
      );
    }
    latestObsMap.set(obs.instanceId, obs);
  }

  // Also seed baseline instances with their latest visual descriptors and bboxes
  for (const inst of baselineInstances) {
    const obs = latestObsMap.get(inst.instanceId);
    if (obs?.visualDescriptor) {
      inst.latestVisualDescriptor = obs.visualDescriptor;
      inst.latestBbox = obs.bbox || undefined;
    }
  }

  console.log(`Loaded ${baselineInstances.length} baseline instances up to Frame 008.`);

  // 2. Decode Frame 009 (frame_1.png)
  const img009Path = path.resolve(__dirname, '../docs/demo_assets/frame_1.png');
  if (!fs.existsSync(img009Path)) {
    throw new Error(`Frame 009 image not found at ${img009Path}`);
  }
  const img009 = decodePngSync(fs.readFileSync(img009Path), zlib);

  // Detections for Frame 009
  // Note: pink holder at [366.6, ...], brown mug on book at [447.2, ...], glass at [1163, ...]
  const rawDetections009: Array<{ id: string; label: string; cat: string; bbox: [number, number, number, number] }> = [
    { id: 'det_9_0', label: 'laptop', cat: 'laptop', bbox: [626.5, 15.4, 1259.9, 546.1] },
    { id: 'det_9_1', label: 'wall', cat: 'wall', bbox: [0, 1.5, 1264, 415] },
    { id: 'det_9_2', label: 'vase', cat: 'vase', bbox: [187.2, 61.3, 330.9, 267.4] },
    { id: 'det_9_3', label: 'cup', cat: 'cup', bbox: [1163, 412.8, 1264, 515.5] }, // Clear glass
    { id: 'det_9_4', label: 'cup', cat: 'cup', bbox: [366.6, 177.8, 514.5, 326.5] }, // Pink pencil holder
    { id: 'det_9_5', label: 'cup', cat: 'cup', bbox: [447.2, 326.4, 629.4, 517.8] }, // Brown mug on book
  ];

  const dets009: SGGObjectDetection[] = rawDetections009.map((d) => {
    const desc = extractVisualDescriptorFromImageData(img009.data, img009.width, img009.height, d.bbox);
    return {
      detectionId: d.id,
      modelLabel: d.label,
      projectCategory: d.cat,
      bbox: d.bbox,
      confidence: 0.9,
      visualDescriptor: desc,
    };
  });

  // Execute Identity Association for Frame 009
  const res009 = IdentityAssociation.associateDetections(
    dets009,
    baselineInstances,
    latestObsMap,
    'ws_fixture',
    'seq_001_frame_009',
    '2026-11-24T00:00:09Z',
    img009.width,
    img009.height
  );

  const brownMug009 = res009.instanceMap.get('det_9_5')!;
  const pinkHolder009 = res009.instanceMap.get('det_9_4')!;
  const clearGlass009 = res009.instanceMap.get('det_9_3')!;

  console.log('Frame 009 Assignment:');
  console.log(`  Brown Mug (det_9_5):    ${brownMug009?.instanceId} (confidence: ${(dets009[5].associationConfidence || 0).toFixed(2)}, reason: ${dets009[5].associationReason})`);
  console.log(`  Pink Holder (det_9_4):  ${pinkHolder009?.instanceId} (confidence: ${(dets009[4].associationConfidence || 0).toFixed(2)}, reason: ${dets009[4].associationReason})`);
  console.log(`  Clear Glass (det_9_3):  ${clearGlass009?.instanceId} (confidence: ${(dets009[3].associationConfidence || 0).toFixed(2)}, reason: ${dets009[3].associationReason})`);

  // Assertions for Frame 009
  if (brownMug009.instanceId !== 'cup_001') {
    throw new Error(`CRITICAL REGRESSION in Frame 009: Brown mug assigned ${brownMug009.instanceId}, expected cup_001`);
  }
  if (clearGlass009.instanceId !== 'cup_002') {
    throw new Error(`CRITICAL REGRESSION in Frame 009: Clear glass assigned ${clearGlass009.instanceId}, expected cup_002`);
  }
  if (pinkHolder009.instanceId !== 'cup_003') {
    throw new Error(`CRITICAL REGRESSION in Frame 009: Pink pencil holder assigned ${pinkHolder009.instanceId}, expected cup_003`);
  }
  if (res009.newInstances.some((i) => i.categoryId === 'cup')) {
    throw new Error('CRITICAL REGRESSION: Phantom cup created in Frame 009');
  }

  // Update latestObsMap for Frame 009
  for (const obs of res009.observations) {
    latestObsMap.set(obs.instanceId, obs);
  }
  // Verify book_001 is preserved in memory despite being unobserved in Frame 009
  const bookInObs = latestObsMap.get('book_001');
  if (!bookInObs) {
    throw new Error('book_001 observation lost from memory map during Frame 009');
  }

  // Merge updated instances
  const currentInstancesMap = new Map<string, PersistentObjectInstance>();
  for (const inst of baselineInstances) currentInstancesMap.set(inst.instanceId, inst);
  for (const inst of res009.updatedInstances) currentInstancesMap.set(inst.instanceId, inst);
  for (const inst of res009.newInstances) currentInstancesMap.set(inst.instanceId, inst);

  // 3. Decode Frame 010 (frame_2.png)
  const img010Path = path.resolve(__dirname, '../docs/demo_assets/frame_2.png');
  if (!fs.existsSync(img010Path)) {
    throw new Error(`Frame 010 image not found at ${img010Path}`);
  }
  const img010 = decodePngSync(fs.readFileSync(img010Path), zlib);

  // Detections for Frame 010
  // Note: Brown mug moved to desk at [819.9, ...], pink holder at [366, ...], glass at [1163, ...], book at [331.3, ...]
  const rawDetections010: Array<{ id: string; label: string; cat: string; bbox: [number, number, number, number] }> = [
    { id: 'det_10_0', label: 'cup', cat: 'cup', bbox: [819.9, 479.7, 1014.7, 694.3] }, // Brown mug on desk
    { id: 'det_10_1', label: 'vase', cat: 'vase', bbox: [187.2, 60.9, 330.9, 267.6] },
    { id: 'det_10_2', label: 'book', cat: 'book', bbox: [331.3, 413.9, 748.6, 634.8] }, // Book reappears!
    { id: 'det_10_3', label: 'wall', cat: 'wall', bbox: [0, 1.6, 1264, 419.3] },
    { id: 'det_10_4', label: 'laptop', cat: 'laptop', bbox: [521.6, 15.7, 1260.3, 526] },
    { id: 'det_10_5', label: 'cup', cat: 'cup', bbox: [366, 177.3, 514.2, 326.7] }, // Pink pencil holder
    { id: 'det_10_6', label: 'cup', cat: 'cup', bbox: [1163, 412.4, 1264, 514.8] }, // Clear glass
  ];

  const dets010: SGGObjectDetection[] = rawDetections010.map((d) => {
    const desc = extractVisualDescriptorFromImageData(img010.data, img010.width, img010.height, d.bbox);
    return {
      detectionId: d.id,
      modelLabel: d.label,
      projectCategory: d.cat,
      bbox: d.bbox,
      confidence: 0.9,
      visualDescriptor: desc,
    };
  });

  const res010 = IdentityAssociation.associateDetections(
    dets010,
    Array.from(currentInstancesMap.values()),
    latestObsMap,
    'ws_fixture',
    'seq_001_frame_010',
    '2026-11-24T00:00:10Z',
    img010.width,
    img010.height
  );

  const brownMug010 = res010.instanceMap.get('det_10_0')!;
  const book010 = res010.instanceMap.get('det_10_2')!;
  const pinkHolder010 = res010.instanceMap.get('det_10_5')!;
  const clearGlass010 = res010.instanceMap.get('det_10_6')!;

  console.log('Frame 010 Assignment:');
  console.log(`  Brown Mug (det_10_0):    ${brownMug010?.instanceId} (confidence: ${(dets010[0].associationConfidence || 0).toFixed(2)}, reason: ${dets010[0].associationReason})`);
  console.log(`  Reappeared Book (det_10_2): ${book010?.instanceId} (confidence: ${(dets010[2].associationConfidence || 0).toFixed(2)}, reason: ${dets010[2].associationReason})`);
  console.log(`  Pink Holder (det_10_5):  ${pinkHolder010?.instanceId} (confidence: ${(dets010[5].associationConfidence || 0).toFixed(2)}, reason: ${dets010[5].associationReason})`);
  console.log(`  Clear Glass (det_10_6):  ${clearGlass010?.instanceId} (confidence: ${(dets010[6].associationConfidence || 0).toFixed(2)}, reason: ${dets010[6].associationReason})`);

  // Assertions for Frame 010
  if (brownMug010.instanceId !== 'cup_001') {
    throw new Error(`CRITICAL REGRESSION in Frame 010: Brown mug assigned ${brownMug010.instanceId}, expected cup_001`);
  }
  if (clearGlass010.instanceId !== 'cup_002') {
    throw new Error(`CRITICAL REGRESSION in Frame 010: Clear glass assigned ${clearGlass010.instanceId}, expected cup_002`);
  }
  if (pinkHolder010.instanceId !== 'cup_003') {
    throw new Error(`CRITICAL REGRESSION in Frame 010: Pink pencil holder assigned ${pinkHolder010.instanceId}, expected cup_003`);
  }
  if (book010.instanceId !== 'book_001') {
    throw new Error(`CRITICAL REGRESSION in Frame 010: Book assigned ${book010.instanceId}, expected book_001`);
  }
  if (res010.newInstances.some((i) => i.categoryId === 'cup')) {
    throw new Error('CRITICAL REGRESSION: Phantom cup_004 created in Frame 010');
  }

  // Check unique instance IDs within Frame 010
  const usedIds010 = new Set<string>();
  for (const det of dets010) {
    if (det.instanceId) {
      if (usedIds010.has(det.instanceId)) {
        throw new Error(`Duplicate instance ID '${det.instanceId}' assigned to multiple detections in Frame 010`);
      }
      usedIds010.add(det.instanceId);
    }
  }

  // 4. Test Graph Cleanliness (No self-loops, no duplicate triplets, gated provisional edges)
  const testRelationships: RelationshipFact[] = [
    {
      relationshipId: 'rel_1',
      frameId: 'seq_001_frame_010',
      workspaceId: 'ws_fixture',
      subjectId: 'workspace_surface_001',
      predicate: 'on',
      objectId: 'workspace_surface_001', // Self-loop!
      confidence: 0.85,
      source: 'reactpp',
      timestamp: '2026-11-24T00:00:10Z',
      provisional: false,
    },
    {
      relationshipId: 'rel_2',
      frameId: 'seq_001_frame_010',
      workspaceId: 'ws_fixture',
      subjectId: 'cup_001',
      predicate: 'on',
      objectId: 'workspace_surface_001',
      confidence: 0.88,
      source: 'reactpp',
      timestamp: '2026-11-24T00:00:10Z',
      provisional: false,
    },
    {
      relationshipId: 'rel_3',
      frameId: 'seq_001_frame_010',
      workspaceId: 'ws_fixture',
      subjectId: 'cup_001',
      predicate: 'on',
      objectId: 'workspace_surface_001', // Duplicate triplet!
      confidence: 0.88,
      source: 'reactpp',
      timestamp: '2026-11-24T00:00:10Z',
      provisional: false,
    },
    {
      relationshipId: 'rel_4',
      frameId: 'seq_001_frame_010',
      workspaceId: 'ws_fixture',
      subjectId: 'cup_001',
      predicate: 'next_to',
      objectId: 'laptop_001',
      confidence: 0.12, // Low confidence (< 0.20)
      source: 'reactpp',
      timestamp: '2026-11-24T00:00:10Z',
      provisional: false,
    },
  ];

  const graph = GraphBuilder.buildGraph(
    'seq_001_frame_010',
    '2026-11-24T00:00:10Z',
    res010.observations,
    testRelationships,
    currentInstancesMap
  );

  // Assert no self-loop edges
  const selfLoops = graph.edges.filter((e) => e.sourceId === e.targetId);
  if (selfLoops.length > 0) {
    throw new Error(`Found ${selfLoops.length} invalid self-loop edges in SceneGraph`);
  }

  // Assert no duplicate triplets
  const edgeKeys = graph.edges.map((e) => `${e.sourceId}__${e.predicate}__${e.targetId}`);
  const uniqueEdgeKeys = new Set(edgeKeys);
  if (edgeKeys.length !== uniqueEdgeKeys.size) {
    throw new Error(`SceneGraph contains duplicate relationship triplets: ${edgeKeys.join(', ')}`);
  }

  // Assert low-confidence edge is marked provisional
  const lowConfEdge = graph.edges.find((e) => e.predicate === 'next_to');
  if (!lowConfEdge || !lowConfEdge.provisional) {
    throw new Error(`Low confidence edge was not marked provisional: ${JSON.stringify(lowConfEdge)}`);
  }

  console.log('✓ Cross-Frame Re-ID Regression Test Passed! (cup_001 persistently preserved, zero self-loops, clean graph facts)');
}
