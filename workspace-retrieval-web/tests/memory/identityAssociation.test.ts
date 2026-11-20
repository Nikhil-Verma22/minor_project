/**
 * Tests for IdentityAssociation.
 * Verifies spatial IoU matching, moved event generation, and new instance ID assignment.
 */

import { IdentityAssociation } from '../../src/memory/identityAssociation';
import { SGGObjectDetection } from '../../src/models/sgg/SGGEngine';
import { PersistentObjectInstance, ObjectObservation } from '../../src/memory/memoryTypes';

export function testIdentityAssociation() {
  console.log('--- Testing IdentityAssociation ---');

  const workspaceId = 'ws_test';
  const frame1 = 'seq_001_frame_001';
  const frame2 = 'seq_001_frame_002';
  const timestamp1 = '2026-11-20T12:00:00+05:30';
  const timestamp2 = '2026-11-20T12:05:00+05:30';

  // 1. Frame 1: Empty initial memory, process two detections
  const detectionsF1: SGGObjectDetection[] = [
    {
      detectionId: 'det_0',
      modelLabel: 'laptop',
      projectCategory: 'laptop',
      bbox: [100, 100, 300, 300],
      confidence: 0.9,
    },
    {
      detectionId: 'det_1',
      modelLabel: 'cell phone',
      projectCategory: 'smartphone',
      bbox: [400, 200, 480, 280],
      confidence: 0.85,
    },
  ];

  const res1 = IdentityAssociation.associateDetections(
    detectionsF1,
    [],
    new Map(),
    workspaceId,
    frame1,
    timestamp1,
    640,
    480
  );

  if (res1.newInstances.length !== 2) {
    throw new Error(`Expected 2 new instances in Frame 1, got ${res1.newInstances.length}`);
  }
  const laptopInstance = res1.newInstances.find((i) => i.categoryId === 'laptop')!;
  if (laptopInstance.instanceId !== 'laptop_001') {
    throw new Error(`Expected instanceId='laptop_001', got ${laptopInstance.instanceId}`);
  }

  // 2. Frame 2: Laptop stays in place (high IoU), phone moves noticeably
  const detectionsF2: SGGObjectDetection[] = [
    {
      detectionId: 'det_2',
      modelLabel: 'laptop',
      projectCategory: 'laptop',
      bbox: [102, 98, 301, 302], // Almost identical bbox
      confidence: 0.92,
    },
    {
      detectionId: 'det_3',
      modelLabel: 'cell phone',
      projectCategory: 'smartphone',
      bbox: [500, 350, 580, 430], // Moved across desk
      confidence: 0.88,
    },
  ];

  const latestObsMap = new Map<string, ObjectObservation>();
  for (const obs of res1.observations) {
    latestObsMap.set(obs.instanceId, obs);
  }

  const res2 = IdentityAssociation.associateDetections(
    detectionsF2,
    res1.newInstances,
    latestObsMap,
    workspaceId,
    frame2,
    timestamp2,
    640,
    480
  );

  // Both should be recognized as existing instances rather than newly minted IDs
  const assignedLaptop = res2.instanceMap.get('det_2')!;
  if (assignedLaptop.instanceId !== 'laptop_001') {
    throw new Error(`Expected reused instanceId='laptop_001', got ${assignedLaptop.instanceId}`);
  }

  // Check moved event for phone
  const movedEvent = res2.events.find((e) => e.eventType === 'moved');
  if (!movedEvent) {
    throw new Error('Expected moved event for displaced phone, but none was triggered');
  }

  console.log('✓ IdentityAssociation tests passed!');
}
