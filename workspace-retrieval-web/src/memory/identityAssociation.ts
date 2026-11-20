/**
 * Identity Association Engine.
 * Associates incoming SGG detections with persistent physical object instances.
 */

import { SGGObjectDetection } from '../models/sgg/SGGEngine';
import { PersistentObjectInstance, ObjectObservation, GraphEvent } from './memoryTypes';
import { generateId, generateInstanceId } from '../utils/ids';
import { WORKSPACE_SURFACE_ENTITY } from '../config/workspaceDomains';

export interface AssociationResult {
  instanceMap: Map<string, PersistentObjectInstance>; // detectionId -> PersistentObjectInstance
  observations: ObjectObservation[];
  newInstances: PersistentObjectInstance[];
  updatedInstances: PersistentObjectInstance[];
  events: GraphEvent[];
}

function calculateIoU(boxA: [number, number, number, number], boxB: [number, number, number, number]): number {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);

  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

  const unionArea = boxAArea + boxBArea - interArea;
  return unionArea > 0 ? interArea / unionArea : 0;
}

function calculateCenterDistance(
  boxA: [number, number, number, number],
  boxB: [number, number, number, number],
  imgW: number,
  imgH: number
): number {
  const cAx = (boxA[0] + boxA[2]) / (2 * imgW);
  const cAy = (boxA[1] + boxA[3]) / (2 * imgH);
  const cBx = (boxB[0] + boxB[2]) / (2 * imgW);
  const cBy = (boxB[1] + boxB[3]) / (2 * imgH);

  return Math.hypot(cAx - cBx, cAy - cBy);
}

export class IdentityAssociation {
  /**
   * Associates current frame detections with existing persistent object instances in the workspace.
   */
  static associateDetections(
    detections: SGGObjectDetection[],
    existingInstances: PersistentObjectInstance[],
    latestObservationsMap: Map<string, ObjectObservation>, // instanceId -> latest ObjectObservation
    workspaceId: string,
    frameId: string,
    timestamp: string,
    imgW: number,
    imgH: number
  ): AssociationResult {
    const instanceMap = new Map<string, PersistentObjectInstance>();
    const observations: ObjectObservation[] = [];
    const newInstances: PersistentObjectInstance[] = [];
    const updatedInstances: PersistentObjectInstance[] = [];
    const events: GraphEvent[] = [];

    // Count existing instances per category to assign clean sequence numbers (e.g. laptop_001)
    const categoryCountMap = new Map<string, number>();
    for (const inst of existingInstances) {
      const count = categoryCountMap.get(inst.categoryId) || 0;
      categoryCountMap.set(inst.categoryId, count + 1);
    }

    const assignedInstanceIds = new Set<string>();

    for (const det of detections) {
      const effectiveCategory = det.projectCategory || det.modelLabel;

      // Special handling for workspace surface if detected as table/counter
      if (['table', 'dining table', 'counter', 'desk-stuff'].includes(det.modelLabel.toLowerCase())) {
        const surfaceInstance: PersistentObjectInstance = {
          instanceId: WORKSPACE_SURFACE_ENTITY.instanceId,
          workspaceId,
          categoryId: WORKSPACE_SURFACE_ENTITY.categoryId,
          displayName: WORKSPACE_SURFACE_ENTITY.displayName,
          attributes: { color: null, type: null },
          firstSeenTimestamp: timestamp,
          lastSeenTimestamp: timestamp,
          firstSeenFrameId: frameId,
          lastSeenFrameId: frameId,
          status: 'active',
        };
        instanceMap.set(det.detectionId, surfaceInstance);
        continue;
      }

      // Find compatible existing instances
      const candidates = existingInstances.filter(
        (inst) => inst.categoryId === effectiveCategory && !assignedInstanceIds.has(inst.instanceId)
      );

      let bestMatch: PersistentObjectInstance | null = null;
      let highestScore = 0;
      let isMoved = false;

      for (const cand of candidates) {
        const lastObs = latestObservationsMap.get(cand.instanceId);
        if (!lastObs || !lastObs.bbox) {
          // If no previous bbox, match by category if sole candidate
          if (candidates.length === 1) {
            bestMatch = cand;
            highestScore = 0.5;
          }
          continue;
        }

        const iou = calculateIoU(det.bbox, lastObs.bbox);
        const dist = calculateCenterDistance(det.bbox, lastObs.bbox, imgW, imgH);

        // Matching heuristic: high IoU or spatial proximity/sole candidate
        if (iou > 0.3) {
          if (iou > highestScore) {
            highestScore = iou;
            bestMatch = cand;
            isMoved = false;
          }
        } else if (dist < 0.45 || candidates.length === 1) {
          const score = Math.max(0.3, 1.0 - dist);
          if (score > highestScore) {
            highestScore = score;
            bestMatch = cand;
            isMoved = dist > 0.08; // Meaningful position shift
          }
        }
      }

      let targetInstance: PersistentObjectInstance;

      if (bestMatch && highestScore > 0.25) {
        // Reuse existing physical instance
        targetInstance = {
          ...bestMatch,
          lastSeenTimestamp: timestamp,
          lastSeenFrameId: frameId,
          status: 'active',
        };
        assignedInstanceIds.add(targetInstance.instanceId);
        updatedInstances.push(targetInstance);

        if (isMoved) {
          events.push({
            eventId: generateId('evt'),
            workspaceId,
            timestamp,
            instanceId: targetInstance.instanceId,
            eventType: 'moved',
            frameId,
            metadata: {
              newBbox: det.bbox,
              confidence: det.confidence,
            },
          });
        }
      } else {
        // Create brand new physical object instance
        const currentCount = categoryCountMap.get(effectiveCategory) || 0;
        const newCount = currentCount + 1;
        categoryCountMap.set(effectiveCategory, newCount);

        const newInstanceId = generateInstanceId(effectiveCategory, newCount);
        targetInstance = {
          instanceId: newInstanceId,
          workspaceId,
          categoryId: effectiveCategory,
          displayName: `${effectiveCategory.replace(/_/g, ' ')} (${newInstanceId})`,
          attributes: {
            color: det.attributes?.color ?? null,
            type: det.attributes?.type ?? null,
          },
          firstSeenTimestamp: timestamp,
          lastSeenTimestamp: timestamp,
          firstSeenFrameId: frameId,
          lastSeenFrameId: frameId,
          status: 'active',
        };

        assignedInstanceIds.add(newInstanceId);
        newInstances.push(targetInstance);

        events.push({
          eventId: generateId('evt'),
          workspaceId,
          timestamp,
          instanceId: newInstanceId,
          eventType: 'first_seen',
          frameId,
          metadata: {
            bbox: det.bbox,
            confidence: det.confidence,
          },
        });
      }

      instanceMap.set(det.detectionId, targetInstance);

      // Create object observation record
      observations.push({
        annotationId: generateId('ann'),
        frameId,
        instanceId: targetInstance.instanceId,
        categoryId: effectiveCategory,
        bbox: det.bbox,
        visibility: 'visible',
        attributes: targetInstance.attributes,
        confidence: det.confidence,
        source: 'reactpp',
        timestamp,
      });
    }

    return {
      instanceMap,
      observations,
      newInstances,
      updatedInstances,
      events,
    };
  }
}
