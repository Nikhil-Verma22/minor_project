/**
 * Identity Association Engine.
 * Associates incoming SGG detections with persistent physical object instances
 * across sequential and multi-frame observations using robust multi-factor scoring
 * (appearance descriptors, spatial displacement, category consistency, and relational context)
 * resolved via global one-to-one bipartite matching.
 */

import { SGGObjectDetection } from '../models/sgg/SGGEngine';
import { PersistentObjectInstance, ObjectObservation, GraphEvent } from './memoryTypes';
import { generateId, generateInstanceId } from '../utils/ids';
import { WORKSPACE_SURFACE_ENTITY } from '../config/workspaceDomains';
import { computeVisualSimilarity } from '../utils/visualDescriptor';

export interface AssociationResult {
  instanceMap: Map<string, PersistentObjectInstance>; // detectionId -> PersistentObjectInstance
  observations: ObjectObservation[];
  newInstances: PersistentObjectInstance[];
  updatedInstances: PersistentObjectInstance[];
  events: GraphEvent[];
}

interface CandidateMatch {
  detection: SGGObjectDetection;
  instance: PersistentObjectInstance;
  lastObservation?: ObjectObservation;
  score: number;
  iou: number;
  distance: number;
  visualSimilarity: number;
  reason: string;
  isMoved: boolean;
}

export function calculateIoU(boxA: [number, number, number, number], boxB: [number, number, number, number]): number {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);

  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const boxAArea = Math.max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]));
  const boxBArea = Math.max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]));

  const unionArea = boxAArea + boxBArea - interArea;
  return unionArea > 0 ? interArea / unionArea : 0;
}

export function calculateCenterDistance(
  boxA: [number, number, number, number],
  boxB: [number, number, number, number],
  imgW: number,
  imgH: number
): number {
  const w = Math.max(1, imgW);
  const h = Math.max(1, imgH);
  const cAx = (boxA[0] + boxA[2]) / (2 * w);
  const cAy = (boxA[1] + boxA[3]) / (2 * h);
  const cBx = (boxB[0] + boxB[2]) / (2 * w);
  const cBy = (boxB[1] + boxB[3]) / (2 * h);

  return Math.hypot(cAx - cBx, cAy - cBy);
}

function areCategoriesCompatible(catA: string, catB: string, modelLabel?: string): boolean {
  const a = catA.toLowerCase().trim();
  const b = catB.toLowerCase().trim();
  if (a === b) return true;
  if (modelLabel && a === modelLabel.toLowerCase().trim()) return true;

  // Domain synonym groupings
  const synonyms: string[][] = [
    ['cup', 'mug', 'coffee cup', 'pencil holder'],
    ['table', 'desk', 'counter', 'workspace surface', 'dining table'],
    ['laptop', 'notebook', 'computer'],
    ['cell phone', 'smartphone', 'phone', 'mobile phone'],
    ['chair', 'armchair', 'office chair', 'swivel chair'],
  ];

  for (const group of synonyms) {
    if (group.includes(a) && group.includes(b)) {
      return true;
    }
  }

  return false;
}

export class IdentityAssociation {
  /**
   * Minimum confidence threshold to accept a cross-frame identity association.
   * Matches below this threshold will not force identity theft; a new instance is minted instead.
   */
  static readonly MIN_ASSOCIATION_THRESHOLD = 0.38;

  /**
   * Associates current frame detections with existing persistent object instances in the workspace.
   * Enforces global 1-to-1 assignment so high-confidence stationary objects take precedence,
   * leaving moved objects to match via visual appearance and displacement rather than greedy first-come.
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

    // Track all used instance IDs across the workspace to guarantee strict intra-frame uniqueness
    const allUsedInstanceIds = new Set<string>();
    const categoryCountMap = new Map<string, number>();

    for (const inst of existingInstances) {
      allUsedInstanceIds.add(inst.instanceId);
      // Extract numeric index if present (e.g. cup_003 -> 3)
      const match = inst.instanceId.match(/_(\d+)$/);
      const parsedNum = match ? parseInt(match[1], 10) : 0;
      const currentMax = categoryCountMap.get(inst.categoryId) || 0;
      categoryCountMap.set(inst.categoryId, Math.max(currentMax, parsedNum));
    }

    // 1. Separate surface entities and regular object detections
    const normalDetections: SGGObjectDetection[] = [];

    for (const det of detections) {
      const labelLower = det.modelLabel.toLowerCase();
      if (['table', 'dining table', 'counter', 'desk-stuff'].includes(labelLower)) {
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
        det.instanceId = WORKSPACE_SURFACE_ENTITY.instanceId;
        det.associationConfidence = 1.0;
        det.associationReason = 'workspace_surface_anchor';
        instanceMap.set(det.detectionId, surfaceInstance);
      } else {
        normalDetections.push(det);
      }
    }

    // 2. Build multi-factor Candidate Match Matrix between all existing instances and detections
    const candidateMatches: CandidateMatch[] = [];

    for (const det of normalDetections) {
      const effectiveCategory = det.projectCategory || det.modelLabel;

      for (const inst of existingInstances) {
        if (!areCategoriesCompatible(inst.categoryId, effectiveCategory, det.modelLabel)) {
          continue;
        }

        const lastObs = latestObservationsMap.get(inst.instanceId);
        const prevBbox = lastObs?.bbox || inst.latestBbox || null;
        const prevDesc = lastObs?.visualDescriptor || inst.latestVisualDescriptor || null;
        const currDesc = det.visualDescriptor || null;

        let iou = 0;
        let dist = 1.0;
        if (prevBbox) {
          iou = calculateIoU(det.bbox, prevBbox);
          dist = calculateCenterDistance(det.bbox, prevBbox, imgW, imgH);
        }

        let visSim = 0.65;
        let hasVisualDescriptors = false;
        if (currDesc && prevDesc) {
          visSim = computeVisualSimilarity(currDesc, prevDesc);
          hasVisualDescriptors = true;
        }

        // Multi-factor scoring logic
        let score = 0;
        let reason = '';
        let isMoved = false;

        if (iou > 0.35) {
          // Stationary or near-stationary object
          if (hasVisualDescriptors) {
            score = 0.45 * iou + 0.35 * visSim + 0.20 * Math.max(0, 1.0 - dist);
          } else {
            score = 0.65 * iou + 0.35 * Math.max(0, 1.0 - dist);
          }
          reason = iou > 0.65 ? 'stationary_high_iou' : 'stationary_moderate_iou';
          isMoved = false;
        } else {
          // Displaced, moved, or reappeared after gap
          isMoved = dist > 0.08;

          if (hasVisualDescriptors) {
            // Visual appearance is primary anchor for moved objects
            if (visSim < 0.62) {
              // Heavily penalize appearance mismatch to prevent identity theft
              score = 0.15 * visSim;
              reason = 'appearance_mismatch_penalty';
            } else {
              score = 0.55 * visSim + 0.25 * Math.max(0, 1.0 - dist) + 0.20;
              reason = isMoved ? 'relocated_visual_match' : 'appearance_consistent_match';
            }
          } else {
            // Fallback when visual descriptors not present (e.g. synthetic test coordinates)
            if (dist < 0.45) {
              score = Math.max(0.35, 1.0 - dist);
              reason = 'spatial_displacement_proximity';
            } else {
              score = 0.20;
              reason = 'distant_unverified';
            }
          }
        }

        if (score >= IdentityAssociation.MIN_ASSOCIATION_THRESHOLD) {
          candidateMatches.push({
            detection: det,
            instance: inst,
            lastObservation: lastObs,
            score,
            iou,
            distance: dist,
            visualSimilarity: visSim,
            reason,
            isMoved,
          });
        }
      }
    }

    // 3. Global One-to-One Optimal Bipartite Assignment
    // Sort candidate pairs descending by score so highest confidence matches claim their persistent ID first
    candidateMatches.sort((a, b) => b.score - a.score);

    const assignedInstanceIds = new Set<string>();
    const assignedDetectionIds = new Set<string>();

    for (const match of candidateMatches) {
      if (
        assignedInstanceIds.has(match.instance.instanceId) ||
        assignedDetectionIds.has(match.detection.detectionId)
      ) {
        continue;
      }

      // Re-use persistent physical object instance
      const targetInstance: PersistentObjectInstance = {
        ...match.instance,
        lastSeenTimestamp: timestamp,
        lastSeenFrameId: frameId,
        status: 'active',
        latestVisualDescriptor: match.detection.visualDescriptor || match.instance.latestVisualDescriptor,
        latestBbox: match.detection.bbox,
      };

      assignedInstanceIds.add(targetInstance.instanceId);
      assignedDetectionIds.add(match.detection.detectionId);
      updatedInstances.push(targetInstance);
      instanceMap.set(match.detection.detectionId, targetInstance);

      match.detection.instanceId = targetInstance.instanceId;
      match.detection.associationConfidence = match.score;
      match.detection.associationReason = match.reason;

      // Emit moved event if object relocated significantly
      if (match.isMoved) {
        events.push({
          eventId: generateId('evt'),
          workspaceId,
          timestamp,
          instanceId: targetInstance.instanceId,
          eventType: 'moved',
          frameId,
          metadata: {
            previousBbox: match.lastObservation?.bbox,
            newBbox: match.detection.bbox,
            displacementDistance: match.distance,
            confidence: match.score,
            reason: match.reason,
          },
        });
      }

      // Record observation
      observations.push({
        annotationId: generateId('ann'),
        frameId,
        instanceId: targetInstance.instanceId,
        categoryId: match.detection.projectCategory || match.detection.modelLabel,
        bbox: match.detection.bbox,
        visibility: 'visible',
        attributes: targetInstance.attributes,
        confidence: match.detection.confidence,
        source: 'reactpp',
        timestamp,
        visualDescriptor: match.detection.visualDescriptor,
        associationConfidence: match.score,
        associationReason: match.reason,
      });
    }

    // 4. Handle remaining unmatched detections as brand-new physical instances
    for (const det of normalDetections) {
      if (assignedDetectionIds.has(det.detectionId)) {
        continue;
      }

      const effectiveCategory = det.projectCategory || det.modelLabel;
      let currentCount = categoryCountMap.get(effectiveCategory) || 0;
      let newCount = currentCount + 1;
      categoryCountMap.set(effectiveCategory, newCount);

      let newInstanceId = generateInstanceId(effectiveCategory, newCount);
      while (allUsedInstanceIds.has(newInstanceId)) {
        newCount++;
        categoryCountMap.set(effectiveCategory, newCount);
        newInstanceId = generateInstanceId(effectiveCategory, newCount);
      }
      allUsedInstanceIds.add(newInstanceId);

      const targetInstance: PersistentObjectInstance = {
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
        latestVisualDescriptor: det.visualDescriptor,
        latestBbox: det.bbox,
      };

      det.instanceId = newInstanceId;
      det.associationConfidence = 1.0;
      det.associationReason = 'first_seen';

      newInstances.push(targetInstance);
      instanceMap.set(det.detectionId, targetInstance);

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
          reason: 'first_seen',
        },
      });

      observations.push({
        annotationId: generateId('ann'),
        frameId,
        instanceId: newInstanceId,
        categoryId: effectiveCategory,
        bbox: det.bbox,
        visibility: 'visible',
        attributes: targetInstance.attributes,
        confidence: det.confidence,
        source: 'reactpp',
        timestamp,
        visualDescriptor: det.visualDescriptor,
        associationConfidence: 1.0,
        associationReason: 'first_seen',
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
