/**
 * Authoritative Frame Selection & Data Isolation Engine.
 *
 * Enforces strict separation between:
 * 1. Persistent Workspace Memory (historical observations and records in IndexedDB)
 * 2. Active Selected-Frame Context (strictly filtered to currently selected frame IDs)
 *
 * Guarantees that unselected frames NEVER contaminate the active Detected Objects,
 * 2D Scene Graph, or query execution context.
 */

import { SGGObjectDetection, SGGRelation } from '../models/sgg/SGGEngine';
import { SceneGraph } from '../graph/graphTypes';

export interface FrameObservationData {
  frameId: string;
  frameIndex: number;
  timestamp: string;
  imageSrc: string;
  imgWidth: number;
  imgHeight: number;
  objects: SGGObjectDetection[];
  relationships: SGGRelation[];
}

export interface AuthoritativeFrameSelection {
  selectedFrames: FrameObservationData[];
  selectedFrameIds: string[];
  detectionsByFrame: Map<string, SGGObjectDetection[]>;
  selectedDetections: SGGObjectDetection[];
  relationshipsByFrame: Map<string, SGGRelation[]>;
  selectedRelationships: SGGRelation[];
  selectedInstanceIds: Set<string>;
  combinedSceneGraph: SceneGraph | null;
  isEmpty: boolean;
}

/**
 * Computes the single authoritative data context for all downstream UI components.
 * If selectedFrameIds is empty, returns an empty context with null graph and empty detection lists.
 */
export function getSelectedFrameData(
  frameCache: Map<string, FrameObservationData>,
  selectedFrameIds: string[]
): AuthoritativeFrameSelection {
  if (!selectedFrameIds || selectedFrameIds.length === 0) {
    return {
      selectedFrames: [],
      selectedFrameIds: [],
      detectionsByFrame: new Map(),
      selectedDetections: [],
      relationshipsByFrame: new Map(),
      selectedRelationships: [],
      selectedInstanceIds: new Set(),
      combinedSceneGraph: null,
      isEmpty: true,
    };
  }

  // Preserve the selection order
  const selectedFrames: FrameObservationData[] = [];
  for (const id of selectedFrameIds) {
    const f = frameCache.get(id);
    if (f) {
      selectedFrames.push(f);
    }
  }

  if (selectedFrames.length === 0) {
    return {
      selectedFrames: [],
      selectedFrameIds,
      detectionsByFrame: new Map(),
      selectedDetections: [],
      relationshipsByFrame: new Map(),
      selectedRelationships: [],
      selectedInstanceIds: new Set(),
      combinedSceneGraph: null,
      isEmpty: true,
    };
  }

  const detectionsByFrame = new Map<string, SGGObjectDetection[]>();
  const relationshipsByFrame = new Map<string, SGGRelation[]>();
  const selectedDetections: SGGObjectDetection[] = [];
  const selectedRelationships: SGGRelation[] = [];
  const selectedInstanceIds = new Set<string>();

  for (const sf of selectedFrames) {
    const frameDets: SGGObjectDetection[] = [];
    for (const obj of sf.objects) {
      const taggedObj: SGGObjectDetection = {
        ...obj,
        frameId: sf.frameId,
      };
      frameDets.push(taggedObj);
      selectedDetections.push(taggedObj);

      if (taggedObj.instanceId) {
        selectedInstanceIds.add(taggedObj.instanceId);
      } else {
        selectedInstanceIds.add(taggedObj.detectionId);
      }
    }
    detectionsByFrame.set(sf.frameId, frameDets);

    const frameRels: SGGRelation[] = [];
    for (const rel of sf.relationships) {
      const taggedRel: SGGRelation = {
        ...rel,
        frameId: sf.frameId,
      };
      frameRels.push(taggedRel);
      selectedRelationships.push(taggedRel);
    }
    relationshipsByFrame.set(sf.frameId, frameRels);
  }

  // Build SceneGraph strictly from selected frames
  let combinedSceneGraph: SceneGraph | null = null;

  if (selectedFrames.length === 1) {
    const sf = selectedFrames[0];
    const nodesMap = new Map();
    const detToIdMap = new Map<string, string>();

    for (const obj of sf.objects) {
      const id = obj.instanceId || obj.detectionId;
      detToIdMap.set(obj.detectionId, id);
      if (obj.instanceId) detToIdMap.set(obj.instanceId, id);

      nodesMap.set(id, {
        id,
        instanceId: id,
        detectionId: obj.detectionId,
        categoryId: obj.projectCategory || obj.modelLabel,
        label: obj.modelLabel,
        bbox: obj.bbox,
        confidence: obj.confidence,
        isSurface: false,
        attributes: obj.attributes || { color: null, type: null },
      });
    }

    const edges: any[] = [];
    const seenEdgeKeys = new Set<string>();

    for (const r of sf.relationships) {
      const srcId = detToIdMap.get(r.subjectDetectionId) || r.subjectDetectionId;
      const tgtId = detToIdMap.get(r.objectDetectionId) || r.objectDetectionId;

      // Block self-loops
      if (srcId === tgtId) {
        continue;
      }

      const edgeKey = `${srcId}__${r.canonicalPredicate}__${tgtId}`;
      if (seenEdgeKeys.has(edgeKey)) {
        continue;
      }
      seenEdgeKeys.add(edgeKey);

      edges.push({
        id: r.relationId,
        sourceId: srcId,
        targetId: tgtId,
        predicate: r.canonicalPredicate,
        confidence: r.confidence,
        source: r.source,
        provisional: r.confidence < 0.20 || Boolean(r.provisional),
        frameId: sf.frameId,
        observedFrames: [sf.frameId],
      });
    }

    combinedSceneGraph = {
      frameId: sf.frameId,
      timestamp: sf.timestamp,
      nodes: Array.from(nodesMap.values()),
      edges,
    };
  } else {
    // Multi-frame union with source attribution
    const nodesMap = new Map();
    const edgeMap = new Map();
    const detToIdMap = new Map<string, string>();

    for (const sf of selectedFrames) {
      for (const obj of sf.objects) {
        const id = obj.instanceId || obj.detectionId;
        detToIdMap.set(obj.detectionId, id);
        if (obj.instanceId) detToIdMap.set(obj.instanceId, id);

        if (!nodesMap.has(id)) {
          nodesMap.set(id, {
            id,
            instanceId: id,
            detectionId: obj.detectionId,
            categoryId: obj.projectCategory || obj.modelLabel,
            label: obj.modelLabel,
            bbox: obj.bbox,
            confidence: obj.confidence,
            isSurface: false,
            attributes: obj.attributes || { color: null, type: null },
          });
        }
      }

      for (const r of sf.relationships) {
        const srcId = detToIdMap.get(r.subjectDetectionId) || r.subjectDetectionId;
        const tgtId = detToIdMap.get(r.objectDetectionId) || r.objectDetectionId;

        // Block self-loops
        if (srcId === tgtId) {
          continue;
        }

        const key = `${srcId}__${r.canonicalPredicate}__${tgtId}`;
        if (edgeMap.has(key)) {
          const existing = edgeMap.get(key);
          if (existing.observedFrames && !existing.observedFrames.includes(sf.frameId)) {
            existing.observedFrames.push(sf.frameId);
          }
        } else {
          edgeMap.set(key, {
            id: r.relationId,
            sourceId: srcId,
            targetId: tgtId,
            predicate: r.canonicalPredicate,
            confidence: r.confidence,
            source: r.source,
            provisional: r.confidence < 0.20 || Boolean(r.provisional),
            frameId: sf.frameId,
            observedFrames: [sf.frameId],
          });
        }
      }
    }

    combinedSceneGraph = {
      frameId: 'combined_selection',
      timestamp: selectedFrames[selectedFrames.length - 1].timestamp,
      nodes: Array.from(nodesMap.values()),
      edges: Array.from(edgeMap.values()),
    };
  }

  return {
    selectedFrames,
    selectedFrameIds,
    detectionsByFrame,
    selectedDetections,
    relationshipsByFrame,
    selectedRelationships,
    selectedInstanceIds,
    combinedSceneGraph,
    isEmpty: false,
  };
}

