/**
 * Scene Graph Builder.
 * Constructs typed 2D Scene Graphs from memory observations and relationship facts.
 */

import { SceneGraph, SceneGraphNode, SceneGraphEdge } from './graphTypes';
import { ObjectObservation, RelationshipFact, PersistentObjectInstance } from '../memory/memoryTypes';
import { WORKSPACE_SURFACE_ENTITY } from '../config/workspaceDomains';

export class GraphBuilder {
  static buildGraph(
    frameId: string,
    timestamp: string,
    observations: ObjectObservation[],
    relationships: RelationshipFact[],
    instancesMap: Map<string, PersistentObjectInstance>
  ): SceneGraph {
    const nodesMap = new Map<string, SceneGraphNode>();

    for (const obs of observations) {
      const inst = instancesMap.get(obs.instanceId);
      const isSurface = obs.instanceId === WORKSPACE_SURFACE_ENTITY.instanceId;

      nodesMap.set(obs.instanceId, {
        id: obs.instanceId,
        instanceId: obs.instanceId,
        detectionId: obs.annotationId,
        categoryId: obs.categoryId,
        label: inst?.displayName || obs.categoryId,
        bbox: obs.bbox,
        confidence: obs.confidence,
        isSurface,
        attributes: obs.attributes,
      });
    }

    // Ensure workspace surface is present if relationships reference it
    const hasSurfaceRelation = relationships.some(
      (r) => r.subjectId === WORKSPACE_SURFACE_ENTITY.instanceId || r.objectId === WORKSPACE_SURFACE_ENTITY.instanceId
    );

    if (hasSurfaceRelation && !nodesMap.has(WORKSPACE_SURFACE_ENTITY.instanceId)) {
      nodesMap.set(WORKSPACE_SURFACE_ENTITY.instanceId, {
        id: WORKSPACE_SURFACE_ENTITY.instanceId,
        instanceId: WORKSPACE_SURFACE_ENTITY.instanceId,
        categoryId: WORKSPACE_SURFACE_ENTITY.categoryId,
        label: WORKSPACE_SURFACE_ENTITY.displayName,
        bbox: null,
        isSurface: true,
        attributes: { color: null, type: null },
      });
    }

    const edges: SceneGraphEdge[] = [];
    const seenEdges = new Set<string>();

    for (const rel of relationships) {
      // Eliminate self-loops: e.g. workspace_surface_001 -> on -> workspace_surface_001
      if (rel.subjectId === rel.objectId) {
        continue;
      }

      // Deduplicate relationship triplets in this graph view
      const edgeKey = `${rel.subjectId}__${rel.predicate}__${rel.objectId}`;
      if (seenEdges.has(edgeKey)) {
        continue;
      }
      seenEdges.add(edgeKey);

      edges.push({
        id: rel.relationshipId,
        sourceId: rel.subjectId,
        targetId: rel.objectId,
        predicate: rel.predicate,
        confidence: rel.confidence,
        source: rel.source,
        provisional: rel.confidence < 0.20 || Boolean(rel.provisional),
        frameId: rel.frameId || frameId,
      });
    }

    return {
      frameId,
      timestamp,
      nodes: Array.from(nodesMap.values()),
      edges,
    };
  }
}
