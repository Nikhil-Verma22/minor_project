/**
 * Graph Reasoner: Deterministic query operations over 2D Scene Graphs.
 */

import { SceneGraph, SceneGraphEdge, SceneGraphNode } from './graphTypes';
import { CanonicalPredicate } from '../config/predicates';

export interface TripletMatch {
  edge: SceneGraphEdge;
  subjectNode: SceneGraphNode;
  objectNode: SceneGraphNode;
}

export class GraphReasoner {
  /**
   * Find matching triplets in a scene graph with optional subject, predicate, object filters.
   */
  static findTriplets(
    graph: SceneGraph,
    filter: {
      subjectId?: string;
      subjectCategory?: string;
      predicate?: CanonicalPredicate;
      objectId?: string;
      objectCategory?: string;
    }
  ): TripletMatch[] {
    const nodeMap = new Map<string, SceneGraphNode>(graph.nodes.map((n) => [n.id, n]));
    const matches: TripletMatch[] = [];

    for (const edge of graph.edges) {
      if (filter.predicate && edge.predicate !== filter.predicate) continue;
      if (filter.subjectId && edge.sourceId !== filter.subjectId) continue;
      if (filter.objectId && edge.targetId !== filter.objectId) continue;

      const subjectNode = nodeMap.get(edge.sourceId);
      const objectNode = nodeMap.get(edge.targetId);

      if (!subjectNode || !objectNode) continue;

      if (filter.subjectCategory && subjectNode.categoryId !== filter.subjectCategory) continue;
      if (filter.objectCategory && objectNode.categoryId !== filter.objectCategory) continue;

      matches.push({
        edge,
        subjectNode,
        objectNode,
      });
    }

    return matches;
  }

  /**
   * Find all spatial neighbors of a node (e.g. next_to, on, in_front_of).
   */
  static getNeighbors(graph: SceneGraph, nodeId: string): { relation: CanonicalPredicate; neighbor: SceneGraphNode; direction: 'outgoing' | 'incoming' }[] {
    const nodeMap = new Map<string, SceneGraphNode>(graph.nodes.map((n) => [n.id, n]));
    const neighbors: { relation: CanonicalPredicate; neighbor: SceneGraphNode; direction: 'outgoing' | 'incoming' }[] = [];

    for (const edge of graph.edges) {
      if (edge.sourceId === nodeId) {
        const target = nodeMap.get(edge.targetId);
        if (target) {
          neighbors.push({ relation: edge.predicate, neighbor: target, direction: 'outgoing' });
        }
      } else if (edge.targetId === nodeId) {
        const source = nodeMap.get(edge.sourceId);
        if (source) {
          neighbors.push({ relation: edge.predicate, neighbor: source, direction: 'incoming' });
        }
      }
    }

    return neighbors;
  }
}
