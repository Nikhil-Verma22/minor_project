/**
 * 2D Scene Graph Types.
 */

import { CanonicalPredicate } from '../config/predicates';

export interface SceneGraphNode {
  id: string; // persistent instanceId (e.g. laptop_001, workspace_surface_001)
  instanceId: string;
  categoryId: string;
  label: string;
  bbox: [number, number, number, number] | null;
  confidence?: number;
  isSurface: boolean;
  attributes: {
    color: string | null;
    type: string | null;
  };
}

export interface SceneGraphEdge {
  id: string;
  sourceId: string; // subject instanceId
  targetId: string; // object instanceId
  predicate: CanonicalPredicate;
  confidence: number;
  source: 'reactpp' | 'vlm' | 'manual';
  provisional: boolean;
}

export interface SceneGraph {
  frameId: string;
  timestamp: string;
  nodes: SceneGraphNode[];
  edges: SceneGraphEdge[];
}
