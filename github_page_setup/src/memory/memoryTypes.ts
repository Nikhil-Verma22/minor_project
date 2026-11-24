/**
 * Memory Types and Database Record Schemas.
 */

import { CanonicalPredicate } from '../config/predicates';
import { VisualCropDescriptor } from '../utils/visualDescriptor';

export interface WorkspaceRecord {
  workspaceId: string;
  domainId: string;
  name: string;
  createdAt: string;
}

export interface SequenceRecord {
  sequenceId: string;
  workspaceId: string;
  cameraId: string;
  viewpoint: string;
  createdAt: string;
}

export interface FrameRecord {
  frameId: string;
  sequenceId: string;
  workspaceId: string;
  frameIndex: number;
  timestamp: string;
  imageWidth: number;
  imageHeight: number;
  thumbnailDataUrl?: string;
  imageDataUrl?: string;
  filename?: string;
  status?: 'pending' | 'processing' | 'completed' | 'failed';
}

export interface PersistentObjectInstance {
  instanceId: string; // e.g. "laptop_001"
  workspaceId: string;
  categoryId: string; // one of the 39 categories or unmapped raw label
  displayName: string;
  attributes: {
    color: string | null;
    type: string | null;
  };
  firstSeenTimestamp: string;
  lastSeenTimestamp: string;
  firstSeenFrameId: string;
  lastSeenFrameId: string;
  status: 'active' | 'moved' | 'missing' | 'uncertain';
  latestVisualDescriptor?: VisualCropDescriptor;
  latestBbox?: [number, number, number, number];
}

export interface ObjectObservation {
  annotationId: string; // e.g. "ann_0001"
  frameId: string;
  instanceId: string;
  categoryId: string;
  bbox: [number, number, number, number] | null;
  visibility: 'visible' | 'occluded' | 'not_observed';
  attributes: {
    color: string | null;
    type: string | null;
  };
  confidence?: number;
  source: 'reactpp' | 'manual' | 'vlm';
  timestamp: string;
  visualDescriptor?: VisualCropDescriptor;
  associationConfidence?: number;
  associationReason?: string;
}

export interface RelationshipFact {
  relationshipId: string; // e.g. "rel_0001"
  frameId: string;
  workspaceId: string;
  subjectId: string; // persistent instanceId
  predicate: CanonicalPredicate;
  objectId: string; // persistent instanceId (e.g. "workspace_surface_001")
  confidence: number;
  source: 'reactpp' | 'vlm' | 'manual';
  timestamp: string;
  provisional: boolean;
}

export type GraphEventType =
  | 'first_seen'
  | 'moved'
  | 'relationship_added'
  | 'relationship_removed'
  | 'became_occluded'
  | 'reappeared'
  | 'uncertain_association';

export interface GraphEvent {
  eventId: string;
  workspaceId: string;
  timestamp: string;
  instanceId: string;
  eventType: GraphEventType;
  frameId: string;
  metadata?: Record<string, any>;
}

export interface VLMCandidate {
  candidateId: string;
  frameId: string;
  workspaceId: string;
  subjectId: string;
  freeTextPredicate: string;
  canonicalPredicate?: CanonicalPredicate;
  objectId: string;
  confidence: number;
  evidence: string;
  status: 'provisional' | 'confirmed' | 'rejected';
  timestamp: string;
}
