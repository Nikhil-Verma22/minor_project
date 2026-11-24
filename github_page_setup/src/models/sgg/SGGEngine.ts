/**
 * SGG Engine Interface & Data Contracts.
 */

import { CanonicalPredicate } from '../../config/predicates';
import { VisualCropDescriptor } from '../../utils/visualDescriptor';

export interface SGGObjectDetection {
  detectionId: string;
  instanceId?: string;
  frameId?: string;
  modelLabel: string;
  projectCategory?: string | null;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  attributes?: {
    color?: string | null;
    type?: string | null;
  };
  featureRef?: string;
  visualDescriptor?: VisualCropDescriptor;
  associationConfidence?: number;
  associationReason?: string;
}

export interface SGGRelation {
  relationId: string;
  subjectDetectionId: string;
  predicateModelLabel: string;
  canonicalPredicate?: CanonicalPredicate;
  objectDetectionId: string;
  confidence: number;
  source: 'reactpp' | 'vlm' | 'manual' | 'user';
  provisional: boolean;
  frameId?: string;
}

export interface SGGInferenceResult {
  frameId: string;
  modelId: string;
  modelVariant: string;
  executionProvider: 'webgpu' | 'wasm' | 'local_service';
  imageWidth: number;
  imageHeight: number;
  objects: SGGObjectDetection[];
  relationships: SGGRelation[];
  timings: {
    preprocessMs: number;
    inferenceMs: number;
    postprocessMs: number;
    totalMs: number;
  };
}

export interface ModelStatus {
  loaded: boolean;
  provider: 'webgpu' | 'wasm' | 'local_service' | 'unavailable';
  variant: string;
  error?: string;
  initTimeMs?: number;
}

export interface ModelInfo {
  name: string;
  variant: string;
  provider: 'webgpu' | 'wasm' | 'local_service';
  objectClassesCount: number;
  relationClassesCount: number;
  inputShape: number[];
}

export interface SGGEngine {
  initialize(): Promise<ModelStatus>;
  infer(
    image: ImageBitmap | HTMLImageElement | HTMLCanvasElement | Blob,
    options?: { boxConf?: number; relConf?: number }
  ): Promise<SGGInferenceResult>;
  getModelInfo(): Promise<ModelInfo>;
  dispose(): Promise<void>;
}
