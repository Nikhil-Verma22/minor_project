/**
 * VLM Engine Interface & Progress Types.
 */

import { VLMStructuredResponse } from './vlmResponseSchema';
import { SceneGraph } from '../../graph/graphTypes';
import { PersistentObjectInstance } from '../../memory/memoryTypes';

export interface VLMProgress {
  stage: 'idle' | 'checking_cache' | 'downloading' | 'compiling' | 'inferring' | 'ready';
  loadedBytes: number;
  totalBytes: number;
  percent: number;
  message: string;
}

export interface VLMStatus {
  loaded: boolean;
  provider: 'webgpu' | 'unavailable';
  error?: string;
  loadTimeMs?: number;
}

export interface VLMEngine {
  initialize(onProgress?: (progress: VLMProgress) => void): Promise<VLMStatus>;
  answer(
    image: ImageBitmap | Blob,
    query: string,
    graphContext: {
      currentGraph: SceneGraph | null;
      activeInstances: PersistentObjectInstance[];
    }
  ): Promise<VLMStructuredResponse>;
  getStatus(): VLMStatus;
  dispose(): Promise<void>;
}
