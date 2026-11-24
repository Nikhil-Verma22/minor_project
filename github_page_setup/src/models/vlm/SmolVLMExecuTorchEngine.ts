/**
 * SmolVLMExecuTorchEngine.
 * Implements VLMEngine interface, coordinating lazy download, WebGPU compilation, and fallback inference.
 */

import { VLMEngine, VLMProgress, VLMStatus } from './VLMEngine';
import { VLMStructuredResponse } from './vlmResponseSchema';
import { SceneGraph } from '../../graph/graphTypes';
import { PersistentObjectInstance } from '../../memory/memoryTypes';
import { generateId } from '../../utils/ids';
import { logMessage } from '../../utils/logging';
import { VLM_CONFIG } from '../../config/modelConfig';

export class SmolVLMExecuTorchEngine implements VLMEngine {
  private worker: Worker | null = null;
  private pendingRequests = new Map<string, { resolve: (val: any) => void; reject: (err: any) => void }>();
  private status: VLMStatus = { loaded: false, provider: 'unavailable' };
  private onProgressCallback?: (progress: VLMProgress) => void;

  async initialize(onProgress?: (progress: VLMProgress) => void): Promise<VLMStatus> {
    if (!VLM_CONFIG.enabled) {
      this.status = {
        loaded: false,
        provider: 'unavailable',
        error: VLM_CONFIG.unavailableReason,
      };
      logMessage('info', 'SmolVLMEngine', VLM_CONFIG.unavailableReason);
      return this.status;
    }

    this.onProgressCallback = onProgress;
    const startTime = performance.now();

    try {
      this.worker = new Worker(new URL('../../workers/smolvlm.worker.ts', import.meta.url), {
        type: 'module',
      });

      this.worker.onmessage = (e: MessageEvent) => {
        const { id, type, payload, error } = e.data;

        if (type === 'PROGRESS') {
          if (this.onProgressCallback) {
            this.onProgressCallback(payload);
          }
          return;
        }

        const pending = this.pendingRequests.get(id);
        if (!pending) return;

        this.pendingRequests.delete(id);
        if (type === 'ERROR' || error) {
          pending.reject(new Error(error || 'VLM Worker error'));
        } else {
          pending.resolve(payload);
        }
      };

      this.worker.onerror = (err) => {
        logMessage('error', 'SmolVLMEngine', 'VLM Worker runtime error', err);
      };

      const res = await this.sendWorkerRequest('INIT', {});
      this.status = {
        loaded: true,
        provider: 'webgpu',
        loadTimeMs: performance.now() - startTime,
      };

      logMessage('info', 'SmolVLMEngine', `SmolVLM initialized on WebGPU in ${Math.round(this.status.loadTimeMs!)}ms`);
      return this.status;
    } catch (err: any) {
      logMessage('warn', 'SmolVLMEngine', 'Failed to initialize SmolVLM on WebGPU', err);
      this.status = {
        loaded: false,
        provider: 'unavailable',
        error: err.message || String(err),
      };
      return this.status;
    }
  }

  async answer(
    image: ImageBitmap | Blob,
    query: string,
    graphContext: {
      currentGraph: SceneGraph | null;
      activeInstances: PersistentObjectInstance[];
    }
  ): Promise<VLMStructuredResponse> {
    if (!this.status.loaded || !this.worker) {
      throw new Error('VLM Engine is not loaded. Call initialize() first.');
    }

    const res = await this.sendWorkerRequest('INFER', {
      query,
      currentGraph: graphContext.currentGraph,
      activeInstances: graphContext.activeInstances,
    });

    return res as VLMStructuredResponse;
  }

  getStatus(): VLMStatus {
    return this.status;
  }

  async dispose(): Promise<void> {
    if (this.worker) {
      await this.sendWorkerRequest('DISPOSE', {});
      this.worker.terminate();
      this.worker = null;
    }
    this.status = { loaded: false, provider: 'unavailable' };
  }

  private sendWorkerRequest(type: string, payload: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = generateId('vlm_req');
      this.pendingRequests.set(id, { resolve, reject });
      this.worker!.postMessage({ id, type, payload });
    });
  }
}
