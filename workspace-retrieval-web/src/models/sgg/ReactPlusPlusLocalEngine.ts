/**
 * ReactPlusPlusLocalEngine.
 * Fallback SGG engine communicating with an optional local Python service on http://127.0.0.1:8000.
 */

import { SGGEngine, SGGInferenceResult, ModelInfo, ModelStatus } from './SGGEngine';
import { generateId } from '../../utils/ids';
import { logMessage } from '../../utils/logging';

export class ReactPlusPlusLocalEngine implements SGGEngine {
  private endpoint = 'http://127.0.0.1:8000';
  private status: ModelStatus = { loaded: false, provider: 'local_service', variant: 'yolo12n' };

  constructor(endpoint?: string) {
    if (endpoint) this.endpoint = endpoint;
  }

  async initialize(): Promise<ModelStatus> {
    try {
      const res = await fetch(`${this.endpoint}/health`, { method: 'GET' });
      if (!res.ok) throw new Error(`Sidecar health check failed: HTTP ${res.status}`);
      const data = await res.json();
      this.status = {
        loaded: true,
        provider: 'local_service',
        variant: data.variant || 'yolo12n',
      };
      return this.status;
    } catch (err: any) {
      logMessage('warn', 'ReactPlusPlusLocalEngine', 'Local sidecar not reachable', err);
      this.status = {
        loaded: false,
        provider: 'unavailable',
        variant: 'yolo12n',
        error: 'Local Python service on 127.0.0.1:8000 is offline',
      };
      return this.status;
    }
  }

  async infer(image: ImageBitmap | Blob): Promise<SGGInferenceResult> {
    const totalStart = performance.now();
    let blob: Blob;

    if (image instanceof Blob) {
      blob = image;
    } else {
      const canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      const ctx = canvas.getContext('2d');
      ctx?.drawImage(image, 0, 0);
      blob = await new Promise<Blob>((res) => canvas.toBlob((b) => res(b!), 'image/jpeg'));
    }

    const formData = new FormData();
    formData.append('file', blob);

    const res = await fetch(`${this.endpoint}/infer`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) throw new Error(`Local sidecar inference failed: HTTP ${res.status}`);
    const data = await res.json();
    const totalMs = performance.now() - totalStart;

    return {
      frameId: generateId('frame'),
      modelId: 'react_pp_local',
      modelVariant: this.status.variant,
      executionProvider: 'local_service',
      imageWidth: data.imageWidth,
      imageHeight: data.imageHeight,
      objects: data.objects,
      relationships: data.relationships,
      timings: {
        preprocessMs: data.timings?.preprocessMs || 0,
        inferenceMs: data.timings?.inferenceMs || 0,
        postprocessMs: data.timings?.postprocessMs || 0,
        totalMs: Math.round(totalMs),
      },
    };
  }

  async getModelInfo(): Promise<ModelInfo> {
    return {
      name: 'REACT++ (Local Python Sidecar)',
      variant: this.status.variant,
      provider: 'local_service',
      objectClassesCount: 133,
      relationClassesCount: 57,
      inputShape: [1, 3, 640, 640],
    };
  }

  async dispose(): Promise<void> {
    this.status = { loaded: false, provider: 'unavailable', variant: 'yolo12n' };
  }
}
