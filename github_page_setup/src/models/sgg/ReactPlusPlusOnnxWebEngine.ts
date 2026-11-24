/**
 * ReactPlusPlusOnnxWebEngine.
 * Orchestrates REACT++ inference in the browser via Web Worker with automatic Direct-Thread Fallback.
 */

import * as ort from 'onnxruntime-web/wasm';
import { SGGEngine, SGGInferenceResult, ModelInfo, ModelStatus } from './SGGEngine';
import { preprocessReactppImage } from './reactppPreprocess';
import { postprocessReactppOutput } from './reactppPostprocess';
import { SGG_VARIANTS, DEFAULT_THRESHOLDS } from '../../config/modelConfig';
import { OPENPSG_OBJ_CLASSES, OPENPSG_REL_CLASSES } from '../../config/openpsgVocab';
import { generateId } from '../../utils/ids';
import { logMessage } from '../../utils/logging';

export class ReactPlusPlusOnnxWebEngine implements SGGEngine {
  private worker: Worker | null = null;
  private directSession: ort.InferenceSession | null = null;
  private pendingRequests = new Map<string, { resolve: (val: any) => void; reject: (err: any) => void; timeout: any }>();
  private status: ModelStatus = { loaded: false, provider: 'unavailable', variant: 'yolo12n' };
  private initPromise: Promise<ModelStatus> | null = null;
  private onProgressCallback?: (progress: { stage: string; percent: number; message: string }) => void;
  private objClasses: string[] = OPENPSG_OBJ_CLASSES;
  private relClasses: string[] = OPENPSG_REL_CLASSES;

  constructor(private variant: 'yolo12n' = 'yolo12n') {}

  async initialize(
    onProgress?: (progress: { stage: string; percent: number; message: string }) => void
  ): Promise<ModelStatus> {
    if (onProgress) this.onProgressCallback = onProgress;
    if (this.status.loaded) return this.status;
    if (this.initPromise) return this.initPromise;

    this.initPromise = this.runInitialize().finally(() => {
      this.initPromise = null;
    });
    return this.initPromise;
  }

  private async runInitialize(): Promise<ModelStatus> {
    const startTime = performance.now();
    const config = SGG_VARIANTS[this.variant];

    // Try Worker First
    try {
      this.worker = new Worker(new URL('../../workers/reactpp.worker.ts', import.meta.url), {
        type: 'module',
      });

      this.worker.onmessage = (e: MessageEvent) => {
        const { id, type, payload, error } = e.data;
        if (type === 'INIT_PROGRESS') {
          this.onProgressCallback?.(payload);
          return;
        }
        const pending = this.pendingRequests.get(id);
        if (!pending) return;

        clearTimeout(pending.timeout);
        this.pendingRequests.delete(id);
        if (type === 'ERROR' || error) {
          pending.reject(new Error(error || 'Worker error'));
        } else {
          pending.resolve(payload);
        }
      };

      this.worker.onerror = (err) => {
        logMessage('warn', 'ReactPlusPlusOnnxWebEngine', 'Worker error, activating direct in-thread execution', err);
        // Reject all pending requests to trigger fallback
        for (const [id, pending] of this.pendingRequests.entries()) {
          clearTimeout(pending.timeout);
          pending.reject(new Error('Worker encountered runtime error'));
          this.pendingRequests.delete(id);
        }
      };

      const origin = typeof window !== 'undefined' ? window.location.origin : '';
      const baseUrl = (typeof import.meta !== 'undefined' && (import.meta as any).env?.BASE_URL) || '/';
      const resolvedModelPath = origin ? new URL(config.path, origin).href : config.path;
      const resolvedWasmPaths = origin
        ? new URL(`${baseUrl}ort-dist/`, origin).href
        : `${baseUrl}ort-dist/`;

      const res = await this.sendWorkerRequest('INIT', {
        modelPath: resolvedModelPath,
        wasmPaths: resolvedWasmPaths,
        preferredProvider: 'wasm',
      }, 240000);

      this.objClasses = res.objClasses;
      this.relClasses = res.relClasses;
      this.status = {
        loaded: true,
        provider: res.provider,
        variant: this.variant,
        initTimeMs: performance.now() - startTime,
      };

      logMessage('info', 'ReactPlusPlusOnnxWebEngine', `Initialized via Web Worker on ${res.provider} in ${Math.round(this.status.initTimeMs!)}ms`);
      return this.status;
    } catch (workerErr) {
      logMessage('warn', 'ReactPlusPlusOnnxWebEngine', 'Web Worker initialization failed or timed out, falling back to direct in-thread session:', workerErr);
      if (this.worker) {
        this.worker.terminate();
        this.worker = null;
      }
      return this.initializeDirect(config, startTime);
    }
  }

  private async initializeDirect(config: any, startTime: number): Promise<ModelStatus> {
    try {
      const baseUrl = (typeof import.meta !== 'undefined' && (import.meta as any).env?.BASE_URL) || '/';
      ort.env.wasm.wasmPaths = `${baseUrl}ort-dist/`;
      const isCrossOriginIsolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
      ort.env.wasm.numThreads = isCrossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

      let epList: string[] = ['wasm'];
      if ('gpu' in navigator && (navigator as any).gpu) {
        try {
          const adapter = await (navigator as any).gpu.requestAdapter();
          if (adapter) epList = ['webgpu', 'wasm'];
        } catch {
          epList = ['wasm'];
        }
      }

      const res = await fetch(config.path);
      if (!res.ok) throw new Error(`HTTP ${res.status} fetching model`);
      const buffer = await res.arrayBuffer();

      try {
        this.directSession = await ort.InferenceSession.create(buffer, {
          executionProviders: epList,
          graphOptimizationLevel: 'all',
        });
      } catch {
        this.directSession = await ort.InferenceSession.create(buffer, {
          executionProviders: ['wasm'],
        });
      }

      // Parse metadata
      try {
        const meta = (this.directSession as any).metadata || (this.directSession as any).handler?.metadata;
        const customMeta = meta?.customMetadata || {};
        if (customMeta.obj_classes) this.objClasses = JSON.parse(customMeta.obj_classes);
        if (customMeta.rel_classes) this.relClasses = JSON.parse(customMeta.rel_classes);
      } catch (e) {
        console.warn('Could not parse metadata in direct mode:', e);
      }

      this.status = {
        loaded: true,
        provider: epList.includes('webgpu') ? 'webgpu' : 'wasm',
        variant: this.variant,
        initTimeMs: performance.now() - startTime,
      };

      logMessage('info', 'ReactPlusPlusOnnxWebEngine', `Initialized direct session on ${this.status.provider} in ${Math.round(this.status.initTimeMs!)}ms`);
      return this.status;
    } catch (err: any) {
      logMessage('error', 'ReactPlusPlusOnnxWebEngine', 'Direct session initialization failed', err);
      this.status = {
        loaded: false,
        provider: 'unavailable',
        variant: this.variant,
        error: err.message || String(err),
      };
      return this.status;
    }
  }

  async infer(
    image: ImageBitmap | HTMLImageElement | HTMLCanvasElement | Blob,
    options?: { boxConf?: number; relConf?: number }
  ): Promise<SGGInferenceResult> {
    if (!this.status.loaded) {
      const initStatus = await this.initialize();
      if (!initStatus.loaded) {
        throw new Error(initStatus.error || 'SGG Engine failed to initialize.');
      }
    }

    const totalStart = performance.now();
    let sourceElement: ImageBitmap | HTMLImageElement | HTMLCanvasElement;
    if (image instanceof Blob) {
      sourceElement = await createImageBitmap(image);
    } else {
      sourceElement = image;
    }

    // 1. Preprocessing
    const preprocess = preprocessReactppImage(sourceElement);

    let boxesData: Float32Array;
    let relsData: Float32Array;
    let inferenceMs = 0;

    // 2. Forward pass (Worker or Direct)
    if (this.worker) {
      const forwardRes = await this.sendWorkerRequest('INFER', {
        tensorData: preprocess.tensorData,
        dims: [1, 3, 640, 640],
      });
      boxesData = new Float32Array(forwardRes.boxesData);
      relsData = new Float32Array(forwardRes.relsData);
      inferenceMs = forwardRes.inferenceMs;
    } else if (this.directSession) {
      const fStart = performance.now();
      const inputTensor = new ort.Tensor('float32', preprocess.tensorData, [1, 3, 640, 640]);
      const results = await this.directSession.run({ input: inputTensor });
      inferenceMs = performance.now() - fStart;
      boxesData = results['boxes'].data as Float32Array;
      relsData = results['rels'].data as Float32Array;
    } else {
      throw new Error('No active inference execution provider available');
    }

    // 3. Postprocessing
    const postprocess = postprocessReactppOutput(
      boxesData,
      relsData,
      preprocess.letterbox,
      this.objClasses,
      this.relClasses,
      {
        boxConf: options?.boxConf ?? DEFAULT_THRESHOLDS.boxConfidence,
        relConf: options?.relConf ?? DEFAULT_THRESHOLDS.relationConfidence,
      }
    );

    const totalMs = performance.now() - totalStart;

    return {
      frameId: generateId('frame'),
      modelId: 'react_pp',
      modelVariant: this.variant,
      executionProvider: this.status.provider as any,
      imageWidth: preprocess.letterbox.origW,
      imageHeight: preprocess.letterbox.origH,
      objects: postprocess.objects,
      relationships: postprocess.relationships,
      timings: {
        preprocessMs: Math.round(preprocess.preprocessTimeMs),
        inferenceMs: Math.round(inferenceMs),
        postprocessMs: Math.round(postprocess.postprocessTimeMs),
        totalMs: Math.round(totalMs),
      },
    };
  }

  async getModelInfo(): Promise<ModelInfo> {
    return {
      name: 'REACT++',
      variant: this.variant,
      provider: this.status.provider as any,
      objectClassesCount: this.objClasses.length,
      relationClassesCount: this.relClasses.length,
      inputShape: [1, 3, 640, 640],
    };
  }

  async dispose(): Promise<void> {
    if (this.worker) {
      await this.sendWorkerRequest('DISPOSE', {}).catch(() => {});
      this.worker.terminate();
      this.worker = null;
    }
    this.directSession = null;
    this.status = { loaded: false, provider: 'unavailable', variant: this.variant };
  }

  private sendWorkerRequest(type: string, payload: any, timeoutMs = 30000): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = generateId('req');
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Worker request '${type}' timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pendingRequests.set(id, { resolve, reject, timeout });
      this.worker!.postMessage({ id, type, payload });
    });
  }
}
