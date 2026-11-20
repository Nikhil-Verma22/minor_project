/**
 * REACT++ ONNX Web Worker.
 * Handles ONNX Runtime Web session initialization and forward passes off the main UI thread.
 */

import * as ort from 'onnxruntime-web';
import { OPENPSG_OBJ_CLASSES, OPENPSG_REL_CLASSES } from '../config/openpsgVocab';

// Configure local WASM binaries
ort.env.wasm.wasmPaths = '/ort-dist/';
const isCrossOriginIsolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
ort.env.wasm.numThreads = isCrossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

let session: ort.InferenceSession | null = null;
let objClasses: string[] = OPENPSG_OBJ_CLASSES;
let relClasses: string[] = OPENPSG_REL_CLASSES;
let activeProvider: 'webgpu' | 'wasm' = 'wasm';

self.onerror = (e) => {
  console.error('[REACT++ Worker] Global error in worker:', e);
};

self.onmessage = async (e: MessageEvent) => {
  const { id, type, payload } = e.data;

  try {
    switch (type) {
      case 'INIT': {
        const { modelPath, preferredProvider } = payload;
        const startTime = performance.now();

        // Check WebGPU availability if requested
        let epList: string[] = ['wasm'];
        if (preferredProvider === 'webgpu' && 'gpu' in navigator && (navigator as any).gpu) {
          try {
            const adapter = await (navigator as any).gpu.requestAdapter();
            if (adapter) {
              epList = ['webgpu', 'wasm'];
            }
          } catch {
            epList = ['wasm'];
          }
        }

        // Fetch model buffer
        const response = await fetch(modelPath);
        if (!response.ok) {
          throw new Error(`Failed to fetch model from ${modelPath}: HTTP ${response.status}`);
        }
        const modelBuffer = await response.arrayBuffer();

        try {
          session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: epList,
            graphOptimizationLevel: 'all',
          });
          activeProvider = epList.includes('webgpu') ? 'webgpu' : 'wasm';
        } catch (gpuErr) {
          console.warn('[REACT++ Worker] Failed to load with preferred provider, falling back to WASM:', gpuErr);
          session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
          });
          activeProvider = 'wasm';
        }

        // Parse custom metadata
        try {
          const meta = (session as any).metadata || (session as any).handler?.metadata;
          const customMeta = meta?.customMetadata || {};
          if (customMeta.obj_classes) {
            objClasses = JSON.parse(customMeta.obj_classes);
          }
          if (customMeta.rel_classes) {
            relClasses = JSON.parse(customMeta.rel_classes);
          }
        } catch (metaErr) {
          console.warn('[REACT++ Worker] Could not parse custom metadata, using fallback vocabularies:', metaErr);
        }

        const initTimeMs = performance.now() - startTime;
        self.postMessage({
          id,
          type: 'INIT_SUCCESS',
          payload: {
            provider: activeProvider,
            objClasses,
            relClasses,
            initTimeMs,
          },
        });
        break;
      }

      case 'INFER': {
        if (!session) throw new Error('REACT++ InferenceSession is not initialized');

        const { tensorData, dims } = payload;
        const startTime = performance.now();

        const inputTensor = new ort.Tensor('float32', tensorData, dims);
        const feeds: Record<string, ort.Tensor> = { input: inputTensor };

        const results = await session.run(feeds);
        const inferenceMs = performance.now() - startTime;

        const boxesTensor = results['boxes'];
        const relsTensor = results['rels'];

        (self as any).postMessage(
          {
            id,
            type: 'INFER_SUCCESS',
            payload: {
              boxesData: boxesTensor.data,
              relsData: relsTensor.data,
              inferenceMs,
            },
          },
          // Transfer typed array buffers for zero-copy performance
          [(boxesTensor.data as Float32Array).buffer, (relsTensor.data as Float32Array).buffer]
        );
        break;
      }

      case 'DISPOSE': {
        session = null;
        self.postMessage({ id, type: 'DISPOSE_SUCCESS' });
        break;
      }

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (err: any) {
    self.postMessage({
      id,
      type: 'ERROR',
      error: err.message || String(err),
    });
  }
};
