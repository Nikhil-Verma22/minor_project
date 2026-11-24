/**
 * REACT++ ONNX Web Worker.
 * Handles ONNX Runtime Web (WASM) session initialization with streaming Cache API caching
 * and forward passes off the main UI thread.
 */

import * as ort from 'onnxruntime-web/wasm';
import { OPENPSG_OBJ_CLASSES, OPENPSG_REL_CLASSES } from '../config/openpsgVocab';

// Configure local WASM binaries (respecting Vite BASE_URL / GitHub Pages subpath)
const DEFAULT_BASE_URL = (typeof import.meta !== 'undefined' && (import.meta as any).env?.BASE_URL) || '/';
ort.env.wasm.wasmPaths = `${DEFAULT_BASE_URL}ort-dist/`;
const isCrossOriginIsolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
ort.env.wasm.numThreads = isCrossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

let session: ort.InferenceSession | null = null;
let objClasses: string[] = OPENPSG_OBJ_CLASSES;
let relClasses: string[] = OPENPSG_REL_CLASSES;
let activeProvider: 'webgpu' | 'wasm' = 'wasm';

self.onerror = (e) => {
  console.error('[REACT++ Worker] Global error in worker:', e);
};

async function fetchModelWithProgress(
  id: string,
  modelPath: string,
  expectedBytes = 45362728
): Promise<ArrayBuffer> {
  // 1. Check Cache API first for instant reload on subsequent visits
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open('reactpp-onnx-v1');
      const cached = await cache.match(modelPath);
      if (cached) {
        self.postMessage({
          id,
          type: 'INIT_PROGRESS',
          payload: { stage: 'loading_cache', percent: 100, message: 'Loading cached REACT++ model...' },
        });
        return await cached.arrayBuffer();
      }
    } catch {
      // Cache API unavailable; fall through to network fetch
    }
  }

  // 2. Network fetch with streaming progress
  const response = await fetch(modelPath);
  if (!response.ok) {
    throw new Error(`Failed to fetch model from ${modelPath}: HTTP ${response.status}`);
  }

  const contentLengthHeader = response.headers.get('content-length');
  const totalBytes = contentLengthHeader ? parseInt(contentLengthHeader, 10) : expectedBytes;

  let modelBuffer: ArrayBuffer;
  if (response.body) {
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let receivedBytes = 0;
    let lastEmit = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        receivedBytes += value.byteLength;
        const now = performance.now();
        if (now - lastEmit > 250 || receivedBytes >= totalBytes) {
          lastEmit = now;
          const pct = Math.min(99, Math.round((receivedBytes / Math.max(1, totalBytes)) * 100));
          const loadedMB = (receivedBytes / (1024 * 1024)).toFixed(1);
          const totalMB = (totalBytes / (1024 * 1024)).toFixed(1);
          self.postMessage({
            id,
            type: 'INIT_PROGRESS',
            payload: {
              stage: 'downloading',
              percent: pct,
              message: `Downloading REACT++ (${loadedMB}/${totalMB} MB • ${pct}%)`,
            },
          });
        }
      }
    }

    const merged = new Uint8Array(receivedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.byteLength;
    }
    modelBuffer = merged.buffer;
  } else {
    modelBuffer = await response.arrayBuffer();
  }

  // 3. Persist downloaded buffer in Cache API for future visits
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open('reactpp-onnx-v1');
      await cache.put(
        modelPath,
        new Response(modelBuffer.slice(0), {
          headers: { 'Content-Type': 'application/octet-stream' },
        })
      );
    } catch {
      // Ignore cache quota errors
    }
  }

  return modelBuffer;
}

self.onmessage = async (e: MessageEvent) => {
  const { id, type, payload } = e.data;

  try {
    switch (type) {
      case 'INIT': {
        const { modelPath, wasmPaths } = payload;
        if (wasmPaths) {
          ort.env.wasm.wasmPaths = wasmPaths;
        }
        const startTime = performance.now();

        const modelBuffer = await fetchModelWithProgress(id, modelPath);

        self.postMessage({
          id,
          type: 'INIT_PROGRESS',
          payload: { stage: 'compiling', percent: 100, message: 'Initializing WASM runtime...' },
        });

        session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        activeProvider = 'wasm';

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
