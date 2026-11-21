/**
 * Image processing utilities for letterbox, scaling, and canvas operations.
 */

export interface LetterboxResult {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  r: number;
  padX: number;
  padY: number;
  newW: number;
  newH: number;
  origW: number;
  origH: number;
}

/**
 * Letterbox an image or ImageBitmap to 640x640 with (114, 114, 114) fill.
 */
export function letterboxImage(
  source: HTMLImageElement | ImageBitmap | HTMLCanvasElement,
  targetSize = 640
): LetterboxResult {
  const origW = 'videoWidth' in source ? (source as any).videoWidth : source.width;
  const origH = 'videoHeight' in source ? (source as any).videoHeight : source.height;

  const r = Math.min(targetSize / origH, targetSize / origW);
  const newW = Math.round(origW * r);
  const newH = Math.round(origH * r);
  const padX = Math.floor((targetSize - newW) / 2);
  const padY = Math.floor((targetSize - newH) / 2);

  let canvas: HTMLCanvasElement | OffscreenCanvas;
  if (typeof OffscreenCanvas !== 'undefined') {
    canvas = new OffscreenCanvas(targetSize, targetSize);
  } else {
    canvas = document.createElement('canvas');
    canvas.width = targetSize;
    canvas.height = targetSize;
  }

  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  if (!ctx) throw new Error('Could not get 2D rendering context');

  // Fill padding background with standard YOLO/OpenPSG gray: rgb(114, 114, 114)
  ctx.fillStyle = 'rgb(114, 114, 114)';
  ctx.fillRect(0, 0, targetSize, targetSize);

  // Draw scaled image centered
  ctx.drawImage(source, 0, 0, origW, origH, padX, padY, newW, newH);

  return {
    canvas,
    r,
    padX,
    padY,
    newW,
    newH,
    origW,
    origH,
  };
}

/**
 * Extracts normalized [1, 3, 640, 640] float32 tensor data from canvas.
 */
export function canvasToFloatTensor(canvas: HTMLCanvasElement | OffscreenCanvas): Float32Array {
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  const width = canvas.width;
  const height = canvas.height;
  const imageData = ctx.getImageData(0, 0, width, height);
  const { data } = imageData;

  const planeSize = width * height;
  const tensorData = new Float32Array(3 * planeSize);

  let p = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255.0;
    const g = data[i + 1] / 255.0;
    const b = data[i + 2] / 255.0;

    tensorData[p] = r;
    tensorData[planeSize + p] = g;
    tensorData[2 * planeSize + p] = b;
    p++;
  }

  return tensorData;
}

/**
 * Reads a Blob as an ImageBitmap.
 */
export async function blobToImageBitmap(blob: Blob): Promise<ImageBitmap> {
  return createImageBitmap(blob);
}

/**
 * Encodes an image source or blob into a data URL for persistent frame storage.
 */
export async function imageSourceToDataUrl(
  source: HTMLImageElement | ImageBitmap | HTMLCanvasElement,
  blob?: Blob | null,
  maxWidth = 960
): Promise<string> {
  if (blob && blob.size < 1_500_000) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve((reader.result as string) || '');
      reader.onerror = () => resolve('');
      reader.readAsDataURL(blob);
    });
  }

  try {
    const origW = 'videoWidth' in source ? (source as any).videoWidth : source.width;
    const origH = 'videoHeight' in source ? (source as any).videoHeight : source.height;
    if (!origW || !origH) return '';

    const scale = Math.min(1.0, maxWidth / Math.max(origW, origH));
    const w = Math.round(origW * scale);
    const h = Math.round(origH * scale);

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    ctx.drawImage(source, 0, 0, origW, origH, 0, 0, w, h);
    return canvas.toDataURL('image/jpeg', 0.85);
  } catch (err) {
    console.warn('[imageSourceToDataUrl] Failed to generate data URL:', err);
    return '';
  }
}
