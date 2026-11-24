/**
 * Visual Crop Descriptor & Similarity Engine.
 *
 * Extracts lightweight, scale-invariant appearance descriptors (color distributions,
 * channel histograms, mean/std RGB, and aspect ratio) from bounding box crops.
 * Works seamlessly in both browser Canvas/ImageBitmap environments and Node.js.
 */

declare const Buffer: any;
declare const require: any;
declare const module: any;

export interface VisualCropDescriptor {
  meanR: number; // 0..255
  meanG: number; // 0..255
  meanB: number; // 0..255
  stdR: number;
  stdG: number;
  stdB: number;
  histogram: number[]; // 16 bins R, 16 bins G, 16 bins B (normalized to 1.0 each, 48 total)
  aspectRatio: number; // width / height
}

const HIST_BINS = 16;

/**
 * Extracts a VisualCropDescriptor directly from a linear RGBA pixel array.
 */
export function extractVisualDescriptorFromImageData(
  pixels: Uint8Array | Uint8ClampedArray,
  imgWidth: number,
  imgHeight: number,
  bbox: [number, number, number, number]
): VisualCropDescriptor {
  const x1 = Math.max(0, Math.min(imgWidth - 1, Math.floor(bbox[0])));
  const y1 = Math.max(0, Math.min(imgHeight - 1, Math.floor(bbox[1])));
  const x2 = Math.max(x1 + 1, Math.min(imgWidth, Math.ceil(bbox[2])));
  const y2 = Math.max(y1 + 1, Math.min(imgHeight, Math.ceil(bbox[3])));

  const cropW = x2 - x1;
  const cropH = y2 - y1;
  const totalPixels = cropW * cropH;

  if (totalPixels <= 0) {
    return {
      meanR: 128,
      meanG: 128,
      meanB: 128,
      stdR: 0,
      stdG: 0,
      stdB: 0,
      histogram: new Array(HIST_BINS * 3).fill(1 / HIST_BINS),
      aspectRatio: 1.0,
    };
  }

  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let rSqSum = 0;
  let gSqSum = 0;
  let bSqSum = 0;

  const rHist = new Float32Array(HIST_BINS);
  const gHist = new Float32Array(HIST_BINS);
  const bHist = new Float32Array(HIST_BINS);

  for (let y = y1; y < y2; y++) {
    const rowOffset = y * imgWidth * 4;
    for (let x = x1; x < x2; x++) {
      const idx = rowOffset + x * 4;
      const r = pixels[idx];
      const g = pixels[idx + 1];
      const b = pixels[idx + 2];

      rSum += r;
      gSum += g;
      bSum += b;
      rSqSum += r * r;
      gSqSum += g * g;
      bSqSum += b * b;

      const rBin = Math.min(HIST_BINS - 1, Math.floor((r / 256) * HIST_BINS));
      const gBin = Math.min(HIST_BINS - 1, Math.floor((g / 256) * HIST_BINS));
      const bBin = Math.min(HIST_BINS - 1, Math.floor((b / 256) * HIST_BINS));

      rHist[rBin]++;
      gHist[gBin]++;
      bHist[bBin]++;
    }
  }

  const meanR = rSum / totalPixels;
  const meanG = gSum / totalPixels;
  const meanB = bSum / totalPixels;

  const stdR = Math.sqrt(Math.max(0, rSqSum / totalPixels - meanR * meanR));
  const stdG = Math.sqrt(Math.max(0, gSqSum / totalPixels - meanG * meanG));
  const stdB = Math.sqrt(Math.max(0, bSqSum / totalPixels - meanB * meanB));

  const histogram = new Array<number>(HIST_BINS * 3);
  for (let i = 0; i < HIST_BINS; i++) {
    histogram[i] = rHist[i] / totalPixels;
    histogram[HIST_BINS + i] = gHist[i] / totalPixels;
    histogram[HIST_BINS * 2 + i] = bHist[i] / totalPixels;
  }

  const aspectRatio = cropH > 0 ? cropW / cropH : 1.0;

  return {
    meanR,
    meanG,
    meanB,
    stdR,
    stdG,
    stdB,
    histogram,
    aspectRatio,
  };
}

// Reusable offscreen canvas for browser extraction
let sharedCanvas: HTMLCanvasElement | null = null;
let sharedCtx: CanvasRenderingContext2D | null = null;

/**
 * Extracts a VisualCropDescriptor from any supported browser or Node image source.
 */
export function extractVisualDescriptor(
  imageSource: any,
  bbox: [number, number, number, number],
  imgWidth?: number,
  imgHeight?: number
): VisualCropDescriptor {
  // If already decoded RGBA pixel buffer
  if (imageSource && imageSource.data && typeof imageSource.width === 'number' && typeof imageSource.height === 'number') {
    return extractVisualDescriptorFromImageData(imageSource.data, imageSource.width, imageSource.height, bbox);
  }

  // Browser environment with DOM Canvas
  if (typeof document !== 'undefined') {
    const srcW = imgWidth || imageSource.videoWidth || imageSource.naturalWidth || imageSource.width || 640;
    const srcH = imgHeight || imageSource.videoHeight || imageSource.naturalHeight || imageSource.height || 480;

    const x1 = Math.max(0, Math.min(srcW - 1, Math.floor(bbox[0])));
    const y1 = Math.max(0, Math.min(srcH - 1, Math.floor(bbox[1])));
    const x2 = Math.max(x1 + 1, Math.min(srcW, Math.ceil(bbox[2])));
    const y2 = Math.max(y1 + 1, Math.min(srcH, Math.ceil(bbox[3])));
    const cropW = x2 - x1;
    const cropH = y2 - y1;

    if (!sharedCanvas) {
      sharedCanvas = document.createElement('canvas');
      sharedCtx = sharedCanvas.getContext('2d', { willReadFrequently: true });
    }

    if (sharedCanvas && sharedCtx && cropW > 0 && cropH > 0) {
      sharedCanvas.width = cropW;
      sharedCanvas.height = cropH;
      sharedCtx.drawImage(imageSource, x1, y1, cropW, cropH, 0, 0, cropW, cropH);
      const imgData = sharedCtx.getImageData(0, 0, cropW, cropH);
      return extractVisualDescriptorFromImageData(imgData.data, cropW, cropH, [0, 0, cropW, cropH]);
    }
  }

  // Fallback default descriptor if pixels cannot be accessed
  return {
    meanR: 128,
    meanG: 128,
    meanB: 128,
    stdR: 10,
    stdG: 10,
    stdB: 10,
    histogram: new Array(HIST_BINS * 3).fill(1 / HIST_BINS),
    aspectRatio: (bbox[2] - bbox[0]) / Math.max(1, bbox[3] - bbox[1]),
  };
}

/**
 * Computes multi-factor visual similarity between two descriptors [0.0 .. 1.0].
 * Combines histogram intersection, normalized Euclidean color distance, and aspect ratio agreement.
 */
export function computeVisualSimilarity(
  descA: VisualCropDescriptor,
  descB: VisualCropDescriptor
): number {
  if (!descA || !descB) return 0.5;

  // 1. Mean RGB Euclidean distance (max possible distance = sqrt(3 * 255^2) ≈ 441.67)
  const dR = descA.meanR - descB.meanR;
  const dG = descA.meanG - descB.meanG;
  const dB = descA.meanB - descB.meanB;
  const colorDist = Math.hypot(dR, dG, dB) / 441.673;
  const colorSimilarity = Math.max(0, 1.0 - colorDist);

  // 2. Histogram intersection over 3 channels (each channel sums to 1.0, total sum = 3.0)
  let intersectionSum = 0;
  const histLen = Math.min(descA.histogram.length, descB.histogram.length);
  for (let i = 0; i < histLen; i++) {
    intersectionSum += Math.min(descA.histogram[i], descB.histogram[i]);
  }
  const histSimilarity = Math.max(0, Math.min(1.0, intersectionSum / 3.0));

  // 3. Aspect ratio consistency
  const arMin = Math.min(descA.aspectRatio, descB.aspectRatio);
  const arMax = Math.max(descA.aspectRatio, descB.aspectRatio);
  const aspectSimilarity = arMax > 0 ? arMin / arMax : 1.0;

  // Weighted combination:
  // Color & Histogram carry 90% of weight, aspect ratio 10%
  return 0.50 * histSimilarity + 0.40 * colorSimilarity + 0.10 * aspectSimilarity;
}

/**
 * Synchronous pure Node.js PNG decoder for test runners and CLI scripts.
 * Decodes PNG IHDR and IDAT chunks using zlib.inflateSync.
 */
export function decodePngSync(
  buffer: Uint8Array,
  zlibModule?: any
): { width: number; height: number; data: Uint8Array } {
  let zlib = zlibModule;
  if (!zlib) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const req = (typeof module !== 'undefined' && (module as any).require) || (typeof require !== 'undefined' ? require : null);
      if (req) zlib = req('zlib');
    } catch {
      // ignore
    }
  }
  if (!zlib) {
    throw new Error('decodePngSync requires Node.js zlib module passed or available');
  }

  const nodeBuf = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer);
  let offset = 8;
  let width = 0;
  let height = 0;
  let colorType = 0;
  const idats: any[] = [];

  while (offset < nodeBuf.length) {
    const len = nodeBuf.readUInt32BE(offset);
    const type = nodeBuf.toString('ascii', offset + 4, offset + 8);
    if (type === 'IHDR') {
      width = nodeBuf.readUInt32BE(offset + 8);
      height = nodeBuf.readUInt32BE(offset + 12);
      colorType = nodeBuf.readUInt8(offset + 17);
    } else if (type === 'IDAT') {
      idats.push(nodeBuf.subarray(offset + 8, offset + 8 + len));
    } else if (type === 'IEND') {
      break;
    }
    offset += 12 + len;
  }

  const decompressed = zlib.inflateSync(Buffer.concat(idats));
  const bpp = colorType === 6 ? 4 : colorType === 2 ? 3 : 1;
  const stride = width * bpp;
  const pixels = new Uint8Array(width * height * 4);
  let srcPos = 0;
  const prevRow = new Uint8Array(stride);
  const currRow = new Uint8Array(stride);

  function paeth(a: number, b: number, c: number): number {
    const p = a + b - c;
    const pa = Math.abs(p - a);
    const pb = Math.abs(p - b);
    const pc = Math.abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
  }

  for (let y = 0; y < height; y++) {
    const filter = decompressed[srcPos++];
    for (let i = 0; i < stride; i++) {
      const val = decompressed[srcPos++];
      const left = i >= bpp ? currRow[i - bpp] : 0;
      const up = prevRow[i];
      const upLeft = i >= bpp ? prevRow[i - bpp] : 0;
      if (filter === 0) currRow[i] = val;
      else if (filter === 1) currRow[i] = (val + left) & 0xff;
      else if (filter === 2) currRow[i] = (val + up) & 0xff;
      else if (filter === 3) currRow[i] = (val + Math.floor((left + up) / 2)) & 0xff;
      else if (filter === 4) currRow[i] = (val + paeth(left, up, upLeft)) & 0xff;
      else currRow[i] = val;
    }
    prevRow.set(currRow);

    const rowOffset = y * width * 4;
    for (let x = 0; x < width; x++) {
      const outIdx = rowOffset + x * 4;
      const inIdx = x * bpp;
      if (bpp === 4) {
        pixels[outIdx] = currRow[inIdx];
        pixels[outIdx + 1] = currRow[inIdx + 1];
        pixels[outIdx + 2] = currRow[inIdx + 2];
        pixels[outIdx + 3] = currRow[inIdx + 3];
      } else if (bpp === 3) {
        pixels[outIdx] = currRow[inIdx];
        pixels[outIdx + 1] = currRow[inIdx + 1];
        pixels[outIdx + 2] = currRow[inIdx + 2];
        pixels[outIdx + 3] = 255;
      }
    }
  }

  return { width, height, data: pixels };
}
