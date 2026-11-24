/**
 * REACT++ Preprocessing.
 * Letterboxes input image to 640x640 with RGB(114,114,114) fill and converts to Float32Array [1, 3, 640, 640].
 */

import { letterboxImage, canvasToFloatTensor, LetterboxResult } from '../../utils/image';

export interface PreprocessOutput {
  tensorData: Float32Array;
  letterbox: LetterboxResult;
  preprocessTimeMs: number;
}

export function preprocessReactppImage(
  source: HTMLImageElement | ImageBitmap | HTMLCanvasElement
): PreprocessOutput {
  const startTime = performance.now();
  const letterbox = letterboxImage(source, 640);
  const tensorData = canvasToFloatTensor(letterbox.canvas);
  const preprocessTimeMs = performance.now() - startTime;

  return {
    tensorData,
    letterbox,
    preprocessTimeMs,
  };
}
