/**
 * Visual Attribute Estimator.
 * Analyzes RGB image pixel data within bounding box regions to estimate
 * real object colors and person clothing attributes (torso/shirt vs pants).
 */

export interface ColorEstimation {
  dominantColor: string;
  colorCategory: string;
  hex: string;
  rgb: [number, number, number];
  confidence: number;
}

export class VisualAttributeEstimator {
  private static canvas: HTMLCanvasElement | null = null;
  private static ctx: CanvasRenderingContext2D | null = null;

  private static getContext(width: number, height: number): CanvasRenderingContext2D | null {
    if (!this.canvas) {
      if (typeof document !== 'undefined') {
        this.canvas = document.createElement('canvas');
      } else {
        return null;
      }
    }
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    return this.ctx;
  }

  /**
   * Samples a bounding box region from an image source and estimates the dominant color.
   */
  static estimateColorFromImage(
    imageSource: CanvasImageSource,
    imgWidth: number,
    imgHeight: number,
    bbox: [number, number, number, number],
    subRegion: 'full' | 'torso' | 'surface' = 'full'
  ): ColorEstimation | null {
    const ctx = this.getContext(imgWidth, imgHeight);
    if (!ctx) return null;

    try {
      ctx.drawImage(imageSource, 0, 0, imgWidth, imgHeight);

      let [x1, y1, x2, y2] = bbox;
      // Clamp to image dimensions
      x1 = Math.max(0, Math.min(imgWidth, x1));
      y1 = Math.max(0, Math.min(imgHeight, y1));
      x2 = Math.max(0, Math.min(imgWidth, x2));
      y2 = Math.max(0, Math.min(imgHeight, y2));

      const boxW = x2 - x1;
      const boxH = y2 - y1;
      if (boxW <= 2 || boxH <= 2) return null;

      let cropX = x1;
      let cropY = y1;
      let cropW = boxW;
      let cropH = boxH;

      if (subRegion === 'torso') {
        // Torso / shirt: vertical 22% to 60%, horizontal central 60%
        cropX = x1 + boxW * 0.2;
        cropY = y1 + boxH * 0.22;
        cropW = Math.max(2, boxW * 0.6);
        cropH = Math.max(2, boxH * 0.38);
      } else if (subRegion === 'surface') {
        // Central 50% of the surface
        cropX = x1 + boxW * 0.25;
        cropY = y1 + boxH * 0.25;
        cropW = Math.max(2, boxW * 0.5);
        cropH = Math.max(2, boxH * 0.5);
      }

      const imgData = ctx.getImageData(
        Math.floor(cropX),
        Math.floor(cropY),
        Math.max(1, Math.floor(cropW)),
        Math.max(1, Math.floor(cropH))
      );

      const pixels = imgData.data;
      if (!pixels || pixels.length === 0) return null;

      let totalR = 0;
      let totalG = 0;
      let totalB = 0;
      let validCount = 0;

      for (let i = 0; i < pixels.length; i += 16) { // step by 4 pixels (16 bytes)
        const r = pixels[i];
        const g = pixels[i + 1];
        const b = pixels[i + 2];
        const a = pixels[i + 3];

        if (a > 50) {
          totalR += r;
          totalG += g;
          totalB += b;
          validCount++;
        }
      }

      if (validCount === 0) return null;

      const avgR = Math.round(totalR / validCount);
      const avgG = Math.round(totalG / validCount);
      const avgB = Math.round(totalB / validCount);

      return this.classifyRgb(avgR, avgG, avgB);
    } catch (e) {
      console.warn('[VisualAttributeEstimator] Error reading pixels:', e);
      return null;
    }
  }

  /**
   * Classify an RGB value into calibrated color category names.
   */
  static classifyRgb(r: number, g: number, b: number): ColorEstimation {
    const rNorm = r / 255;
    const gNorm = g / 255;
    const bNorm = b / 255;

    const max = Math.max(rNorm, gNorm, bNorm);
    const min = Math.min(rNorm, gNorm, bNorm);
    const delta = max - min;

    let h = 0;
    if (delta !== 0) {
      if (max === rNorm) {
        h = ((gNorm - bNorm) / delta) % 6;
      } else if (max === gNorm) {
        h = (bNorm - rNorm) / delta + 2;
      } else {
        h = (rNorm - gNorm) / delta + 4;
      }
      h = Math.round(h * 60);
      if (h < 0) h += 360;
    }

    const s = max === 0 ? 0 : delta / max;
    const v = max;

    const hex = `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;

    // 1. Grayscale checks (Black, White, Gray)
    if (s < 0.16) {
      if (v < 0.22) {
        return { dominantColor: 'black', colorCategory: 'black', hex, rgb: [r, g, b], confidence: 0.9 };
      }
      if (v > 0.78) {
        return { dominantColor: 'white', colorCategory: 'white', hex, rgb: [r, g, b], confidence: 0.88 };
      }
      if (v < 0.45) {
        return { dominantColor: 'dark gray', colorCategory: 'gray', hex, rgb: [r, g, b], confidence: 0.85 };
      }
      return { dominantColor: 'gray', colorCategory: 'gray', hex, rgb: [r, g, b], confidence: 0.85 };
    }

    if (v < 0.18) {
      return { dominantColor: 'black', colorCategory: 'black', hex, rgb: [r, g, b], confidence: 0.88 };
    }

    // 2. Brown / Wood Tone check (warm hues, low to medium saturation/value)
    if (h >= 14 && h <= 45 && s < 0.75 && v < 0.72) {
      if (v < 0.4) {
        return { dominantColor: 'dark brown', colorCategory: 'brown', hex, rgb: [r, g, b], confidence: 0.85 };
      }
      if (s < 0.45) {
        return { dominantColor: 'tan/wood', colorCategory: 'brown', hex, rgb: [r, g, b], confidence: 0.82 };
      }
      return { dominantColor: 'brown', colorCategory: 'brown', hex, rgb: [r, g, b], confidence: 0.85 };
    }

    // 3. Chromatic classification by hue angle
    if (h >= 345 || h < 14) {
      return { dominantColor: 'red', colorCategory: 'red', hex, rgb: [r, g, b], confidence: 0.84 };
    }
    if (h >= 14 && h < 45) {
      if (s > 0.4) {
        return { dominantColor: r > 180 && g > 130 ? 'peach/orange' : 'orange', colorCategory: 'orange', hex, rgb: [r, g, b], confidence: 0.86 };
      }
      return { dominantColor: 'brown', colorCategory: 'brown', hex, rgb: [r, g, b], confidence: 0.8 };
    }
    if (h >= 45 && h < 70) {
      return { dominantColor: 'yellow', colorCategory: 'yellow', hex, rgb: [r, g, b], confidence: 0.82 };
    }
    if (h >= 70 && h < 165) {
      return { dominantColor: 'green', colorCategory: 'green', hex, rgb: [r, g, b], confidence: 0.85 };
    }
    if (h >= 165 && h < 195) {
      return { dominantColor: 'cyan/teal', colorCategory: 'cyan', hex, rgb: [r, g, b], confidence: 0.83 };
    }
    if (h >= 195 && h < 260) {
      if (v < 0.35) {
        return { dominantColor: 'navy blue', colorCategory: 'blue', hex, rgb: [r, g, b], confidence: 0.88 };
      }
      return { dominantColor: 'blue', colorCategory: 'blue', hex, rgb: [r, g, b], confidence: 0.86 };
    }
    if (h >= 260 && h < 315) {
      return { dominantColor: 'purple', colorCategory: 'purple', hex, rgb: [r, g, b], confidence: 0.82 };
    }
    if (h >= 315 && h < 345) {
      return { dominantColor: 'pink', colorCategory: 'pink', hex, rgb: [r, g, b], confidence: 0.84 };
    }

    return { dominantColor: 'gray', colorCategory: 'gray', hex, rgb: [r, g, b], confidence: 0.6 };
  }
}
