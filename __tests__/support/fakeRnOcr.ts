/**
 * JavaScript implementation of the package's native JSI object, installed as
 * `globalThis.__rnocr_jsi__` before `src/` is imported. Each op mirrors the
 * C++ kernel's contract (buffer in, buffer out, explicit dimensions) so the
 * TypeScript wrappers are exercised exactly as on a device.
 *
 * The fakes reproduce the kernels' semantics closely enough for the pipeline
 * tests: `warpByGrid` samples with nearest or bilinear filtering (cubic falls
 * back to bilinear) and `extractCraftTextBoxes` runs a plain 4-connected
 * component pass without OpenCV's dilation and rotated-rectangle fit, so its
 * boxes are the components' axis-aligned extents with angle 0.
 */
import type { CraftDecodeOptions, WarpByGridOptions } from '../../src/native/ops';
import type { NativeBuffer, RnOcrJsi } from '../../src/native/types';

function view(buffer: NativeBuffer, op: string): Uint8Array {
  if (buffer instanceof ArrayBuffer) return new Uint8Array(buffer);
  if (ArrayBuffer.isView(buffer)) {
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  }
  throw new Error(`${op}: expected an ArrayBuffer or a typed array`);
}

function requireBytes(buffer: NativeBuffer, expected: number, op: string): Uint8Array {
  const bytes = view(buffer, op);
  if (bytes.byteLength !== expected) {
    throw new Error(`${op}: expected ${expected} bytes, got ${bytes.byteLength}`);
  }
  return bytes;
}

function requireFloats(buffer: NativeBuffer, expectedCount: number, op: string): Float32Array {
  const bytes = requireBytes(buffer, expectedCount * 4, op);
  return new Float32Array(bytes.buffer, bytes.byteOffset, expectedCount);
}

function requirePositiveInt(value: number, what: string): number {
  if (!Number.isInteger(value) || value <= 0)
    throw new Error(`${what}: expected a positive integer`);
  return value;
}

function requireChannels(channels: number, op: string, max = 512): number {
  if (!Number.isInteger(channels) || channels < 1 || channels > max) {
    throw new Error(`${op}: unsupported channel count ${channels} (expected 1 to ${max})`);
  }
  return channels;
}

export const fakeRnOcr: RnOcrJsi = {
  isEmulator: false,

  rotate(src, height, width, channels, degrees) {
    requirePositiveInt(height, 'rotate: height');
    requirePositiveInt(width, 'rotate: width');
    requireChannels(channels, 'rotate');
    const input = requireBytes(src, height * width * channels, 'rotate');
    const swap = degrees !== 180;
    const outH = swap ? width : height;
    const outW = swap ? height : width;
    const out = new Uint8Array(outH * outW * channels);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let ox: number;
        let oy: number;
        if (degrees === 90) {
          ox = height - 1 - y;
          oy = x;
        } else if (degrees === 180) {
          ox = width - 1 - x;
          oy = height - 1 - y;
        } else if (degrees === 270) {
          ox = y;
          oy = width - 1 - x;
        } else {
          throw new Error('rotate: degreesClockwise must be 90, 180 or 270');
        }
        const from = (y * width + x) * channels;
        const to = (oy * outW + ox) * channels;
        for (let c = 0; c < channels; c++) out[to + c] = input[from + c]!;
      }
    }
    return out.buffer;
  },

  crop(src, height, width, channels, x0, y0, x1, y1) {
    requirePositiveInt(height, 'crop: height');
    requirePositiveInt(width, 'crop: width');
    requireChannels(channels, 'crop');
    const input = requireBytes(src, height * width * channels, 'crop');
    if (x0 < 0 || y0 < 0 || x1 > width || y1 > height) {
      throw new Error(
        `crop: box [${x0}, ${y0}, ${x1}, ${y1}] exceeds the ${width}x${height} image`
      );
    }
    if (x1 <= x0 || y1 <= y0) throw new Error('crop: box must have positive width and height');
    const outW = x1 - x0;
    const outH = y1 - y0;
    const out = new Uint8Array(outH * outW * channels);
    for (let y = 0; y < outH; y++) {
      const from = ((y0 + y) * width + x0) * channels;
      out.set(input.subarray(from, from + outW * channels), y * outW * channels);
    }
    return out.buffer;
  },

  warpByGrid(src, srcH, srcW, channels, grid, gridH, gridW, dstH, dstW, options) {
    requirePositiveInt(srcH, 'warpByGrid: srcHeight');
    requirePositiveInt(srcW, 'warpByGrid: srcWidth');
    requirePositiveInt(gridH, 'warpByGrid: gridHeight');
    requirePositiveInt(gridW, 'warpByGrid: gridWidth');
    requirePositiveInt(dstH, 'warpByGrid: dstHeight');
    requirePositiveInt(dstW, 'warpByGrid: dstWidth');
    requireChannels(channels, 'warpByGrid', 4);
    const input = requireBytes(src, srcH * srcW * channels, 'warpByGrid: src');
    const field = requireFloats(grid, 2 * gridH * gridW, 'warpByGrid: grid');
    const opts: Required<WarpByGridOptions> = {
      interpolation: options?.interpolation ?? 'linear',
      borderMode: options?.borderMode ?? 'replicate',
      borderValue: options?.borderValue ?? 0,
    };
    if (!['nearest', 'linear', 'cubic'].includes(opts.interpolation)) {
      throw new Error(`warpByGrid: options.interpolation: unsupported '${opts.interpolation}'`);
    }
    if (!['replicate', 'constant'].includes(opts.borderMode)) {
      throw new Error(`warpByGrid: options.borderMode: unsupported '${opts.borderMode}'`);
    }

    const plane = gridH * gridW;
    const sampleGrid = (c: number, gx: number, gy: number): number => {
      const x0 = Math.min(Math.max(Math.floor(gx), 0), gridW - 1);
      const y0 = Math.min(Math.max(Math.floor(gy), 0), gridH - 1);
      const x1 = Math.min(x0 + 1, gridW - 1);
      const y1 = Math.min(y0 + 1, gridH - 1);
      const dx = gx - x0;
      const dy = gy - y0;
      const base = c * plane;
      const top =
        field[base + y0 * gridW + x0]! +
        (field[base + y0 * gridW + x1]! - field[base + y0 * gridW + x0]!) * dx;
      const bottom =
        field[base + y1 * gridW + x0]! +
        (field[base + y1 * gridW + x1]! - field[base + y1 * gridW + x0]!) * dx;
      return top + (bottom - top) * dy;
    };

    const pixel = (x: number, y: number, c: number): number => {
      if (x < 0 || y < 0 || x >= srcW || y >= srcH) {
        if (opts.borderMode === 'constant') return opts.borderValue;
        x = Math.min(Math.max(x, 0), srcW - 1);
        y = Math.min(Math.max(y, 0), srcH - 1);
      }
      return input[(y * srcW + x) * channels + c]!;
    };

    const out = new Uint8Array(dstH * dstW * channels);
    const stepY = dstH > 1 ? (gridH - 1) / (dstH - 1) : 0;
    const stepX = dstW > 1 ? (gridW - 1) / (dstW - 1) : 0;
    for (let oy = 0; oy < dstH; oy++) {
      for (let ox = 0; ox < dstW; ox++) {
        const gx = ox * stepX;
        const gy = oy * stepY;
        const sx = ((sampleGrid(0, gx, gy) + 1) / 2) * (srcW - 1);
        const sy = ((sampleGrid(1, gx, gy) + 1) / 2) * (srcH - 1);
        const to = (oy * dstW + ox) * channels;
        if (opts.interpolation === 'nearest') {
          const nx = Math.round(sx);
          const ny = Math.round(sy);
          for (let c = 0; c < channels; c++) out[to + c] = pixel(nx, ny, c);
        } else {
          const x0 = Math.floor(sx);
          const y0 = Math.floor(sy);
          const fx = sx - x0;
          const fy = sy - y0;
          for (let c = 0; c < channels; c++) {
            const top = pixel(x0, y0, c) * (1 - fx) + pixel(x0 + 1, y0, c) * fx;
            const bottom = pixel(x0, y0 + 1, c) * (1 - fx) + pixel(x0 + 1, y0 + 1, c) * fx;
            out[to + c] = Math.round(top * (1 - fy) + bottom * fy);
          }
        }
      }
    }
    return out.buffer;
  },

  padCornerMean(strip, height, width, channels, targetWidth) {
    requirePositiveInt(height, 'padCornerMean: height');
    requirePositiveInt(width, 'padCornerMean: width');
    requireChannels(channels, 'padCornerMean', 4);
    const input = requireBytes(strip, height * width * channels, 'padCornerMean');
    if (targetWidth < width) {
      throw new Error(
        `padCornerMean: targetWidth (${targetWidth}) must be at least the strip width (${width})`
      );
    }
    const patch = Math.max(1, Math.floor(Math.min(height, width) / 30));
    const corners: [number, number][] = [
      [0, 0],
      [width - patch, 0],
      [0, height - patch],
      [width - patch, height - patch],
    ];
    const mean = new Array<number>(channels).fill(0);
    for (const [cx, cy] of corners) {
      for (let y = cy; y < cy + patch; y++) {
        for (let x = cx; x < cx + patch; x++) {
          for (let c = 0; c < channels; c++) mean[c]! += input[(y * width + x) * channels + c]!;
        }
      }
    }
    const fill = mean.map((sum) => Math.round(sum / (4 * patch * patch)));

    const out = new Uint8Array(height * targetWidth * channels);
    for (let y = 0; y < height; y++) {
      out.set(
        input.subarray(y * width * channels, (y + 1) * width * channels),
        y * targetWidth * channels
      );
      for (let x = width; x < targetWidth; x++) {
        for (let c = 0; c < channels; c++) out[(y * targetWidth + x) * channels + c] = fill[c]!;
      }
    }
    return out.buffer;
  },

  extractCraftTextBoxes(heat, heatH, heatW, options) {
    requirePositiveInt(heatH, 'extractCraftTextBoxes: heatHeight');
    requirePositiveInt(heatW, 'extractCraftTextBoxes: heatWidth');
    const data = requireFloats(heat, heatH * heatW * 2, 'extractCraftTextBoxes: heat');
    const opts: CraftDecodeOptions = options;
    for (const key of [
      'textThreshold',
      'linkThreshold',
      'lowTextThreshold',
      'minArea',
      'scale',
    ] as const) {
      if (typeof opts[key] !== 'number' || !Number.isFinite(opts[key])) {
        throw new Error(`extractCraftTextBoxes: options.${key}: expected a finite number`);
      }
    }
    if (typeof opts.charLevel !== 'boolean') {
      throw new Error('extractCraftTextBoxes: options.charLevel: expected a boolean');
    }
    if (!Number.isInteger(opts.minArea) || opts.minArea < 0) {
      throw new Error('extractCraftTextBoxes: options.minArea: expected a non-negative integer');
    }
    if (!(opts.scale > 0))
      throw new Error('extractCraftTextBoxes: scale must be a positive number');

    // Binarize: text + affinity links glyphs, text - affinity separates them.
    const binary = new Uint8Array(heatH * heatW);
    for (let i = 0; i < heatH * heatW; i++) {
      const text = data[i * 2]! >= opts.textThreshold ? 1 : 0;
      const link = data[i * 2 + 1]! >= opts.linkThreshold ? 1 : 0;
      binary[i] = opts.charLevel ? (text - link > 0 ? 1 : 0) : text + link > 0 ? 1 : 0;
    }

    // 4-connected components by flood fill.
    const labels = new Int32Array(heatH * heatW);
    const boxes: number[] = [];
    let next = 1;
    const stack: number[] = [];
    for (let start = 0; start < heatH * heatW; start++) {
      if (binary[start] === 0 || labels[start] !== 0) continue;
      const label = next++;
      let area = 0;
      let peak = -Infinity;
      let minX = heatW;
      let minY = heatH;
      let maxX = -1;
      let maxY = -1;
      stack.push(start);
      labels[start] = label;
      while (stack.length > 0) {
        const i = stack.pop()!;
        const x = i % heatW;
        const y = (i - x) / heatW;
        area++;
        peak = Math.max(peak, data[i * 2]!);
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
        const neighbors = [
          x > 0 ? i - 1 : -1,
          x < heatW - 1 ? i + 1 : -1,
          y > 0 ? i - heatW : -1,
          y < heatH - 1 ? i + heatW : -1,
        ];
        for (const n of neighbors) {
          if (n >= 0 && binary[n] === 1 && labels[n] === 0) {
            labels[n] = label;
            stack.push(n);
          }
        }
      }
      if (area < opts.minArea || peak < opts.lowTextThreshold) continue;
      boxes.push(
        minX * opts.scale,
        minY * opts.scale,
        (maxX + 1) * opts.scale,
        (maxY + 1) * opts.scale,
        0
      );
    }
    return Float32Array.from(boxes).buffer;
  },
};

export function installFakeRnOcr(): void {
  // eslint-disable-next-line camelcase
  (globalThis as { __rnocr_jsi__?: RnOcrJsi }).__rnocr_jsi__ = fakeRnOcr;
}
