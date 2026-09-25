import { tensor } from 'react-native-executorch';
import {
  cropTensor,
  extractCraftTextBoxes,
  isEmulator,
  padCornerMean,
  rotateTensor,
  warpByGridTensor,
  type CraftDecodeOptions,
} from '../../src';

/** Builds an `[h, w, ch]` image whose byte value is its row-major index. */
function ramp(h: number, w: number, ch = 1) {
  return tensor(
    'uint8',
    [h, w, ch],
    Uint8Array.from({ length: h * w * ch }, (_, i) => i % 256)
  );
}

function bytesOf(t: { getData: (dst: Uint8Array) => Uint8Array; numel: number }): number[] {
  return Array.from(t.getData(new Uint8Array(t.numel)));
}

const codedError = (code: string) => expect.objectContaining({ name: 'RnExecuTorchError', code });

describe('rotateTensor', () => {
  it('rotates 90 degrees clockwise and swaps the axes', () => {
    const src = ramp(2, 3); // [[0,1,2],[3,4,5]]
    const out = rotateTensor(src, 90);
    expect(out.shape).toEqual([3, 2, 1]);
    // Clockwise: the first row becomes the last column.
    expect(bytesOf(out)).toEqual([3, 0, 4, 1, 5, 2]);
    src.dispose();
    out.dispose();
  });

  it('rotates 180 degrees and keeps the axes', () => {
    const src = ramp(2, 3);
    const out = rotateTensor(src, 180);
    expect(out.shape).toEqual([2, 3, 1]);
    expect(bytesOf(out)).toEqual([5, 4, 3, 2, 1, 0]);
    src.dispose();
    out.dispose();
  });

  it('rotates 270 degrees clockwise', () => {
    const src = ramp(2, 3);
    const out = rotateTensor(src, 270);
    expect(out.shape).toEqual([3, 2, 1]);
    expect(bytesOf(out)).toEqual([2, 5, 1, 4, 0, 3]);
    src.dispose();
    out.dispose();
  });

  it('carries every channel', () => {
    const src = tensor('uint8', [1, 2, 3], Uint8Array.from([1, 2, 3, 4, 5, 6]));
    const out = rotateTensor(src, 90);
    expect(out.shape).toEqual([2, 1, 3]);
    expect(bytesOf(out)).toEqual([1, 2, 3, 4, 5, 6]);
    src.dispose();
    out.dispose();
  });

  it('rejects tensors that are not uint8 HWC images', () => {
    const src = tensor('float32', [2, 3, 1]);
    expect(() => rotateTensor(src, 90)).toThrow(codedError('INVALID_ARGUMENT'));
    src.dispose();
  });
});

describe('cropTensor', () => {
  it('copies the region', () => {
    // 3x4: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    const src = ramp(3, 4);
    const out = cropTensor(src, { format: 'xyxy', xmin: 1, ymin: 1, xmax: 3, ymax: 3 });
    expect(out.shape).toEqual([2, 2, 1]);
    expect(bytesOf(out)).toEqual([5, 6, 9, 10]);
    src.dispose();
    out.dispose();
  });

  it('keeps every channel', () => {
    const src = ramp(2, 2, 3);
    const out = cropTensor(src, { format: 'xyxy', xmin: 1, ymin: 1, xmax: 2, ymax: 2 });
    expect(out.shape).toEqual([1, 1, 3]);
    expect(bytesOf(out)).toEqual([9, 10, 11]);
    src.dispose();
    out.dispose();
  });

  it('snaps a fractional box outwards and clamps it to the image', () => {
    const src = ramp(3, 4);
    const out = cropTensor(src, { format: 'xyxy', xmin: 0.6, ymin: -2, xmax: 2.2, ymax: 9 });
    // x: floor(0.6)=0 .. ceil(2.2)=3; y: clamped to 0 .. 3.
    expect(out.shape).toEqual([3, 3, 1]);
    expect(bytesOf(out)).toEqual([0, 1, 2, 4, 5, 6, 8, 9, 10]);
    src.dispose();
    out.dispose();
  });

  it('rejects boxes that miss the image', () => {
    const src = ramp(3, 4);
    expect(() => cropTensor(src, { format: 'xyxy', xmin: 4, ymin: 0, xmax: 6, ymax: 2 })).toThrow(
      codedError('INVALID_ARGUMENT')
    );
    expect(() => cropTensor(src, { format: 'xyxy', xmin: 2, ymin: 1, xmax: 1, ymax: 2 })).toThrow(
      codedError('INVALID_ARGUMENT')
    );
    src.dispose();
  });

  it('rejects tensors that are not uint8 HWC images', () => {
    const src = tensor('uint8', [3, 4]);
    expect(() => cropTensor(src, { format: 'xyxy', xmin: 0, ymin: 0, xmax: 1, ymax: 1 })).toThrow(
      codedError('INVALID_ARGUMENT')
    );
    src.dispose();
  });
});

describe('warpByGridTensor', () => {
  /** A `[2, gh, gw]` identity field: x runs -1..1 across, y runs -1..1 down. */
  function identityGrid(gh: number, gw: number, batched = false) {
    const data = new Float32Array(2 * gh * gw);
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        data[y * gw + x] = gw > 1 ? -1 + (2 * x) / (gw - 1) : 0;
        data[gh * gw + y * gw + x] = gh > 1 ? -1 + (2 * y) / (gh - 1) : 0;
      }
    }
    return tensor('float32', batched ? [1, 2, gh, gw] : [2, gh, gw], data);
  }

  it('copies the image through an identity grid', () => {
    const src = ramp(4, 5, 3);
    const grid = identityGrid(2, 2);
    const out = warpByGridTensor(src, grid, { interpolation: 'nearest' });
    expect(out.shape).toEqual([4, 5, 3]);
    expect(bytesOf(out)).toEqual(bytesOf(src));
    src.dispose();
    grid.dispose();
    out.dispose();
  });

  it('accepts a batched grid and the default linear filter', () => {
    const src = ramp(3, 3);
    const grid = identityGrid(3, 3, true);
    const out = warpByGridTensor(src, grid);
    expect(bytesOf(out)).toEqual(bytesOf(src));
    src.dispose();
    grid.dispose();
    out.dispose();
  });

  it('flips horizontally through a mirrored grid', () => {
    const src = ramp(2, 4); // [[0,1,2,3],[4,5,6,7]]
    const grid = tensor('float32', [2, 2, 2], Float32Array.from([1, -1, 1, -1, -1, -1, 1, 1]));
    const out = warpByGridTensor(src, grid, { interpolation: 'nearest' });
    expect(bytesOf(out)).toEqual([3, 2, 1, 0, 7, 6, 5, 4]);
    src.dispose();
    grid.dispose();
    out.dispose();
  });

  it('fills out-of-source samples with the constant border value', () => {
    const src = ramp(2, 3);
    const grid = tensor('float32', [2, 2, 2], Float32Array.from([3, 3, 3, 3, -1, -1, 1, 1]));
    const out = warpByGridTensor(src, grid, { borderMode: 'constant', borderValue: 7 });
    expect(bytesOf(out)).toEqual([7, 7, 7, 7, 7, 7]);
    src.dispose();
    grid.dispose();
    out.dispose();
  });

  it('rejects grids that are not float32 [2, gh, gw]', () => {
    const src = ramp(2, 3);
    const bad = tensor('float32', [3, 2, 2]);
    expect(() => warpByGridTensor(src, bad)).toThrow(codedError('INVALID_ARGUMENT'));
    const wrongType = tensor('uint8', [2, 2, 2]);
    expect(() => warpByGridTensor(src, wrongType)).toThrow(codedError('INVALID_ARGUMENT'));
    src.dispose();
    bad.dispose();
    wrongType.dispose();
  });
});

describe('padCornerMean', () => {
  it('fills the new columns with the corner mean', () => {
    // 4x6 strip of 50s with corners 8, 12, 8, 12 -> mean 10.
    const strip = new Uint8Array(4 * 6).fill(50);
    strip[0] = 8;
    strip[5] = 12;
    strip[3 * 6] = 8;
    strip[3 * 6 + 5] = 12;
    const out = padCornerMean(strip, 4, 6, 1, 10);
    expect(out.length).toBe(4 * 10);
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 10; x++) {
        expect(out[y * 10 + x]).toBe(x < 6 ? strip[y * 6 + x] : 10);
      }
    }
  });

  it('averages per channel', () => {
    const strip = Uint8Array.from([10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30]); // 2x2x3
    const out = padCornerMean(strip, 2, 2, 3, 3);
    expect(Array.from(out.subarray(6, 9))).toEqual([10, 20, 30]);
    expect(Array.from(out.subarray(15, 18))).toEqual([10, 20, 30]);
  });

  it('returns a copy when the width is unchanged', () => {
    const strip = Uint8Array.from([1, 2, 3, 4]);
    expect(Array.from(padCornerMean(strip, 2, 2, 1, 2))).toEqual([1, 2, 3, 4]);
  });

  it('rejects strips whose size does not match the dimensions', () => {
    expect(() => padCornerMean(new Uint8Array(5), 2, 2, 1, 4)).toThrow(
      codedError('INVALID_ARGUMENT')
    );
  });
});

describe('extractCraftTextBoxes', () => {
  const options: CraftDecodeOptions = {
    textThreshold: 0.4,
    linkThreshold: 0.4,
    lowTextThreshold: 0.7,
    minArea: 10,
    charLevel: false,
    scale: 1,
  };

  /** A `[1, h, w, 2]` heatmap with text painted over inclusive rectangles. */
  function heat(h: number, w: number, blobs: [number, number, number, number, number?][]) {
    const data = new Float32Array(h * w * 2);
    for (const [x0, y0, x1, y1, value = 1] of blobs) {
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) data[(y * w + x) * 2] = value;
    }
    return tensor('float32', [1, h, w, 2], data);
  }

  it('returns one box per blob, scaled to the detector input', () => {
    const map = heat(32, 32, [[8, 8, 15, 15]]);
    const boxes = extractCraftTextBoxes(map, { ...options, scale: 2 });
    expect(boxes).toHaveLength(1);
    const box = boxes[0]!;
    expect(box.x0).toBeLessThanOrEqual(16);
    expect(box.y0).toBeLessThanOrEqual(16);
    expect(box.x1).toBeGreaterThanOrEqual(32);
    expect(box.y1).toBeGreaterThanOrEqual(32);
    expect(box.angle).toBe(0);
    map.dispose();
  });

  it('finds every disjoint blob and nothing on an empty map', () => {
    const two = heat(48, 48, [
      [4, 4, 15, 15],
      [30, 30, 43, 43],
    ]);
    expect(extractCraftTextBoxes(two, options)).toHaveLength(2);
    two.dispose();
    const empty = heat(16, 16, []);
    expect(extractCraftTextBoxes(empty, options)).toEqual([]);
    empty.dispose();
  });

  it('drops weak and tiny components', () => {
    const weak = heat(32, 32, [[8, 8, 15, 15, 0.5]]);
    expect(extractCraftTextBoxes(weak, options)).toEqual([]);
    weak.dispose();
    const tiny = heat(32, 32, [[8, 8, 10, 10]]); // 9 px
    expect(extractCraftTextBoxes(tiny, options)).toEqual([]);
    expect(extractCraftTextBoxes(tiny, { ...options, minArea: 9 })).toHaveLength(1);
    tiny.dispose();
  });

  it('rejects heatmaps that are not float32 [.., h, w, 2]', () => {
    const wrongChannels = tensor('float32', [1, 8, 8, 1]);
    expect(() => extractCraftTextBoxes(wrongChannels, options)).toThrow(
      codedError('INVALID_ARGUMENT')
    );
    wrongChannels.dispose();
    const wrongType = tensor('uint8', [8, 8, 2]);
    expect(() => extractCraftTextBoxes(wrongType, options)).toThrow(codedError('INVALID_ARGUMENT'));
    wrongType.dispose();
  });
});

describe('isEmulator', () => {
  it('reads the native flag', () => {
    expect(isEmulator()).toBe(false);
  });
});
