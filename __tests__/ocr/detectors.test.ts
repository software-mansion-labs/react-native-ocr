import { cv, schema, tensor } from 'react-native-executorch';
import {
  craftExtractBoxes,
  dbnetExtractBoxes,
  makeCraftExtractBoxes,
  makeDbnetExtractBoxes,
} from '../../src/ocr/detectors';
import { setFakeDbnetTextQuads, type DbnetDecodeOptions } from '../support/cvMock';
import { fakeRnOcr } from '../support/fakeRnOcr';

const codedError = (code: string) => expect.objectContaining({ name: 'RnExecuTorchError', code });

const rect = (x0: number, y0: number, x1: number, y1: number): cv.Quad => [
  { x: x0, y: y0 },
  { x: x1, y: y0 },
  { x: x1, y: y1 },
  { x: x0, y: y1 },
];

/** A `[1, h, w, 2]` CRAFT heatmap with region score 1 inside the given blobs. */
function heatmapOf(h: number, w: number, blobs: [number, number, number, number][]) {
  const data = new Float32Array(h * w * 2);
  for (const [x0, y0, x1, y1] of blobs) {
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) data[(y * w + x) * 2] = 1;
    }
  }
  return tensor('float32', [1, h, w, 2], data);
}

describe('makeCraftExtractBoxes', () => {
  it('declares a half-resolution heatmap that needs even input sizes', () => {
    expect(craftExtractBoxes.inputAlignment).toBe(2);
    expect(craftExtractBoxes.supportsCharLevel).toBe(true);
    expect(craftExtractBoxes.outputShapes({ width: 64, height: 32 })).toEqual([[1, 16, 32, 2]]);
    expect(() => craftExtractBoxes.outputShapes({ width: 63, height: 32 })).toThrow(
      codedError('INVALID_ARGUMENT')
    );
    expect(() => craftExtractBoxes.outputShapes({ width: 64, height: 31 })).toThrow(
      codedError('INVALID_ARGUMENT')
    );
  });

  it('describes its detect output with the caller-chosen dimension kind', () => {
    expect(craftExtractBoxes.detectOutputSpec(schema.DynamicDim)).toEqual([
      {
        kind: 'Tensor',
        dtype: 'float32',
        shape: [
          { kind: 'constant', value: 1 },
          { kind: 'dynamic', symbol: 'detOutH' },
          { kind: 'dynamic', symbol: 'detOutW' },
          { kind: 'constant', value: 2 },
        ],
      },
    ]);
    expect(craftExtractBoxes.detectOutputSpec(schema.StaticDim)[0]).toMatchObject({
      kind: 'Tensor',
      shape: [
        { kind: 'constant', value: 1 },
        { kind: 'static', symbol: 'detOutH' },
        { kind: 'static', symbol: 'detOutW' },
        { kind: 'constant', value: 2 },
      ],
    });
  });

  it('decodes a heatmap blob into one line quad in detector-input pixels', () => {
    const heat = heatmapOf(16, 32, [[4, 4, 28, 12]]);
    const quads = craftExtractBoxes.extract([heat], { width: 64, height: 32 }, false);
    heat.dispose();
    expect(quads).toEqual([rect(8, 8, 56, 24)]);
  });

  it('passes the thresholds and the heatmap scale to the native op', () => {
    const spy = jest.spyOn(fakeRnOcr, 'extractCraftTextBoxes');
    const heat = heatmapOf(16, 32, []);
    makeCraftExtractBoxes({ textThreshold: 0.5, minArea: 3 }).extract(
      [heat],
      { width: 64, height: 32 },
      true
    );
    heat.dispose();
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0]!.slice(1)).toEqual([
      16,
      32,
      {
        textThreshold: 0.5,
        linkThreshold: 0.4,
        lowTextThreshold: 0.7,
        minArea: 3,
        charLevel: true,
        scale: 2,
      },
    ]);
  });

  it('skips line grouping at char level so small glyphs survive', () => {
    const heat = heatmapOf(16, 32, [
      [2, 2, 6, 6],
      [10, 2, 14, 6],
    ]);
    // Line mode: 8 px glyph boxes are below the readable-line minimum.
    expect(craftExtractBoxes.extract([heat], { width: 64, height: 32 }, false)).toEqual([]);
    expect(craftExtractBoxes.extract([heat], { width: 64, height: 32 }, true)).toEqual([
      rect(4, 4, 12, 12),
      rect(20, 4, 28, 12),
    ]);
    heat.dispose();
  });
});

describe('makeDbnetExtractBoxes', () => {
  afterEach(() => setFakeDbnetTextQuads(null));

  it('declares a full-resolution probability map with no alignment or char level', () => {
    expect(dbnetExtractBoxes.inputAlignment).toBeUndefined();
    expect(dbnetExtractBoxes.supportsCharLevel).toBeFalsy();
    expect(dbnetExtractBoxes.outputShapes({ width: 64, height: 32 })).toEqual([[1, 1, 32, 64]]);
    expect(dbnetExtractBoxes.detectOutputSpec(schema.DynamicDim)).toEqual([
      {
        kind: 'Tensor',
        dtype: 'float32',
        shape: [
          { kind: 'constant', value: 1 },
          { kind: 'constant', value: 1 },
          { kind: 'dynamic', symbol: 'detOutH' },
          { kind: 'dynamic', symbol: 'detOutW' },
        ],
      },
    ]);
  });

  it("forwards the default thresholds to react-native-executorch's decoder", () => {
    let seen: DbnetDecodeOptions | null = null;
    const quad = rect(1, 2, 30, 12);
    setFakeDbnetTextQuads((_map, options) => {
      seen = options;
      return [quad];
    });
    const probs = tensor('float32', [1, 1, 32, 64]);
    expect(dbnetExtractBoxes.extract([probs], { width: 64, height: 32 }, true)).toEqual([quad]);
    probs.dispose();
    expect(seen).toEqual({
      binThreshold: 0.3,
      boxThreshold: 0.6,
      unclipRatio: 1.5,
      minBoxSide: 3,
      maxCandidates: 1000,
    });
  });

  it('forwards overrides', () => {
    let seen: DbnetDecodeOptions | null = null;
    setFakeDbnetTextQuads((_map, options) => {
      seen = options;
      return [];
    });
    const probs = tensor('float32', [1, 1, 8, 8]);
    makeDbnetExtractBoxes({ binThreshold: 0.2, maxCandidates: 5 }).extract(
      [probs],
      { width: 8, height: 8 },
      false
    );
    probs.dispose();
    expect(seen).toEqual({
      binThreshold: 0.2,
      boxThreshold: 0.6,
      unclipRatio: 1.5,
      minBoxSide: 3,
      maxCandidates: 5,
    });
  });
});
