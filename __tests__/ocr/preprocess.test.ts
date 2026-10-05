import { tensor } from 'react-native-executorch';
import { createTensorPreprocessor } from '../../src/ocr/preprocess';
import { tensorTracker } from '../support/fakeTensor';

const codedError = (code: string) => expect.objectContaining({ name: 'RnExecuTorchError', code });

const OPTIONS = {
  resizeMode: 'stretch',
  interpolation: 'linear',
  normalizeOpts: { alpha: 1 / 255, beta: 0 },
} as const;

describe('createTensorPreprocessor', () => {
  it('rejects a shape that is not RGB channels-first, allocating nothing', () => {
    expect(() => createTensorPreprocessor(OPTIONS, [1, 1, 4, 4])).toThrow(
      codedError('SCHEMA_MISMATCH')
    );
    expect(() => createTensorPreprocessor(OPTIONS, [4, 4, 3])).toThrow(
      codedError('SCHEMA_MISMATCH')
    );
    expect(tensorTracker.liveCount()).toBe(0);
  });

  it('resizes, reorders to channels-first and normalizes an RGB page', () => {
    const preprocessor = createTensorPreprocessor(OPTIONS, [1, 3, 2, 2]);
    // 4x4 page, uniform per channel: R 255, G 51, B 0.
    const data = new Uint8Array(4 * 4 * 3);
    for (let i = 0; i < data.length; i += 3) data.set([255, 51, 0], i);
    const page = tensor('uint8', [4, 4, 3], data);
    const out = preprocessor.process(page, 'rgb');
    expect(out.shape).toEqual([1, 3, 2, 2]);
    const values = [...out.getData(new Float32Array(12))];
    expect(values.slice(0, 4)).toEqual([1, 1, 1, 1]);
    values.slice(4, 8).forEach((v) => expect(v).toBeCloseTo(0.2));
    expect(values.slice(8)).toEqual([0, 0, 0, 0]);
    page.dispose();
    preprocessor.dispose();
  });

  it('converts a BGRA page to RGB', () => {
    const preprocessor = createTensorPreprocessor(OPTIONS, [3, 1, 1]);
    const page = tensor('uint8', [1, 1, 4], Uint8Array.from([0, 51, 255, 255]));
    const values = [...preprocessor.process(page, 'bgra').getData(new Float32Array(3))];
    expect(values[0]).toBeCloseTo(1);
    expect(values[1]).toBeCloseTo(0.2);
    expect(values[2]).toBeCloseTo(0);
    page.dispose();
    preprocessor.dispose();
  });

  it('letterboxes with the pad value', () => {
    const preprocessor = createTensorPreprocessor(
      { ...OPTIONS, resizeMode: 'letterbox', padValue: 255 },
      [3, 4, 4]
    );
    // A 4x2 black page in a 4x4 input: one padded row above and below.
    const page = tensor('uint8', [2, 4, 3]);
    const values = [...preprocessor.process(page, 'rgb').getData(new Float32Array(48))];
    expect(values.slice(0, 16)).toEqual([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]);
    page.dispose();
    preprocessor.dispose();
  });

  it('keeps only its managed tensors alive between runs and frees them on dispose', () => {
    const preprocessor = createTensorPreprocessor(OPTIONS, [1, 3, 2, 2]);
    const page = tensor('uint8', [4, 4, 4]);
    preprocessor.process(page, 'rgba');
    preprocessor.process(page, 'rgba');
    page.dispose();
    expect(tensorTracker.liveCount()).toBe(4);
    preprocessor.dispose();
    expect(tensorTracker.liveCount()).toBe(0);
  });

  it('frees the per-run scratch when an op throws', () => {
    const preprocessor = createTensorPreprocessor(OPTIONS, [1, 3, 2, 2]);
    // Two channels match no pixel format, so the chain fails after the resize.
    const page = tensor('uint8', [4, 4, 2]);
    expect(() => preprocessor.process(page, 'rgb')).toThrow();
    page.dispose();
    expect(tensorTracker.liveCount()).toBe(4);
    preprocessor.dispose();
  });
});
