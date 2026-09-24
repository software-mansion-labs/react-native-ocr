import { tensor } from 'react-native-executorch';
import { isEmulator, rotateTensor } from '../../src';

/** Builds an `[h, w, 1]` image whose pixel value is its row-major index. */
function ramp(h: number, w: number) {
  return tensor(
    'uint8',
    [h, w, 1],
    Uint8Array.from({ length: h * w }, (_, i) => i)
  );
}

describe('rotateTensor', () => {
  it('rotates 90 degrees clockwise and swaps the axes', () => {
    const src = ramp(2, 3); // [[0,1,2],[3,4,5]]
    const out = rotateTensor(src, 90);
    expect(out.shape).toEqual([3, 2, 1]);
    // Clockwise: the first row becomes the last column.
    expect(Array.from(out.getData(new Uint8Array(6)))).toEqual([3, 0, 4, 1, 5, 2]);
    src.dispose();
    out.dispose();
  });

  it('rotates 180 degrees and keeps the axes', () => {
    const src = ramp(2, 3);
    const out = rotateTensor(src, 180);
    expect(out.shape).toEqual([2, 3, 1]);
    expect(Array.from(out.getData(new Uint8Array(6)))).toEqual([5, 4, 3, 2, 1, 0]);
    src.dispose();
    out.dispose();
  });

  it('rotates 270 degrees clockwise', () => {
    const src = ramp(2, 3);
    const out = rotateTensor(src, 270);
    expect(out.shape).toEqual([3, 2, 1]);
    expect(Array.from(out.getData(new Uint8Array(6)))).toEqual([2, 5, 1, 4, 0, 3]);
    src.dispose();
    out.dispose();
  });

  it('carries every channel', () => {
    const src = tensor('uint8', [1, 2, 3], Uint8Array.from([1, 2, 3, 4, 5, 6]));
    const out = rotateTensor(src, 90);
    expect(out.shape).toEqual([2, 1, 3]);
    expect(Array.from(out.getData(new Uint8Array(6)))).toEqual([1, 2, 3, 4, 5, 6]);
    src.dispose();
    out.dispose();
  });

  it('rejects tensors that are not uint8 HWC images', () => {
    const src = tensor('float32', [2, 3, 1]);
    expect(() => rotateTensor(src, 90)).toThrow(
      expect.objectContaining({ name: 'RnExecuTorchError', code: 'INVALID_ARGUMENT' })
    );
    src.dispose();
  });
});

describe('isEmulator', () => {
  it('reads the native flag', () => {
    expect(isEmulator()).toBe(false);
  });
});
