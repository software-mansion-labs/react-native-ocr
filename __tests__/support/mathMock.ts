/**
 * Mock of `react-native-executorch/math`: the tensor ops `src/` uses, over
 * {@link FakeTensor}. Only the last axis is supported, which is all the
 * pipeline needs; anything else throws so a test cannot pass by accident.
 */
import type { FakeTensor } from './fakeTensor';

function requireLastAxis(axis: number, rank: number, op: string): void {
  if (axis !== -1 && axis !== rank - 1) {
    throw new Error(`${op}: the mock supports the last axis only, got ${axis}`);
  }
}

export function argmax(src: FakeTensor, dst: FakeTensor, axis: number = -1): FakeTensor {
  requireLastAxis(axis, src.shape.length, 'argmax');
  const inner = src.shape[src.shape.length - 1]!;
  const outer = src.numel / inner;
  if (dst.dtype !== 'int32' || dst.numel !== outer) {
    throw new Error(
      `argmax: dst must be int32 with ${outer} elements, got ${dst.dtype}[${dst.shape}]`
    );
  }
  const data = src.data as ArrayLike<number | bigint>;
  const out = dst.data as Int32Array;
  for (let o = 0; o < outer; o++) {
    let best = 0;
    let bestValue = -Infinity;
    for (let i = 0; i < inner; i++) {
      const value = Number(data[o * inner + i]);
      if (value > bestValue) {
        bestValue = value;
        best = i;
      }
    }
    out[o] = best;
  }
  return dst;
}

export function gather(
  src: FakeTensor,
  indices: FakeTensor,
  dst: FakeTensor,
  axis: number = -1
): FakeTensor {
  requireLastAxis(axis, src.shape.length, 'gather');
  const inner = src.shape[src.shape.length - 1]!;
  const outer = src.numel / inner;
  const picks = indices.shape[indices.shape.length - 1]!;
  if (indices.numel !== outer * picks) {
    throw new Error(`gather: indices [${indices.shape}] do not match src [${src.shape}]`);
  }
  if (dst.dtype !== src.dtype || dst.numel !== indices.numel) {
    throw new Error(
      `gather: dst must be ${src.dtype}[${indices.shape}], got ${dst.dtype}[${dst.shape}]`
    );
  }
  const data = src.data as ArrayLike<number | bigint>;
  const idx = indices.data as ArrayLike<number | bigint>;
  const out = dst.data as { [i: number]: number | bigint };
  for (let o = 0; o < outer; o++) {
    for (let j = 0; j < picks; j++) {
      const at = Number(idx[o * picks + j]);
      if (at < 0 || at >= inner) throw new Error(`gather: index ${at} out of range [0, ${inner})`);
      out[o * picks + j] = data[o * inner + at]!;
    }
  }
  return dst;
}
