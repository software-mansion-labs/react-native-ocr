/**
 * JavaScript implementation of the package's native JSI object, installed as
 * `globalThis.__rnocr_jsi__` before `src/` is imported. Each op mirrors the
 * C++ kernel's contract (buffer in, buffer out, explicit dimensions) so the
 * TypeScript wrappers are exercised exactly as on a device.
 */
import type { RnOcrJsi } from '../../src/native/types';

function requireBytes(buffer: ArrayBuffer, expected: number, op: string): Uint8Array {
  if (!(buffer instanceof ArrayBuffer)) throw new Error(`${op}: expected an ArrayBuffer`);
  if (buffer.byteLength !== expected) {
    throw new Error(`${op}: expected ${expected} bytes, got ${buffer.byteLength}`);
  }
  return new Uint8Array(buffer);
}

export const fakeRnOcr: RnOcrJsi = {
  isEmulator: false,

  rotate(src, height, width, channels, degrees) {
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
          throw new Error('rotate: degrees must be 90, 180 or 270');
        }
        const from = (y * width + x) * channels;
        const to = (oy * outW + ox) * channels;
        for (let c = 0; c < channels; c++) out[to + c] = input[from + c]!;
      }
    }
    return out.buffer;
  },
};

export function installFakeRnOcr(): void {
  // eslint-disable-next-line camelcase
  (globalThis as { __rnocr_jsi__?: RnOcrJsi }).__rnocr_jsi__ = fakeRnOcr;
}
