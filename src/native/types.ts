/**
 * The shape of the native JSI object this package installs as
 * `globalThis.__rnocr_jsi__`.
 *
 * Every function takes and returns plain `ArrayBuffer`s with explicit
 * dimensions. react-native-executorch tensors are JSI host objects that only
 * its own binary can construct or unwrap, so the boundary between the two
 * packages is raw bytes; `ops.ts` does the tensor bridging.
 *
 * Kept in sync by hand with `cpp/ops/bindings.cpp`.
 * @internal
 */
import type { CraftDecodeOptions, WarpByGridOptions } from './ops';

/**
 * A buffer argument: a bare `ArrayBuffer` or any typed-array view of one. The
 * native side unwraps a view through its `buffer`, `byteOffset` and
 * `byteLength`, so a sub-range of a larger buffer can be passed without a copy.
 * @internal
 */
export type NativeBuffer = ArrayBuffer | ArrayBufferView;

/** @internal */
export type RnOcrJsi = {
  /** Whether the app runs on an Android emulator or an iOS simulator. */
  readonly isEmulator: boolean;

  /**
   * Rotates a `uint8` `[height, width, channels]` image by a multiple of 90
   * degrees clockwise.
   * @returns A buffer holding the rotated image: `[width, height, channels]`
   * for 90 and 270, `[height, width, channels]` for 180.
   */
  rotate(
    src: NativeBuffer,
    height: number,
    width: number,
    channels: number,
    degreesClockwise: 90 | 180 | 270
  ): ArrayBuffer;

  /**
   * Copies the region `[x0, x1) x [y0, y1)` of a `uint8`
   * `[height, width, channels]` image. The bounds must lie inside the image
   * and enclose at least one pixel.
   * @returns A buffer holding `[y1 - y0, x1 - x0, channels]`.
   */
  crop(
    src: NativeBuffer,
    height: number,
    width: number,
    channels: number,
    x0: number,
    y0: number,
    x1: number,
    y1: number
  ): ArrayBuffer;

  /**
   * Resamples a `uint8` `[srcHeight, srcWidth, channels]` image through a
   * `float32` `[2, gridHeight, gridWidth]` backward field normalized to
   * `[-1, 1]` (`align_corners = true`), into `[dstHeight, dstWidth, channels]`.
   * Images of 1 to 4 channels are supported.
   */
  warpByGrid(
    src: NativeBuffer,
    srcHeight: number,
    srcWidth: number,
    channels: number,
    grid: NativeBuffer,
    gridHeight: number,
    gridWidth: number,
    dstHeight: number,
    dstWidth: number,
    options?: WarpByGridOptions
  ): ArrayBuffer;

  /**
   * Widens a `uint8` `[height, width, channels]` strip to `targetWidth`,
   * filling the new columns with the mean of its four corner patches.
   * Images of 1 to 4 channels are supported.
   * @returns A buffer holding `[height, targetWidth, channels]`.
   */
  padCornerMean(
    strip: NativeBuffer,
    height: number,
    width: number,
    channels: number,
    targetWidth: number
  ): ArrayBuffer;

  /**
   * Decodes a `float32` `[heatHeight, heatWidth, 2]` CRAFT heatmap (text and
   * affinity interleaved) into component boxes.
   * @returns A buffer of `float32` values, five per box:
   * `x0, y0, x1, y1, angle`.
   */
  extractCraftTextBoxes(
    heat: NativeBuffer,
    heatHeight: number,
    heatWidth: number,
    options: CraftDecodeOptions
  ): ArrayBuffer;
};
