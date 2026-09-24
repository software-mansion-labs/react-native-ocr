/**
 * The shape of the native JSI object this package installs as
 * `globalThis.__rnocr_jsi__`.
 *
 * Every function takes and returns plain `ArrayBuffer`s with explicit
 * dimensions. react-native-executorch tensors are JSI host objects that only
 * its own binary can construct or unwrap, so the boundary between the two
 * packages is raw bytes; `ops.ts` does the tensor bridging.
 *
 * Kept in sync by hand with `cpp/RnOcr.cpp`.
 * @internal
 */
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
    src: ArrayBuffer,
    height: number,
    width: number,
    channels: number,
    degreesClockwise: 90 | 180 | 270
  ): ArrayBuffer;
};
