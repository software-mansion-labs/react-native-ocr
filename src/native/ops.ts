/**
 * Typed wrappers over the native ops.
 *
 * Each wrapper is a worklet that moves bytes between react-native-executorch
 * tensors and the package's native kernels: `getData` out of the input tensor,
 * one native call on plain buffers, `tensor(...)` for the result. Page-sized
 * ops therefore cost one copy each way, which is acceptable for the once-per-
 * page operations (orientation, dewarp) and the small per-region crops.
 */
import { RnExecuTorchError, tensor, type Tensor } from 'react-native-executorch';
import { rnocrJsi } from './bridge';

/**
 * A clockwise rotation by a multiple of 90 degrees.
 * @category Types
 */
export type RotationDegrees = 90 | 180 | 270;

/**
 * Asserts that `t` is a `uint8` `[height, width, channels]` image tensor.
 * @param t The tensor to check.
 * @param op The calling op, for the error message.
 * @returns The three dimensions.
 */
function requireHwcUint8(t: Tensor, op: string): [number, number, number] {
  'worklet';
  if (t.dtype !== 'uint8' || t.shape.length !== 3) {
    throw RnExecuTorchError(
      'INVALID_ARGUMENT',
      `${op}: expected a uint8 [height, width, channels] tensor, got ${t.dtype}[${t.shape.join(', ')}]`
    );
  }
  return [t.shape[0]!, t.shape[1]!, t.shape[2]!];
}

/**
 * Rotates a `uint8` `[height, width, channels]` image tensor clockwise by 90,
 * 180 or 270 degrees.
 *
 * Runs `cv::rotate` natively; the result is a new tensor of shape
 * `[width, height, channels]` for 90 and 270 and `[height, width, channels]`
 * for 180. The caller owns both tensors.
 * @category Functions
 * @param image The image to rotate.
 * @param degrees The clockwise rotation.
 * @returns A newly allocated tensor holding the rotated image.
 * @throws {RnExecuTorchError} With code `INVALID_ARGUMENT` when `image` is not
 * a `uint8` rank-3 tensor.
 */
export function rotateTensor(image: Tensor, degrees: RotationDegrees): Tensor {
  'worklet';
  const [height, width, channels] = requireHwcUint8(image, 'rotateTensor');
  const src = image.getData(new Uint8Array(height * width * channels));
  const out = rnocrJsi.rotate(src.buffer, height, width, channels, degrees);
  const swap = degrees !== 180;
  return tensor(
    'uint8',
    [swap ? width : height, swap ? height : width, channels],
    new Uint8Array(out)
  );
}

/**
 * Reports whether the app runs on an Android emulator or an iOS simulator.
 *
 * Mirrors the check react-native-executorch uses internally when it picks a
 * model's `DEFAULT` variant, which that package does not export.
 * @category Functions
 * @returns `true` on an emulator or simulator.
 */
export function isEmulator(): boolean {
  'worklet';
  return rnocrJsi.isEmulator;
}
