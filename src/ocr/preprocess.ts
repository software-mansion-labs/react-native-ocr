/**
 * Model-input preprocessing for a page that already lives in a tensor.
 * react-native-executorch's `createImagePreprocessor` only accepts an
 * `ImageBuffer`; the pipeline keeps the (possibly corrected) page on the native
 * side, so it needs the same chain starting from a tensor.
 */
import { RnExecuTorchError, tensor, type Tensor } from 'react-native-executorch';
import {
  cvtColor,
  normalize,
  resize,
  toChannelsFirst,
  type ImageFormat,
  type InterpolationMethod,
  type NormalizeOptions,
  type ResizeMode,
} from 'react-native-executorch/cv';
import { FORMAT_CHANNELS, TO_RGB } from './imageFormat';

const RGB_CHANNELS = 3;

/**
 * Options of {@link createTensorPreprocessor}.
 */
export type TensorPreprocessorOptions = {
  /** How the page is fitted to the model's input size. */
  readonly resizeMode: ResizeMode;
  /** Interpolation used when resizing. */
  readonly interpolation: InterpolationMethod;
  /** Normalization applied to the RGB pixels. */
  readonly normalizeOpts: NormalizeOptions;
  /** Background fill when letterboxing. Default 0. */
  readonly padValue?: number;
};

/**
 * A reusable preprocessing chain bound to one model input shape.
 */
export type TensorPreprocessor = {
  /**
   * Resizes, converts to RGB, reorders to channels-first and normalizes `src`
   * into the managed output tensor. Must run on the thread that uses the
   * result; the returned tensor belongs to the preprocessor, so do not dispose
   * it.
   * @param src The page, `uint8` `[H, W, C]`.
   * @param format The pixel format of `src`.
   * @returns The managed `float32` tensor of the shape given at creation.
   */
  readonly process: (src: Tensor, format: ImageFormat) => Tensor;
  /** Releases the scratch and output tensors. */
  readonly dispose: () => void;
};

/**
 * Creates a preprocessing chain (`resize`, `cvtColor` when the source is not
 * RGB, `toChannelsFirst`, `normalize`) for a model input of a fixed shape. The
 * scratch tensors that depend only on that shape are allocated once; the one
 * that depends on the source's channel count is allocated per call.
 * @param options Resize mode, interpolation, normalization and padding.
 * @param outputShape The model input shape, `[1, 3, H, W]` or `[3, H, W]`.
 * @returns The `process` runner and a `dispose` method.
 * @throws {RnExecuTorchError} With code `SCHEMA_MISMATCH` when `outputShape`
 * is not an RGB channels-first shape.
 */
export function createTensorPreprocessor(
  options: TensorPreprocessorOptions,
  outputShape: number[]
): TensorPreprocessor {
  const isRank3 = outputShape.length === 3 && outputShape[0] === RGB_CHANNELS;
  const isRank4 = outputShape.length === 4 && outputShape[1] === RGB_CHANNELS;
  if (!isRank3 && !isRank4) {
    throw RnExecuTorchError(
      'SCHEMA_MISMATCH',
      `createTensorPreprocessor: got shape [${outputShape}], expected ` +
        `[${RGB_CHANNELS}, H, W] or [1, ${RGB_CHANNELS}, H, W]`
    );
  }

  const targetH = outputShape[outputShape.length - 2]!;
  const targetW = outputShape[outputShape.length - 1]!;
  const tensors = [
    tensor('uint8', [targetH, targetW, RGB_CHANNELS]),
    tensor('uint8', [RGB_CHANNELS, targetH, targetW]),
    tensor('float32', [RGB_CHANNELS, targetH, targetW]),
    tensor('float32', outputShape),
  ] as const;
  const [tColor, tChanFirst, tNorm, tOutput] = tensors;
  const { resizeMode, interpolation, normalizeOpts, padValue } = options;

  const dispose = () => tensors.forEach((t) => t.dispose());

  const process = (src: Tensor, format: ImageFormat): Tensor => {
    'worklet';
    const numChannels = FORMAT_CHANNELS[format];
    const colorCode = TO_RGB[format];
    const tResize = tensor('uint8', [targetH, targetW, numChannels]);
    try {
      src
        .through(resize, tResize, { mode: resizeMode, interpolation, padValue })
        .throughIf(colorCode !== null, cvtColor, tColor, colorCode!)
        .through(toChannelsFirst, tChanFirst)
        .through(normalize, tNorm, normalizeOpts)
        .copyTo(tOutput);
    } finally {
      tResize.dispose();
    }
    return tOutput;
  };

  return { process, dispose };
}
