/**
 * Typed wrappers over the native ops.
 *
 * Each wrapper is a worklet that moves bytes between react-native-executorch
 * tensors and the package's native kernels: `getData` out of the input tensor,
 * one native call on plain buffers, `tensor(...)` for the result. Page-sized
 * ops therefore cost one copy each way, which is acceptable for the once-per-
 * page operations (orientation, dewarp) and the small per-region crops.
 */
import { RnExecuTorchError, tensor, type Tensor, type cv } from 'react-native-executorch';
import { rnocrJsi } from './bridge';

/**
 * A clockwise rotation by a multiple of 90 degrees.
 * @category Types
 */
export type RotationDegrees = 90 | 180 | 270;

/**
 * How {@link warpByGridTensor} samples between source pixels.
 * @category Types
 */
export type WarpInterpolation = 'nearest' | 'linear' | 'cubic';

/**
 * How {@link warpByGridTensor} fills samples that fall outside the source.
 * @category Types
 */
export type WarpBorderMode = 'replicate' | 'constant';

/**
 * Options for {@link warpByGridTensor}.
 * @category Types
 */
export type WarpByGridOptions = {
  /** Sampling filter. Default `'linear'`. */
  readonly interpolation?: WarpInterpolation;
  /** Treatment of samples outside the source. Default `'replicate'`. */
  readonly borderMode?: WarpBorderMode;
  /** Fill value for every channel under `borderMode: 'constant'`. Default 0. */
  readonly borderValue?: number;
};

/**
 * Thresholds for {@link extractCraftTextBoxes}.
 * @category Types
 */
export type CraftDecodeOptions = {
  /** Region-score threshold for a pixel to count as text. */
  readonly textThreshold: number;
  /** Affinity-score threshold for linking adjacent glyphs into a line. */
  readonly linkThreshold: number;
  /** Minimum peak region score for a component to survive. */
  readonly lowTextThreshold: number;
  /** Components covering fewer heatmap pixels than this are dropped. */
  readonly minArea: number;
  /**
   * `false` links glyphs into line-level components and keeps each one's
   * angle; `true` breaks lines into upright per-glyph components (angle 0).
   */
  readonly charLevel: boolean;
  /**
   * Multiplier from heatmap pixels to the returned coordinates. CRAFT's
   * heatmap is half the detector input, so pass `inputHeight / heatHeight`.
   */
  readonly scale: number;
};

/**
 * One connected text component from a CRAFT heatmap: an axis-aligned box
 * (`x0, y0` = min corner, `x1, y1` = max corner) in the coordinate space
 * selected by {@link CraftDecodeOptions.scale}, plus the angle in degrees of
 * the rotated rectangle fitted to the component.
 * @category Types
 */
export type CraftBox = {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
  readonly angle: number;
};

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
 * Reads a `uint8` HWC tensor's bytes into a fresh array.
 * @param t The tensor.
 * @param height Its height.
 * @param width Its width.
 * @param channels Its channel count.
 * @returns The pixel bytes.
 */
function readHwc(t: Tensor, height: number, width: number, channels: number): Uint8Array {
  'worklet';
  return t.getData(new Uint8Array(height * width * channels));
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
  const src = readHwc(image, height, width, channels);
  const out = rnocrJsi.rotate(src, height, width, channels, degrees);
  const swap = degrees !== 180;
  return tensor(
    'uint8',
    [swap ? width : height, swap ? height : width, channels],
    new Uint8Array(out)
  );
}

/**
 * Copies an axis-aligned region of a `uint8` `[height, width, channels]` image
 * tensor into a new tensor.
 *
 * The box is snapped outwards to whole pixels and clamped to the image, so a
 * detector's fractional region can be passed as is; the result has shape
 * `[ymax - ymin, xmax - xmin, channels]` after that adjustment. The caller
 * owns both tensors.
 * @category Functions
 * @param image The image to crop.
 * @param box The region in image pixels, `xyxy` format.
 * @returns A newly allocated tensor holding the region.
 * @throws {RnExecuTorchError} With code `INVALID_ARGUMENT` when `image` is not
 * a `uint8` rank-3 tensor or the box does not intersect it.
 */
export function cropTensor(image: Tensor, box: cv.BoundingBox<'xyxy'>): Tensor {
  'worklet';
  const [height, width, channels] = requireHwcUint8(image, 'cropTensor');
  const x0 = Math.max(0, Math.floor(box.xmin));
  const y0 = Math.max(0, Math.floor(box.ymin));
  const x1 = Math.min(width, Math.ceil(box.xmax));
  const y1 = Math.min(height, Math.ceil(box.ymax));
  if (!(x1 > x0 && y1 > y0)) {
    throw RnExecuTorchError(
      'INVALID_ARGUMENT',
      `cropTensor: box [${box.xmin}, ${box.ymin}, ${box.xmax}, ${box.ymax}] does not intersect the ${width}x${height} image`
    );
  }
  const src = readHwc(image, height, width, channels);
  const out = rnocrJsi.crop(src, height, width, channels, x0, y0, x1, y1);
  return tensor('uint8', [y1 - y0, x1 - x0, channels], new Uint8Array(out));
}

/**
 * Resamples a `uint8` `[height, width, channels]` image tensor through a
 * `grid_sample`-style backward field, as produced by the UVDoc dewarping
 * model, into a new tensor of the same size.
 *
 * `grid` is a `float32` tensor of shape `[2, gridHeight, gridWidth]` (a
 * leading batch dimension of 1 is accepted): channel 0 holds x and channel 1
 * holds y, both normalized to `[-1, 1]` with `align_corners = true`. The grid
 * is bilinearly upsampled to the image size natively, so it may be much
 * coarser than the image. Sampling is `cv::remap`. Images of 1 to 4 channels
 * are supported. The caller owns every tensor involved.
 * @category Functions
 * @param image The image to warp.
 * @param grid The sampling grid.
 * @param options Interpolation and border handling.
 * @returns A newly allocated tensor of the same shape as `image`.
 * @throws {RnExecuTorchError} With code `INVALID_ARGUMENT` when the tensors
 * are not of the expected dtype and rank.
 */
export function warpByGridTensor(image: Tensor, grid: Tensor, options?: WarpByGridOptions): Tensor {
  'worklet';
  const [height, width, channels] = requireHwcUint8(image, 'warpByGridTensor');
  const gridShape = grid.shape;
  const gridRank = gridShape.length;
  const gridOk =
    grid.dtype === 'float32' &&
    (gridRank === 3 || (gridRank === 4 && gridShape[0] === 1)) &&
    gridShape[gridRank - 3] === 2;
  if (!gridOk) {
    throw RnExecuTorchError(
      'INVALID_ARGUMENT',
      `warpByGridTensor: expected a float32 [2, gridHeight, gridWidth] grid, got ${grid.dtype}[${gridShape.join(', ')}]`
    );
  }
  const gridHeight = gridShape[gridRank - 2]!;
  const gridWidth = gridShape[gridRank - 1]!;
  const src = readHwc(image, height, width, channels);
  const field = grid.getData(new Float32Array(2 * gridHeight * gridWidth));
  const out = rnocrJsi.warpByGrid(
    src,
    height,
    width,
    channels,
    field,
    gridHeight,
    gridWidth,
    height,
    width,
    options
  );
  return tensor('uint8', [height, width, channels], new Uint8Array(out));
}

/**
 * Widens a recognizer strip to `targetWidth`, filling the new columns with the
 * mean color of the strip's four corner patches so the padding continues the
 * background instead of introducing a seam some recognizers read as a glyph.
 *
 * Works on raw bytes rather than tensors because the strip is an intermediate
 * that the caller writes straight into a pre-allocated canvas tensor with
 * `setData`. Images of 1 to 4 channels are supported.
 * @category Functions
 * @param strip The `[height, width, channels]` uint8 pixels.
 * @param height The strip height.
 * @param width The strip width.
 * @param channels The channel count.
 * @param targetWidth The output width, at least `width`.
 * @returns The `[height, targetWidth, channels]` pixels.
 * @throws {RnExecuTorchError} With code `INVALID_ARGUMENT` when `strip` does
 * not hold `height * width * channels` bytes or `targetWidth < width`.
 */
export function padCornerMean(
  strip: Uint8Array,
  height: number,
  width: number,
  channels: number,
  targetWidth: number
): Uint8Array {
  'worklet';
  if (strip.byteLength !== height * width * channels) {
    throw RnExecuTorchError(
      'INVALID_ARGUMENT',
      `padCornerMean: strip holds ${strip.byteLength} bytes, expected ${height * width * channels} for [${height}, ${width}, ${channels}]`
    );
  }
  return new Uint8Array(rnocrJsi.padCornerMean(strip, height, width, channels, targetWidth));
}

/**
 * Decodes a CRAFT region and affinity heatmap into text component boxes:
 * thresholds the two score maps, links (or, with `charLevel`, separates)
 * glyphs into connected components and fits a rotated rectangle to each.
 *
 * Line grouping and reading order are computed in TypeScript on the returned
 * boxes.
 * @category Functions
 * @param heatmap The detector's `float32` output of shape
 * `[heatHeight, heatWidth, 2]` (a leading batch dimension of 1 is accepted),
 * region and affinity scores interleaved.
 * @param options Decode thresholds and the heatmap-to-input scale.
 * @returns One box per surviving component, in no particular order.
 * @throws {RnExecuTorchError} With code `INVALID_ARGUMENT` when the heatmap
 * is not of the expected dtype and shape or an option is out of range.
 */
export function extractCraftTextBoxes(heatmap: Tensor, options: CraftDecodeOptions): CraftBox[] {
  'worklet';
  const shape = heatmap.shape;
  const rank = shape.length;
  const shapeOk =
    heatmap.dtype === 'float32' &&
    (rank === 3 || (rank === 4 && shape[0] === 1)) &&
    shape[rank - 1] === 2;
  if (!shapeOk) {
    throw RnExecuTorchError(
      'INVALID_ARGUMENT',
      `extractCraftTextBoxes: expected a float32 [heatHeight, heatWidth, 2] heatmap, got ${heatmap.dtype}[${shape.join(', ')}]`
    );
  }
  const heatHeight = shape[rank - 3]!;
  const heatWidth = shape[rank - 2]!;
  const heat = heatmap.getData(new Float32Array(heatHeight * heatWidth * 2));
  const flat = new Float32Array(
    rnocrJsi.extractCraftTextBoxes(heat, heatHeight, heatWidth, options)
  );
  const boxes: CraftBox[] = [];
  for (let i = 0; i + 4 < flat.length; i += 5) {
    boxes.push({
      x0: flat[i]!,
      y0: flat[i + 1]!,
      x1: flat[i + 2]!,
      y1: flat[i + 3]!,
      angle: flat[i + 4]!,
    });
  }
  return boxes;
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
