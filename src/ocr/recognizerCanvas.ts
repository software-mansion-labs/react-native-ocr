/**
 * Fills the recognizer's input canvas from one text quad. The content is
 * rectified to the canvas height at its natural width and sits on the left;
 * what remains of the canvas width is padding, filled according to the
 * model's pad mode.
 */
import { tensor, type Tensor } from 'react-native-executorch';
import { rectifyQuad, type Quad } from 'react-native-executorch/cv';
import { padCornerMean } from '../native/ops';

/**
 * How {@link fillRecognizerCanvas} lays a quad out on the canvas.
 */
export type RecognizerCanvasOptions = {
  /** Width in px the rectified content occupies, at most the canvas width. */
  readonly contentWidth: number;
  /**
   * Padding fill: the flat `padValue`, or the mean of the content's four
   * corner patches, which avoids a seam some recognizers read as a glyph.
   */
  readonly padMode: 'constant' | 'cornerMean';
  /** The fill for `'constant'`. */
  readonly padValue: number;
};

/**
 * Rectifies `quad` out of `src` into `canvas` and pads the unused width.
 *
 * `'constant'` is a single native pass. `'cornerMean'` rectifies into a strip
 * exactly as wide as the content, pads that strip natively to the canvas
 * width and uploads the result, which costs one round trip of a strip-sized
 * buffer.
 * @param src The page, `uint8` `[H, W, C]`.
 * @param canvas The pre-allocated canvas, `uint8` `[canvasH, canvasW, C]`.
 * @param quad The region corners (TL, TR, BR, BL) in `src` pixels.
 * @param options Content width and padding.
 * @returns `canvas`.
 */
export function fillRecognizerCanvas(
  src: Tensor,
  canvas: Tensor,
  quad: Quad,
  options: RecognizerCanvasOptions
): Tensor {
  'worklet';
  const { contentWidth, padMode, padValue } = options;
  const canvasHeight = canvas.shape[0]!;
  const canvasWidth = canvas.shape[1]!;
  const channels = canvas.shape[2]!;
  // Content that fills the canvas leaves nothing to pad.
  if (padMode === 'constant' || contentWidth >= canvasWidth) {
    return rectifyQuad(src, canvas, quad, { contentWidth, align: 'left', padValue });
  }

  const tStrip = tensor('uint8', [canvasHeight, contentWidth, channels]);
  try {
    rectifyQuad(src, tStrip, quad, { contentWidth, align: 'left', padValue });
    const strip = tStrip.getData(new Uint8Array(canvasHeight * contentWidth * channels));
    return canvas.setData(padCornerMean(strip, canvasHeight, contentWidth, channels, canvasWidth));
  } finally {
    tStrip.dispose();
  }
}
