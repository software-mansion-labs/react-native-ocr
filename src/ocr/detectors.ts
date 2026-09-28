/**
 * Built-in detector box-extraction strategies (CRAFT, DBNet): each turns one
 * architecture's `detect` output into quads. The pipeline is model-agnostic
 * and only ever calls `OcrModelOptions.extractBoxes`.
 */
import { RnExecuTorchError, type Tensor } from 'react-native-executorch';
import { extractDbnetTextQuads, type Quad } from 'react-native-executorch/cv';
import { f32, type ParamSpec, type SymbolicDim } from 'react-native-executorch/schema';
import { extractCraftTextBoxes } from '../native/ops';
import { boxToQuad, groupBoxes } from './geometry';

/**
 * A detector's box-extraction strategy. Plug a new detector architecture into
 * the OCR pipeline by supplying an object of this type (a built-in below, or
 * your own). `outputShapes` and `extract` MUST be worklets.
 * @category Types
 */
export type TextBoxExtractor = {
  /**
   * The detector input dims (W and H) must be a multiple of this; the pipeline
   * snaps the letterbox size up to it before allocating, so `outputShapes`
   * never receives a size it must reject. Default 1 (no alignment). CRAFT
   * needs 2 (its heatmap is half-resolution).
   */
  readonly inputAlignment?: number;
  /**
   * Whether `extract(.., charLevel = true)` yields per-glyph boxes. When false
   * the stacked-text (vertical) pass skips its char-level re-detection split
   * and reads those boxes horizontally instead. Default false.
   */
  readonly supportsCharLevel?: boolean;
  /**
   * The allowed spec of the `detect` outputs this strategy decodes, used to
   * validate the model at load. Dimensions that track the detector input size
   * must be built with the supplied `dim` factory: the pipeline passes
   * `schema.DynamicDim` when validating a size-varying detector and
   * `schema.StaticDim` when validating a fixed-size one, so one declaration
   * covers both.
   * @param dim Symbol factory for every input-size-dependent dimension.
   * @returns One param spec per `detect` output, in order.
   */
  readonly detectOutputSpec: (dim: (symbol: string) => SymbolicDim) => ParamSpec<SymbolicDim>[];
  /**
   * The `float32` output tensor shapes the `detect` method produces for a
   * given detector input size, so the caller can pre-allocate them. One shape
   * per `detect` output, in order.
   * @param inputSize The detector input size the image was letterboxed to.
   * @returns One shape per `detect` output tensor.
   */
  readonly outputShapes: (inputSize: {
    readonly width: number;
    readonly height: number;
  }) => number[][];
  /**
   * Turns the model's `detect` output tensors into oriented quads in
   * detector-input pixel space.
   * @param outputs The model's `detect` output tensors, in order.
   * @param inputSize The detector input size the image was letterboxed to.
   * @param charLevel Emit one box per glyph instead of grouped lines;
   * strategies without a char-level mode ignore it.
   * @returns Oriented quads (TL, TR, BR, BL) in detector-input pixel space.
   */
  readonly extract: (
    outputs: readonly Tensor[],
    inputSize: { readonly width: number; readonly height: number },
    charLevel: boolean
  ) => Quad[];
};

/**
 * CRAFT decode thresholds: tuning knobs for {@link makeCraftExtractBoxes}. All
 * optional; the defaults suit the built-in CRAFT models.
 * @category Types
 */
export type CraftExtractOptions = {
  /** Region-score threshold for a pixel to count as text. Default 0.4. */
  readonly textThreshold?: number;
  /** Affinity-score threshold for linking adjacent glyphs into a line. Default 0.4. */
  readonly linkThreshold?: number;
  /** Minimum peak region score for a component to survive. Default 0.7. */
  readonly lowTextThreshold?: number;
  /** Components covering fewer heatmap pixels than this are dropped. Default 10. */
  readonly minArea?: number;
};

/**
 * Builds a CRAFT box-extraction strategy: native region + affinity heatmap
 * decode (`outputs[0]` = `[1, Hd, Wd, 2]`), then line grouping and de-skew in
 * TypeScript; per-glyph boxes when `charLevel`. Detector input dims must be
 * even (half-resolution heatmap). {@link craftExtractBoxes} is this with the
 * default thresholds.
 * @category Functions
 * @param options Threshold overrides; omit for the built-in defaults.
 * @returns A {@link TextBoxExtractor} bound to those thresholds.
 */
export function makeCraftExtractBoxes(options: CraftExtractOptions = {}): TextBoxExtractor {
  const textThreshold = options.textThreshold ?? 0.4;
  const linkThreshold = options.linkThreshold ?? 0.4;
  const lowTextThreshold = options.lowTextThreshold ?? 0.7;
  const minArea = options.minArea ?? 10;
  return {
    // Half-resolution heatmap: input dims must be even. The pipeline snaps to
    // this, so the guard below only ever fires on a hand-rolled caller.
    inputAlignment: 2,
    supportsCharLevel: true,
    // Half-resolution NHWC heatmap: [1, H/2, W/2, 2] (region + affinity).
    detectOutputSpec: (dim) => [f32(1, dim('detOutH'), dim('detOutW'), 2)],
    outputShapes: ({ width, height }) => {
      'worklet';
      if (width % 2 !== 0 || height % 2 !== 0) {
        throw RnExecuTorchError(
          'INVALID_ARGUMENT',
          'craftExtractBoxes: detect input dimensions must be even ' +
            `(half-resolution heatmap), got ${width}x${height}.`
        );
      }
      return [[1, height / 2, width / 2, 2]];
    },
    extract: (outputs, inputSize, charLevel) => {
      'worklet';
      const heatmap = outputs[0]!;
      const heatHeight = heatmap.shape[heatmap.shape.length - 3]!;
      const boxes = extractCraftTextBoxes(heatmap, {
        textThreshold,
        linkThreshold,
        lowTextThreshold,
        minArea,
        charLevel,
        // Heatmap pixels to detector-input pixels (2 for CRAFT's half-res map).
        scale: inputSize.height / heatHeight,
      });
      // Char-level glyphs are read individually, so they skip line grouping.
      return (charLevel ? boxes : groupBoxes(boxes)).map(boxToQuad);
    },
  };
}

/**
 * CRAFT box-extraction strategy with the default thresholds.
 * @category Constants
 */
export const craftExtractBoxes: TextBoxExtractor = makeCraftExtractBoxes();

/**
 * DBNet decode thresholds: tuning knobs for {@link makeDbnetExtractBoxes}. All
 * optional; the defaults suit the built-in DBNet models.
 * @category Types
 */
export type DbnetExtractOptions = {
  /** Binarization threshold on the probability map. Default 0.3. */
  readonly binThreshold?: number;
  /** Minimum mean box score to keep a candidate. Default 0.6. */
  readonly boxThreshold?: number;
  /** How far to expand (unclip) each shrunk box. Default 1.5. */
  readonly unclipRatio?: number;
  /** Discard boxes with a side smaller than this (px). Default 3. */
  readonly minBoxSide?: number;
  /** Cap on contour candidates scored per map. Default 1000. */
  readonly maxCandidates?: number;
};

/**
 * Builds a DBNet box-extraction strategy: thresholds and unclips the
 * full-resolution post-sigmoid probability map (`outputs[0]` = `[1, 1, H, W]`)
 * into oriented quads with react-native-executorch's native decoder. No
 * char-level mode. {@link dbnetExtractBoxes} is this with the defaults.
 * @category Functions
 * @param options Threshold overrides; omit for the built-in defaults.
 * @returns A {@link TextBoxExtractor} bound to those thresholds.
 */
export function makeDbnetExtractBoxes(options: DbnetExtractOptions = {}): TextBoxExtractor {
  const binThreshold = options.binThreshold ?? 0.3;
  const boxThreshold = options.boxThreshold ?? 0.6;
  const unclipRatio = options.unclipRatio ?? 1.5;
  const minBoxSide = options.minBoxSide ?? 3;
  const maxCandidates = options.maxCandidates ?? 1000;
  return {
    // Full-resolution NCHW probability map: [1, 1, H, W].
    detectOutputSpec: (dim) => [f32(1, 1, dim('detOutH'), dim('detOutW'))],
    outputShapes: ({ width, height }) => {
      'worklet';
      return [[1, 1, height, width]];
    },
    extract: (outputs) => {
      'worklet';
      return extractDbnetTextQuads(outputs[0]!, {
        binThreshold,
        boxThreshold,
        unclipRatio,
        minBoxSide,
        maxCandidates,
      });
    },
  };
}

/**
 * DBNet box-extraction strategy with the default thresholds.
 * @category Constants
 */
export const dbnetExtractBoxes: TextBoxExtractor = makeDbnetExtractBoxes();
