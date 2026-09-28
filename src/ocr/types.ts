/**
 * Public configuration and result types of the OCR pipeline. The pipeline
 * itself (`createOcr`, the engine, the document helpers) follows in later
 * phases; the model registry only needs the shapes.
 */
import type { ObjectDetection, ObjectDetectorModel, Tensor } from 'react-native-executorch';
import type { BoundingBox, ImageBuffer, NormalizeOptions, Quad } from 'react-native-executorch/cv';
import type { DocumentModelsConfig } from '../document/types';
import type { TextBoxExtractor } from './detectors';

/**
 * Model-specific OCR options. The pipeline validates the detect/recognize
 * contract and resolves legal input sizes from the model metadata itself;
 * these cover everything it cannot infer.
 * @category Types
 */
export type OcrModelOptions = {
  /** Recognizer charset (string = one codepoint per index; array = taken verbatim). */
  readonly charset: string | readonly string[];
  /** Maps raw `detect` outputs to quads: `craftExtractBoxes` or `dbnetExtractBoxes`. */
  readonly extractBoxes: TextBoxExtractor;
  /**
   * Drop detections below this recognition confidence. Default 0. Confidence
   * is the mean per-timestep max probability, so this requires the recognizer
   * to export a softmaxed head (values in `[0, 1]`); on raw logits it is
   * meaningless.
   */
  readonly minConfidence?: number;
  /** Detector norm on uint8 RGB. Default ImageNet; must match the model's training norm. */
  readonly detectorNorm?: NormalizeOptions;
  /** Recognizer norm applied after the warp. Default `(x / 255 - 0.5) / 0.5`, into `[-1, 1]`. */
  readonly recognizerNorm?: NormalizeOptions;
  /** Recognizer canvas padding fill. Default 128 (neutral gray). */
  readonly recognizerPadValue?: number;
  /** Strip padding fill: `'constant'` (default) or `'cornerMean'` (background-matched). */
  readonly recognizerPadMode?: 'constant' | 'cornerMean';
  /**
   * Custom decode replacing greedy CTC; receives softmaxed `[1, T, V]` probs.
   * Must be a worklet. The probs tensor is pre-allocated from the
   * recognizer's width-to-timesteps relation, so `T` always matches the run's
   * input width.
   */
  readonly decode?: (
    probs: Tensor,
    charset: readonly string[]
  ) => { readonly text: string; readonly confidence: number };
};

/**
 * Model configuration for the OCR pipeline: one fused detect/recognize PTE
 * plus its options, with optional `layout` (group text into regions) and
 * `documentModels` (orientation, dewarp, table) stages. Omit both for
 * whole-page OCR.
 * @category Types
 * @typeParam L The layout region class-label type.
 */
export type OcrModel<L = never> = {
  /** Local path or remote URL of the fused detect/recognize `.pte`. */
  readonly modelPath: string;
  /** Recognizer charset, detector strategy and the other model-level knobs. */
  readonly modelOpts: OcrModelOptions;
  /** A layout detector; when set, text is grouped into its regions. */
  readonly layout?: ObjectDetectorModel<'xyxy', L>;
  /** The document-helper model for orientation, dewarp and table structure. */
  readonly documentModels?: DocumentModelsConfig;
  /** Layout labels carrying no text; OCR skips these regions. Default `['image', 'chart', 'seal']`. */
  readonly visualLabels?: readonly string[];
  /** The layout label treated as a table (structure recognition + cell fill). Default `'table'`. */
  readonly tableLabel?: string;
};

/**
 * Per-run OCR options, toggled without reloading. `orientation` and `dewarp`
 * are no-ops without loaded `documentModels` and default to its config flags.
 * @category Types
 */
export type RunOcrOptions = {
  /**
   * Also read upright glyph stacks (vertical text), at extra compute. Fully
   * supported with CRAFT; partial with DBNet (glyph columns are still read,
   * but the stacked-box char-level re-detection split is skipped).
   */
  readonly vertical?: boolean;
  /** Height/width ratio above which a box is a vertical stack. Default 1.5. */
  readonly tallCropRatio?: number;
  /** Max stacked-box re-detection passes per page. Default 8. */
  readonly maxRedetections?: number;
  /** Drop vertical reads below this confidence. Default 0.4; set 0 to keep all. */
  readonly verticalMinConfidence?: number;
  /** Rotate the page upright first. No-op without loaded `documentModels`; defaults to its config flag. */
  readonly orientation?: boolean;
  /** Flatten a physically warped page first. No-op without loaded `documentModels`; defaults to its config flag. */
  readonly dewarp?: boolean;
  /** Recognize table structure in table regions. Needs loaded `documentModels` (with a table config) plus layout. Default true. */
  readonly tables?: boolean;
};

/**
 * A single recognized text region. `quad` is the oriented (TL, TR, BR, BL)
 * box in the result image's pixels; its axis-aligned bounds are
 * `cv.boundingBoxOfPoints(quad, 'xyxy')`.
 * @category Types
 */
export type OcrDetection = {
  readonly text: string;
  readonly confidence: number;
  readonly quad: Quad;
};

/**
 * One assembled document block: a layout region (or an ungrouped catch-all)
 * with its OCR lines in reading order and their texts joined by newlines. All
 * coordinates are in the {@link OcrResult} `image` frame.
 * @category Types
 * @typeParam L The layout region class-label type.
 */
export type DocumentBlock<L> = {
  readonly regionType: L | 'ungrouped';
  readonly bbox: BoundingBox<'xyxy'>;
  /** The layout model's confidence for the region (1 for `'ungrouped'`). */
  readonly layoutConfidence: number;
  readonly text: string;
  readonly lines: readonly OcrDetection[];
  readonly isTable: boolean;
  /** The recognized table structure HTML with OCR text filled into its cells. */
  readonly tableHtml?: string;
};

/**
 * The result of one OCR run. `detections` is always the flat list of
 * recognized regions; `blocks` and `regions` carry the layout grouping (one
 * `'ungrouped'` block without a layout model); `image` is the frame all
 * coordinates are relative to: the corrected page after any orientation or
 * dewarp pass, or the input.
 * @category Types
 * @typeParam L The layout region class-label type.
 */
export type OcrResult<L = never> = {
  readonly detections: OcrDetection[];
  readonly blocks: DocumentBlock<L>[];
  readonly regions: ObjectDetection<'xyxy', L>[];
  readonly image: ImageBuffer;
};
