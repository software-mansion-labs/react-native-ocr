/**
 * The document helpers: one fused model carrying page-orientation
 * classification, the UVDoc dewarping grid and, optionally, SLANet table
 * structure. `createDocumentModels` loads it once and returns a worklet per
 * capability; `createOcr` composes them around the OCR pass.
 */
import {
  createResourceScope,
  IMAGENET_NORM,
  loadModel,
  RnExecuTorchError,
  tensor,
  wrapAsync,
  type Model,
  type Tensor,
} from 'react-native-executorch';
import type { ImageFormat } from 'react-native-executorch/cv';
import { f32, method, validateSpec } from 'react-native-executorch/schema';
import type { WorkletRuntime } from 'react-native-worklets';
import { warpByGridTensor } from '../native/ops';
import { FORMAT_CHANNELS } from '../ocr/imageFormat';
import { createTensorPreprocessor, type TensorPreprocessor } from '../ocr/preprocess';
import type { DocumentModelsConfig, TableConfig } from './types';

export type { DocumentModelsConfig, TableConfig } from './types';

/**
 * A detected page orientation: the clockwise rotation the page shows, and the
 * classifier's confidence in `[0, 1]`.
 * @category Types
 */
export type Orientation = {
  readonly rotationCW: 0 | 90 | 180 | 270;
  readonly confidence: number;
};

/**
 * The document-helper runner {@link createDocumentModels} returns. Each
 * worklet runs on the thread that calls it; `page` is a caller-owned `uint8`
 * `[H, W, C]` tensor.
 */
export type DocumentModels = {
  /** Releases the model and the pre-allocated tensors. */
  readonly dispose: () => void;
  /** Classifies the page's orientation. */
  readonly detectOrientationWorklet: (page: Tensor, format: ImageFormat) => Orientation;
  /**
   * Flattens the page through the predicted sampling grid. Returns `page`
   * itself when the warp is declined; the caller owns whichever is returned.
   */
  readonly dewarpWorklet: (page: Tensor, format: ImageFormat) => Tensor;
  /**
   * Decodes a table's `<tr>`/`<td>` skeleton (empty cells). Throws
   * `INVALID_STATE` when the runner was created without a table config.
   */
  readonly recognizeTableWorklet: (page: Tensor, format: ImageFormat) => string;
};

/**
 * Index of the largest value in `arr`.
 * @param arr The values.
 * @returns The index of the maximum (the first one on ties).
 */
function argmax(arr: ArrayLike<number>): number {
  'worklet';
  let index = 0;
  let best = arr[0]!;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i]! > best) {
      best = arr[i]!;
      index = i;
    }
  }
  return index;
}

// ─── Orientation ─────────────────────────────────────────────────────────────

// The orientation head's class count: [0°, 90°, 180°, 270°] clockwise.
const ORIENTATION_CLASSES = 4;

/**
 * Runs the orientation head and turns its logits into a class and a softmax
 * confidence.
 * @param model The helper model.
 * @param preprocessor The chain bound to the orientation input shape.
 * @param tOri The pre-allocated logits tensor.
 * @param oriBuf The staging buffer for the logits.
 * @param page The page tensor.
 * @param format The pixel format of `page`.
 * @returns The detected orientation.
 */
function detectPageOrientation(
  model: Model,
  preprocessor: TensorPreprocessor,
  tOri: Tensor,
  oriBuf: Float32Array,
  page: Tensor,
  format: ImageFormat
): Orientation {
  'worklet';
  const tInput = preprocessor.process(page, format);
  model.execute('orientation', [tInput], [tOri]);
  tOri.getData(oriBuf);
  const cls = argmax(oriBuf);
  const best = oriBuf[cls]!;
  let sumExp = 0;
  for (let i = 0; i < oriBuf.length; i++) {
    sumExp += Math.exp(oriBuf[i]! - best);
  }
  return { rotationCW: (cls * 90) as Orientation['rotationCW'], confidence: 1 / sumExp };
}

// ─── Dewarp ──────────────────────────────────────────────────────────────────

// Default for DocumentModelsConfig.dewarpMinVarianceRatio: a grid estimated on
// a page without clear borders can collapse the output to near-blank, so the
// warp is declined when it keeps less than this fraction of the source
// variance. A collapsed page keeps close to none; a page photographed on a
// dark table keeps about a third once the table is cropped away (the example's
// curved page: 0.35), which the reference's 0.5 wrongly declined.
const DEFAULT_DEWARP_MIN_VARIANCE_RATIO = 0.1;
const VARIANCE_SAMPLE_STRIDE = 31;

/**
 * Variance of one channel, sampled sparsely: a cheap proxy for image content.
 * @param data The `[H, W, C]` pixels.
 * @param channels The channel count.
 * @returns The sampled variance of channel 0.
 */
function pixelVariance(data: Uint8Array, channels: number): number {
  'worklet';
  let n = 0;
  let sum = 0;
  let sumSq = 0;
  const step = channels * VARIANCE_SAMPLE_STRIDE;
  for (let i = 0; i < data.length; i += step) {
    sum += data[i]!;
    sumSq += data[i]! * data[i]!;
    n++;
  }
  if (n === 0) {
    return 0;
  }
  return sumSq / n - (sum / n) * (sum / n);
}

/**
 * Estimates the sampling field and resamples the page through it. Returns
 * `page` unchanged when the warp is declined; the caller owns whichever
 * tensor is returned.
 * @param model The helper model.
 * @param preprocessor The chain bound to the dewarp input shape.
 * @param tGrid The pre-allocated grid tensor.
 * @param page The page tensor.
 * @param format The pixel format of `page`.
 * @param minVarianceRatio The fraction of the source variance the result must keep.
 * @returns The flattened page, or `page` when the warp was declined.
 */
function dewarpPage(
  model: Model,
  preprocessor: TensorPreprocessor,
  tGrid: Tensor,
  page: Tensor,
  format: ImageFormat,
  minVarianceRatio: number
): Tensor {
  'worklet';
  const tInput = preprocessor.process(page, format);
  model.execute('dewarp', [tInput], [tGrid]);
  const h = page.shape[0]!;
  const w = page.shape[1]!;
  const ch = FORMAT_CHANNELS[format];
  const tDst = warpByGridTensor(page, tGrid);
  try {
    // Known cost: both full frames are read back just to sample about 3% of
    // one channel for the variance ratio (some 70 MB of copies on a 12 MP
    // page). Acceptable while dewarp is opt-in and rare; a native variance op
    // is the fix if it ever matters.
    const out = tDst.getData(new Uint8Array(w * h * ch));
    const src = page.getData(new Uint8Array(w * h * ch));
    if (pixelVariance(out, ch) < minVarianceRatio * pixelVariance(src, ch)) {
      tDst.dispose();
      return page;
    }
    return tDst;
  } catch (e) {
    tDst.dispose(); // the caller cannot free it on the throw path
    throw e;
  }
}

// ─── Table structure ─────────────────────────────────────────────────────────

/** The table decoder's pre-allocated tensors and staging buffers. */
type TableDecodeState = {
  readonly tFeatures: Tensor;
  readonly tHidden: Tensor;
  readonly tOnehot: Tensor;
  readonly tProbs: Tensor;
  readonly tNewHidden: Tensor;
  readonly zeroHidden: Float32Array;
  readonly zeroVocab: Float32Array;
  readonly onehotBuf: Float32Array;
  readonly probsBuf: Float32Array;
};

/**
 * Decodes the table's `<tr>`/`<td>` HTML skeleton token by token; the cell
 * text is filled in later from the region's OCR lines (`fillTableCells`).
 * @param model The helper model.
 * @param preprocessor The chain bound to the table encoder's input shape.
 * @param state The decoder's tensors and buffers.
 * @param tableConfig The vocabulary, the end token and the step cap.
 * @param page The table crop.
 * @param format The pixel format of `page`.
 * @returns The structure HTML with empty cells.
 */
function recognizeTableStructure(
  model: Model,
  preprocessor: TensorPreprocessor,
  state: TableDecodeState,
  tableConfig: TableConfig,
  page: Tensor,
  format: ImageFormat
): string {
  'worklet';
  const { structureVocab, eosTokenId, maxSteps } = tableConfig;
  const { tFeatures, tHidden, tOnehot, tProbs, tNewHidden } = state;
  const tInput = preprocessor.process(page, format);
  model.execute('table_encode', [tInput], [tFeatures]);
  tHidden.setData(state.zeroHidden);
  // The first step feeds an all-zero one-hot, not a 'sos' (index 0) one-hot:
  // the exported decoder embeds its own start token, so the first real input
  // is the zero vector. (Validated on device: a 'sos' one-hot here offsets
  // every decoded token by one. Revisit if a re-export changes the start.)
  tOnehot.setData(state.zeroVocab);
  let html = '';
  for (let step = 0; step < maxSteps; step++) {
    model.execute('table_decode_step', [tFeatures, tHidden, tOnehot], [tProbs, tNewHidden]);
    tProbs.getData(state.probsBuf);
    const tok = argmax(state.probsBuf);
    if (tok === eosTokenId) {
      break;
    }
    // The reserved start/end token range stays out of the skeleton.
    if (tok > 0 && tok < eosTokenId && tok < structureVocab.length) {
      html += structureVocab[tok]!;
    }
    tNewHidden.copyTo(tHidden);
    state.onehotBuf.fill(0);
    state.onehotBuf[tok] = 1;
    tOnehot.setData(state.onehotBuf);
  }
  return html;
}

// ─── Contract and factory ────────────────────────────────────────────────────

/** The tensor shapes the factory pre-allocates from, per capability. */
type DocumentModelsShapes = {
  readonly orientation: { readonly input: number[]; readonly output: number[] };
  readonly dewarp: { readonly input: number[]; readonly grid: number[] };
  readonly table?: {
    readonly input: number[];
    readonly features: number[];
    readonly hidden: number[];
    readonly probs: number[];
  };
};

/**
 * Validates the orientation and dewarp methods always, and the table methods
 * only when a table vocabulary size is given. Every method of this model is
 * exported at a fixed size, so each dimension binds to a static symbol; a
 * symbol shared across methods enforces the model's internal agreements (the
 * decoder's one-hot input vocab equals its output vocab, its hidden state
 * round-trips, and it consumes exactly the features the encoder produces).
 * @param model The loaded helper model.
 * @param tableVocabSize The structure vocabulary size, when tables are wanted.
 * @returns The shapes to pre-allocate from.
 * @throws {RnExecuTorchError} With code `SCHEMA_MISMATCH` when the model does
 * not carry the required methods at the expected shapes.
 */
export function resolveDocumentModelsShapes(
  model: Model,
  tableVocabSize?: number
): DocumentModelsShapes {
  const imageMethods = {
    // 4 orientation classes, in [0°, 90°, 180°, 270°] order.
    ...method('orientation', [f32(1, 3, 'oriH', 'oriW')], [f32(1, ORIENTATION_CLASSES)]),
    ...method('dewarp', [f32(1, 3, 'dewH', 'dewW')], [f32(1, 2, 'gridH', 'gridW')]),
  };
  if (tableVocabSize === undefined) {
    const { dims } = validateSpec(model.schema, { default: imageMethods });
    const [oriH, oriW, dewH, dewW, gridH, gridW] = dims.constant(
      'oriH',
      'oriW',
      'dewH',
      'dewW',
      'gridH',
      'gridW'
    );
    return {
      orientation: { input: [1, 3, oriH, oriW], output: [1, ORIENTATION_CLASSES] },
      dewarp: { input: [1, 3, dewH, dewW], grid: [1, 2, gridH, gridW] },
    };
  }

  const { dims } = validateSpec(model.schema, {
    default: {
      ...imageMethods,
      ...method('table_encode', [f32(1, 3, 'tabH', 'tabW')], [f32(1, 'feat', 'featDim')]),
      ...method(
        'table_decode_step',
        [f32(1, 'feat', 'featDim'), f32(1, 'hidden'), f32(1, tableVocabSize)],
        [f32(1, tableVocabSize), f32(1, 'hidden')]
      ),
    },
  });
  const [oriH, oriW, dewH, dewW, gridH, gridW, tabH, tabW, feat, featDim, hidden] = dims.constant(
    'oriH',
    'oriW',
    'dewH',
    'dewW',
    'gridH',
    'gridW',
    'tabH',
    'tabW',
    'feat',
    'featDim',
    'hidden'
  );
  return {
    orientation: { input: [1, 3, oriH, oriW], output: [1, ORIENTATION_CLASSES] },
    dewarp: { input: [1, 3, dewH, dewW], grid: [1, 2, gridH, gridW] },
    table: {
      input: [1, 3, tabH, tabW],
      features: [1, feat, featDim],
      hidden: [1, hidden],
      probs: [1, tableVocabSize],
    },
  };
}

/**
 * Loads the document-helper model once and returns a worklet per capability.
 * Orientation and dewarp are always wired (their methods are always present);
 * the table decoder only with a `table` config.
 * @param config The model path, the stage defaults and the table config.
 * @param runtime Optional worklet runtime to load the model on.
 * @returns A promise resolving to the runner.
 * @throws {RnExecuTorchError} With code `LOAD_FAILED` if the model cannot be
 * loaded and `SCHEMA_MISMATCH` if it does not carry the enabled methods at
 * the expected shapes; nothing is leaked on either.
 */
export async function createDocumentModels(
  config: DocumentModelsConfig,
  runtime?: WorkletRuntime
): Promise<DocumentModels> {
  const { modelPath, table } = config;
  const dewarpMinVarianceRatio = config.dewarpMinVarianceRatio ?? DEFAULT_DEWARP_MIN_VARIANCE_RATIO;

  // Everything is tracked as it is built, so a failure part-way through
  // releases what exists, and the runner's `dispose` is the same teardown.
  const scope = createResourceScope();
  const dispose = scope.dispose;
  try {
    const model = scope.track(await wrapAsync(loadModel, runtime)(modelPath));
    const s = resolveDocumentModelsShapes(model, table?.structureVocab.length);

    const orientationPreprocessor = scope.track(
      createTensorPreprocessor(
        { resizeMode: 'stretch', interpolation: 'linear', normalizeOpts: IMAGENET_NORM },
        s.orientation.input
      )
    );
    const dewarpPreprocessor = scope.track(
      createTensorPreprocessor(
        {
          resizeMode: 'stretch',
          interpolation: 'linear',
          normalizeOpts: { alpha: 1 / 255, beta: 0 },
        },
        s.dewarp.input
      )
    );
    const tOri = scope.track(tensor('float32', s.orientation.output));
    const tGrid = scope.track(tensor('float32', s.dewarp.grid));
    const oriBuf = new Float32Array(s.orientation.output[1]!);

    let tablePreprocessor: TensorPreprocessor | null = null;
    let decodeState: TableDecodeState | null = null;
    if (table && s.table) {
      tablePreprocessor = scope.track(
        createTensorPreprocessor(
          { resizeMode: 'stretch', interpolation: 'linear', normalizeOpts: IMAGENET_NORM },
          s.table.input
        )
      );
      const hidLen = s.table.hidden[1]!;
      const vocabLen = s.table.probs[1]!;
      decodeState = {
        tFeatures: scope.track(tensor('float32', s.table.features)),
        tHidden: scope.track(tensor('float32', s.table.hidden)),
        tOnehot: scope.track(tensor('float32', s.table.probs)),
        tProbs: scope.track(tensor('float32', s.table.probs)),
        tNewHidden: scope.track(tensor('float32', s.table.hidden)),
        zeroHidden: new Float32Array(hidLen),
        zeroVocab: new Float32Array(vocabLen),
        onehotBuf: new Float32Array(vocabLen),
        probsBuf: new Float32Array(vocabLen),
      };
    }

    // Thin bindings; the logic lives in the top-level worklets above.
    const detectOrientationWorklet = (page: Tensor, format: ImageFormat): Orientation => {
      'worklet';
      return detectPageOrientation(model, orientationPreprocessor, tOri, oriBuf, page, format);
    };
    const dewarpWorklet = (page: Tensor, format: ImageFormat): Tensor => {
      'worklet';
      return dewarpPage(model, dewarpPreprocessor, tGrid, page, format, dewarpMinVarianceRatio);
    };
    const recognizeTableWorklet = (page: Tensor, format: ImageFormat): string => {
      'worklet';
      if (!tablePreprocessor || !decodeState || !table) {
        throw RnExecuTorchError(
          'INVALID_STATE',
          'recognizeTableWorklet: these document models were loaded without a table config.'
        );
      }
      return recognizeTableStructure(model, tablePreprocessor, decodeState, table, page, format);
    };

    return { dispose, detectOrientationWorklet, dewarpWorklet, recognizeTableWorklet };
  } catch (e) {
    dispose();
    throw e;
  }
}
