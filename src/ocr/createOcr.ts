/**
 * `createOcr`: loads a fused detect/recognize model, validates it and returns
 * the OCR runner.
 */
import { IMAGENET_NORM, loadModel, RnExecuTorchError, wrapAsync } from 'react-native-executorch';
import {
  boundingBoxOfPoints,
  type BoundingBox,
  type ImageBuffer,
} from 'react-native-executorch/cv';
import type { WorkletRuntime } from 'react-native-worklets';
import { RECOGNIZER_NORM, RECOGNIZER_PAD_VALUE } from '../constants';
import { resolveOcrContract, runOcrPass, type OcrEngine } from './engine';
import { orderByReadingOrder } from './geometry';
import type { DocumentBlock, OcrDetection, OcrModel, OcrResult, RunOcrOptions } from './types';

export type {
  DocumentBlock,
  OcrDetection,
  OcrModel,
  OcrModelOptions,
  OcrResult,
  RunOcrOptions,
} from './types';

/**
 * The OCR runner {@link createOcr} returns.
 * @category Types
 * @typeParam L The layout region class-label type.
 */
export type Ocr<L = never> = {
  /** Releases the model and every other native resource of the runner. */
  readonly dispose: () => void;
  /**
   * Reads the text of an image on the runner's worklet runtime.
   * @param input The image to read.
   * @param options Per-run toggles. See {@link RunOcrOptions}.
   * @returns A promise resolving to the recognized regions in reading order.
   */
  readonly runOcr: (input: ImageBuffer, options?: RunOcrOptions) => Promise<OcrResult<L>>;
  /**
   * Synchronous version of `runOcr`, to be called on a worklet runtime.
   */
  readonly runOcrWorklet: (input: ImageBuffer, options?: RunOcrOptions) => OcrResult<L>;
};

function makeBlock<L>(
  regionType: L | 'ungrouped',
  bbox: BoundingBox<'xyxy'>,
  layoutConfidence: number,
  lines: OcrDetection[],
  isTable: boolean
): DocumentBlock<L> {
  'worklet';
  const sorted = orderByReadingOrder(lines);
  return {
    regionType,
    bbox,
    layoutConfidence,
    isTable,
    lines: sorted,
    text: sorted.map((l) => l.text).join('\n'),
  };
}

/**
 * Creates the OCR runner for a detect and recognize model (EasyOCR or
 * PaddleOCR). Each run detects the text quads of the whole page, reads every
 * one and returns them in reading order, both flat and as one `'ungrouped'`
 * block.
 *
 * The `layout` and `documentModels` stages of {@link OcrModel} are not
 * implemented yet; a config that sets either is rejected.
 * @category Tasks
 * @typeParam L The layout region class-label type.
 * @param config Model path and OCR options.
 * @param runtime Optional worklet runtime to load and run the model on.
 * @returns A promise resolving to recognition and disposal controls.
 * @throws {RnExecuTorchError} With code `LOAD_FAILED` if the model cannot be
 * loaded, `SCHEMA_MISMATCH` if its `detect`/`recognize` methods match none of
 * the pipeline's spec variants, declare a width-to-timestep relation the
 * pipeline cannot satisfy, or disagree with the charset size, and
 * `INVALID_ARGUMENT` if the config asks for a stage that is not implemented.
 */
export async function createOcr<L = never>(
  config: OcrModel<L>,
  runtime?: WorkletRuntime
): Promise<Ocr<L>> {
  const { modelPath, modelOpts } = config;
  // Ignoring these would return a result that looks complete and is not.
  if (config.layout || config.documentModels) {
    throw RnExecuTorchError(
      'INVALID_ARGUMENT',
      `createOcr: '${config.layout ? 'layout' : 'documentModels'}' is not supported yet; ` +
        'only whole-page OCR is implemented.'
    );
  }

  const model = await wrapAsync(loadModel, runtime)(modelPath);

  // Contract validation can throw; a bad config must not leak the model.
  let engine: OcrEngine;
  try {
    const contract = resolveOcrContract(model, modelOpts.charset, modelOpts.extractBoxes);
    engine = {
      model,
      extractBoxes: modelOpts.extractBoxes,
      det: { ...contract.det, norm: modelOpts.detectorNorm ?? IMAGENET_NORM },
      rec: {
        ...contract.rec,
        norm: modelOpts.recognizerNorm ?? RECOGNIZER_NORM,
        padValue: modelOpts.recognizerPadValue ?? RECOGNIZER_PAD_VALUE,
        padMode: modelOpts.recognizerPadMode ?? 'constant',
      },
      charset: contract.charset,
      minConfidence: modelOpts.minConfidence ?? 0,
      decode: modelOpts.decode,
    };
  } catch (e) {
    model.dispose();
    throw e;
  }

  const dispose = () => {
    model.dispose();
  };

  const runOcrWorklet = (input: ImageBuffer, options?: RunOcrOptions): OcrResult<L> => {
    'worklet';
    // `options` is structurally a superset of the engine's pass options.
    const detections = runOcrPass(engine, input, options);
    const blocks = detections.length
      ? [
          makeBlock<L>(
            'ungrouped',
            boundingBoxOfPoints(
              detections.flatMap((d) => d.quad),
              'xyxy'
            ),
            1,
            detections,
            false
          ),
        ]
      : [];
    return { detections, blocks, regions: [], image: input };
  };

  const runOcr = wrapAsync(runOcrWorklet, runtime);

  return { runOcr, runOcrWorklet, dispose };
}
