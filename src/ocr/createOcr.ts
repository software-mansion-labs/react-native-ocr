/**
 * `createOcr`: loads a fused detect/recognize model (and the optional layout
 * detector), validates it and returns the OCR runner.
 */
import {
  createObjectDetector,
  IMAGENET_NORM,
  loadModel,
  RnExecuTorchError,
  tensor,
  wrapAsync,
  type ObjectDetector,
} from 'react-native-executorch';
import { boundingBoxOfPoints, type ImageBuffer } from 'react-native-executorch/cv';
import type { WorkletRuntime } from 'react-native-worklets';
import { RECOGNIZER_NORM, RECOGNIZER_PAD_VALUE } from '../constants';
import { cropTensor } from '../native/ops';
import { resolveOcrContract, runOcrPass, runOcrPassOnTensor, type OcrEngine } from './engine';
import { orderByReadingOrder } from './geometry';
import { FORMAT_CHANNELS } from './imageFormat';
import {
  DEFAULT_TABLE_LABEL,
  DEFAULT_VISUAL_LABELS,
  makeBlock,
  mergeContainedRegions,
} from './layout';
import { quadFromBounds, translateQuad } from './quad';
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

/**
 * Creates the OCR runner for a detect and recognize model (EasyOCR or
 * PaddleOCR), optionally composed with a layout detector. Each run detects
 * the text quads, reads every one and returns them in reading order, both
 * flat and as blocks: one `'ungrouped'` block for whole-page OCR, or one
 * block per layout region, each read on its own crop.
 *
 * The `documentModels` stage of {@link OcrModel} is not implemented yet; a
 * config that sets it is rejected.
 * @category Tasks
 * @typeParam L The layout region class-label type.
 * @param config Model path and OCR options, plus the optional layout model.
 * @param runtime Optional worklet runtime to load and run the models on.
 * @returns A promise resolving to recognition and disposal controls.
 * @throws {RnExecuTorchError} With code `LOAD_FAILED` if a model cannot be
 * loaded, `SCHEMA_MISMATCH` if the OCR model's `detect`/`recognize` methods
 * match none of the pipeline's spec variants, declare a width-to-timestep
 * relation the pipeline cannot satisfy, or disagree with the charset size
 * (or the layout model matches no object-detector variant), and
 * `INVALID_ARGUMENT` if the config asks for a stage that is not implemented.
 */
export async function createOcr<L = never>(
  config: OcrModel<L>,
  runtime?: WorkletRuntime
): Promise<Ocr<L>> {
  const { modelPath, modelOpts } = config;
  // Ignoring this would return a result that looks complete and is not.
  if (config.documentModels) {
    throw RnExecuTorchError(
      'INVALID_ARGUMENT',
      "createOcr: 'documentModels' is not supported yet; only whole-page and layout OCR are implemented."
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

  // The layout stage failing to build must not leak the OCR model.
  let layout: ObjectDetector<'xyxy', L> | null = null;
  if (config.layout) {
    try {
      layout = await createObjectDetector<'xyxy', L>(config.layout, runtime);
    } catch (e) {
      model.dispose();
      throw e;
    }
  }

  const visualLabels = config.visualLabels ?? DEFAULT_VISUAL_LABELS;
  const tableLabel = config.tableLabel ?? DEFAULT_TABLE_LABEL;

  const dispose = () => {
    model.dispose();
    layout?.dispose();
  };

  const runOcrWorklet = (input: ImageBuffer, options?: RunOcrOptions): OcrResult<L> => {
    'worklet';
    // `options` is structurally a superset of the engine's pass options.
    if (!layout) {
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
    }

    // The layout path crops each region natively out of the page tensor,
    // straight into the recognizer: no JS crop and re-upload per region.
    const regions = mergeContainedRegions(layout.detectObjectsWorklet(input), visualLabels);
    const blocks: DocumentBlock<L>[] = [];
    const tPage = tensor('uint8', [input.height, input.width, FORMAT_CHANNELS[input.format]]);
    try {
      tPage.setData(input.data);
      for (const region of regions) {
        if (visualLabels.includes(String(region.label))) {
          continue;
        }
        const isTable = String(region.label) === tableLabel;
        const xmin = Math.max(0, Math.min(Math.round(region.box.xmin), input.width));
        const ymin = Math.max(0, Math.min(Math.round(region.box.ymin), input.height));
        const xmax = Math.max(0, Math.min(Math.round(region.box.xmax), input.width));
        const ymax = Math.max(0, Math.min(Math.round(region.box.ymax), input.height));
        if (xmax <= xmin || ymax <= ymin) {
          continue;
        }
        // Each region is read on its own crop for better dense-page recall;
        // the lines are offset back to page coordinates.
        const tCrop = cropTensor(tPage, { format: 'xyxy', xmin, ymin, xmax, ymax });
        let lines: OcrDetection[];
        try {
          lines = runOcrPassOnTensor(
            engine,
            tCrop,
            xmax - xmin,
            ymax - ymin,
            input.format,
            options
          ).map((d) => ({ ...d, quad: translateQuad(d.quad, xmin, ymin) }));
        } finally {
          tCrop.dispose();
        }
        // A table region is a block even when nothing was read: its structure
        // is still a result once the table stage exists.
        if (lines.length === 0 && !isTable) {
          continue;
        }
        blocks.push(makeBlock<L>(region.label, region.box, region.confidence, lines, isTable));
      }
    } finally {
      tPage.dispose();
    }
    // The flat detections follow the ordered blocks, so the two views of the
    // page agree.
    const orderedBlocks = orderByReadingOrder(
      blocks.map((block) => ({ block, quad: quadFromBounds(block.bbox) }))
    ).map((w) => w.block);
    return {
      detections: orderedBlocks.flatMap((block) => block.lines),
      blocks: orderedBlocks,
      regions,
      image: input,
    };
  };

  const runOcr = wrapAsync(runOcrWorklet, runtime);

  return { runOcr, runOcrWorklet, dispose };
}
