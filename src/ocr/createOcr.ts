/**
 * `createOcr`: loads a fused detect/recognize model (and the optional layout
 * detector and document helpers), validates them and returns the OCR runner.
 */
import {
  createObjectDetector,
  IMAGENET_NORM,
  loadModel,
  tensor,
  wrapAsync,
  type ObjectDetector,
  type Tensor,
} from 'react-native-executorch';
import { boundingBoxOfPoints, type ImageBuffer } from 'react-native-executorch/cv';
import type { WorkletRuntime } from 'react-native-worklets';
import { RECOGNIZER_NORM, RECOGNIZER_PAD_VALUE } from '../constants';
import { createDocumentModels, type DocumentModels } from '../document/documentModels';
import { fillTableCells } from '../document/table';
import { cropTensor, rotateTensor, type RotationDegrees } from '../native/ops';
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

// Default for DocumentModelsConfig.orientationMinConfidence.
const DEFAULT_ORIENTATION_MIN_CONFIDENCE = 0.85;

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
 * PaddleOCR), optionally composed with a layout detector and the document
 * helpers. Each run goes orientation → dewarp → (layout ? per-region :
 * whole-page) OCR → reading-ordered blocks, with table regions filling their
 * recognized structure: the text quads are detected, every one is read and
 * they are returned in reading order, both flat and as blocks (one
 * `'ungrouped'` block for whole-page OCR, or one block per layout region,
 * each read on its own crop).
 * @category Tasks
 * @typeParam L The layout region class-label type.
 * @param config Model path and OCR options, plus the optional layout model
 * and document helpers.
 * @param runtime Optional worklet runtime to load and run the models on.
 * @returns A promise resolving to recognition and disposal controls.
 * @throws {RnExecuTorchError} With code `LOAD_FAILED` if a model cannot be
 * loaded, and `SCHEMA_MISMATCH` if the OCR model's `detect`/`recognize`
 * methods match none of the pipeline's spec variants, declare a
 * width-to-timestep relation the pipeline cannot satisfy, or disagree with
 * the charset size, if the layout model matches no object-detector variant,
 * or if the helper model lacks a method the config enables.
 */
export async function createOcr<L = never>(
  config: OcrModel<L>,
  runtime?: WorkletRuntime
): Promise<Ocr<L>> {
  const { modelPath, modelOpts } = config;
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

  // A later stage failing to build must not leak the ones already built.
  let layout: ObjectDetector<'xyxy', L> | null = null;
  let documentModels: DocumentModels | null = null;
  try {
    layout = config.layout ? await createObjectDetector<'xyxy', L>(config.layout, runtime) : null;
    documentModels = config.documentModels
      ? await createDocumentModels(config.documentModels, runtime)
      : null;
  } catch (e) {
    layout?.dispose();
    model.dispose();
    throw e;
  }

  const dm = config.documentModels;
  const hasTable = !!dm?.table;
  const defaultOrientation = !!dm?.orientation;
  const defaultDewarp = !!dm?.dewarp;
  const orientationMinConfidence =
    dm?.orientationMinConfidence ?? DEFAULT_ORIENTATION_MIN_CONFIDENCE;
  const visualLabels = config.visualLabels ?? DEFAULT_VISUAL_LABELS;
  const tableLabel = config.tableLabel ?? DEFAULT_TABLE_LABEL;

  const dispose = () => {
    model.dispose();
    layout?.dispose();
    documentModels?.dispose();
  };

  const runOcrWorklet = (input: ImageBuffer, options?: RunOcrOptions): OcrResult<L> => {
    'worklet';
    // `options` is structurally a superset of the engine's pass options.
    const useOrientation = !!documentModels && (options?.orientation ?? defaultOrientation);
    const useDewarp = !!documentModels && (options?.dewarp ?? defaultDewarp);
    // Tables are on whenever the model supports them (unlike orientation and
    // dewarp, which are opt-in); a per-run `tables: false` skips the decoder.
    const useTables = hasTable && (options?.tables ?? true);
    const ch = FORMAT_CHANNELS[input.format];

    // Orientation and dewarp produce the corrected frame all coordinates are
    // relative to; without them the frame is the input unchanged. A corrected
    // page is kept alive as a tensor (`tPage`) so the OCR pass reads it
    // directly instead of round-tripping through the materialized `img`
    // buffer; the outer `finally` disposes it.
    let img = input;
    let tPage: Tensor | null = null;
    try {
      if (documentModels && (useOrientation || useDewarp)) {
        let changed = false;
        tPage = tensor('uint8', [input.height, input.width, ch]);
        tPage.setData(input.data);
        if (useOrientation) {
          const orientation = documentModels.detectOrientationWorklet(tPage, input.format);
          const deg = (360 - orientation.rotationCW) % 360;
          if (deg !== 0 && orientation.confidence >= orientationMinConfidence) {
            const tRotated = rotateTensor(tPage, deg as RotationDegrees);
            tPage.dispose();
            tPage = tRotated;
            changed = true;
          }
        }
        if (useDewarp) {
          // dewarp returns the input tensor itself when it declines the warp.
          const tDewarped = documentModels.dewarpWorklet(tPage, input.format);
          if (tDewarped !== tPage) {
            tPage.dispose();
            tPage = tDewarped;
            changed = true;
          }
        }
        // A new frame is materialized only when a correction was applied;
        // otherwise the input buffer already is that frame.
        if (changed) {
          const height = tPage.shape[0]!;
          const width = tPage.shape[1]!;
          const data = tPage.getData(new Uint8Array(width * height * ch));
          img = { data, width, height, format: input.format, layout: input.layout };
        }
      }

      if (!layout) {
        const detections = tPage
          ? runOcrPassOnTensor(engine, tPage, img.width, img.height, img.format, options)
          : runOcrPass(engine, img, options);
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
        return { detections, blocks, regions: [], image: img };
      }

      // The layout path crops each region natively out of the page tensor,
      // straight into the recognizer: no JS crop and re-upload per region.
      // The corrected page is reused when there is one; otherwise `img` is
      // uploaded once.
      if (!tPage) {
        tPage = tensor('uint8', [img.height, img.width, ch]);
        tPage.setData(img.data);
      }
      const regions = mergeContainedRegions(layout.detectObjectsWorklet(img), visualLabels);
      const blocks: DocumentBlock<L>[] = [];
      for (const region of regions) {
        if (visualLabels.includes(String(region.label))) {
          continue;
        }
        const isTable = String(region.label) === tableLabel;
        const xmin = Math.max(0, Math.min(Math.round(region.box.xmin), img.width));
        const ymin = Math.max(0, Math.min(Math.round(region.box.ymin), img.height));
        const xmax = Math.max(0, Math.min(Math.round(region.box.xmax), img.width));
        const ymax = Math.max(0, Math.min(Math.round(region.box.ymax), img.height));
        if (xmax <= xmin || ymax <= ymin) {
          continue;
        }
        // Each region is read on its own crop for better dense-page recall;
        // the lines are offset back to page coordinates. The table encoder
        // reads the same crop, so it stays alive until the region is done.
        const tCrop = cropTensor(tPage, { format: 'xyxy', xmin, ymin, xmax, ymax });
        try {
          const lines: OcrDetection[] = runOcrPassOnTensor(
            engine,
            tCrop,
            xmax - xmin,
            ymax - ymin,
            img.format,
            options
          ).map((d) => ({ ...d, quad: translateQuad(d.quad, xmin, ymin) }));
          // A table region is a block even when nothing was read: its
          // structure is still a result.
          if (lines.length === 0 && !isTable) {
            continue;
          }
          let block = makeBlock<L>(region.label, region.box, region.confidence, lines, isTable);
          if (isTable && documentModels && useTables) {
            const skeleton = documentModels.recognizeTableWorklet(tCrop, img.format);
            block = { ...block, tableHtml: fillTableCells(skeleton, block.lines) };
          }
          blocks.push(block);
        } finally {
          tCrop.dispose();
        }
      }
      // The flat detections follow the ordered blocks, so the two views of
      // the page agree.
      const orderedBlocks = orderByReadingOrder(
        blocks.map((block) => ({ block, quad: quadFromBounds(block.bbox) }))
      ).map((w) => w.block);
      return {
        detections: orderedBlocks.flatMap((block) => block.lines),
        blocks: orderedBlocks,
        regions,
        image: img,
      };
    } finally {
      tPage?.dispose();
    }
  };

  const runOcr = wrapAsync(runOcrWorklet, runtime);

  return { runOcr, runOcrWorklet, dispose };
}
