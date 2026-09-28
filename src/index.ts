/**
 * Advanced on-device OCR for React Native, built on react-native-executorch.
 *
 * Phases 1 to 3 of the build-out ship the native module with its typed
 * wrappers, the model registry, the detector strategies and the pipeline's
 * public types; `createOcr` and `useOcr` follow in later phases.
 * @packageDocumentation
 */

export {
  cropTensor,
  extractCraftTextBoxes,
  isEmulator,
  padCornerMean,
  rotateTensor,
  warpByGridTensor,
  type CraftBox,
  type CraftDecodeOptions,
  type RotationDegrees,
  type WarpBorderMode,
  type WarpByGridOptions,
  type WarpInterpolation,
} from './native/ops';

export { models } from './models';
export { alphabets, PPOCR_SYMBOLS } from './charsets';
export { DOC_LAYOUT_LABELS, SLANET_STRUCTURE_VOCAB, type DocLayoutLabel } from './constants';

export {
  craftExtractBoxes,
  dbnetExtractBoxes,
  makeCraftExtractBoxes,
  makeDbnetExtractBoxes,
  type CraftExtractOptions,
  type DbnetExtractOptions,
  type TextBoxExtractor,
} from './ocr/detectors';

export { greedyCtcDecode } from './ocr/ctc';

export type {
  DocumentBlock,
  OcrDetection,
  OcrModel,
  OcrModelOptions,
  OcrResult,
  RunOcrOptions,
} from './ocr/types';
export type { DocumentModelsConfig, TableConfig } from './document/types';

export type { ImageBuffer, NormalizeOptions, Quad } from 'react-native-executorch/cv';
