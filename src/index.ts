/**
 * Advanced on-device OCR for React Native, built on react-native-executorch.
 *
 * `createOcr` and `useOcr` run whole-page OCR with the models in `models.ocr`,
 * or per-region OCR with a layout model from `models.layoutDetection`; the
 * document helpers follow in a later phase.
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
export { mergeContainedRegions } from './ocr/layout';

export { createOcr, type Ocr } from './ocr/createOcr';
export { useOcr } from './hooks/useOcr';

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
