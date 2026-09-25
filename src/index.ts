/**
 * Advanced on-device OCR for React Native, built on react-native-executorch.
 *
 * Phases 1 and 2 of the build-out ship the native module and its typed
 * wrappers; the OCR pipeline, hooks and model registry follow in later phases.
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
