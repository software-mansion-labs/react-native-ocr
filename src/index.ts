/**
 * Advanced on-device OCR for React Native, built on react-native-executorch.
 *
 * Phase 1 of the build-out ships the native module and its first typed
 * wrappers; the OCR pipeline, hooks and model registry follow in later phases.
 * @packageDocumentation
 */

export { isEmulator, rotateTensor, type RotationDegrees } from './native/ops';
