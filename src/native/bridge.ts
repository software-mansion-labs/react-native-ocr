import { RnExecuTorchError } from 'react-native-executorch';
import type { RnOcrJsi } from './types';

const globalObj = globalThis as { __rnocr_jsi__?: RnOcrJsi };

if (!globalObj.__rnocr_jsi__) {
  const NativeRnOcr = require('./NativeRnOcr').default;
  if (NativeRnOcr) NativeRnOcr.install();
}

/**
 * Direct reference to the package's native JSI object (`__rnocr_jsi__`).
 *
 * Worklets capture this module-level constant in their closure, the same way
 * react-native-executorch captures its own JSI object, so the host functions
 * are reachable from a background worklet runtime without touching
 * `globalThis` there.
 * @internal
 */
export const rnocrJsi: RnOcrJsi = globalObj.__rnocr_jsi__ as RnOcrJsi;

if (!rnocrJsi) {
  throw RnExecuTorchError(
    'UNKNOWN',
    "JSI global object '__rnocr_jsi__' is not registered. Make sure @swmansion/react-native-ocr is autolinked and the app was rebuilt."
  );
}
