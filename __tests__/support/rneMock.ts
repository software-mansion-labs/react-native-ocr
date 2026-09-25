/**
 * Mock of `react-native-executorch`: the subset of its public API that `src/`
 * uses, backed by {@link FakeTensor}. Grows with the package.
 */
import * as cv from './cvMock';
import { FakeTensor, type DType } from './fakeTensor';

export { cv };

export type Tensor = FakeTensor;
export type { DType };

export const VALID_ERROR_CODES = [
  'LOAD_FAILED',
  'EXECUTION_FAILED',
  'SCHEMA_MISMATCH',
  'INVALID_ARGUMENT',
  'INVALID_STATE',
  'RESOURCE_DISPOSED',
  'RESOURCE_BUSY',
  'DOWNLOAD_FAILED',
  'DOWNLOAD_ABORTED',
  'UNKNOWN',
] as const;

export type RnExecuTorchErrorCode = (typeof VALID_ERROR_CODES)[number];

export type RnExecuTorchError<C extends RnExecuTorchErrorCode = RnExecuTorchErrorCode> = Error & {
  name: 'RnExecuTorchError';
  code: C;
  etRuntimeErrorCode?: number;
};

export function RnExecuTorchError<C extends RnExecuTorchErrorCode>(
  code: C,
  message: string,
  etRuntimeErrorCode?: number
): RnExecuTorchError<C> {
  const err = new Error(message) as RnExecuTorchError<C>;
  err.name = 'RnExecuTorchError';
  err.code = code;
  if (etRuntimeErrorCode !== undefined) err.etRuntimeErrorCode = etRuntimeErrorCode;
  return err;
}

export function isRnExecuTorchError<C extends RnExecuTorchErrorCode = RnExecuTorchErrorCode>(
  err: unknown,
  code?: C
): err is RnExecuTorchError<C> {
  if (err === null || typeof err !== 'object') return false;
  if (!('name' in err) || err.name !== 'RnExecuTorchError') return false;
  if (!('code' in err) || typeof err.code !== 'string') return false;
  if (!(VALID_ERROR_CODES as readonly string[]).includes(err.code)) return false;
  return code === undefined || err.code === code;
}

export function tensor(
  dtype: DType,
  shape: number[],
  src?: Float32Array | Uint8Array | Int32Array | BigInt64Array
): Tensor {
  const t = new FakeTensor(dtype, shape);
  if (src) t.setData(src);
  return t;
}

export const defaultWorkletRuntime = { name: 'FakeExecuTorchRuntime' };

export function wrapAsync<Args extends any[], R>(fn: (...args: Args) => R) {
  return async (...args: Args): Promise<R> => fn(...args);
}

export type NativeResource = { dispose: () => void };

export function createResourceScope() {
  const allocated: NativeResource[] = [];
  return {
    track: <R extends NativeResource>(resource: R): R => {
      allocated.push(resource);
      return resource;
    },
    dispose: (): void => {
      for (const resource of allocated.splice(0).reverse()) resource.dispose();
    },
  };
}
