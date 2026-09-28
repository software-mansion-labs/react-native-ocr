/**
 * Mock of `react-native-executorch`: the subset of its public API that `src/`
 * uses, backed by {@link FakeTensor}. Grows with the package.
 *
 * The `cv` and `math` namespaces are the sibling mocks (also mapped to the
 * `react-native-executorch/cv` and `/math` subpaths); `schema` is the real
 * module, which is plain TypeScript with no native dependency.
 */
import * as schema from 'react-native-executorch/schema';
import * as cv from './cvMock';
import { FakeTensor, type DType } from './fakeTensor';
import * as math from './mathMock';

export { cv, math, schema };

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

export type ModelInput = Tensor | number | boolean | null;
export type ModelOutput = Tensor | number | boolean | string | null;

/** What one exported method does when executed: it fills `outputs` in place. */
export type FakeMethod = (inputs: ModelInput[], outputs: Tensor[]) => void;

export type FakeModelDefinition = {
  readonly schema: schema.ModelSpec<schema.ConcreteDim>;
  readonly methods: Record<string, FakeMethod>;
};

const modelDefinitions = new Map<string, FakeModelDefinition>();
const liveModels = new Set<FakeModel>();

export class FakeModel {
  readonly schema: schema.ModelSpec<schema.ConcreteDim>;
  readonly backends: Record<string, readonly string[]> = {};
  disposed = false;

  constructor(
    readonly path: string,
    private readonly definition: FakeModelDefinition
  ) {
    this.schema = definition.schema;
    liveModels.add(this);
  }

  execute(methodName: string, inputs: ModelInput[], outputTensors: Tensor[]): ModelOutput[] {
    if (this.disposed) {
      throw RnExecuTorchError('RESOURCE_DISPOSED', `execute: model '${this.path}' was disposed`);
    }
    const run = this.definition.methods[methodName];
    if (!run) {
      throw RnExecuTorchError('INVALID_ARGUMENT', `execute: no method '${methodName}'`);
    }
    for (const t of [...inputs, ...outputTensors]) {
      if (t instanceof FakeTensor && t.disposed) {
        throw RnExecuTorchError('RESOURCE_DISPOSED', `execute: tensor ${t.id} was disposed`);
      }
    }
    run(inputs, outputTensors);
    return outputTensors;
  }

  dispose(): void {
    this.disposed = true;
    liveModels.delete(this);
  }
}

export type Model = FakeModel;

/**
 * Registers what `loadModel(path)` returns: the exported schema and the
 * behavior of each method. Loading an unregistered path fails like a missing
 * file does.
 */
export function setFakeModel(path: string, definition: FakeModelDefinition): void {
  modelDefinitions.set(path, definition);
}

export function loadModel(modelPath: string): Model {
  const definition = modelDefinitions.get(modelPath);
  if (!definition) {
    throw RnExecuTorchError('LOAD_FAILED', `loadModel: no fake model registered at '${modelPath}'`);
  }
  return new FakeModel(modelPath, definition);
}

export const modelTracker = {
  liveCount(): number {
    return liveModels.size;
  },
  livePaths(): string[] {
    return [...liveModels].map((m) => m.path);
  },
  reset(): void {
    liveModels.clear();
    modelDefinitions.clear();
  },
};

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

export const IMAGENET_NORM = {
  alpha: [1 / (255.0 * 0.229), 1 / (255.0 * 0.224), 1 / (255.0 * 0.225)],
  beta: [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
} as const;

let registeredBackends: readonly string[] = [];

/**
 * Sets what `getRegisteredBackends` reports. Modules that resolve a `DEFAULT`
 * read it once at import, so pair this with `jest.isolateModules`.
 */
export function setFakeRegisteredBackends(names: readonly string[]): void {
  registeredBackends = names;
}

export function getRegisteredBackends(): readonly string[] {
  return registeredBackends;
}
