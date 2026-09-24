/**
 * A typed-array-backed stand-in for react-native-executorch's native `Tensor`.
 *
 * Only what `src/` uses is implemented, with the byte-level `setData` and
 * `getData` semantics of the real thing, plus allocation tracking so a test can
 * assert that everything it allocated was disposed.
 */

export type DType = 'float32' | 'uint8' | 'int32' | 'int64' | 'bool';
type Storage = Float32Array | Uint8Array | Int32Array | BigInt64Array;

const STORAGE: Record<DType, new (length: number) => Storage> = {
  float32: Float32Array,
  uint8: Uint8Array,
  int32: Int32Array,
  int64: BigInt64Array,
  bool: Uint8Array,
};

const BYTES_PER_ELEMENT: Record<DType, number> = {
  float32: 4,
  uint8: 1,
  int32: 4,
  int64: 8,
  bool: 1,
};

const live = new Map<number, FakeTensor>();
let nextId = 0;

export class FakeTensor {
  readonly id: number;
  readonly dtype: DType;
  readonly shape: readonly number[];
  readonly numel: number;
  readonly data: Storage;
  disposed = false;

  constructor(dtype: DType, shape: readonly number[]) {
    if (!(dtype in STORAGE)) throw new Error(`createTensor: unsupported dtype '${dtype}'`);
    if (shape.some((d) => !Number.isInteger(d) || d <= 0)) {
      throw new Error(`createTensor: invalid shape [${shape}]`);
    }
    this.id = nextId++;
    this.dtype = dtype;
    this.shape = Object.freeze([...shape]);
    this.numel = shape.reduce((a, b) => a * b, 1);
    this.data = new STORAGE[dtype](this.numel);
    live.set(this.id, this);
  }

  private assertLive(op: string): void {
    if (this.disposed) throw new Error(`${op}: tensor ${this.id} has already been disposed`);
  }

  private get byteLength(): number {
    return this.numel * BYTES_PER_ELEMENT[this.dtype];
  }

  setData(src: Storage): FakeTensor {
    this.assertLive('setData');
    if (src.byteLength !== this.byteLength) {
      throw new Error(
        `setData: source is ${src.byteLength} bytes, tensor holds ${this.byteLength}`
      );
    }
    new Uint8Array(this.data.buffer).set(
      new Uint8Array(src.buffer as ArrayBuffer, src.byteOffset, src.byteLength)
    );
    return this;
  }

  getData<T extends Storage>(dst: T): T {
    this.assertLive('getData');
    if (dst.byteLength !== this.byteLength) {
      throw new Error(
        `getData: destination is ${dst.byteLength} bytes, tensor holds ${this.byteLength}`
      );
    }
    new Uint8Array(dst.buffer as ArrayBuffer, dst.byteOffset, dst.byteLength).set(
      new Uint8Array(this.data.buffer)
    );
    return dst;
  }

  copyTo(dst: FakeTensor, options?: { offset?: number; length?: number }): FakeTensor {
    this.assertLive('copyTo');
    dst.assertLive('copyTo');
    if (dst.dtype !== this.dtype) {
      throw new Error(`copyTo: dtype mismatch ('${this.dtype}' -> '${dst.dtype}')`);
    }
    const offset = options?.offset ?? 0;
    const length = options?.length ?? this.numel - offset;
    if (offset < 0 || length < 0 || offset + length > this.numel) {
      throw new Error(`copyTo: window [${offset}, ${offset + length}) is out of bounds`);
    }
    if (length > dst.numel) {
      throw new Error(`copyTo: destination holds ${dst.numel} elements, need ${length}`);
    }
    for (let i = 0; i < length; i++) (dst.data as any)[i] = (this.data as any)[offset + i];
    return dst;
  }

  through<R, Args extends unknown[]>(fn: (t: FakeTensor, ...args: Args) => R, ...args: Args): R {
    this.assertLive('through');
    return fn(this, ...args);
  }

  throughIf<Args extends unknown[]>(
    pred: boolean,
    fn: (t: FakeTensor, ...args: Args) => FakeTensor,
    ...args: Args
  ): FakeTensor {
    return pred ? this.through(fn, ...args) : this;
  }

  dispose(): void {
    this.disposed = true;
    live.delete(this.id);
  }
}

export const tensorTracker = {
  liveCount(): number {
    return live.size;
  },
  liveDescriptions(): string[] {
    return [...live.values()].map((t) => `#${t.id} ${t.dtype}[${t.shape}]`);
  },
  reset(): void {
    live.clear();
  },
};
