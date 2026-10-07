/**
 * A fake document-helper model (orientation, dewarp, table structure) for the
 * pipeline tests. Every method is exported at a fixed size, like the real
 * PaddleHelpers program; what each one computes is set per test: the
 * orientation logits, the dewarp grid and the token sequence the table
 * decoder replays.
 */
import {
  ConstantDim,
  type ConcreteDim,
  type MethodSpec,
  type ModelSpec,
} from 'react-native-executorch/schema';
import type { FakeTensor } from './fakeTensor';
import { setFakeModel, type FakeMethod } from './rneMock';

const f32 = (...shape: number[]) =>
  ({ kind: 'Tensor', dtype: 'float32', shape: shape.map((d) => ConstantDim(d)) }) as const;

const spec = (inputs: ReturnType<typeof f32>[], outputs: ReturnType<typeof f32>[]) =>
  ({ inputs, outputs, runtimeConstraints: [] }) satisfies MethodSpec<ConcreteDim>;

/** The fixed sizes of the fake program. */
export const HELPERS = {
  oriH: 16,
  oriW: 16,
  dewH: 8,
  dewW: 8,
  gridH: 3,
  gridW: 3,
  tabH: 8,
  tabW: 8,
  feat: 4,
  featDim: 2,
  hidden: 3,
  vocab: 8,
} as const;

/**
 * The fake's structure vocabulary: `eos` is the last token, so `eosTokenId`
 * is 7 and every id in between is a real fragment.
 */
export const FAKE_STRUCTURE_VOCAB = [
  'sos',
  '<tr>',
  '</tr>',
  '<td></td>',
  '<td',
  ' colspan="2"',
  '></td>',
  'eos',
] as const;
export const FAKE_EOS = FAKE_STRUCTURE_VOCAB.indexOf('eos');

export type FakeHelpersSchemaOptions = {
  /** Leave the table methods out of the export. */
  readonly withoutTable?: boolean;
  /** Overrides of the fixed sizes. */
  readonly sizes?: Partial<typeof HELPERS>;
};

/**
 * The schema of the fake program, PaddleHelpers-shaped.
 */
export function fakeHelpersSchema(options: FakeHelpersSchemaOptions = {}): ModelSpec<ConcreteDim> {
  const s = { ...HELPERS, ...options.sizes };
  const imageMethods: ModelSpec<ConcreteDim> = {
    orientation: spec([f32(1, 3, s.oriH, s.oriW)], [f32(1, 4)]),
    dewarp: spec([f32(1, 3, s.dewH, s.dewW)], [f32(1, 2, s.gridH, s.gridW)]),
  };
  if (options.withoutTable) {
    return imageMethods;
  }
  return {
    ...imageMethods,
    ['table_encode']: spec([f32(1, 3, s.tabH, s.tabW)], [f32(1, s.feat, s.featDim)]),
    ['table_decode_step']: spec(
      [f32(1, s.feat, s.featDim), f32(1, s.hidden), f32(1, s.vocab)],
      [f32(1, s.vocab), f32(1, s.hidden)]
    ),
  };
}

/**
 * An `align_corners` identity sampling grid of the fake's grid size: channel 0
 * is x and channel 1 is y, both spanning `[-1, 1]`.
 */
export function identityGrid(gridH = HELPERS.gridH, gridW = HELPERS.gridW): Float32Array {
  const grid = new Float32Array(2 * gridH * gridW);
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      grid[y * gridW + x] = gridW > 1 ? -1 + (2 * x) / (gridW - 1) : 0;
      grid[gridH * gridW + y * gridW + x] = gridH > 1 ? -1 + (2 * y) / (gridH - 1) : 0;
    }
  }
  return grid;
}

/** The identity grid mirrored left to right. */
export function mirroredGrid(gridH = HELPERS.gridH, gridW = HELPERS.gridW): Float32Array {
  const grid = identityGrid(gridH, gridW);
  for (let i = 0; i < gridH * gridW; i++) grid[i] = -grid[i]!;
  return grid;
}

export type FakeHelpersModelOptions = {
  readonly schema?: ModelSpec<ConcreteDim>;
  /** The orientation logits, `[0°, 90°, 180°, 270°]`. Default: 0° with certainty. */
  readonly logits?: readonly number[];
  /** The dewarp grid. Default: the identity. */
  readonly grid?: Float32Array;
  /**
   * The tokens the table decoder emits, one per step, as the index of the
   * one-hot winner. Default: `eos` at the first step (an empty skeleton).
   */
  readonly tokens?: readonly number[];
};

export type FakeHelpersCalls = {
  orientation: number;
  dewarp: number;
  tableEncode: number;
  /** The one-hot token fed at every decode step (-1 for the all-zero start). */
  readonly decodeInputs: number[];
};

/**
 * Registers the fake helper model at `path` and returns the log of its
 * executions.
 */
export function installFakeHelpersModel(
  path: string,
  options: FakeHelpersModelOptions = {}
): FakeHelpersCalls {
  const calls: FakeHelpersCalls = { orientation: 0, dewarp: 0, tableEncode: 0, decodeInputs: [] };
  const logits = options.logits ?? [20, 0, 0, 0];
  const grid = options.grid ?? identityGrid();
  const tokens = options.tokens ?? [FAKE_EOS];

  const orientation: FakeMethod = (_inputs, outputs) => {
    calls.orientation++;
    (outputs[0]!.data as Float32Array).set(logits);
  };
  const dewarp: FakeMethod = (_inputs, outputs) => {
    calls.dewarp++;
    const out = outputs[0]!;
    if (out.numel !== grid.length) {
      throw new Error(`fake dewarp: grid has ${grid.length} values, the output holds ${out.numel}`);
    }
    (out.data as Float32Array).set(grid);
  };
  const tableEncode: FakeMethod = (_inputs, outputs) => {
    calls.tableEncode++;
    (outputs[0]!.data as Float32Array).fill(0.5);
  };
  // The hidden state counts the steps, so the token to emit is read off it.
  const tableDecodeStep: FakeMethod = (inputs, outputs) => {
    const hidden = (inputs[1] as FakeTensor).data as Float32Array;
    const onehot = (inputs[2] as FakeTensor).data as Float32Array;
    calls.decodeInputs.push(onehot.indexOf(1));
    const step = hidden[0]!;
    const probs = outputs[0]!.data as Float32Array;
    probs.fill(0);
    probs[tokens[step] ?? FAKE_EOS] = 1;
    const newHidden = outputs[1]!.data as Float32Array;
    newHidden.set(hidden);
    newHidden[0] = step + 1;
  };

  setFakeModel(path, {
    schema: options.schema ?? fakeHelpersSchema(),
    methods: {
      orientation,
      dewarp,
      ['table_encode']: tableEncode,
      ['table_decode_step']: tableDecodeStep,
    },
  });
  return calls;
}
