/**
 * A fake fused detect/recognize model for the pipeline tests, plus the page
 * encoding its recognizer reads.
 *
 * Text is painted as `CELL`-sized squares whose gray level is `10 * class`, so
 * a line rectified to the recognizer height at scale 1 puts one class per CTC
 * timestep. The fake `recognize` turns each timestep's pixel back into a
 * one-hot probability row; any level that is not a multiple of 10 (the default
 * gray padding, for one) reads as the blank. `detect` computes nothing: the
 * quads come from the extractor under test.
 */
import {
  ConstantDim,
  constraint,
  RangeDim,
  type ConcreteDim,
  type MethodSpec,
  type ModelSpec,
  type RuntimeConstraint,
} from 'react-native-executorch/schema';
import type { ImageBuffer, ImageFormat } from './cvMock';
import type { FakeTensor } from './fakeTensor';
import { setFakeModel, type FakeMethod } from './rneMock';

/** Side of one painted glyph cell, the recognizer height and its CTC stride. */
export const CELL = 8;

export const FAKE_CHARSET = 'abcdefghij';
/** Charset plus the blank. */
export const FAKE_VOCAB = FAKE_CHARSET.length + 1;

export type FakeOcrSchemaOptions = {
  readonly detH?: ConcreteDim;
  readonly detW?: ConcreteDim;
  /** Shape of the single `detect` output given the input dims. */
  readonly detOut?: (h: ConcreteDim, w: ConcreteDim) => ConcreteDim[];
  readonly recH?: number;
  readonly recW?: ConcreteDim;
  readonly recT?: ConcreteDim;
  readonly vocab?: number;
  readonly recConstraints?: RuntimeConstraint[];
};

const f32 = (...shape: ConcreteDim[]) => ({ kind: 'Tensor', dtype: 'float32', shape }) as const;

export const REC_WIDTH = { paramSide: 'input', tensorIdx: 0, dimIdx: 3 } as const;
export const REC_TIMESTEPS = { paramSide: 'output', tensorIdx: 0, dimIdx: 1 } as const;

/**
 * An exported schema shaped like PP-OCRv6 (range dims, DBNet output, stride
 * `(CELL, 0)`), with every part replaceable.
 */
export function fakeOcrSchema(options: FakeOcrSchemaOptions = {}): ModelSpec<ConcreteDim> {
  const detH = options.detH ?? RangeDim(32, 960, 32);
  const detW = options.detW ?? RangeDim(32, 960, 32);
  const detOut = options.detOut ?? ((h, w) => [ConstantDim(1), ConstantDim(1), h, w]);
  const recW = options.recW ?? RangeDim(CELL, 32 * CELL, CELL);
  const recT = options.recT ?? RangeDim(1, 32, 1);
  const detect: MethodSpec<ConcreteDim> = {
    inputs: [f32(ConstantDim(1), ConstantDim(3), detH, detW)],
    outputs: [f32(...detOut(detH, detW))],
    runtimeConstraints: [],
  };
  const recognize: MethodSpec<ConcreteDim> = {
    inputs: [f32(ConstantDim(1), ConstantDim(3), ConstantDim(options.recH ?? CELL), recW)],
    outputs: [f32(ConstantDim(1), recT, ConstantDim(options.vocab ?? FAKE_VOCAB))],
    runtimeConstraints: options.recConstraints ?? [
      constraint.linear(REC_WIDTH, REC_TIMESTEPS, CELL, 0),
    ],
  };
  return { detect, recognize };
}

export type FakeOcrModelOptions = {
  readonly schema?: ModelSpec<ConcreteDim>;
  /** Probability written for the class each timestep reads. Default 1. */
  readonly probability?: number;
  readonly detect?: FakeMethod;
  readonly recognize?: FakeMethod;
};

export type FakeOcrModelCalls = {
  /** `[H, W]` of every detector input, in call order. */
  readonly detect: number[][];
  /** Width of every recognizer input, in call order. */
  readonly recognize: number[];
};

/**
 * Registers the fake model at `path` and returns the log of its executions.
 */
export function installFakeOcrModel(
  path: string,
  options: FakeOcrModelOptions = {}
): FakeOcrModelCalls {
  const calls: FakeOcrModelCalls = { detect: [], recognize: [] };
  const probability = options.probability ?? 1;

  const recognize: FakeMethod = (inputs, outputs) => {
    const input = inputs[0] as FakeTensor;
    const probs = outputs[0]!;
    const [, , height, width] = input.shape as unknown as [number, number, number, number];
    const [, timesteps, vocab] = probs.shape as unknown as [number, number, number];
    if (width !== timesteps * CELL) {
      throw new Error(`fake recognize: width ${width} does not match ${timesteps} timesteps`);
    }
    const pixels = input.data as Float32Array;
    const out = probs.data as Float32Array;
    out.fill(0);
    const row = Math.floor(height / 2);
    for (let t = 0; t < timesteps; t++) {
      // Channel 0, normalized with the default recognizer norm (x / 127.5 - 1).
      const level = Math.round((pixels[row * width + t * CELL + CELL / 2]! + 1) * 127.5);
      const cls = level % 10 === 0 && level / 10 < vocab ? level / 10 : 0;
      out[t * vocab + cls] = probability;
    }
  };

  setFakeModel(path, {
    schema: options.schema ?? fakeOcrSchema(),
    methods: {
      detect: (inputs, outputs) => {
        const input = inputs[0] as FakeTensor;
        calls.detect.push([input.shape[2]!, input.shape[3]!]);
        options.detect?.(inputs, outputs);
      },
      recognize: (inputs, outputs) => {
        calls.recognize.push((inputs[0] as FakeTensor).shape[3]!);
        (options.recognize ?? recognize)(inputs, outputs);
      },
    },
  });
  return calls;
}

const CHANNELS: Record<ImageFormat, number> = { rgb: 3, bgr: 3, rgba: 4, bgra: 4, gray: 1 };

/** A blank page (level 5, which reads as the CTC blank). */
export function blankPage(width: number, height: number, format: ImageFormat = 'rgb'): ImageBuffer {
  const channels = CHANNELS[format];
  const data = new Uint8Array(width * height * channels).fill(5);
  if (channels === 4) {
    for (let i = 3; i < data.length; i += 4) data[i] = 255;
  }
  return { data, width, height, format, layout: 'hwc' };
}

function paintCell(page: ImageBuffer, x0: number, y0: number, char: string): void {
  const channels = CHANNELS[page.format];
  const level = char === ' ' ? 0 : (FAKE_CHARSET.indexOf(char) + 1) * 10;
  for (let y = y0; y < y0 + CELL; y++) {
    for (let x = x0; x < x0 + CELL; x++) {
      for (let c = 0; c < Math.min(channels, 3); c++) {
        page.data[(y * page.width + x) * channels + c] = level;
      }
    }
  }
}

type Corner = { readonly x: number; readonly y: number };
type PaintedQuad = readonly [Corner, Corner, Corner, Corner];

const rect = (x0: number, y0: number, x1: number, y1: number): PaintedQuad => [
  { x: x0, y: y0 },
  { x: x1, y: y0 },
  { x: x1, y: y1 },
  { x: x0, y: y1 },
];

/**
 * Paints `text` left to right from `(x, y)`, one cell per character (a space
 * paints the blank), and returns the quad around it.
 */
export function paintLine(page: ImageBuffer, x: number, y: number, text: string): PaintedQuad {
  [...text].forEach((char, i) => paintCell(page, x + i * CELL, y, char));
  return rect(x, y, x + text.length * CELL, y + CELL);
}

/**
 * Paints `text` top to bottom from `(x, y)` with `gap` px between glyphs and
 * returns one quad per glyph.
 */
export function paintColumn(
  page: ImageBuffer,
  x: number,
  y: number,
  text: string,
  gap = 0
): PaintedQuad[] {
  return [...text].map((char, i) => {
    const top = y + i * (CELL + gap);
    paintCell(page, x, top, char);
    return rect(x, top, x + CELL, top + CELL);
  });
}
