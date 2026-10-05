import { loadModel, tensor, type Tensor } from 'react-native-executorch';
import type { Quad } from 'react-native-executorch/cv';
import { ConstantDim, constraint, EnumDim, f32, RangeDim } from 'react-native-executorch/schema';
import {
  craftExtractBoxes,
  dbnetExtractBoxes,
  type TextBoxExtractor,
} from '../../src/ocr/detectors';
import {
  resolveOcrContract,
  runOcrPass,
  runOcrPassOnTensor,
  type OcrEngine,
} from '../../src/ocr/engine';
import {
  blankPage,
  CELL,
  FAKE_CHARSET,
  fakeOcrSchema,
  installFakeOcrModel,
  paintColumn,
  paintLine,
  REC_TIMESTEPS,
  REC_WIDTH,
  type FakeOcrModelOptions,
  type FakeOcrSchemaOptions,
} from '../support/fakeOcrModel';
import { tensorTracker, type FakeTensor } from '../support/fakeTensor';

const codedError = (code: string) => expect.objectContaining({ name: 'RnExecuTorchError', code });

const PATH = '/models/ocr.pte';

/** An extractor whose quads the test supplies, in detector-input pixels. */
function extractorOf(
  quadsFor: (charLevel: boolean, inputSize: { width: number; height: number }) => Quad[],
  extra: Partial<TextBoxExtractor> = {}
): TextBoxExtractor {
  return {
    detectOutputSpec: (dim) => [f32(1, 1, dim('detOutH'), dim('detOutW'))],
    outputShapes: ({ width, height }) => [[1, 1, height, width]],
    extract: (_outputs, inputSize, charLevel) => quadsFor(charLevel, inputSize),
    ...extra,
  };
}

function contractOf(schemaOptions: FakeOcrSchemaOptions, extractor = dbnetExtractBoxes) {
  installFakeOcrModel(PATH, { schema: fakeOcrSchema(schemaOptions) });
  const model = loadModel(PATH);
  try {
    return resolveOcrContract(model, FAKE_CHARSET, extractor);
  } finally {
    model.dispose();
  }
}

/** Loads the fake model and runs `body` with an engine over it. */
function withEngine<R>(
  extractBoxes: TextBoxExtractor,
  body: (engine: OcrEngine, calls: ReturnType<typeof installFakeOcrModel>) => R,
  options: FakeOcrModelOptions &
    Partial<Pick<OcrEngine, 'minConfidence' | 'decode'>> & {
      padMode?: 'constant' | 'cornerMean';
    } = {}
): R {
  const calls = installFakeOcrModel(PATH, options);
  const model = loadModel(PATH);
  try {
    const contract = resolveOcrContract(model, FAKE_CHARSET, extractBoxes);
    return body(
      {
        model,
        extractBoxes,
        det: { ...contract.det, norm: { alpha: 1 / 255, beta: 0 } },
        rec: {
          ...contract.rec,
          norm: { alpha: 1 / 127.5, beta: -1 },
          padValue: 128,
          padMode: options.padMode ?? 'constant',
        },
        charset: contract.charset,
        minConfidence: options.minConfidence ?? 0,
        decode: options.decode,
      },
      calls
    );
  } finally {
    model.dispose();
  }
}

describe('resolveOcrContract', () => {
  it('resolves the domains, the stride and the charset of a size-varying model', () => {
    const contract = contractOf({});
    expect(contract.det).toEqual({ h: RangeDim(32, 960, 32), w: RangeDim(32, 960, 32) });
    expect(contract.rec).toEqual({
      recH: CELL,
      maxW: 32 * CELL,
      width: RangeDim(CELL, 32 * CELL, CELL),
      stride: { pixelsPerStep: CELL, offset: 0 },
      vocab: FAKE_CHARSET.length + 1,
    });
    expect(contract.charset).toEqual(['[blank]', ...FAKE_CHARSET]);
  });

  it('takes an array charset verbatim', () => {
    installFakeOcrModel(PATH, { schema: fakeOcrSchema({ vocab: 3 }) });
    const model = loadModel(PATH);
    const contract = resolveOcrContract(model, ['ff', 'ij'], dbnetExtractBoxes);
    model.dispose();
    expect(contract.charset).toEqual(['[blank]', 'ff', 'ij']);
  });

  it.each([
    ['fixed detector, varying recognizer', { detH: ConstantDim(64), detW: ConstantDim(96) }],
    ['varying detector, fixed recognizer', { recW: ConstantDim(64), recT: ConstantDim(8) }],
    [
      'both fixed',
      {
        detH: ConstantDim(64),
        detW: ConstantDim(96),
        recW: ConstantDim(64),
        recT: ConstantDim(8),
      },
    ],
    [
      'enum domains',
      {
        detH: EnumDim([320, 640]),
        detW: EnumDim([320, 640]),
        recW: EnumDim([64, 128]),
        recT: EnumDim([8, 16]),
      },
    ],
  ] as [string, FakeOcrSchemaOptions][])('accepts %s', (_name, schemaOptions) => {
    const contract = contractOf(schemaOptions);
    expect(contract.det.h).toEqual(schemaOptions.detH ?? RangeDim(32, 960, 32));
    expect(contract.rec.width).toEqual(schemaOptions.recW ?? RangeDim(CELL, 32 * CELL, CELL));
  });

  it('reads an affine stride from the declared constraint', () => {
    const contract = contractOf({
      recW: ConstantDim(68),
      recT: ConstantDim(16),
      recConstraints: [constraint.linear(REC_WIDTH, REC_TIMESTEPS, 4, 4)],
    });
    expect(contract.rec.stride).toEqual({ pixelsPerStep: 4, offset: 4 });
    expect(contract.rec.maxW).toBe(68);
  });

  it('accepts the CRAFT layout with even detector sizes', () => {
    const half = RangeDim(16, 480, 16);
    const contract = contractOf(
      { detOut: () => [ConstantDim(1), half, half, ConstantDim(2)] },
      craftExtractBoxes
    );
    expect(contract.det.w).toEqual(RangeDim(32, 960, 32));
  });

  it.each([
    ['no linear constraint', { recConstraints: [] }],
    [
      'a constraint over other dimensions',
      {
        recConstraints: [constraint.linear({ ...REC_WIDTH, dimIdx: 2 }, REC_TIMESTEPS, CELL, 0)],
      },
    ],
    [
      'a non-positive width per timestep',
      { recConstraints: [constraint.linear(REC_WIDTH, REC_TIMESTEPS, 0, 8)] },
    ],
    [
      'a second, unknown constraint',
      {
        recConstraints: [
          constraint.linear(REC_WIDTH, REC_TIMESTEPS, CELL, 0),
          constraint.equality(REC_WIDTH, REC_TIMESTEPS),
        ],
      },
    ],
    ['a charset of the wrong size', { vocab: 5 }],
    ['a detect output of another layout', { detOut: () => [ConstantDim(1), ConstantDim(7)] }],
    [
      'a recognizer width off the timestep lattice',
      { recW: ConstantDim(30), recT: ConstantDim(3) },
    ],
    ['a recognizer that mixes a fixed width with varying timesteps', { recW: ConstantDim(64) }],
  ])('rejects %s', (_name, schemaOptions) => {
    expect(() => contractOf(schemaOptions)).toThrow(codedError('SCHEMA_MISMATCH'));
  });

  it('rejects a detector whose sizes the extractor alignment excludes', () => {
    const schemaOptions = {
      detH: ConstantDim(64),
      detW: ConstantDim(33),
      detOut: () => [ConstantDim(1), ConstantDim(32), ConstantDim(16), ConstantDim(2)],
    };
    expect(() => contractOf(schemaOptions, craftExtractBoxes)).toThrow(
      codedError('SCHEMA_MISMATCH')
    );
  });

  it('rejects a model without the methods', () => {
    installFakeOcrModel(PATH, { schema: { forward: fakeOcrSchema().detect! } });
    const model = loadModel(PATH);
    expect(() => resolveOcrContract(model, FAKE_CHARSET, dbnetExtractBoxes)).toThrow(
      codedError('SCHEMA_MISMATCH')
    );
    model.dispose();
  });
});

describe('runOcrPass', () => {
  it('reads every detected line and returns them in reading order', () => {
    const page = blankPage(64, 32);
    const lower = paintLine(page, 8, 20, 'cab');
    const upper = paintLine(page, 0, 4, 'bad');
    withEngine(
      extractorOf(() => [lower, upper]),
      (engine, calls) => {
        expect(runOcrPass(engine, page)).toEqual([
          { text: 'bad', confidence: 1, quad: upper },
          { text: 'cab', confidence: 1, quad: lower },
        ]);
        expect(calls.detect).toEqual([[32, 64]]);
        expect(calls.recognize).toEqual([3 * CELL, 3 * CELL]);
      }
    );
  });

  it('collapses repeats and blanks like CTC', () => {
    const page = blankPage(64, 32);
    const line = paintLine(page, 0, 0, 'aa bb b');
    withEngine(
      extractorOf(() => [line]),
      (engine) => expect(runOcrPass(engine, page).map((d) => d.text)).toEqual(['abb'])
    );
  });

  it.each(['rgba', 'bgra', 'gray'] as const)('reads a %s page', (format) => {
    const page = blankPage(64, 32, format);
    const line = paintLine(page, 16, 8, 'hid');
    withEngine(
      extractorOf(() => [line]),
      (engine) => expect(runOcrPass(engine, page).map((d) => d.text)).toEqual(['hid'])
    );
  });

  it('maps quads from the letterboxed detector input back to the page', () => {
    // A 128x64 page against a detector fixed at 64x64: scale 0.5, 16 px of
    // padding above and below.
    const page = blankPage(128, 64);
    const line = paintLine(page, 16, 32, 'fig');
    withEngine(
      extractorOf((_charLevel, inputSize) => {
        expect(inputSize).toEqual({ width: 64, height: 64 });
        return [
          [
            { x: 8, y: 32 },
            { x: 20, y: 32 },
            { x: 20, y: 36 },
            { x: 8, y: 36 },
          ],
        ];
      }),
      (engine) => {
        expect(runOcrPass(engine, page)).toEqual([{ text: 'fig', confidence: 1, quad: line }]);
      },
      { schema: fakeOcrSchema({ detH: ConstantDim(64), detW: ConstantDim(64) }) }
    );
  });

  it('orders the corners of a quad the detector returned rotated', () => {
    const page = blankPage(64, 32);
    const line = paintLine(page, 0, 0, 'abc');
    const rotated: Quad = [line[2], line[3], line[0], line[1]];
    withEngine(
      extractorOf(() => [rotated]),
      (engine) => {
        expect(runOcrPass(engine, page)).toEqual([{ text: 'abc', confidence: 1, quad: line }]);
      }
    );
  });

  it('returns nothing, and recognizes nothing, when no text is detected', () => {
    withEngine(
      extractorOf(() => []),
      (engine, calls) => {
        expect(runOcrPass(engine, blankPage(64, 32))).toEqual([]);
        expect(calls.recognize).toEqual([]);
      }
    );
  });

  it('skips boxes with a side under 3 px and lines that read empty', () => {
    const page = blankPage(64, 32);
    const thin: Quad = [
      { x: 0, y: 0 },
      { x: 40, y: 0 },
      { x: 40, y: 2 },
      { x: 0, y: 2 },
    ];
    const empty = paintLine(page, 0, 16, '   ');
    withEngine(
      extractorOf(() => [thin, empty]),
      (engine, calls) => {
        expect(runOcrPass(engine, page)).toEqual([]);
        expect(calls.recognize).toEqual([3 * CELL]);
      }
    );
  });

  it('drops reads under the minimum confidence', () => {
    const page = blankPage(64, 32);
    const line = paintLine(page, 0, 0, 'abc');
    const extractor = extractorOf(() => [line]);
    withEngine(extractor, (engine) => expect(runOcrPass(engine, page)).toEqual([]), {
      probability: 0.4,
      minConfidence: 0.5,
    });
    withEngine(
      extractor,
      (engine) => {
        const [detection] = runOcrPass(engine, page);
        expect(detection!.text).toBe('abc');
        expect(detection!.confidence).toBeCloseTo(0.4);
      },
      { probability: 0.4, minConfidence: 0.4 - 1e-6 }
    );
  });

  it('pads a narrow line up to a fixed recognizer width', () => {
    const page = blankPage(64, 32);
    const line = paintLine(page, 0, 0, 'ab');
    for (const padMode of ['constant', 'cornerMean'] as const) {
      withEngine(
        extractorOf(() => [line]),
        (engine, calls) => {
          expect(runOcrPass(engine, page).map((d) => d.text)).toEqual(['ab']);
          expect(calls.recognize).toEqual([64]);
        },
        { schema: fakeOcrSchema({ recW: ConstantDim(64), recT: ConstantDim(8) }), padMode }
      );
    }
  });

  describe('with a recognizer at most 32 px wide', () => {
    const narrow = { schema: fakeOcrSchema({ recW: RangeDim(8, 32, 8), recT: RangeDim(1, 4, 1) }) };

    it('squishes a line up to 15% wider than the maximum', () => {
      const page = blankPage(64, 32);
      const line = paintLine(page, 0, 0, 'abcd');
      const wider: Quad = [line[0], { x: 36, y: 0 }, { x: 36, y: 8 }, line[3]];
      withEngine(
        extractorOf(() => [wider]),
        (engine, calls) => {
          expect(runOcrPass(engine, page)).toHaveLength(1);
          expect(calls.recognize).toEqual([32]);
        },
        narrow
      );
    });

    it('splits a wider line and joins the segments with a length-weighted confidence', () => {
      const page = blankPage(128, 32);
      const line = paintLine(page, 0, 0, 'abcdefg     ');
      // Segment 1 reads 4 glyphs at 1.0, segment 2 reads 3 at 0.3, segment 3
      // reads nothing and must not count.
      const probabilities = [1, 0.3, 0.9];
      let call = 0;
      withEngine(
        extractorOf(() => [line]),
        (engine, calls) => {
          const [detection] = runOcrPass(engine, page);
          expect(calls.recognize).toEqual([32, 32, 32]);
          expect(detection!.text).toBe('abcdefg');
          expect(detection!.confidence).toBeCloseTo((4 * 1 + 3 * 0.3) / 7);
          expect(detection!.quad).toEqual(line);
        },
        {
          ...narrow,
          decode: (probs) => {
            const segment = call++;
            const data = (probs as Tensor).getData(new Float32Array(4 * 11));
            let text = '';
            for (let t = 0; t < 4; t++) {
              const cls = [...data.slice(t * 11, (t + 1) * 11)].findIndex((p) => p > 0);
              if (cls > 0) text += FAKE_CHARSET[cls - 1];
            }
            return { text, confidence: probabilities[segment]! };
          },
        }
      );
    });
  });

  it('uses the custom decode with the charset including the blank', () => {
    const page = blankPage(64, 32);
    const line = paintLine(page, 0, 0, 'abc');
    const decode = jest.fn(() => ({ text: 'custom', confidence: 0.75 }));
    withEngine(
      extractorOf(() => [line]),
      (engine) => {
        expect(runOcrPass(engine, page)).toEqual([
          { text: 'custom', confidence: 0.75, quad: line },
        ]);
        const [probs, charset] = decode.mock.calls[0] as unknown as [Tensor, string[]];
        expect(probs.shape).toEqual([1, 3, FAKE_CHARSET.length + 1]);
        expect(charset).toEqual(['[blank]', ...FAKE_CHARSET]);
      },
      { decode }
    );
  });

  it('leaks nothing when the extractor rejects the detector size', () => {
    const extractor = extractorOf(() => [], {
      outputShapes: () => {
        throw new Error('bad size');
      },
    });
    withEngine(extractor, (engine) => {
      expect(() => runOcrPass(engine, blankPage(64, 32))).toThrow('bad size');
      expect(tensorTracker.liveCount()).toBe(0);
    });
  });

  it.each(['detect', 'recognize'] as const)('leaks nothing when %s fails', (failing) => {
    const page = blankPage(64, 32, 'rgba');
    const line = paintLine(page, 0, 0, 'abc');
    withEngine(
      extractorOf(() => [line]),
      (engine) => {
        expect(() => runOcrPass(engine, page)).toThrow('native failure');
        expect(tensorTracker.liveCount()).toBe(0);
      },
      {
        [failing]: () => {
          throw new Error('native failure');
        },
      }
    );
  });
});

describe('runOcrPassOnTensor', () => {
  it('reads a caller-owned page tensor and leaves it alive', () => {
    const page = blankPage(64, 32);
    const line = paintLine(page, 0, 0, 'jig');
    const tPage = tensor('uint8', [32, 64, 3], page.data);
    withEngine(
      extractorOf(() => [line]),
      (engine) => {
        expect(runOcrPassOnTensor(engine, tPage, 64, 32, 'rgb').map((d) => d.text)).toEqual([
          'jig',
        ]);
        expect((tPage as unknown as FakeTensor).disposed).toBe(false);
        expect(tensorTracker.liveCount()).toBe(1);
      }
    );
    tPage.dispose();
  });
});

describe('vertical text', () => {
  it('reads separately detected glyphs as three lines without the vertical pass', () => {
    const page = blankPage(64, 64);
    const glyphs = paintColumn(page, 8, 0, 'dig', 2);
    withEngine(
      extractorOf(() => glyphs),
      (engine) => expect(runOcrPass(engine, page).map((d) => d.text)).toEqual(['d', 'i', 'g'])
    );
  });

  it('joins a stack of glyphs into one read, top to bottom', () => {
    const page = blankPage(64, 64);
    const glyphs = paintColumn(page, 8, 0, 'dig', 2);
    const line = paintLine(page, 24, 40, 'bead');
    withEngine(
      extractorOf(() => [glyphs[2]!, line, glyphs[0]!, glyphs[1]!]),
      (engine, calls) => {
        expect(runOcrPass(engine, page, { vertical: true })).toEqual([
          {
            text: 'dig',
            confidence: 1,
            quad: [
              { x: 8, y: 0 },
              { x: 16, y: 0 },
              { x: 16, y: 28 },
              { x: 8, y: 28 },
            ],
          },
          { text: 'bead', confidence: 1, quad: line },
        ]);
        expect(calls.recognize).toEqual([CELL, CELL, CELL, 4 * CELL]);
      }
    );
  });

  it('gates vertical reads with their own confidence threshold', () => {
    const page = blankPage(64, 64);
    const glyphs = paintColumn(page, 8, 0, 'dig', 2);
    const extractor = extractorOf(() => glyphs);
    withEngine(
      extractor,
      (engine) => {
        expect(runOcrPass(engine, page, { vertical: true })).toEqual([]);
        expect(
          runOcrPass(engine, page, { vertical: true, verticalMinConfidence: 0.2 }).map(
            (d) => d.text
          )
        ).toEqual(['dig']);
      },
      { probability: 0.3 }
    );
  });

  describe('a tall box merged from stacked glyphs', () => {
    const tall: Quad = [
      { x: 8, y: 0 },
      { x: 16, y: 0 },
      { x: 16, y: 24 },
      { x: 8, y: 24 },
    ];
    // The 8x24 crop is letterboxed into the 32x32 detector input.
    const scale = 32 / 24;
    const offsetX = (32 - 8 * scale) / 2;
    const glyphInDetector = (i: number): Quad => [
      { x: offsetX, y: i * CELL * scale },
      { x: offsetX + CELL * scale, y: i * CELL * scale },
      { x: offsetX + CELL * scale, y: (i + 1) * CELL * scale },
      { x: offsetX, y: (i + 1) * CELL * scale },
    ];
    const charLevelExtractor = (supportsCharLevel: boolean, glyphs = [2, 0, 1]) =>
      extractorOf((charLevel) => (charLevel ? glyphs.map(glyphInDetector) : [tall]), {
        supportsCharLevel,
      });

    function page() {
      const p = blankPage(64, 32);
      paintColumn(p, 8, 0, 'hid');
      return p;
    }

    it('is re-detected at char level and read glyph by glyph', () => {
      withEngine(charLevelExtractor(true), (engine, calls) => {
        expect(runOcrPass(engine, page(), { vertical: true })).toEqual([
          { text: 'hid', confidence: 1, quad: tall },
        ]);
        expect(calls.detect).toEqual([
          [32, 64],
          [32, 32],
        ]);
        expect(calls.recognize).toEqual([CELL, CELL, CELL]);
      });
    });

    it('is read from an RGBA page too', () => {
      const p = blankPage(64, 32, 'rgba');
      paintColumn(p, 8, 0, 'hid');
      withEngine(charLevelExtractor(true), (engine) => {
        expect(runOcrPass(engine, p, { vertical: true }).map((d) => d.text)).toEqual(['hid']);
      });
    });

    it('is read horizontally when the extractor has no char-level mode', () => {
      withEngine(charLevelExtractor(false), (engine, calls) => {
        runOcrPass(engine, page(), { vertical: true });
        expect(calls.detect).toEqual([[32, 64]]);
        expect(calls.recognize).toEqual([CELL]);
      });
    });

    it('is read horizontally when the re-detection finds no glyphs', () => {
      withEngine(charLevelExtractor(true, []), (engine, calls) => {
        runOcrPass(engine, page(), { vertical: true });
        expect(calls.detect).toHaveLength(2);
        expect(calls.recognize).toEqual([CELL]);
      });
    });

    it('is read horizontally once the re-detection budget is spent', () => {
      withEngine(charLevelExtractor(true), (engine, calls) => {
        runOcrPass(engine, page(), { vertical: true, maxRedetections: 0 });
        expect(calls.detect).toEqual([[32, 64]]);
      });
    });

    it('is left alone under the tall-crop ratio', () => {
      withEngine(charLevelExtractor(true), (engine, calls) => {
        runOcrPass(engine, page(), { vertical: true, tallCropRatio: 4 });
        expect(calls.detect).toEqual([[32, 64]]);
      });
    });
  });
});
