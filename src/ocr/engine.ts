/**
 * The detect and recognize engine behind `createOcr`: the execution of one OCR
 * pass over a page tensor, and the load-time contract that validates a model
 * and resolves its legal input sizes.
 *
 * Worklet source order matters: a worklet must be defined above its callers.
 */
import { RnExecuTorchError, tensor, type Model, type Tensor } from 'react-native-executorch';
import {
  cvtColor,
  normalize,
  orderQuad,
  quadSize,
  rectifyQuad,
  resize,
  scaleQuad,
  toChannelsFirst,
  type ImageBuffer,
  type ImageFormat,
  type NormalizeOptions,
  type Quad,
} from 'react-native-executorch/cv';
import {
  constraint,
  DynamicDim,
  f32,
  method,
  StaticDim,
  validateSpec,
  type ConcreteDim,
  type DimRef,
  type LinearConstraint,
  type ModelSpec,
  type SymbolicDim,
  type TensorSpec,
} from 'react-native-executorch/schema';
import { greedyCtcDecode } from './ctc';
import type { TextBoxExtractor } from './detectors';
import { groupVerticalStacks, orderByReadingOrder } from './geometry';
import { FORMAT_CHANNELS, TO_RGB } from './imageFormat';
import { boundingQuadOf, splitTallQuad, splitWideQuad } from './quad';
import { fillRecognizerCanvas } from './recognizerCanvas';
import {
  ctcTimesteps,
  dimExtent,
  domainAdmits,
  onCtcLattice,
  resolveDetectorSize,
  resolveRecWidth,
  type CtcStride,
} from './sizing';
import type { OcrDetection } from './types';

// ─── Execution ───────────────────────────────────────────────────────────────

// A line at most this factor wider than the recognizer max is squished into it;
// anything wider is split and read piecewise (a hard clamp is unreadable).
const WIDE_SQUISH_TOLERANCE = 1.15;

// Boxes smaller than this (px, either side) are not worth recognizing.
const MIN_BOX_SIDE = 3;

// The recognizer input is contractually RGB (enforced by resolveOcrContract).
const REC_CHANNELS = 3;

const TALL_CROP_RATIO = 1.5;
const MAX_VERTICAL_REDETECTIONS = 8;
// Vertical reads are hallucination-prone, so they default to a modest gate; a
// per-run `verticalMinConfidence` tunes it (0 keeps everything).
const DEFAULT_VERTICAL_MIN_CONFIDENCE = 0.4;

/**
 * The resolved, model-level state one detect and recognize pass needs: the
 * model itself, the validated contract, the CTC charset and the model options.
 * Built once by `createOcr`, reused across every page and per-region call.
 */
export type OcrEngine = {
  readonly model: Model;
  readonly extractBoxes: TextBoxExtractor;
  readonly det: {
    /** The exported domain of the detector input's height (NCHW dim 2). */
    readonly h: ConcreteDim;
    /** The exported domain of the detector input's width (NCHW dim 3). */
    readonly w: ConcreteDim;
    readonly norm: NormalizeOptions;
  };
  readonly rec: {
    readonly recH: number;
    readonly maxW: number;
    /** The exported domain of the recognizer input's width (NCHW dim 3). */
    readonly width: ConcreteDim;
    readonly stride: CtcStride;
    /** Recognizer output vocab size (charset length including the blank). */
    readonly vocab: number;
    readonly norm: NormalizeOptions;
    readonly padValue: number;
    readonly padMode: 'constant' | 'cornerMean';
  };
  readonly charset: string[];
  readonly minConfidence: number;
  readonly decode?: (
    probs: Tensor,
    charset: readonly string[]
  ) => { readonly text: string; readonly confidence: number };
};

/**
 * Per-run knobs of one OCR pass: the OCR-level subset of `RunOcrOptions`,
 * which is structurally assignable, so callers pass it directly.
 */
export type OcrPassOptions = {
  readonly vertical?: boolean;
  readonly tallCropRatio?: number;
  readonly maxRedetections?: number;
  readonly verticalMinConfidence?: number;
};

type Reading = { text: string; confidence: number };

// Detects text boxes in `src`, returning quads in `src` pixel space.
// `charLevel` requests per-glyph boxes for the stacked-text pass.
function detectQuads(
  engine: OcrEngine,
  src: Tensor,
  width: number,
  height: number,
  format: ImageFormat,
  charLevel = false
): Quad[] {
  'worklet';
  const numChannels = FORMAT_CHANNELS[format];
  const toRgbCode = TO_RGB[format];
  const align = Math.max(1, engine.extractBoxes.inputAlignment ?? 1);
  const { detW, detH } = resolveDetectorSize(engine.det, width, height, align);
  // Resolve output shapes before allocating: the extractor may throw (CRAFT on
  // odd dims), and nothing must leak on that path.
  const outputShapes = engine.extractBoxes.outputShapes({ width: detW, height: detH });
  const tResize = tensor('uint8', [detH, detW, numChannels]);
  const tColor = toRgbCode !== null ? tensor('uint8', [detH, detW, 3]) : null;
  const tCF = tensor('uint8', [3, detH, detW]);
  const tNorm = tensor('float32', [3, detH, detW]);
  const tInput = tensor('float32', [1, 3, detH, detW]);
  const tOutputs = outputShapes.map((shape) => tensor('float32', shape));
  try {
    src
      .through(resize, tResize, { mode: 'letterbox', interpolation: 'area', padValue: 0 })
      .throughIf(tColor !== null, cvtColor, tColor!, toRgbCode!)
      .through(toChannelsFirst, tCF)
      .through(normalize, tNorm, engine.det.norm)
      .copyTo(tInput);

    engine.model.execute('detect', [tInput], tOutputs);
    const quads = engine.extractBoxes.extract(tOutputs, { width: detW, height: detH }, charLevel);
    return quads.map((q) =>
      scaleQuad(q, {
        from: { width: detW, height: detH },
        to: { width, height },
        resizeMode: 'letterbox',
      })
    );
  } finally {
    tResize.dispose();
    tColor?.dispose();
    tCF.dispose();
    tNorm.dispose();
    tInput.dispose();
    for (const tOut of tOutputs) {
      tOut.dispose();
    }
  }
}

// Recognizes a canvas already sized to a legal recognizer width: the custom
// `decode` if provided, else greedy CTC.
function recognizeCanvas(engine: OcrEngine, tCanvas: Tensor, snappedW: number): Reading {
  'worklet';
  const { recH, stride, vocab, norm } = engine.rec;
  const tCF = tensor('uint8', [REC_CHANNELS, recH, snappedW]);
  const tNorm = tensor('float32', [REC_CHANNELS, recH, snappedW]);
  const tInput = tensor('float32', [1, REC_CHANNELS, recH, snappedW]);
  // The width the caller snapped to sits on the timestep lattice by
  // construction, so the dynamically sized probs output can be pre-allocated
  // exactly.
  const timesteps = ctcTimesteps(snappedW, stride);
  const tProbs = tensor('float32', [1, timesteps, vocab]);
  try {
    tCanvas.through(toChannelsFirst, tCF).through(normalize, tNorm, norm).copyTo(tInput);
    engine.model.execute('recognize', [tInput], [tProbs]);
    const read = engine.decode
      ? engine.decode(tProbs, engine.charset)
      : greedyCtcDecode(tProbs, engine.charset);
    return { text: read.text, confidence: read.confidence };
  } finally {
    tCF.dispose();
    tNorm.dispose();
    tInput.dispose();
    tProbs.dispose();
  }
}

// Recognizes one quad that fits the recognizer width (a mild squish is fine);
// wider lines go through recognizeQuad instead.
function recognizeNarrowQuad(engine: OcrEngine, src: Tensor, quad: Quad): Reading {
  'worklet';
  const { recH, maxW, width, stride, padValue, padMode } = engine.rec;
  const size = quadSize(quad);
  // Aspect-preserving width of the content at recognizer height (>= 1 px).
  const aspectWidth = Math.max(1, Math.round((recH * size.width) / Math.max(1, size.height)));
  const contentW = Math.min(aspectWidth, maxW);
  const snappedW = resolveRecWidth(width, contentW, stride);
  const tCanvas = tensor('uint8', [recH, snappedW, REC_CHANNELS]);
  try {
    fillRecognizerCanvas(src, tCanvas, quad, {
      // A constant or enum domain can snap below the content; the canvas caps it.
      contentWidth: Math.min(contentW, snappedW),
      padMode,
      padValue,
    });
    return recognizeCanvas(engine, tCanvas, snappedW);
  } finally {
    tCanvas.dispose();
  }
}

// Recognizes one ordered (TL, TR, BR, BL) quad. A line wider than the
// tolerance is split into segments read piecewise; confidence is the
// length-weighted mean.
function recognizeQuad(engine: OcrEngine, src: Tensor, quad: Quad): Reading {
  'worklet';
  const { recH, maxW } = engine.rec;
  const size = quadSize(quad);

  const desiredW = Math.max(1, Math.round((recH * size.width) / Math.max(1, size.height)));
  if (desiredW <= maxW * WIDE_SQUISH_TOLERANCE) {
    return recognizeNarrowQuad(engine, src, quad);
  }
  // Known limitation: segments abut with no overlap, so a glyph straddling a
  // cut can be mangled or dropped at the seam. An overlap margin plus de-dup
  // would fix it; acceptable for now since a line this wide is the rare case.
  const segments = splitWideQuad(quad, Math.ceil(desiredW / maxW));
  let text = '';
  let weightedConfidence = 0;
  let weight = 0;
  for (const segment of segments) {
    const r = recognizeNarrowQuad(engine, src, segment);
    if (r.text.length > 0) {
      text += r.text;
      weightedConfidence += r.confidence * r.text.length;
      weight += r.text.length;
    }
  }
  return { text, confidence: weight === 0 ? 0 : weightedConfidence / weight };
}

// Reads a vertical stack of glyph quads top to bottom: each glyph is
// recognized on its OWN and the single-char results joined. Composing the
// glyphs into one horizontal strip makes the recognizer hallucinate a word out
// of a non-word column (its context was trained on horizontal words); reading
// them individually drops that bias. A box the detector merged from several
// stacked letters is first split into roughly square cells. Confidence is the
// length-weighted mean; null when nothing usable.
function recognizeGlyphColumn(
  engine: OcrEngine,
  src: Tensor,
  glyphs: readonly Quad[]
): Reading | null {
  'worklet';
  let text = '';
  let weightedConfidence = 0;
  let weight = 0;
  for (const glyph of glyphs) {
    const glyphSize = quadSize(glyph);
    if (glyphSize.width < 1 || glyphSize.height < 1) {
      continue;
    }
    const parts = Math.max(1, Math.round(glyphSize.height / Math.max(1, glyphSize.width)));
    for (const cell of splitTallQuad(glyph, parts)) {
      const cellSize = quadSize(cell);
      if (cellSize.width < 1 || cellSize.height < 1) {
        continue;
      }
      const r = recognizeNarrowQuad(engine, src, cell);
      if (r.text.length > 0) {
        text += r.text;
        weightedConfidence += r.confidence * r.text.length;
        weight += r.text.length;
      }
    }
  }
  return weight === 0 ? null : { text, confidence: weightedConfidence / weight };
}

// Reads one tall box the detector merged from stacked glyphs: crop upright,
// re-detect the glyphs (char-level), recognize top to bottom. Null (the caller
// reads it horizontally) when too small, out of re-detect budget, or no glyphs
// found.
function readStackedBox(
  engine: OcrEngine,
  page: Tensor,
  format: ImageFormat,
  ordered: Quad,
  size: { width: number; height: number },
  budget: { remaining: number }
): Reading | null {
  'worklet';
  const boxW = Math.round(size.width);
  const boxH = Math.round(size.height);
  if (boxW < MIN_BOX_SIDE || boxH < MIN_BOX_SIDE || budget.remaining <= 0) {
    return null;
  }
  budget.remaining--;
  const numChannels = FORMAT_CHANNELS[format];
  const toRgbCode = TO_RGB[format];
  const tBoxRaw = tensor('uint8', [boxH, boxW, numChannels]);
  let tRecBox: Tensor | null = null;
  try {
    rectifyQuad(page, tBoxRaw, ordered, { contentWidth: boxW, align: 'left', padValue: 0 });
    const charQuads = detectQuads(engine, tBoxRaw, boxW, boxH, format, /* charLevel */ true);
    if (charQuads.length === 0) {
      return null;
    }
    let boxSrc = tBoxRaw;
    if (toRgbCode !== null) {
      tRecBox = tensor('uint8', [boxH, boxW, REC_CHANNELS]);
      boxSrc = cvtColor(tBoxRaw, tRecBox, toRgbCode);
    }
    const glyphs = charQuads.map((q) => orderQuad(q)).sort((a, b) => a[0].y - b[0].y);
    return recognizeGlyphColumn(engine, boxSrc, glyphs);
  } finally {
    tBoxRaw.dispose();
    tRecBox?.dispose();
  }
}

// ─── OCR pass ────────────────────────────────────────────────────────────────

function pushDetection(
  out: OcrDetection[],
  threshold: number,
  text: string,
  confidence: number,
  quad: Quad
): void {
  'worklet';
  if (text.length > 0 && confidence >= threshold) {
    out.push({ text, confidence, quad });
  }
}

/**
 * Runs one detect, recognize and reading-order pass over a page tensor,
 * returning the recognized regions in reading order. Does NOT own `page` (the
 * caller allocates and disposes it), so the orientation and dewarp path can
 * feed its corrected page tensor straight in without a round trip through an
 * `ImageBuffer`. The horizontal pass reads every quad; the opt-in vertical
 * pass additionally reads upright glyph stacks.
 * @param engine The resolved OCR engine (contract, model, model options).
 * @param page The page tensor to read (`uint8` `[H, W, C]`); caller-owned.
 * @param width The page width in pixels.
 * @param height The page height in pixels.
 * @param format The pixel format of `page` (channels and color conversion).
 * @param options Per-pass toggles (vertical text, tall-crop ratio, re-detect cap).
 * @returns The recognized regions, in reading order.
 */
export function runOcrPassOnTensor(
  engine: OcrEngine,
  page: Tensor,
  width: number,
  height: number,
  format: ImageFormat,
  options?: OcrPassOptions
): OcrDetection[] {
  'worklet';
  const vertical = options?.vertical ?? false;
  const tallCropRatio = options?.tallCropRatio ?? TALL_CROP_RATIO;
  const maxRedetections = options?.maxRedetections ?? MAX_VERTICAL_REDETECTIONS;
  const verticalMinConfidence = options?.verticalMinConfidence ?? DEFAULT_VERTICAL_MIN_CONFIDENCE;
  const toRgbCode = TO_RGB[format];

  let tRecImage: Tensor | null = null;
  try {
    const quads = detectQuads(engine, page, width, height, format);
    if (quads.length === 0) {
      return [];
    }

    let recSrc = page;
    if (toRgbCode !== null) {
      tRecImage = tensor('uint8', [height, width, REC_CHANNELS]);
      recSrc = cvtColor(page, tRecImage, toRgbCode);
    }
    const budget = { remaining: maxRedetections };

    const detections: OcrDetection[] = [];
    const ordered: Quad[] = [];
    for (const quad of quads) {
      const orderedQuad = orderQuad(quad);
      const size = quadSize(orderedQuad);
      if (size.width >= MIN_BOX_SIDE && size.height >= MIN_BOX_SIDE) {
        ordered.push(orderedQuad);
      }
    }

    if (!vertical) {
      for (const orderedQuad of ordered) {
        const { text, confidence } = recognizeQuad(engine, recSrc, orderedQuad);
        pushDetection(detections, engine.minConfidence, text, confidence, orderedQuad);
      }
      return orderByReadingOrder(detections);
    }

    // The stacked-box path re-detects glyphs char-level; only extractors that
    // support that mode take it, others read the tall box horizontally.
    const canReadStacks = engine.extractBoxes.supportsCharLevel ?? false;
    const { stacks, singles } = groupVerticalStacks(ordered);
    for (const stack of stacks) {
      const column = recognizeGlyphColumn(engine, recSrc, stack);
      if (column) {
        pushDetection(
          detections,
          verticalMinConfidence,
          column.text,
          column.confidence,
          boundingQuadOf(stack)
        );
      }
    }
    for (const orderedQuad of singles) {
      const size = quadSize(orderedQuad);
      if (canReadStacks && size.height >= size.width * tallCropRatio) {
        const stacked = readStackedBox(engine, page, format, orderedQuad, size, budget);
        if (stacked) {
          pushDetection(
            detections,
            verticalMinConfidence,
            stacked.text,
            stacked.confidence,
            orderedQuad
          );
          continue;
        }
      }
      const { text, confidence } = recognizeQuad(engine, recSrc, orderedQuad);
      pushDetection(detections, engine.minConfidence, text, confidence, orderedQuad);
    }
    return orderByReadingOrder(detections);
  } finally {
    tRecImage?.dispose();
  }
}

/**
 * Runs one OCR pass over an `ImageBuffer`, uploading it to a page tensor and
 * delegating to {@link runOcrPassOnTensor}.
 * @param engine The resolved OCR engine (contract, model, model options).
 * @param input The page (or region crop) to read.
 * @param options Per-pass toggles (vertical text, tall-crop ratio, re-detect cap).
 * @returns The recognized regions, in reading order.
 */
export function runOcrPass(
  engine: OcrEngine,
  input: ImageBuffer,
  options?: OcrPassOptions
): OcrDetection[] {
  'worklet';
  const numChannels = FORMAT_CHANNELS[input.format];
  const tPage = tensor('uint8', [input.height, input.width, numChannels]);
  try {
    tPage.setData(input.data);
    return runOcrPassOnTensor(engine, tPage, input.width, input.height, input.format, options);
  } finally {
    tPage.dispose();
  }
}

// ─── Contract ────────────────────────────────────────────────────────────────

type DimFactory = (symbol: string) => SymbolicDim;

// Where the recognizer's width-to-timesteps constraint lives in `recognize`:
// the input width is NCHW dim 3 of the only input tensor, the timestep count
// is dim 1 of the only output tensor.
const REC_WIDTH_REF: DimRef = { paramSide: 'input', tensorIdx: 0, dimIdx: 3 };
const REC_TIMESTEPS_REF: DimRef = { paramSide: 'output', tensorIdx: 0, dimIdx: 1 };

function refsEqual(a: DimRef, b: DimRef): boolean {
  return a.paramSide === b.paramSide && a.tensorIdx === b.tensorIdx && a.dimIdx === b.dimIdx;
}

// Reads the width-to-timesteps relation the recognizer declares. It is
// discovered rather than assumed because it differs per architecture;
// `resolveOcrContract` feeds the result back into the allowed spec so
// `validateSpec` still matches the constraint one to one and rejects any
// others the model may declare.
function readCtcStride(schema: ModelSpec<ConcreteDim>): CtcStride {
  const constraints = schema.recognize?.runtimeConstraints ?? [];
  const linear = constraints.find(
    (c): c is LinearConstraint =>
      c.kind === 'linear' &&
      refsEqual(c.dimLhs, REC_WIDTH_REF) &&
      refsEqual(c.dimRhs, REC_TIMESTEPS_REF)
  );
  if (linear === undefined) {
    throw RnExecuTorchError(
      'SCHEMA_MISMATCH',
      'resolveOcrContract: the recognizer must declare a linear runtime constraint relating ' +
        'its input width (input 0, dim 3) to its CTC timesteps (output 0, dim 1); the pipeline ' +
        'needs it to pre-size the probs output.'
    );
  }
  const [pixelsPerStep, offset] = linear.coefficients;
  if (pixelsPerStep <= 0) {
    throw RnExecuTorchError(
      'SCHEMA_MISMATCH',
      `resolveOcrContract: the recognizer's width-per-timestep coefficient must be positive, ` +
        `but the model declares ${pixelsPerStep}.`
    );
  }
  return { pixelsPerStep, offset };
}

// The exported domain of one tensor dimension. `validateSpec` has already
// proved the method exists with this rank, so the lookup cannot miss.
function exportedDim(schema: ModelSpec<ConcreteDim>, methodName: string, ref: DimRef): ConcreteDim {
  const methodSpec = schema[methodName]!;
  const params = ref.paramSide === 'input' ? methodSpec.inputs : methodSpec.outputs;
  const tensors = params.filter((p): p is TensorSpec<ConcreteDim> => p.kind === 'Tensor');
  return tensors[ref.tensorIdx]!.shape[ref.dimIdx]!;
}

/**
 * The model-derived part of an {@link OcrEngine}: everything but the options
 * `createOcr` adds.
 */
export type OcrContract = {
  readonly det: Omit<OcrEngine['det'], 'norm'>;
  readonly rec: Omit<OcrEngine['rec'], 'norm' | 'padValue' | 'padMode'>;
  readonly charset: string[];
};

/**
 * Validates the model's `detect` (RGB `[1, 3, H, W]` in, extractor-defined
 * outputs) and `recognize` (RGB `[1, 3, H, W]` in, `[1, T, V]` probs out; only
 * the width may vary) methods against the exported schema, resolves their
 * legal input sizes, and builds the CTC charset. Runs at construction.
 * @param model The loaded fused detect/recognize model.
 * @param charsetOption The recognizer charset (string = one codepoint per
 * index; array = taken verbatim).
 * @param extractBoxes The detector's box extractor: it declares the `detect`
 * output layout it decodes, and its alignment is checked against the model's
 * legal input sizes.
 * @returns The model-derived detect/recognize contract.
 * @throws {RnExecuTorchError} With code `SCHEMA_MISMATCH` on any contract
 * mismatch.
 */
export function resolveOcrContract(
  model: Model,
  charsetOption: string | readonly string[],
  extractBoxes: TextBoxExtractor
): OcrContract {
  const schema = model.schema;
  const stride = readCtcStride(schema);

  // Either method may be exported at a fixed size or with varying sizes, and
  // the two choices are independent (EasyOCR pairs a size-varying detector
  // with a fixed-width recognizer), so all four combinations are offered as
  // variants. A `static` symbol binds only to a constant dimension and a
  // `dynamic` one only to a range or enum, which is exactly what separates
  // them.
  const variant = (detDim: DimFactory, recDim: DimFactory) => ({
    ...method(
      'detect',
      [f32(1, 3, detDim('detH'), detDim('detW'))],
      extractBoxes.detectOutputSpec(detDim)
    ),
    ...method(
      'recognize',
      [f32(1, REC_CHANNELS, 'recH', recDim('recW'))],
      [f32(1, recDim('recT'), 'vocab')],
      [constraint.linear(REC_WIDTH_REF, REC_TIMESTEPS_REF, stride.pixelsPerStep, stride.offset)]
    ),
  });

  const { dims } = validateSpec(schema, {
    dynamicDetectDynamicRec: variant(DynamicDim, DynamicDim),
    dynamicDetectFixedRec: variant(DynamicDim, StaticDim),
    fixedDetectDynamicRec: variant(StaticDim, DynamicDim),
    fixedDetectFixedRec: variant(StaticDim, StaticDim),
  });
  const [recH, vocabSize] = dims.constant('recH', 'vocab');

  const det = {
    h: exportedDim(schema, 'detect', { paramSide: 'input', tensorIdx: 0, dimIdx: 2 }),
    w: exportedDim(schema, 'detect', { paramSide: 'input', tensorIdx: 0, dimIdx: 3 }),
  };
  const width = exportedDim(schema, 'recognize', REC_WIDTH_REF);

  // Reject at load, not per run, a detector whose legal sizes the extractor
  // cannot decode: `resolveDetectorSize` only ever returns a size the domain
  // admits, so if no such size is a multiple of the extractor's alignment,
  // every run fails.
  const detAlign = Math.max(1, extractBoxes.inputAlignment ?? 1);
  if (detAlign > 1) {
    for (const [axis, dim] of [
      ['height', det.h],
      ['width', det.w],
    ] as const) {
      if (!domainAdmits(dim, (value) => value % detAlign === 0)) {
        throw RnExecuTorchError(
          'SCHEMA_MISMATCH',
          `resolveOcrContract: no legal detector input ${axis} is a multiple of the extractor's ` +
            `required alignment (${detAlign}), so decoding would fail on every run.`
        );
      }
    }
  }

  // Likewise, `recognizeCanvas` pre-allocates the probs tensor as
  // [1, ctcTimesteps(width), V], so the width the runtime picks must land on
  // the timestep lattice: the domain has to admit at least one such width.
  if (!domainAdmits(width, (value) => onCtcLattice(value, stride))) {
    throw RnExecuTorchError(
      'SCHEMA_MISMATCH',
      `resolveOcrContract: no legal recognizer width yields a whole number of CTC timesteps ` +
        `(width = ${stride.pixelsPerStep} * timesteps + ${stride.offset}), so the probs tensor ` +
        `cannot be pre-sized.`
    );
  }

  // CTC lookup: index 0 is the blank, then the model's characters (a string
  // splits into codepoints; an array is taken verbatim, preserving ligatures).
  const charset = [
    '[blank]',
    ...(typeof charsetOption === 'string' ? Array.from(charsetOption) : charsetOption),
  ];
  if (charset.length !== vocabSize) {
    throw RnExecuTorchError(
      'SCHEMA_MISMATCH',
      `resolveOcrContract: charset size (${charset.length}, incl. blank) must match the ` +
        `recognizer output vocab (${vocabSize}).`
    );
  }

  return {
    det,
    rec: { recH, maxW: dimExtent(width)[1], width, stride, vocab: vocabSize },
    charset,
  };
}
