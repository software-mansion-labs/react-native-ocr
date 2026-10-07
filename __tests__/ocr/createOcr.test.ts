import { IMAGENET_NORM, tensor } from 'react-native-executorch';
import * as cv from 'react-native-executorch/cv';
import { ConstantDim } from 'react-native-executorch/schema';
import { rotateTensor, type RotationDegrees } from '../../src/native/ops';
import { createOcr } from '../../src/ocr/createOcr';
import { makeDbnetExtractBoxes } from '../../src/ocr/detectors';
import type { OcrModel } from '../../src/ocr/types';
import { setFakeDbnetTextQuads, type ImageBuffer } from '../support/cvMock';
import {
  FAKE_EOS,
  FAKE_STRUCTURE_VOCAB,
  HELPERS,
  installFakeHelpersModel,
  mirroredGrid,
} from '../support/fakeHelpersModel';
import {
  blankPage,
  FAKE_CHARSET,
  fakeOcrSchema,
  installFakeOcrModel,
  paintLine,
} from '../support/fakeOcrModel';
import { tensorTracker } from '../support/fakeTensor';
import { modelTracker, setFakeLayoutRegions, setFakeModel } from '../support/rneMock';

const codedError = (code: string) => expect.objectContaining({ name: 'RnExecuTorchError', code });

const PATH = '/models/ocr.pte';

const CONFIG: OcrModel = {
  modelPath: PATH,
  modelOpts: { charset: FAKE_CHARSET, extractBoxes: makeDbnetExtractBoxes() },
};

afterEach(() => setFakeDbnetTextQuads(null));

describe('createOcr', () => {
  it('reads a page into flat detections and one ungrouped block', async () => {
    installFakeOcrModel(PATH);
    const page = blankPage(64, 32);
    const second = paintLine(page, 8, 20, 'cab');
    const first = paintLine(page, 0, 4, 'bad');
    setFakeDbnetTextQuads(() => [second, first]);

    const ocr = await createOcr(CONFIG);
    const result = await ocr.runOcr(page);
    ocr.dispose();

    const detections = [
      { text: 'bad', confidence: 1, quad: first },
      { text: 'cab', confidence: 1, quad: second },
    ];
    expect(result).toEqual({
      detections,
      blocks: [
        {
          regionType: 'ungrouped',
          bbox: { format: 'xyxy', xmin: 0, ymin: 4, xmax: 32, ymax: 28 },
          layoutConfidence: 1,
          isTable: false,
          lines: detections,
          text: 'bad\ncab',
        },
      ],
      regions: [],
      image: page,
    });
  });

  it('returns no block for a page without text', async () => {
    installFakeOcrModel(PATH);
    setFakeDbnetTextQuads(() => []);
    const page = blankPage(64, 32);
    const ocr = await createOcr(CONFIG);
    expect(ocr.runOcrWorklet(page)).toEqual({
      detections: [],
      blocks: [],
      regions: [],
      image: page,
    });
    ocr.dispose();
  });

  it('applies the model options and their defaults', async () => {
    installFakeOcrModel(PATH, {
      probability: 0.6,
      schema: fakeOcrSchema({ recW: ConstantDim(64), recT: ConstantDim(8) }),
    });
    const page = blankPage(64, 32);
    const line = paintLine(page, 0, 0, 'abc');
    setFakeDbnetTextQuads(() => [line]);
    const normalize = jest.spyOn(cv, 'normalize');
    const rectify = jest.spyOn(cv, 'rectifyQuad');

    const defaults = await createOcr(CONFIG);
    expect(defaults.runOcrWorklet(page).detections).toHaveLength(1);
    defaults.dispose();
    expect(normalize.mock.calls.map((call) => call[2])).toEqual([
      IMAGENET_NORM,
      { alpha: 1 / 127.5, beta: -1 },
    ]);
    expect(rectify.mock.calls[0]![3]).toMatchObject({ padValue: 128 });

    normalize.mockClear();
    rectify.mockClear();
    const detectorNorm = { alpha: 1 / 255, beta: 0 };
    const tuned = await createOcr({
      ...CONFIG,
      modelOpts: {
        ...CONFIG.modelOpts,
        minConfidence: 0.7,
        detectorNorm,
        recognizerPadValue: 5,
      },
    });
    expect(tuned.runOcrWorklet(page).detections).toEqual([]);
    tuned.dispose();
    expect(normalize.mock.calls[0]![2]).toBe(detectorNorm);
    expect(rectify.mock.calls[0]![3]).toMatchObject({ padValue: 5 });
  });

  it('passes the per-run options to the pass', async () => {
    installFakeOcrModel(PATH, { probability: 0.3 });
    const page = blankPage(64, 64);
    const glyphs = ['d', 'i', 'g'].map((char, i) => paintLine(page, 8, i * 10, char));
    setFakeDbnetTextQuads(() => glyphs);
    const ocr = await createOcr(CONFIG);
    expect(ocr.runOcrWorklet(page).detections.map((d) => d.text)).toEqual(['d', 'i', 'g']);
    expect(ocr.runOcrWorklet(page, { vertical: true }).detections).toEqual([]);
    expect(
      ocr
        .runOcrWorklet(page, { vertical: true, verticalMinConfidence: 0 })
        .detections.map((d) => d.text)
    ).toEqual(['dig']);
    ocr.dispose();
  });

  it('releases the model on dispose', async () => {
    installFakeOcrModel(PATH);
    const ocr = await createOcr(CONFIG);
    expect(modelTracker.liveCount()).toBe(1);
    ocr.dispose();
    expect(modelTracker.liveCount()).toBe(0);
    expect(() => ocr.runOcrWorklet(blankPage(64, 32))).toThrow(codedError('RESOURCE_DISPOSED'));
    expect(tensorTracker.liveCount()).toBe(0);
  });

  it('fails with LOAD_FAILED when the model cannot be loaded', async () => {
    await expect(createOcr(CONFIG)).rejects.toEqual(codedError('LOAD_FAILED'));
  });

  it('releases the model when it does not match the contract', async () => {
    installFakeOcrModel(PATH, { schema: fakeOcrSchema({ vocab: 4 }) });
    await expect(createOcr(CONFIG)).rejects.toEqual(codedError('SCHEMA_MISMATCH'));
    expect(modelTracker.liveCount()).toBe(0);
  });
});

describe('createOcr with a layout model', () => {
  const LAYOUT_PATH = '/models/layout.pte';
  type Label = 'text' | 'image' | 'table';
  const LAYOUT_CONFIG: OcrModel<Label> = {
    ...CONFIG,
    layout: {
      modelPath: LAYOUT_PATH,
      modelOpts: { labels: ['text', 'image', 'table'], boxFormat: 'xyxy' },
    } as unknown as NonNullable<OcrModel<Label>['layout']>,
  };
  const installFakeLayoutModel = () =>
    setFakeModel(LAYOUT_PATH, { schema: fakeOcrSchema(), methods: {} });
  const box = (xmin: number, ymin: number, xmax: number, ymax: number) => ({
    format: 'xyxy' as const,
    xmin,
    ymin,
    xmax,
    ymax,
  });
  const region = (label: Label, bounds: ReturnType<typeof box>, confidence: number) => ({
    box: bounds,
    label,
    confidence,
  });

  it('reads each region on its own crop and assembles blocks in reading order', async () => {
    const calls = installFakeOcrModel(PATH);
    installFakeLayoutModel();
    // A 160x96 page of 64x32 cells (one detector size each, so quads map 1:1)
    // in two columns around a 32 px gutter: text/text, text/image, empty/table.
    const page = blankPage(160, 96);
    const bad = paintLine(page, 0, 4, 'bad');
    const cab = paintLine(page, 104, 20, 'cab');
    const dig = paintLine(page, 8, 36, 'dig');
    paintLine(page, 104, 40, 'fee'); // inside the image region: never read
    const left1 = region('text', box(0, 0, 64, 32), 0.9);
    const nested = region('text', box(8, 0, 40, 32), 0.6); // a line inside left1
    const right1 = region('text', box(95.6, -2, 160.4, 32.2), 0.8); // rounded and clamped
    const left2 = region('text', box(0, 32, 64, 64), 0.7);
    const image = region('image', box(96, 32, 160, 64), 0.95);
    const empty = region('text', box(0, 64, 64, 96), 0.5);
    const table = region('table', box(96, 64, 160, 96), 0.85);
    setFakeLayoutRegions(() => [left1, nested, right1, left2, image, empty, table]);
    // The crops are read in region order: left1, right1, left2, empty, table.
    const queue = [[bad], [shift(cab, -96, 0)], [shift(dig, 0, -32)], [], []];
    setFakeDbnetTextQuads(() => queue.shift() ?? []);

    const ocr = await createOcr<Label>(LAYOUT_CONFIG);
    const result = await ocr.runOcr(page);
    ocr.dispose();

    const line = (text: string, quad: typeof bad) => ({ text, confidence: 1, quad });
    expect(calls.detect).toHaveLength(5);
    expect(result.detections).toEqual([line('bad', bad), line('dig', dig), line('cab', cab)]);
    expect(result.blocks).toEqual([
      {
        regionType: 'text',
        bbox: left1.box,
        layoutConfidence: 0.9,
        isTable: false,
        lines: [line('bad', bad)],
        text: 'bad',
      },
      {
        regionType: 'text',
        bbox: left2.box,
        layoutConfidence: 0.7,
        isTable: false,
        lines: [line('dig', dig)],
        text: 'dig',
      },
      {
        regionType: 'text',
        bbox: right1.box,
        layoutConfidence: 0.8,
        isTable: false,
        lines: [line('cab', cab)],
        text: 'cab',
      },
      {
        regionType: 'table',
        bbox: table.box,
        layoutConfidence: 0.85,
        isTable: true,
        lines: [],
        text: '',
      },
    ]);
    expect(result.regions).toEqual([left1, right1, left2, image, empty, table]);
    expect(result.image).toBe(page);
  });

  it('returns no blocks for a page without regions', async () => {
    installFakeOcrModel(PATH);
    installFakeLayoutModel();
    const page = blankPage(64, 32);
    const ocr = await createOcr<Label>(LAYOUT_CONFIG);
    expect(ocr.runOcrWorklet(page)).toEqual({
      detections: [],
      blocks: [],
      regions: [],
      image: page,
    });
    ocr.dispose();
  });

  it('skips a region that rounds to nothing', async () => {
    const calls = installFakeOcrModel(PATH);
    installFakeLayoutModel();
    const page = blankPage(64, 32);
    setFakeLayoutRegions(() => [
      region('text', box(10, 10, 10.4, 20), 0.9),
      region('text', box(70, 0, 90, 20), 0.9),
    ]);
    const ocr = await createOcr<Label>(LAYOUT_CONFIG);
    expect(ocr.runOcrWorklet(page).blocks).toEqual([]);
    expect(calls.detect).toHaveLength(0);
    ocr.dispose();
  });

  it('honors custom visual and table labels', async () => {
    installFakeOcrModel(PATH);
    installFakeLayoutModel();
    const page = blankPage(128, 32);
    const figure = region('text', box(0, 0, 64, 32), 0.9);
    const grid = region('image', box(64, 0, 128, 32), 0.9);
    setFakeLayoutRegions(() => [figure, grid]);
    setFakeDbnetTextQuads(() => []);
    const ocr = await createOcr<Label>({
      ...LAYOUT_CONFIG,
      visualLabels: ['text'],
      tableLabel: 'image',
    });
    const result = ocr.runOcrWorklet(page);
    ocr.dispose();
    expect(result.blocks).toEqual([
      expect.objectContaining({ regionType: 'image', isTable: true, lines: [] }),
    ]);
  });

  it('releases both models on dispose', async () => {
    installFakeOcrModel(PATH);
    installFakeLayoutModel();
    const ocr = await createOcr<Label>(LAYOUT_CONFIG);
    expect(modelTracker.liveCount()).toBe(2);
    ocr.dispose();
    expect(modelTracker.liveCount()).toBe(0);
    expect(() => ocr.runOcrWorklet(blankPage(64, 32))).toThrow(codedError('RESOURCE_DISPOSED'));
    expect(tensorTracker.liveCount()).toBe(0);
  });

  it('releases the OCR model when the layout model cannot be loaded', async () => {
    installFakeOcrModel(PATH);
    await expect(createOcr<Label>(LAYOUT_CONFIG)).rejects.toEqual(codedError('LOAD_FAILED'));
    expect(modelTracker.liveCount()).toBe(0);
  });
});

describe('createOcr with document models', () => {
  const HELPERS_PATH = '/models/helpers.pte';
  const TABLE = { structureVocab: FAKE_STRUCTURE_VOCAB, eosTokenId: FAKE_EOS, maxSteps: 20 };
  const DM_CONFIG: OcrModel = {
    ...CONFIG,
    documentModels: { modelPath: HELPERS_PATH, table: TABLE },
  };
  const line = (text: string, quad: ReturnType<typeof paintLine>) => ({
    text,
    confidence: 1,
    quad,
  });
  // The 90° class with certainty, and the same class at a confidence below the gate.
  const SHOWS_90 = [0, 20, 0, 0];
  const MAYBE_90 = [0, 1, 0, 0];
  const ZERO_GRID = new Float32Array(2 * HELPERS.gridH * HELPERS.gridW);

  /** A 64x32 page reading `bad`, and the same page turned 90° clockwise. */
  function uprightAndTurned() {
    const upright = blankPage(64, 32);
    const bad = paintLine(upright, 0, 4, 'bad');
    const turned = rotate(upright, 90);
    return { upright, bad, turned };
  }

  it('rotates a page upright before reading it when orientation is on', async () => {
    const calls = installFakeHelpersModel(HELPERS_PATH, { logits: SHOWS_90 });
    installFakeOcrModel(PATH);
    const { upright, bad, turned } = uprightAndTurned();
    setFakeDbnetTextQuads(() => [bad]);
    const ocr = await createOcr({
      ...DM_CONFIG,
      documentModels: { ...DM_CONFIG.documentModels!, orientation: true },
    });
    const result = await ocr.runOcr(turned);
    ocr.dispose();
    expect(calls.orientation).toBe(1);
    expect(result.image).toEqual(upright);
    expect(result.image).not.toBe(turned);
    expect(result.detections).toEqual([line('bad', bad)]);
    expect(result.blocks[0]).toMatchObject({ regionType: 'ungrouped', text: 'bad' });
    expect(tensorTracker.liveCount()).toBe(0);
  });

  it('leaves the page alone when the classifier is not confident enough', async () => {
    installFakeHelpersModel(HELPERS_PATH, { logits: MAYBE_90 });
    installFakeOcrModel(PATH);
    const { turned } = uprightAndTurned();
    setFakeDbnetTextQuads(() => []);
    const ocr = await createOcr(DM_CONFIG);
    expect(ocr.runOcrWorklet(turned, { orientation: true }).image).toBe(turned);
    ocr.dispose();

    const lenient = await createOcr({
      ...DM_CONFIG,
      documentModels: { ...DM_CONFIG.documentModels!, orientationMinConfidence: 0.4 },
    });
    const result = lenient.runOcrWorklet(turned, { orientation: true });
    lenient.dispose();
    expect(result.image).toMatchObject({ width: 64, height: 32 });
  });

  it('leaves a page that is already upright alone', async () => {
    installFakeHelpersModel(HELPERS_PATH, { logits: [20, 0, 0, 0] });
    installFakeOcrModel(PATH);
    const { upright } = uprightAndTurned();
    setFakeDbnetTextQuads(() => []);
    const ocr = await createOcr(DM_CONFIG);
    expect(ocr.runOcrWorklet(upright, { orientation: true }).image).toBe(upright);
    ocr.dispose();
  });

  it('defaults the per-run orientation and dewarp flags to the config', async () => {
    const calls = installFakeHelpersModel(HELPERS_PATH, { logits: [20, 0, 0, 0] });
    installFakeOcrModel(PATH);
    const page = blankPage(64, 32);
    setFakeDbnetTextQuads(() => []);
    const off = await createOcr(DM_CONFIG);
    off.runOcrWorklet(page);
    off.dispose();
    expect(calls).toMatchObject({ orientation: 0, dewarp: 0 });

    const on = await createOcr({
      ...DM_CONFIG,
      documentModels: { ...DM_CONFIG.documentModels!, orientation: true, dewarp: true },
    });
    on.runOcrWorklet(page);
    expect(calls).toMatchObject({ orientation: 1, dewarp: 1 });
    on.runOcrWorklet(page, { orientation: false, dewarp: false });
    on.dispose();
    expect(calls).toMatchObject({ orientation: 1, dewarp: 1 });
  });

  it('reads the dewarped page and returns it as the result frame', async () => {
    installFakeHelpersModel(HELPERS_PATH, { grid: mirroredGrid() });
    installFakeOcrModel(PATH);
    const page = blankPage(64, 32);
    const bad = paintLine(page, 0, 4, 'bad');
    // Mirrored, the line sits at the right edge and reads backwards.
    const mirrored = blankPage(64, 32);
    const dab = paintLine(mirrored, 40, 4, 'dab');
    setFakeDbnetTextQuads(() => [dab]);
    const ocr = await createOcr(DM_CONFIG);
    const result = ocr.runOcrWorklet(page, { dewarp: true });
    ocr.dispose();
    expect(result.image).toEqual(mirrored);
    expect(result.detections).toEqual([line('dab', dab)]);
    expect(bad[0]).toEqual({ x: 0, y: 4 });
    expect(tensorTracker.liveCount()).toBe(0);
  });

  it('keeps the input frame when the dewarp is declined', async () => {
    const calls = installFakeHelpersModel(HELPERS_PATH, { grid: ZERO_GRID });
    installFakeOcrModel(PATH);
    const page = blankPage(64, 32);
    const bad = paintLine(page, 0, 4, 'bad');
    setFakeDbnetTextQuads(() => [bad]);
    const ocr = await createOcr(DM_CONFIG);
    const result = ocr.runOcrWorklet(page, { dewarp: true });
    ocr.dispose();
    expect(calls.dewarp).toBe(1);
    expect(result.image).toBe(page);
    expect(result.detections).toEqual([line('bad', bad)]);
    expect(tensorTracker.liveCount()).toBe(0);
  });

  it('rotates and then flattens, with the layout detector seeing the corrected frame', async () => {
    installFakeHelpersModel(HELPERS_PATH, { logits: SHOWS_90, grid: mirroredGrid() });
    installFakeOcrModel(PATH);
    const LAYOUT_PATH = '/models/layout.pte';
    setFakeModel(LAYOUT_PATH, { schema: fakeOcrSchema(), methods: {} });
    const { turned } = uprightAndTurned();
    const seen: ImageBuffer[] = [];
    const regionBox = { format: 'xyxy' as const, xmin: 32, ymin: 0, xmax: 64, ymax: 32 };
    setFakeLayoutRegions((input) => {
      seen.push(input);
      return [{ box: regionBox, label: 'text', confidence: 0.9 }];
    });
    const dab = paintLine(blankPage(64, 32), 8, 4, 'dab'); // in crop coordinates
    setFakeDbnetTextQuads(() => [dab]);
    const ocr = await createOcr<'text'>({
      ...DM_CONFIG,
      layout: {
        modelPath: LAYOUT_PATH,
        modelOpts: { labels: ['text'], boxFormat: 'xyxy' },
      } as unknown as NonNullable<OcrModel<'text'>['layout']>,
    });
    const result = ocr.runOcrWorklet(turned, { orientation: true, dewarp: true });
    ocr.dispose();
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ width: 64, height: 32 });
    expect(result.image).toBe(seen[0]);
    expect(result.blocks).toEqual([
      expect.objectContaining({ regionType: 'text', bbox: regionBox, text: 'dab' }),
    ]);
    expect(result.detections[0]!.quad[0]).toEqual({ x: 40, y: 4 });
    expect(tensorTracker.liveCount()).toBe(0);
  });

  describe('tables', () => {
    const LAYOUT_PATH = '/models/layout.pte';
    type Label = 'text' | 'table';
    const TABLE_CONFIG: OcrModel<Label> = {
      ...DM_CONFIG,
      layout: {
        modelPath: LAYOUT_PATH,
        modelOpts: { labels: ['text', 'table'], boxFormat: 'xyxy' },
      } as unknown as NonNullable<OcrModel<Label>['layout']>,
    };
    const tableRegion = {
      box: { format: 'xyxy' as const, xmin: 0, ymin: 0, xmax: 64, ymax: 32 },
      label: 'table' as const,
      confidence: 0.9,
    };
    // A one-row, two-cell skeleton.
    const ROW_OF_TWO = [1, 3, 3, 2, FAKE_EOS];

    it('fills the recognized structure of a table region with its lines', async () => {
      const calls = installFakeHelpersModel(HELPERS_PATH, { tokens: ROW_OF_TWO });
      installFakeOcrModel(PATH);
      setFakeModel(LAYOUT_PATH, { schema: fakeOcrSchema(), methods: {} });
      const page = blankPage(64, 32);
      const bad = paintLine(page, 0, 12, 'bad');
      const cab = paintLine(page, 40, 12, 'cab');
      setFakeLayoutRegions(() => [tableRegion]);
      setFakeDbnetTextQuads(() => [cab, bad]);
      const ocr = await createOcr<Label>(TABLE_CONFIG);
      const result = ocr.runOcrWorklet(page);
      ocr.dispose();
      expect(calls.tableEncode).toBe(1);
      expect(result.blocks).toEqual([
        {
          regionType: 'table',
          bbox: tableRegion.box,
          layoutConfidence: 0.9,
          isTable: true,
          lines: [line('bad', bad), line('cab', cab)],
          text: 'bad\ncab',
          tableHtml: '<tr><td>bad</td><td>cab</td></tr>',
        },
      ]);
      expect(tensorTracker.liveCount()).toBe(0);
    });

    it('still recognizes the structure of a table without any text', async () => {
      installFakeHelpersModel(HELPERS_PATH, { tokens: ROW_OF_TWO });
      installFakeOcrModel(PATH);
      setFakeModel(LAYOUT_PATH, { schema: fakeOcrSchema(), methods: {} });
      setFakeLayoutRegions(() => [tableRegion]);
      setFakeDbnetTextQuads(() => []);
      const ocr = await createOcr<Label>(TABLE_CONFIG);
      const result = ocr.runOcrWorklet(blankPage(64, 32));
      ocr.dispose();
      expect(result.blocks[0]).toMatchObject({
        isTable: true,
        lines: [],
        tableHtml: '<tr><td></td><td></td></tr>',
      });
    });

    it('skips the decoder with tables: false or without a table config', async () => {
      const calls = installFakeHelpersModel(HELPERS_PATH, { tokens: ROW_OF_TWO });
      installFakeOcrModel(PATH);
      setFakeModel(LAYOUT_PATH, { schema: fakeOcrSchema(), methods: {} });
      setFakeLayoutRegions(() => [tableRegion]);
      setFakeDbnetTextQuads(() => []);
      const ocr = await createOcr<Label>(TABLE_CONFIG);
      const result = ocr.runOcrWorklet(blankPage(64, 32), { tables: false });
      ocr.dispose();
      expect(result.blocks[0]).toMatchObject({ isTable: true });
      expect(result.blocks[0]).not.toHaveProperty('tableHtml');

      const noTable = await createOcr<Label>({
        ...TABLE_CONFIG,
        documentModels: { modelPath: HELPERS_PATH },
      });
      expect(noTable.runOcrWorklet(blankPage(64, 32)).blocks[0]).not.toHaveProperty('tableHtml');
      noTable.dispose();
      expect(calls.tableEncode).toBe(0);
    });
  });

  it('releases all three models on dispose', async () => {
    installFakeHelpersModel(HELPERS_PATH);
    installFakeOcrModel(PATH);
    const LAYOUT_PATH = '/models/layout.pte';
    setFakeModel(LAYOUT_PATH, { schema: fakeOcrSchema(), methods: {} });
    const ocr = await createOcr({
      ...DM_CONFIG,
      layout: {
        modelPath: LAYOUT_PATH,
        modelOpts: { labels: [], boxFormat: 'xyxy' },
      } as unknown as NonNullable<OcrModel['layout']>,
    });
    expect(modelTracker.liveCount()).toBe(3);
    ocr.dispose();
    expect(modelTracker.liveCount()).toBe(0);
    expect(tensorTracker.liveCount()).toBe(0);
  });

  it('releases the other models when the helpers cannot be loaded', async () => {
    installFakeOcrModel(PATH);
    const LAYOUT_PATH = '/models/layout.pte';
    setFakeModel(LAYOUT_PATH, { schema: fakeOcrSchema(), methods: {} });
    await expect(
      createOcr({
        ...DM_CONFIG,
        layout: {
          modelPath: LAYOUT_PATH,
          modelOpts: { labels: [], boxFormat: 'xyxy' },
        } as unknown as NonNullable<OcrModel['layout']>,
      })
    ).rejects.toEqual(codedError('LOAD_FAILED'));
    expect(modelTracker.liveCount()).toBe(0);
  });
});

/** Turns a page clockwise through the fake native op. */
function rotate(page: ImageBuffer, degrees: RotationDegrees): ImageBuffer {
  const channels = page.data.length / (page.width * page.height);
  const src = tensor('uint8', [page.height, page.width, channels], page.data);
  const out = rotateTensor(src, degrees);
  const [height, width] = out.shape as [number, number, number];
  const data = out.getData(new Uint8Array(page.data.length));
  out.dispose();
  src.dispose();
  return { ...page, data, width, height };
}

/** Moves a painted quad, as the detector sees it on a crop. */
function shift<Q extends readonly { x: number; y: number }[]>(quad: Q, dx: number, dy: number): Q {
  return quad.map((p) => ({ x: p.x + dx, y: p.y + dy })) as unknown as Q;
}
