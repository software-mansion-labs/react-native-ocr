import { IMAGENET_NORM } from 'react-native-executorch';
import * as cv from 'react-native-executorch/cv';
import { ConstantDim } from 'react-native-executorch/schema';
import { createOcr } from '../../src/ocr/createOcr';
import { makeDbnetExtractBoxes } from '../../src/ocr/detectors';
import type { OcrModel } from '../../src/ocr/types';
import { setFakeDbnetTextQuads } from '../support/cvMock';
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

  it('rejects a config with documentModels before loading anything', async () => {
    installFakeOcrModel(PATH);
    const config = { ...CONFIG, documentModels: { modelPath: '/models/helpers.pte' } };
    await expect(createOcr(config)).rejects.toEqual(codedError('INVALID_ARGUMENT'));
    await expect(createOcr(config)).rejects.toThrow('documentModels');
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

/** Moves a painted quad, as the detector sees it on a crop. */
function shift<Q extends readonly { x: number; y: number }[]>(quad: Q, dx: number, dy: number): Q {
  return quad.map((p) => ({ x: p.x + dx, y: p.y + dy })) as unknown as Q;
}
