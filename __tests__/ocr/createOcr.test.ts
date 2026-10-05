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
import { modelTracker } from '../support/rneMock';

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

  it.each([
    ['layout', { layout: { modelPath: '/models/layout.pte' } }],
    ['documentModels', { documentModels: { modelPath: '/models/helpers.pte' } }],
  ])('rejects a config with %s before loading anything', async (stage, extra) => {
    installFakeOcrModel(PATH);
    const config = { ...CONFIG, ...extra } as unknown as OcrModel;
    await expect(createOcr(config)).rejects.toEqual(codedError('INVALID_ARGUMENT'));
    await expect(createOcr(config)).rejects.toThrow(stage);
    expect(modelTracker.liveCount()).toBe(0);
  });
});
