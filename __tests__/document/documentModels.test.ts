import { tensor } from 'react-native-executorch';
import { createDocumentModels } from '../../src/document/documentModels';
import type { DocumentModelsConfig } from '../../src/document/types';
import {
  FAKE_EOS,
  FAKE_STRUCTURE_VOCAB,
  fakeHelpersSchema,
  HELPERS,
  identityGrid,
  installFakeHelpersModel,
  mirroredGrid,
} from '../support/fakeHelpersModel';
import { tensorTracker } from '../support/fakeTensor';
import { modelTracker } from '../support/rneMock';

const codedError = (code: string) => expect.objectContaining({ name: 'RnExecuTorchError', code });

const PATH = '/models/helpers.pte';
const TABLE = { structureVocab: FAKE_STRUCTURE_VOCAB, eosTokenId: FAKE_EOS, maxSteps: 20 };
const CONFIG: DocumentModelsConfig = { modelPath: PATH, table: TABLE };

/**
 * A 64x16 RGB page, a dark left half and a light right half: large enough for
 * the dewarp's sparse variance sample to see both halves.
 */
function halfPage() {
  const width = 64;
  const height = 16;
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      data.fill(x < width / 2 ? 20 : 220, (y * width + x) * 3, (y * width + x) * 3 + 3);
    }
  }
  return { tensor: tensor('uint8', [height, width, 3], data), data, width, height };
}

describe('createDocumentModels', () => {
  it('loads the model and releases everything on dispose', async () => {
    installFakeHelpersModel(PATH);
    const dm = await createDocumentModels(CONFIG);
    expect(modelTracker.liveCount()).toBe(1);
    expect(tensorTracker.liveCount()).toBeGreaterThan(0);
    dm.dispose();
    expect(modelTracker.liveCount()).toBe(0);
    expect(tensorTracker.liveCount()).toBe(0);
    dm.dispose(); // a second dispose is a no-op
  });

  it('fails with LOAD_FAILED when the model cannot be loaded', async () => {
    await expect(createDocumentModels(CONFIG)).rejects.toEqual(codedError('LOAD_FAILED'));
  });

  it('rejects a model without the table methods when a table config is given', async () => {
    installFakeHelpersModel(PATH, { schema: fakeHelpersSchema({ withoutTable: true }) });
    await expect(createDocumentModels(CONFIG)).rejects.toEqual(codedError('SCHEMA_MISMATCH'));
    expect(modelTracker.liveCount()).toBe(0);
    expect(tensorTracker.liveCount()).toBe(0);
  });

  it('rejects a table vocabulary that does not match the decoder', async () => {
    installFakeHelpersModel(PATH);
    const config = { ...CONFIG, table: { ...TABLE, structureVocab: ['sos', 'eos'] } };
    await expect(createDocumentModels(config)).rejects.toEqual(codedError('SCHEMA_MISMATCH'));
    expect(modelTracker.liveCount()).toBe(0);
  });

  it('accepts a model without the table methods when no table is wanted', async () => {
    installFakeHelpersModel(PATH, { schema: fakeHelpersSchema({ withoutTable: true }) });
    const dm = await createDocumentModels({ modelPath: PATH });
    const page = halfPage();
    expect(() => dm.recognizeTableWorklet(page.tensor, 'rgb')).toThrow(codedError('INVALID_STATE'));
    page.tensor.dispose();
    dm.dispose();
  });

  describe('orientation', () => {
    it('reads the class off the logits with a softmax confidence', async () => {
      const calls = installFakeHelpersModel(PATH, { logits: [0, 0, 3, 0] });
      const dm = await createDocumentModels(CONFIG);
      const page = halfPage();
      const orientation = dm.detectOrientationWorklet(page.tensor, 'rgb');
      expect(orientation.rotationCW).toBe(180);
      expect(orientation.confidence).toBeCloseTo(Math.exp(3) / (Math.exp(3) + 3), 6);
      expect(calls.orientation).toBe(1);
      page.tensor.dispose();
      dm.dispose();
    });

    it('reports 0 degrees for the first class', async () => {
      installFakeHelpersModel(PATH, { logits: [5, 1, 1, 1] });
      const dm = await createDocumentModels(CONFIG);
      const page = halfPage();
      expect(dm.detectOrientationWorklet(page.tensor, 'rgb').rotationCW).toBe(0);
      page.tensor.dispose();
      dm.dispose();
    });
  });

  describe('dewarp', () => {
    it('resamples the page through the predicted grid', async () => {
      installFakeHelpersModel(PATH, { grid: mirroredGrid() });
      const dm = await createDocumentModels(CONFIG);
      const page = halfPage();
      const out = dm.dewarpWorklet(page.tensor, 'rgb');
      expect(out).not.toBe(page.tensor);
      expect(out.shape).toEqual([page.height, page.width, 3]);
      const pixels = out.getData(new Uint8Array(page.data.length));
      // Mirrored: the dark half is now on the right.
      expect(pixels[0]).toBe(220);
      expect(pixels[(page.width - 1) * 3]).toBe(20);
      out.dispose();
      page.tensor.dispose();
      dm.dispose();
    });

    it('declines a warp that collapses the page and returns the input', async () => {
      // An all-zero grid samples the page center everywhere: no variance left.
      installFakeHelpersModel(PATH, { grid: new Float32Array(2 * HELPERS.gridH * HELPERS.gridW) });
      const dm = await createDocumentModels(CONFIG);
      const page = halfPage();
      const before = tensorTracker.liveCount();
      expect(dm.dewarpWorklet(page.tensor, 'rgb')).toBe(page.tensor);
      expect(tensorTracker.liveCount()).toBe(before);
      page.tensor.dispose();
      dm.dispose();
    });

    it('honours dewarpMinVarianceRatio', async () => {
      installFakeHelpersModel(PATH, { grid: new Float32Array(2 * HELPERS.gridH * HELPERS.gridW) });
      const dm = await createDocumentModels({ ...CONFIG, dewarpMinVarianceRatio: 0 });
      const page = halfPage();
      const out = dm.dewarpWorklet(page.tensor, 'rgb');
      expect(out).not.toBe(page.tensor);
      out.dispose();
      page.tensor.dispose();
      dm.dispose();
    });

    it('keeps an identity warp', async () => {
      installFakeHelpersModel(PATH, { grid: identityGrid() });
      const dm = await createDocumentModels(CONFIG);
      const page = halfPage();
      const out = dm.dewarpWorklet(page.tensor, 'rgb');
      expect(out.getData(new Uint8Array(page.data.length))).toEqual(page.data);
      out.dispose();
      page.tensor.dispose();
      dm.dispose();
    });
  });

  describe('table structure', () => {
    it('decodes the skeleton token by token, feeding each token back one-hot', async () => {
      const tokens = [1, 3, 4, 5, 6, 2, FAKE_EOS, 3];
      const calls = installFakeHelpersModel(PATH, { tokens });
      const dm = await createDocumentModels(CONFIG);
      const page = halfPage();
      expect(dm.recognizeTableWorklet(page.tensor, 'rgb')).toBe(
        '<tr><td></td><td colspan="2"></td></tr>'
      );
      expect(calls.tableEncode).toBe(1);
      // The first step gets the all-zero vector, then each emitted token.
      expect(calls.decodeInputs).toEqual([-1, 1, 3, 4, 5, 6, 2]);
      page.tensor.dispose();
      dm.dispose();
    });

    it('stops at maxSteps and skips the reserved tokens', async () => {
      installFakeHelpersModel(PATH, { tokens: [0, 1, 3, 2] });
      const dm = await createDocumentModels({ ...CONFIG, table: { ...TABLE, maxSteps: 3 } });
      const page = halfPage();
      expect(dm.recognizeTableWorklet(page.tensor, 'rgb')).toBe('<tr><td></td>');
      page.tensor.dispose();
      dm.dispose();
    });

    it('decodes a second table from a fresh state', async () => {
      const calls = installFakeHelpersModel(PATH, { tokens: [1, 2, FAKE_EOS] });
      const dm = await createDocumentModels(CONFIG);
      const page = halfPage();
      expect(dm.recognizeTableWorklet(page.tensor, 'rgb')).toBe('<tr></tr>');
      expect(dm.recognizeTableWorklet(page.tensor, 'rgb')).toBe('<tr></tr>');
      expect(calls.decodeInputs).toEqual([-1, 1, 2, -1, 1, 2]);
      page.tensor.dispose();
      dm.dispose();
    });
  });
});
