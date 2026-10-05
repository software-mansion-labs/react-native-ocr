import { tensor } from 'react-native-executorch';
import * as cv from 'react-native-executorch/cv';
import { fillRecognizerCanvas } from '../../src/ocr/recognizerCanvas';
import { fakeRnOcr } from '../support/fakeRnOcr';
import { tensorTracker } from '../support/fakeTensor';

const QUAD: cv.Quad = [
  { x: 0, y: 0 },
  { x: 4, y: 0 },
  { x: 4, y: 2 },
  { x: 0, y: 2 },
];

/** A 2x4 single-channel page: 10 in the left half, 50 in the right half. */
function page() {
  return tensor('uint8', [2, 4, 1], Uint8Array.from([10, 10, 50, 50, 10, 10, 50, 50]));
}

function run(padMode: 'constant' | 'cornerMean', canvasWidth: number, contentWidth = 4) {
  const src = page();
  const canvas = tensor('uint8', [2, canvasWidth, 1]);
  try {
    const returned = fillRecognizerCanvas(src, canvas, QUAD, {
      contentWidth,
      padMode,
      padValue: 128,
    });
    expect(returned).toBe(canvas);
    return [...canvas.getData(new Uint8Array(2 * canvasWidth))];
  } finally {
    src.dispose();
    canvas.dispose();
  }
}

describe('fillRecognizerCanvas', () => {
  it("pads with the flat value in 'constant' mode, in one native pass", () => {
    const rectify = jest.spyOn(cv, 'rectifyQuad');
    const pad = jest.spyOn(fakeRnOcr, 'padCornerMean');
    expect(run('constant', 6)).toEqual([10, 10, 50, 50, 128, 128, 10, 10, 50, 50, 128, 128]);
    expect(rectify).toHaveBeenCalledTimes(1);
    expect(rectify.mock.calls[0]![3]).toEqual({ contentWidth: 4, align: 'left', padValue: 128 });
    expect(pad).not.toHaveBeenCalled();
  });

  it("pads with the content's corner mean in 'cornerMean' mode", () => {
    const pad = jest.spyOn(fakeRnOcr, 'padCornerMean');
    // Corners are 10, 50, 10, 50: mean 30.
    expect(run('cornerMean', 6)).toEqual([10, 10, 50, 50, 30, 30, 10, 10, 50, 50, 30, 30]);
    expect(pad).toHaveBeenCalledTimes(1);
    expect(pad.mock.calls[0]!.slice(1)).toEqual([2, 4, 1, 6]);
  });

  it('skips the strip round trip when the content fills the canvas', () => {
    const pad = jest.spyOn(fakeRnOcr, 'padCornerMean');
    expect(run('cornerMean', 4)).toEqual([10, 10, 50, 50, 10, 10, 50, 50]);
    expect(pad).not.toHaveBeenCalled();
  });

  it('squeezes the content into a narrower content width', () => {
    expect(run('constant', 4, 2)).toEqual([10, 50, 128, 128, 10, 50, 128, 128]);
  });

  it('frees the strip when padding throws', () => {
    jest.spyOn(fakeRnOcr, 'padCornerMean').mockImplementation(() => {
      throw new Error('boom');
    });
    const src = page();
    const canvas = tensor('uint8', [2, 6, 1]);
    expect(() =>
      fillRecognizerCanvas(src, canvas, QUAD, {
        contentWidth: 4,
        padMode: 'cornerMean',
        padValue: 128,
      })
    ).toThrow('boom');
    expect(tensorTracker.liveCount()).toBe(2);
    src.dispose();
    canvas.dispose();
  });
});
