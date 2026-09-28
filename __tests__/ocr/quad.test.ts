import type { cv } from 'react-native-executorch';
import { boundingQuadOf, quadFromBounds, splitTallQuad, splitWideQuad } from '../../src/ocr/quad';

const rect = (x0: number, y0: number, x1: number, y1: number): cv.Quad => [
  { x: x0, y: y0 },
  { x: x1, y: y0 },
  { x: x1, y: y1 },
  { x: x0, y: y1 },
];

describe('quadFromBounds', () => {
  it('lists the corners TL, TR, BR, BL', () => {
    expect(quadFromBounds({ format: 'xyxy', xmin: 1, ymin: 2, xmax: 5, ymax: 7 })).toEqual(
      rect(1, 2, 5, 7)
    );
  });
});

describe('splitWideQuad', () => {
  it('returns the quad itself for one part or fewer', () => {
    const q = rect(0, 0, 90, 30);
    expect(splitWideQuad(q, 1)).toEqual([q]);
    expect(splitWideQuad(q, 0)).toEqual([q]);
  });

  it('splits into equal abutting segments left to right', () => {
    const parts = splitWideQuad(rect(0, 0, 90, 30), 3);
    expect(parts).toEqual([rect(0, 0, 30, 30), rect(30, 0, 60, 30), rect(60, 0, 90, 30)]);
    expect(parts[0]![1]).toEqual(parts[1]![0]);
    expect(parts[0]![2]).toEqual(parts[1]![3]);
  });

  it('follows a skewed quad along its top and bottom edges', () => {
    const skewed: cv.Quad = [
      { x: 0, y: 0 },
      { x: 100, y: 10 },
      { x: 100, y: 40 },
      { x: 0, y: 30 },
    ];
    const [left, right] = splitWideQuad(skewed, 2);
    expect(left).toEqual([
      { x: 0, y: 0 },
      { x: 50, y: 5 },
      { x: 50, y: 35 },
      { x: 0, y: 30 },
    ]);
    expect(right).toEqual([
      { x: 50, y: 5 },
      { x: 100, y: 10 },
      { x: 100, y: 40 },
      { x: 50, y: 35 },
    ]);
  });
});

describe('splitTallQuad', () => {
  it('returns the quad itself for one part or fewer', () => {
    const q = rect(0, 0, 30, 90);
    expect(splitTallQuad(q, 1)).toEqual([q]);
  });

  it('splits into equal abutting bands top to bottom', () => {
    const parts = splitTallQuad(rect(0, 0, 30, 90), 3);
    expect(parts).toEqual([rect(0, 0, 30, 30), rect(0, 30, 30, 60), rect(0, 60, 30, 90)]);
    expect(parts[0]![3]).toEqual(parts[1]![0]);
    expect(parts[0]![2]).toEqual(parts[1]![1]);
  });
});

describe('boundingQuadOf', () => {
  it('encloses every corner of every quad', () => {
    expect(boundingQuadOf([rect(10, 10, 20, 20), rect(0, 5, 15, 30)])).toEqual(rect(0, 5, 20, 30));
  });

  it('returns a zero quad for no quads', () => {
    expect(boundingQuadOf([])).toEqual(rect(0, 0, 0, 0));
  });
});
