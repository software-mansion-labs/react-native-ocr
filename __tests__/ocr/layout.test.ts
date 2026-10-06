import type { ObjectDetection } from 'react-native-executorch';
import {
  coveredFraction,
  DEFAULT_VISUAL_LABELS,
  makeBlock,
  mergeContainedRegions,
  REGION_CONTAINMENT,
} from '../../src/ocr/layout';
import { quadFromBounds } from '../../src/ocr/quad';

const box = (xmin: number, ymin: number, xmax: number, ymax: number) => ({
  format: 'xyxy' as const,
  xmin,
  ymin,
  xmax,
  ymax,
});

const region = (
  label: string,
  bounds: ReturnType<typeof box>,
  confidence = 0.9
): ObjectDetection<'xyxy', string> => ({ box: bounds, label, confidence });

describe('coveredFraction', () => {
  it('is the intersection over the inner box', () => {
    expect(coveredFraction(box(0, 0, 10, 10), box(0, 0, 10, 10))).toBe(1);
    expect(coveredFraction(box(0, 0, 10, 10), box(5, 0, 20, 10))).toBe(0.5);
    expect(coveredFraction(box(0, 0, 10, 10), box(5, 5, 20, 20))).toBe(0.25);
    expect(coveredFraction(box(2, 2, 4, 4), box(0, 0, 10, 10))).toBe(1);
  });

  it('is 0 for disjoint, touching and empty boxes', () => {
    expect(coveredFraction(box(0, 0, 10, 10), box(10, 0, 20, 10))).toBe(0);
    expect(coveredFraction(box(0, 0, 10, 10), box(20, 20, 30, 30))).toBe(0);
    expect(coveredFraction(box(5, 5, 5, 5), box(0, 0, 10, 10))).toBe(0);
  });
});

describe('mergeContainedRegions', () => {
  const paragraph = region('text', box(0, 0, 100, 100));
  const line = region('text', box(10, 10, 90, 20));

  it('drops a region covered by a larger one and keeps the original order', () => {
    const kept = mergeContainedRegions([line, paragraph], DEFAULT_VISUAL_LABELS);
    expect(kept).toEqual([paragraph]);
    const other = region('text', box(200, 0, 300, 50));
    expect(mergeContainedRegions([other, line, paragraph], DEFAULT_VISUAL_LABELS)).toEqual([
      other,
      paragraph,
    ]);
  });

  it('keeps a region that is not covered enough', () => {
    const straddling = region('text', box(50, 50, 150, 70)); // half inside
    expect(mergeContainedRegions([paragraph, straddling], DEFAULT_VISUAL_LABELS)).toEqual([
      paragraph,
      straddling,
    ]);
    // Exactly at the threshold counts as covered.
    const w = 100 / REGION_CONTAINMENT;
    const atThreshold = region('text', box(0, 10, w, 20));
    expect(mergeContainedRegions([paragraph, atThreshold], DEFAULT_VISUAL_LABELS)).toEqual([
      paragraph,
    ]);
  });

  it('never lets a visual region swallow another one', () => {
    const figure = region('image', box(0, 0, 100, 100));
    const caption = region('figure_title', box(10, 80, 90, 95));
    expect(mergeContainedRegions([figure, caption], DEFAULT_VISUAL_LABELS)).toEqual([
      figure,
      caption,
    ]);
    // Unless the caller says the label is not visual.
    expect(mergeContainedRegions([figure, caption], [])).toEqual([figure]);
  });

  it('still drops a visual region inside a text region', () => {
    const inset = region('seal', box(70, 70, 90, 90));
    expect(mergeContainedRegions([paragraph, inset], DEFAULT_VISUAL_LABELS)).toEqual([paragraph]);
  });

  it('does not let a dropped region drop others', () => {
    // `line` is dropped by `paragraph`; `word` sits inside `line` but is
    // judged against `paragraph` only, which it also lies in.
    const word = region('text', box(20, 12, 40, 18));
    expect(mergeContainedRegions([word, line, paragraph], DEFAULT_VISUAL_LABELS)).toEqual([
      paragraph,
    ]);
    // `shifted` is 89% inside `paragraph` and dropped; `outside` lies inside
    // `shifted` but past `paragraph`'s edge, so it survives.
    const shifted = region('text', box(20, 0, 110, 100));
    const outside = region('text', box(102, 10, 108, 20));
    expect(mergeContainedRegions([outside, shifted, paragraph], DEFAULT_VISUAL_LABELS)).toEqual([
      outside,
      paragraph,
    ]);
  });

  it('handles the empty list', () => {
    expect(mergeContainedRegions([], DEFAULT_VISUAL_LABELS)).toEqual([]);
  });
});

describe('makeBlock', () => {
  it('orders the lines and joins their texts', () => {
    const lower = { text: 'second', confidence: 0.5, quad: quadFromBounds(box(0, 20, 50, 30)) };
    const upper = { text: 'first', confidence: 0.9, quad: quadFromBounds(box(0, 0, 50, 10)) };
    const block = makeBlock<string>('text', box(0, 0, 50, 30), 0.8, [lower, upper], false);
    expect(block).toEqual({
      regionType: 'text',
      bbox: box(0, 0, 50, 30),
      layoutConfidence: 0.8,
      isTable: false,
      lines: [upper, lower],
      text: 'first\nsecond',
    });
  });

  it('builds an empty table block', () => {
    expect(makeBlock<string>('table', box(0, 0, 50, 30), 0.7, [], true)).toMatchObject({
      isTable: true,
      lines: [],
      text: '',
    });
  });
});
