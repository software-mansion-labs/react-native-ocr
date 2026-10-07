import { clusterByGaps, fillTableCells } from '../../src/document/table';
import type { OcrDetection } from '../../src/ocr/types';

const line = (text: string, x: number, y: number, w = 20, h = 10): OcrDetection => ({
  text,
  confidence: 1,
  quad: [
    { x, y },
    { x: x + w, y },
    { x: x + w, y: y + h },
    { x, y: y + h },
  ],
});

describe('clusterByGaps', () => {
  it('splits at the widest gaps and returns the group means in order', () => {
    expect(clusterByGaps([10, 12, 50, 52, 54, 90], 3)).toEqual([11, 52, 90]);
  });

  it('returns the sorted values themselves when there are at most k of them', () => {
    expect(clusterByGaps([30, 10], 2)).toEqual([10, 30]);
    expect(clusterByGaps([30, 10], 5)).toEqual([10, 30]);
  });

  it('returns one mean for k = 1 and nothing for k < 1', () => {
    expect(clusterByGaps([1, 2, 3], 1)).toEqual([2]);
    expect(clusterByGaps([1, 2, 3], 0)).toEqual([]);
  });
});

describe('fillTableCells', () => {
  const skeleton = '<tr><td></td><td></td></tr><tr><td></td><td></td></tr>';

  it('routes each line to the cell of its nearest row and column', () => {
    const lines = [line('d', 100, 50), line('a', 0, 0), line('c', 0, 50), line('b', 100, 0)];
    expect(fillTableCells(skeleton, lines)).toBe(
      '<tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr>'
    );
  });

  it('joins the lines that land in one cell, in their given order', () => {
    const lines = [line('one', 0, 0), line('two', 2, 4), line('x', 100, 50)];
    expect(fillTableCells(skeleton, lines)).toBe(
      '<tr><td>one two</td><td></td></tr><tr><td></td><td>x</td></tr>'
    );
  });

  it('counts a colspan as several columns and skips the spanned cells', () => {
    const html = '<tr><td colspan="2"></td></tr><tr><td></td><td></td></tr>';
    const lines = [line('head', 0, 0), line('l', 0, 50), line('r', 100, 50)];
    expect(fillTableCells(html, lines)).toBe(
      '<tr><td colspan="2">head</td></tr><tr><td>l</td><td>r</td></tr>'
    );
  });

  it('keeps the columns under a rowspan occupied on the rows it covers', () => {
    const html =
      '<tr><td rowspan="2"></td><td></td></tr><tr><td></td></tr><tr><td></td><td></td></tr>';
    const lines = [
      line('tall', 0, 0),
      line('r1', 100, 0),
      line('r2', 100, 50),
      line('l3', 0, 100),
      line('r3', 100, 100),
    ];
    expect(fillTableCells(html, lines)).toBe(
      '<tr><td rowspan="2">tall</td><td>r1</td></tr><tr><td>r2</td></tr><tr><td>l3</td><td>r3</td></tr>'
    );
  });

  it('fills the cells in document order when there is no grid to place by', () => {
    expect(fillTableCells(skeleton, [])).toBe(skeleton);
    expect(fillTableCells('<td></td><td></td>', [line('a', 0, 0), line('b', 9, 9)])).toBe(
      '<td>a</td><td>b</td>'
    );
  });

  it('merges every line into a single cell', () => {
    expect(fillTableCells('<tr><td></td></tr>', [line('a', 0, 0), line('b', 100, 0)])).toBe(
      '<tr><td>a b</td></tr>'
    );
  });
});
