import type { cv } from 'react-native-executorch';
import type { CraftBox } from '../../src/native/ops';
import {
  boxToQuad,
  groupBoxes,
  groupVerticalStacks,
  orderByReadingOrder,
} from '../../src/ocr/geometry';

const box = (x0: number, y0: number, x1: number, y1: number, angle = 0): CraftBox => ({
  x0,
  y0,
  x1,
  y1,
  angle,
});

const rect = (x0: number, y0: number, x1: number, y1: number): cv.Quad => [
  { x: x0, y: y0 },
  { x: x1, y: y0 },
  { x: x1, y: y1 },
  { x: x0, y: y1 },
];

const byTop = (a: CraftBox, b: CraftBox) => a.y0 - b.y0;

describe('groupBoxes', () => {
  it('merges word boxes on one baseline into a single line', () => {
    const words = [box(0, 0, 60, 20), box(70, 0, 130, 20), box(140, 0, 200, 20)];
    const lines = groupBoxes(words);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({ x0: 0, y0: 0, x1: 200, y1: 20 });
    expect(lines[0]!.angle).toBeCloseTo(0);
  });

  it('keeps separate baselines as separate lines', () => {
    const lines = groupBoxes([
      box(0, 0, 60, 20),
      box(70, 0, 130, 20),
      box(0, 50, 60, 70),
      box(70, 50, 130, 70),
    ]).sort(byTop);
    expect(lines).toHaveLength(2);
    expect(lines[0]).toMatchObject({ x0: 0, y0: 0, x1: 130, y1: 20 });
    expect(lines[1]).toMatchObject({ x0: 0, y0: 50, x1: 130, y1: 70 });
  });

  it('does not bridge a gap wider than twice the candidate height', () => {
    const lines = groupBoxes([box(0, 0, 60, 20), box(120, 0, 180, 20)]);
    expect(lines).toHaveLength(2);
  });

  it('does not merge boxes of very different heights', () => {
    // Candidate height 20; the seed is 80 tall: |80 - 20| = 60 >= 2 * 20.
    const lines = groupBoxes([box(0, 0, 100, 80), box(105, 30, 165, 50)]);
    expect(lines).toHaveLength(2);
  });

  it('drops lines too thin or too short to read', () => {
    expect(groupBoxes([box(0, 0, 100, 10)])).toEqual([]); // short side 10 <= 15
    expect(groupBoxes([box(0, 0, 25, 20)])).toEqual([]); // long side 25 <= 30
    expect(groupBoxes([box(0, 0, 40, 16)])).toHaveLength(1);
  });

  it('marks a tall box as vertical with angle -90', () => {
    const [line] = groupBoxes([box(0, 0, 20, 100)]);
    expect(line).toMatchObject({ x0: 0, y0: 0, x1: 20, y1: 100, angle: -90 });
  });

  it('stacks square glyphs under each other into one tall line', () => {
    // Four 40x50 glyphs, 10 px apart, centred on x = 100 (angle 90: an
    // upright rectangle as minAreaRect reports it).
    const glyphs = [0, 1, 2, 3].map((i) => box(80, 100 + i * 60, 120, 150 + i * 60, 90));
    const lines = groupBoxes(glyphs);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({ x0: 80, y0: 100, x1: 120, y1: 330, angle: -90 });
  });

  it('does not pull a wide word above a glyph column into the stack', () => {
    // Glyphs (50x70) larger than the heading word (60x30) above them, all
    // centred on x = 75: the column seeds first and its vertical line fit
    // would otherwise absorb the word, which sits within the gap threshold.
    const word = box(45, 0, 105, 30, 90);
    const glyphs = [0, 1, 2, 3].map((i) => box(50, 60 + i * 80, 100, 130 + i * 80, 90));
    const lines = groupBoxes([word, ...glyphs]).sort(byTop);
    expect(lines).toHaveLength(2);
    expect(lines[0]).toMatchObject({ x0: 45, y0: 0, x1: 105, y1: 30 });
    expect(lines[1]).toMatchObject({ x0: 50, y0: 60, x1: 100, y1: 370 });
  });

  it('does not mutate its input', () => {
    const input = [box(0, 0, 60, 20, 12)];
    const snapshot = JSON.stringify(input);
    groupBoxes(input);
    expect(JSON.stringify(input)).toBe(snapshot);
  });
});

describe('boxToQuad', () => {
  it('returns the corners TL, TR, BR, BL for an upright box', () => {
    expect(boxToQuad(box(1, 2, 5, 4))).toEqual(rect(1, 2, 5, 4));
  });

  it('rotates the corners about the center by the box angle', () => {
    const quad = boxToQuad(box(0, 0, 40, 20, 30));
    const c = Math.cos(Math.PI / 6);
    const s = Math.sin(Math.PI / 6);
    // TL (0,0) relative to center (20,10) is (-20,-10).
    expect(quad[0].x).toBeCloseTo(-20 * c + 10 * s + 20);
    expect(quad[0].y).toBeCloseTo(-20 * s - 10 * c + 10);
    // Rotation preserves side lengths and the center.
    expect(Math.hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y)).toBeCloseTo(40);
    expect(Math.hypot(quad[3].x - quad[0].x, quad[3].y - quad[0].y)).toBeCloseTo(20);
    const cx = (quad[0].x + quad[2].x) / 2;
    const cy = (quad[0].y + quad[2].y) / 2;
    expect(cx).toBeCloseTo(20);
    expect(cy).toBeCloseTo(10);
  });

  it('keeps near-vertical boxes upright', () => {
    expect(boxToQuad(box(0, 0, 20, 100, -90))).toEqual(rect(0, 0, 20, 100));
    expect(boxToQuad(box(0, 0, 20, 100, 60))).toEqual(rect(0, 0, 20, 100));
  });
});

describe('orderByReadingOrder', () => {
  const item = (name: string, x0: number, y0: number, x1: number, y1: number) => ({
    name,
    quad: rect(x0, y0, x1, y1),
  });
  const names = (items: { name: string }[]) => items.map((it) => it.name);

  // A paper-like page: title and abstract across both columns, two body
  // columns with headings, a centered footer. Heights are about 25 px.
  const page = [
    item('footer', 317, 1038, 483, 1058),
    item('right text', 416, 368, 674, 532),
    item('2 Method', 417, 327, 524, 354),
    item('left text 2', 55, 677, 325, 785),
    item('abstract', 56, 206, 659, 288),
    item('3 Results', 56, 637, 162, 663),
    item('left text 1', 55, 367, 327, 589),
    item('1 Introduction', 58, 328, 209, 355),
    item('caption', 417, 772, 686, 795),
    item('authors', 236, 107, 564, 132),
    item('Abstract', 57, 167, 153, 194),
    item('title', 129, 46, 669, 90),
  ];

  it('returns the input array itself for one item or none', () => {
    const empty: { quad: cv.Quad }[] = [];
    expect(orderByReadingOrder(empty)).toBe(empty);
    const single = [item('a', 0, 0, 10, 10)];
    expect(orderByReadingOrder(single)).toBe(single);
  });

  it('reads the columns under a spanning title one after the other', () => {
    expect(names(orderByReadingOrder(page))).toEqual([
      'title',
      'authors',
      'Abstract',
      'abstract',
      '1 Introduction',
      'left text 1',
      '3 Results',
      'left text 2',
      '2 Method',
      'right text',
      'caption',
      'footer',
    ]);
  });

  it('keeps a single column with short headings in top-to-bottom order', () => {
    // The short headings and the centered footer leave an x gap between them
    // that is not a gutter: the paragraphs crossing it are the bands.
    const items = [
      item('footer', 300, 500, 400, 520),
      item('para 2', 0, 300, 700, 450),
      item('heading 2', 0, 260, 90, 285),
      item('para 1', 0, 60, 700, 220),
      item('heading 1', 0, 20, 100, 45),
    ];
    expect(names(orderByReadingOrder(items))).toEqual([
      'heading 1',
      'para 1',
      'heading 2',
      'para 2',
      'footer',
    ]);
  });

  it('separates a footer sitting in the gutter from the columns above it', () => {
    const items = [
      item('footer', 90, 300, 210, 320),
      item('R1', 200, 0, 300, 100),
      item('L2', 0, 120, 100, 220),
      item('R2', 200, 120, 300, 220),
      item('L1', 0, 0, 100, 100),
    ];
    expect(names(orderByReadingOrder(items))).toEqual(['L1', 'L2', 'R1', 'R2', 'footer']);
  });

  it('reads the left column before the right one', () => {
    const items = [
      item('R1', 200, 0, 300, 20),
      item('L2', 0, 30, 100, 50),
      item('R2', 200, 30, 300, 50),
      item('L1', 0, 0, 100, 20),
    ];
    expect(names(orderByReadingOrder(items))).toEqual(['L1', 'L2', 'R1', 'R2']);
  });

  it('keeps a small box on the line of a tall neighbor it overlaps', () => {
    const items = [
      item('below', 0, 45, 100, 60),
      item('small', 55, 10, 100, 25),
      item('tall', 0, 0, 50, 40),
    ];
    expect(names(orderByReadingOrder(items))).toEqual(['tall', 'small', 'below']);
  });

  it('does not treat the left margin as a gutter', () => {
    const items = [item('b', 500, 30, 600, 50), item('a', 500, 0, 600, 20)];
    expect(names(orderByReadingOrder(items))).toEqual(['a', 'b']);
  });

  it('does not split a column at its line spacing', () => {
    // Lines 20 px tall with 10 px between them, the lower ones shorter.
    const items = [
      item('c', 0, 60, 60, 80),
      item('a', 0, 0, 200, 20),
      item('b', 0, 30, 200, 50),
      item('d', 0, 90, 60, 110),
    ];
    expect(names(orderByReadingOrder(items))).toEqual(['a', 'b', 'c', 'd']);
  });

  it('reads words within a line left to right once no gap is wide enough', () => {
    const items = [
      item('C1', 110, 0, 160, 20),
      item('B2', 55, 30, 105, 50),
      item('A1', 0, 0, 50, 20),
      item('A2', 0, 30, 50, 50),
      item('B1', 55, 0, 105, 20),
      item('C2', 110, 30, 160, 50),
    ];
    expect(names(orderByReadingOrder(items))).toEqual(['A1', 'B1', 'C1', 'A2', 'B2', 'C2']);
  });
});

describe('groupVerticalStacks', () => {
  it('clusters aligned square glyphs into one top-to-bottom stack', () => {
    const g1 = rect(0, 0, 20, 20);
    const g2 = rect(1, 25, 21, 45);
    const g3 = rect(0, 50, 20, 70);
    const { stacks, singles } = groupVerticalStacks([g3, g1, g2]);
    expect(stacks).toEqual([[g1, g2, g3]]);
    expect(singles).toEqual([]);
  });

  it('passes wide lines and lone glyphs through as singles', () => {
    const wide = rect(0, 0, 100, 20);
    const lone = rect(300, 300, 320, 320);
    const g1 = rect(0, 50, 20, 70);
    const g2 = rect(0, 75, 20, 95);
    const { stacks, singles } = groupVerticalStacks([wide, g1, lone, g2]);
    expect(stacks).toEqual([[g1, g2]]);
    expect(singles).toEqual([wide, lone]);
  });

  it('breaks a stack at a gap wider than 2.5 glyph heights', () => {
    const g1 = rect(0, 0, 20, 20);
    const g2 = rect(0, 25, 20, 45);
    const g3 = rect(0, 120, 20, 140);
    const g4 = rect(0, 145, 20, 165);
    const { stacks, singles } = groupVerticalStacks([g1, g2, g3, g4]);
    expect(stacks).toEqual([
      [g1, g2],
      [g3, g4],
    ]);
    expect(singles).toEqual([]);
  });

  it('keeps side-by-side columns apart', () => {
    const a1 = rect(0, 0, 20, 20);
    const a2 = rect(0, 25, 20, 45);
    const b1 = rect(40, 0, 60, 20);
    const b2 = rect(40, 25, 60, 45);
    const { stacks } = groupVerticalStacks([a1, b1, a2, b2]);
    expect(stacks).toEqual([
      [a1, a2],
      [b1, b2],
    ]);
  });
});
