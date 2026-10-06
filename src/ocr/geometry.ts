/**
 * Box geometry for the OCR pipeline: CRAFT component grouping and de-skew,
 * reading order, and vertical-stack clustering. Pure functions over plain
 * numbers, no tensors and no model. Worklet source order matters: a callee is
 * defined above its callers.
 */
import { boundingBoxOfPoints, type Point, type Quad } from 'react-native-executorch/cv';
import type { CraftBox } from '../native/ops';

/**
 * An axis-aligned box (`x0, y0` = min corner, `x1, y1` = max corner) plus the
 * rotation angle in degrees carried from the detector's rotated rectangle.
 */
type Box = CraftBox;

// ─── CRAFT box grouping ──────────────────────────────────────────────────────

// Empirically tuned CRAFT grouping constants (detector-input pixels). Tune as a
// group, not one by one.
const CENTER_THRESHOLD = 0.5; // line-fit lateral tolerance (x candidate min side)
const DISTANCE_THRESHOLD = 2.0; // max inter-box gap to merge a line (x candidate height)
const HEIGHT_THRESHOLD = 2.0; // max height mismatch to merge (x candidate height)
const MIN_SHORT_SIDE = 15; // discard merged lines thinner than this (px)
const MIN_LONG_SIDE = 30; // discard merged lines shorter than this (px)
const VERTICAL_THRESHOLD = 20; // |dx| of short-side midpoints below this => vertical
// A box wider than this multiple of its height is a horizontal word, never a
// stacked glyph: a vertical line fit does not absorb it (so a heading sitting
// right above a glyph column stays its own line), and the stack clustering
// below leaves it out. Narrow tall boxes join horizontal lines freely, since
// `l` and `I` are ordinary glyphs.
const STACK_GLYPH_ASPECT = 1.6;

const boxWidth = (b: Box): number => {
  'worklet';
  return b.x1 - b.x0;
};
const boxHeight = (b: Box): number => {
  'worklet';
  return b.y1 - b.y0;
};
const minSide = (b: Box): number => {
  'worklet';
  return Math.min(boxWidth(b), boxHeight(b));
};
const maxSide = (b: Box): number => {
  'worklet';
  return Math.max(boxWidth(b), boxHeight(b));
};
const center = (b: Box): Point => {
  'worklet';
  return { x: (b.x0 + b.x1) / 2, y: (b.y0 + b.y1) / 2 };
};
const dist = (a: Point, b: Point): number => {
  'worklet';
  return Math.hypot(b.x - a.x, b.y - a.y);
};

function corners(b: Box): [Point, Point, Point, Point] {
  'worklet';
  return [
    { x: b.x0, y: b.y0 },
    { x: b.x1, y: b.y0 },
    { x: b.x1, y: b.y1 },
    { x: b.x0, y: b.y1 },
  ];
}

function rotateAround(p: Point, ctr: Point, rad: number): Point {
  'worklet';
  const tx = p.x - ctr.x;
  const ty = p.y - ctr.y;
  return {
    x: tx * Math.cos(rad) - ty * Math.sin(rad) + ctr.x,
    y: tx * Math.sin(rad) + ty * Math.cos(rad) + ctr.y,
  };
}

// Fit a line to the two shortest sides' midpoints. The least-squares fit over
// two points is just the line through them (cv::fitLine reduced to slope form).
function fitLineToShortestSides(b: Box): {
  slope: number;
  intercept: number;
  isVertical: boolean;
} {
  'worklet';
  const pts = corners(b);
  const sides = pts.map((p, i) => {
    const q = pts[(i + 1) % 4]!;
    return { len: dist(p, q), mid: { x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 } };
  });
  sides.sort((a, c) => a.len - c.len);
  let m1 = sides[0]!.mid;
  let m2 = sides[1]!.mid;
  const isVertical = Math.abs(m2.x - m1.x) < VERTICAL_THRESHOLD;
  if (isVertical) {
    m1 = { x: m1.y, y: m1.x };
    m2 = { x: m2.y, y: m2.x };
  }
  // Coincident midpoints (degenerate box) leave dx = 0; fall back to a flat line
  // rather than propagating Infinity/NaN into every candidate distance.
  const dx = m2.x - m1.x;
  const slope = Math.abs(dx) < 1e-6 ? 0 : (m2.y - m1.y) / dx;
  return { slope, intercept: m1.y - slope * m1.x, isVertical };
}

// Axis-aligned bounds of `b` after rotating its corners by `angleDeg`.
function rotateBox(b: Box, angleDeg: number): Box {
  'worklet';
  const ctr = center(b);
  const rad = (angleDeg * Math.PI) / 180;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of corners(b)) {
    const r = rotateAround(p, ctr, rad);
    minX = Math.min(minX, r.x);
    minY = Math.min(minY, r.y);
    maxX = Math.max(maxX, r.x);
    maxY = Math.max(maxY, r.y);
  }
  return { x0: minX, y0: minY, x1: maxX, y1: maxY, angle: b.angle };
}

function minDistanceBetween(a: Box, b: Box): number {
  'worklet';
  let md = Infinity;
  for (const c1 of corners(a)) {
    for (const c2 of corners(b)) {
      md = Math.min(md, dist(c1, c2));
    }
  }
  return md;
}

// Every not-yet-consumed box whose center falls within the line's lateral
// tolerance, sorted nearest-center-first (ties broken by original index, the
// same order the old nearest-first scan produced). The caller walks this list
// once per line state instead of re-scanning per rejected box.
function findLineCandidates(
  boxes: readonly Box[],
  consumed: readonly boolean[],
  current: Box,
  isVertical: boolean,
  slope: number,
  intercept: number
): { index: number; height: number; dist: number }[] {
  'worklet';
  const cc = center(current);
  const out: { index: number; height: number; dist: number }[] = [];
  for (let i = 0; i < boxes.length; i++) {
    if (consumed[i]) {
      continue;
    }
    const pc = center(boxes[i]!);
    const h = minSide(boxes[i]!);
    const lineDistance = isVertical
      ? Math.abs(pc.x - (slope * pc.y + intercept))
      : Math.abs(pc.y - (slope * pc.x + intercept));
    if (lineDistance < h * CENTER_THRESHOLD) {
      out.push({ index: i, height: h, dist: dist(cc, pc) });
    }
  }
  out.sort((a, b) => a.dist - b.dist || a.index - b.index);
  return out;
}

const mergeBoxes = (a: Box, b: Box): Box => {
  'worklet';
  return {
    x0: Math.min(a.x0, b.x0),
    y0: Math.min(a.y0, b.y0),
    x1: Math.max(a.x1, b.x1),
    y1: Math.max(a.y1, b.y1),
    angle: a.angle,
  };
};

/**
 * Merges CRAFT component boxes into reading lines: greedily take the largest
 * box, fit a line through its short sides, absorb the nearest aligned box of
 * similar height, repeat; drop lines too small to read. Reading order is
 * derived later by {@link orderByReadingOrder}.
 *
 * `consumed[]` marks boxes already pulled into a line; candidates are gathered
 * once per line state and re-gathered only after a successful merge refits it.
 * @param input The detector's component boxes, in any order.
 * @returns One box per line, carrying the fitted line angle.
 */
export function groupBoxes(input: readonly Box[]): Box[] {
  'worklet';
  const boxes = [...input].sort((a, b) => maxSide(b) - maxSide(a));
  const consumed = new Array<boolean>(boxes.length).fill(false);
  const merged: Box[] = [];
  for (let seed = 0; seed < boxes.length; seed++) {
    if (consumed[seed]) {
      continue;
    }
    consumed[seed] = true;
    let current = boxes[seed]!;
    const normalizedAngle = current.angle > 45 ? current.angle - 90 : current.angle;
    let lineAngle = 0;
    for (;;) {
      const { slope, intercept, isVertical } = fitLineToShortestSides(current);
      lineAngle = isVertical ? -90 : (Math.atan(slope) * 180) / Math.PI;
      const candidates = findLineCandidates(boxes, consumed, current, isVertical, slope, intercept);
      let mergedOne = false;
      for (const candidateEntry of candidates) {
        let candidate = boxes[candidateEntry.index]!;
        if (isVertical && boxWidth(candidate) > boxHeight(candidate) * STACK_GLYPH_ASPECT) {
          continue;
        }
        // Only re-orient a candidate the detector emitted (near-)axis-aligned
        // (angle ~ 0 or 90, within the epsilon below) onto the seed's line; a
        // box with any real skew already carries its own angle and is left as is.
        if (
          (Math.abs(candidate.angle - 90) < 1e-3 && !isVertical) ||
          (Math.abs(candidate.angle) < 1e-3 && isVertical)
        ) {
          candidate = rotateBox(candidate, normalizedAngle);
        }
        const gap = minDistanceBetween(candidate, current);
        const currentHeight = minSide(current);
        if (
          gap < DISTANCE_THRESHOLD * candidateEntry.height &&
          Math.abs(currentHeight - candidateEntry.height) < candidateEntry.height * HEIGHT_THRESHOLD
        ) {
          current = mergeBoxes(current, candidate);
          consumed[candidateEntry.index] = true;
          mergedOne = true;
          break; // line grew: refit and re-gather
        }
      }
      if (!mergedOne) {
        break;
      }
    }
    // Push a fresh box rather than writing `angle` through to `current`: for an
    // unmerged seed `current` is still the caller's input element.
    merged.push({ ...current, angle: lineAngle });
  }
  return merged.filter((b) => minSide(b) > MIN_SHORT_SIDE && maxSide(b) > MIN_LONG_SIDE);
}

/**
 * De-skews a box into an oriented quad by rotating its corners about its
 * center by the box angle. Near-vertical boxes (|angle| > 45 degrees) stay
 * upright rather than being laid flat, so a glyph stack remains a tall quad
 * for the vertical reader.
 * @param b The box to convert.
 * @returns The corners ordered TL, TR, BR, BL.
 */
export function boxToQuad(b: Box): Quad {
  'worklet';
  const boxCenter = center(b);
  const rad = Math.abs(b.angle) > 45 ? 0 : (b.angle * Math.PI) / 180;
  const [tl, tr, br, bl] = corners(b);
  return [
    rotateAround(tl, boxCenter, rad),
    rotateAround(tr, boxCenter, rad),
    rotateAround(br, boxCenter, rad),
    rotateAround(bl, boxCenter, rad),
  ];
}

// ─── Reading order and vertical grouping ─────────────────────────────────────

// A gutter must be at least this fraction of the content width to split
// columns; two boxes share a line when their vertical extents overlap by at
// least this fraction of the shorter box's height.
const COLUMN_GAP_FRACTION = 0.06;
const LINE_OVERLAP_FRACTION = 0.3;
// A blank band must be at least this many times the smallest item height tall
// to split a page into a top and a bottom part (`orderByXyCut`): one blank
// line's worth, more than the gap between a heading and its paragraph.
const ROW_GAP_HEIGHTS = 1;
// Items at least this fraction of the content width are left out when looking
// for a gutter that spanning items hide (`orderByXyCut`).
const WIDE_FRACTION = 0.5;

type Bounds = {
  readonly xmin: number;
  readonly ymin: number;
  readonly xmax: number;
  readonly ymax: number;
};

// Sweeps the boxes' extents along one axis and returns the midpoint of the
// widest gap that no box covers and that is at least `minGap` wide, or `null`.
function widestGap(
  boxes: readonly Bounds[],
  indices: readonly number[],
  axis: 'x' | 'y',
  minGap: number
): { at: number; width: number } | null {
  'worklet';
  const edges: { at: number; delta: number }[] = [];
  for (const i of indices) {
    const box = boxes[i]!;
    edges.push({ at: axis === 'x' ? box.xmin : box.ymin, delta: 1 });
    edges.push({ at: axis === 'x' ? box.xmax : box.ymax, delta: -1 });
  }
  // Starts before ends at equal positions, so touching boxes leave no gap.
  edges.sort((a, b) => a.at - b.at || b.delta - a.delta);
  let best: { at: number; width: number } | null = null;
  let coverage = 0;
  // The margin before the first box is not a gap: the sweep starts on it.
  let gapStart = edges.length > 0 ? edges[0]!.at : 0;
  for (const edge of edges) {
    const before = coverage;
    coverage += edge.delta;
    if (before > 0 && coverage === 0) {
      gapStart = edge.at;
    } else if (before === 0 && coverage > 0) {
      const width = edge.at - gapStart;
      if (width >= minGap && (best === null || width > best.width)) {
        best = { at: (gapStart + edge.at) / 2, width };
      }
    }
  }
  return best;
}

// Orders boxes that form one column: boxes whose vertical extents overlap
// enough sit on the same text line; lines go top to bottom and boxes within a
// line left to right.
function orderColumn(boxes: readonly Bounds[], column: number[]): number[] {
  'worklet';
  column.sort((a, b) => boxes[a]!.ymin - boxes[b]!.ymin);
  const lines: { items: number[]; ymin: number; ymax: number }[] = [];
  for (const i of column) {
    const box = boxes[i]!;
    let placed = false;
    for (const line of lines) {
      // Overlap is measured against the SHORTER of the two heights, so a
      // small box beside a tall one still joins its line.
      const overlap = Math.min(line.ymax, box.ymax) - Math.max(line.ymin, box.ymin);
      const minHeight = Math.min(line.ymax - line.ymin, box.ymax - box.ymin);
      if (overlap >= LINE_OVERLAP_FRACTION * Math.max(1, minHeight)) {
        line.items.push(i);
        line.ymin = Math.min(line.ymin, box.ymin);
        line.ymax = Math.max(line.ymax, box.ymax);
        placed = true;
        break;
      }
    }
    if (!placed) {
      lines.push({ items: [i], ymin: box.ymin, ymax: box.ymax });
    }
  }
  lines.sort((a, b) => a.ymin - b.ymin);
  const order: number[] = [];
  for (const line of lines) {
    line.items.sort((a, b) => boxes[a]!.xmin + boxes[a]!.xmax - (boxes[b]!.xmin + boxes[b]!.xmax));
    order.push(...line.items);
  }
  return order;
}

/**
 * Reorders `{ quad }` items the way a person reads the page: leftmost column
 * first (top to bottom), then the next column. Detectors emit boxes in
 * arbitrary order, so detections are ordered through this.
 *
 * Columns are the gaps in the x projection of all boxes, so a line that spans
 * the page (a title, a wide heading) hides the gutter below it and the
 * columns under it interleave; {@link orderByXyCut} handles that for layout
 * blocks. This is the order react-native-executorch's PaddleOCR task returns.
 * @typeParam T The item type; anything carrying a `quad`.
 * @param items The items to order.
 * @returns The same items in reading order (the input array itself when it
 * holds at most one item).
 */
export function orderByReadingOrder<T extends { readonly quad: Quad }>(items: T[]): T[] {
  'worklet';
  const count = items.length;
  if (count <= 1) {
    return items;
  }

  const boxes = items.map((it) => boundingBoxOfPoints(it.quad, 'xyxy'));

  // Column gutters: x ranges no box covers, wide enough relative to the
  // content width. A full sweep of the x axis finds every gutter at once.
  let minX = Infinity;
  let maxX = -Infinity;
  for (const box of boxes) {
    if (box.xmin < minX) minX = box.xmin;
    if (box.xmax > maxX) maxX = box.xmax;
  }
  const minGap = COLUMN_GAP_FRACTION * Math.max(1, maxX - minX);

  const edges: { x: number; delta: number }[] = [];
  for (const box of boxes) {
    edges.push({ x: box.xmin, delta: 1 });
    edges.push({ x: box.xmax, delta: -1 });
  }
  edges.sort((a, b) => a.x - b.x || b.delta - a.delta);
  const cuts: number[] = [];
  let coverage = 0;
  let gutterStart = edges.length > 0 ? edges[0]!.x : 0;
  for (const edge of edges) {
    const before = coverage;
    coverage += edge.delta;
    if (before > 0 && coverage === 0) {
      gutterStart = edge.x;
    } else if (before === 0 && coverage > 0 && edge.x - gutterStart >= minGap) {
      cuts.push((gutterStart + edge.x) / 2);
    }
  }

  // A box belongs to column k when exactly k cuts lie left of its center.
  const columns: number[][] = Array.from({ length: cuts.length + 1 }, () => []);
  for (let i = 0; i < count; i++) {
    const centerX = (boxes[i]!.xmin + boxes[i]!.xmax) / 2;
    let column = 0;
    for (const cut of cuts) {
      if (centerX > cut) column++;
    }
    columns[column]!.push(i);
  }

  const order: number[] = [];
  for (const column of columns) {
    order.push(...orderColumn(boxes, column));
  }
  return order.map((i) => items[i]!);
}

/**
 * Reorders `{ quad }` items the way a person reads a page with spanning
 * elements, by recursive cuts. A part is split at the widest column gutter
 * no item crosses (left before right). When some items cross the gutter (a
 * title, an abstract, a wide heading), the gutter is looked for among the
 * items narrower than half the content, and the crossing items divide the
 * part into bands read top to bottom, each band split again. Failing both,
 * the part is split at the widest blank band (top before bottom), which
 * separates a footer sitting in the gutter from the columns above it. What
 * cannot be split is read as one column. Unlike {@link orderByReadingOrder},
 * the columns under a title therefore still read left column first. Used
 * for layout blocks.
 *
 * A gutter must be {@link COLUMN_GAP_FRACTION} of the content width; a blank
 * band must be as tall as the smallest item (one line of the smallest text),
 * so ordinary line spacing and the gap under a heading never split.
 * @typeParam T The item type; anything carrying a `quad`.
 * @param items The items to order.
 * @returns The same items in reading order (the input array itself when it
 * holds at most one item).
 */
export function orderByXyCut<T extends { readonly quad: Quad }>(items: T[]): T[] {
  'worklet';
  if (items.length <= 1) {
    return items;
  }
  const boxes = items.map((it) => boundingBoxOfPoints(it.quad, 'xyxy'));

  // Depth first; the stack holds the parts still to order, the next one on top.
  const order: number[] = [];
  const stack: number[][] = [items.map((_, i) => i)];
  while (stack.length > 0) {
    const part = stack.pop()!;
    if (part.length <= 1) {
      order.push(...part);
      continue;
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minHeight = Infinity;
    for (const i of part) {
      const box = boxes[i]!;
      if (box.xmin < minX) minX = box.xmin;
      if (box.xmax > maxX) maxX = box.xmax;
      if (box.ymax - box.ymin < minHeight) minHeight = box.ymax - box.ymin;
    }
    const contentWidth = Math.max(1, maxX - minX);
    const minGutter = COLUMN_GAP_FRACTION * contentWidth;

    // 1. A gutter across the whole part: columns.
    const gutter = widestGap(boxes, part, 'x', minGutter);
    if (gutter !== null) {
      const left: number[] = [];
      const right: number[] = [];
      for (const i of part) {
        (boxes[i]!.xmin < gutter.at ? left : right).push(i);
      }
      stack.push(right, left);
      continue;
    }

    // 2. A gutter hidden by spanning items: bands between them.
    const narrow = part.filter(
      (i) => boxes[i]!.xmax - boxes[i]!.xmin < WIDE_FRACTION * contentWidth
    );
    const hidden = narrow.length >= 2 ? widestGap(boxes, narrow, 'x', minGutter) : null;
    if (hidden !== null) {
      const spanning = part.filter((i) => boxes[i]!.xmin < hidden.at && boxes[i]!.xmax > hidden.at);
      spanning.sort((a, b) => boxes[a]!.ymin - boxes[b]!.ymin);
      // An item belongs to band k when exactly k spanning items start above
      // its center.
      const bands: number[][] = Array.from({ length: spanning.length + 1 }, () => []);
      for (const i of part) {
        if (spanning.includes(i)) {
          continue;
        }
        const centerY = (boxes[i]!.ymin + boxes[i]!.ymax) / 2;
        let band = 0;
        for (const j of spanning) {
          if (boxes[j]!.ymin < centerY) band++;
        }
        bands[band]!.push(i);
      }
      // Top to bottom: band 0, the first spanning item, band 1, ...
      const sequence: number[][] = [];
      for (let k = 0; k < bands.length; k++) {
        if (bands[k]!.length > 0) sequence.push(bands[k]!);
        if (k < spanning.length) sequence.push([spanning[k]!]);
      }
      for (let k = sequence.length - 1; k >= 0; k--) {
        stack.push(sequence[k]!);
      }
      continue;
    }

    // 3. A blank band: top and bottom.
    const band = widestGap(boxes, part, 'y', ROW_GAP_HEIGHTS * Math.max(1, minHeight));
    if (band !== null) {
      const top: number[] = [];
      const bottom: number[] = [];
      for (const i of part) {
        (boxes[i]!.ymin < band.at ? top : bottom).push(i);
      }
      stack.push(bottom, top);
      continue;
    }

    // 4. One column.
    order.push(...orderColumn(boxes, part));
  }
  return order.map((i) => items[i]!);
}

const STACK_X_OVERLAP = 0.25;
const STACK_Y_GAP = 2.5;
const STACK_UPWARD_OVERLAP = 0.5; // a box may reach back up this fraction of its height into the stack

/**
 * Clusters glyph-like, x-aligned, vertically stacked boxes into stacks (each
 * read top to bottom); wide lines and isolated boxes fall through as
 * `singles`. A "stack" is character-under-character vertical text, unrelated
 * to the page-layout columns of {@link orderByReadingOrder}.
 * @param quads The detected quads.
 * @returns The stacks (top to bottom within each) and the remaining singles.
 */
export function groupVerticalStacks(quads: readonly Quad[]): {
  stacks: Quad[][];
  singles: Quad[];
} {
  'worklet';
  type Candidate = {
    quad: Quad;
    xmin: number;
    xmax: number;
    ymin: number;
    ymax: number;
    width: number;
    height: number;
  };
  const candidates: Candidate[] = [];
  const singles: Quad[] = [];
  for (const q of quads) {
    const { xmin, ymin, xmax, ymax } = boundingBoxOfPoints(q, 'xyxy');
    const width = xmax - xmin;
    const height = ymax - ymin;
    if (width > height * STACK_GLYPH_ASPECT) {
      singles.push(q);
    } else {
      candidates.push({ quad: q, xmin, xmax, ymin, ymax, width, height });
    }
  }
  // Grow each stack top to bottom from its current bottom box, checking
  // alignment against the stack's accumulated x-range so a narrow glyph
  // between wider ones doesn't break the run.
  candidates.sort((a, b) => a.ymin - b.ymin);
  type Stack = { boxes: Candidate[]; xmin: number; xmax: number; bottom: number };
  const stacks: Stack[] = [];
  for (const box of candidates) {
    let placed = false;
    for (const stack of stacks) {
      const overlap = Math.min(box.xmax, stack.xmax) - Math.max(box.xmin, stack.xmin);
      const aligned = overlap > STACK_X_OVERLAP * Math.min(box.width, stack.xmax - stack.xmin);
      const gap = box.ymin - stack.bottom;
      if (aligned && gap < STACK_Y_GAP * box.height && gap > -STACK_UPWARD_OVERLAP * box.height) {
        stack.boxes.push(box);
        stack.xmin = Math.min(stack.xmin, box.xmin);
        stack.xmax = Math.max(stack.xmax, box.xmax);
        stack.bottom = box.ymax;
        placed = true;
        break;
      }
    }
    if (!placed) {
      stacks.push({ boxes: [box], xmin: box.xmin, xmax: box.xmax, bottom: box.ymax });
    }
  }
  const grouped: Quad[][] = [];
  for (const stack of stacks) {
    if (stack.boxes.length >= 2) {
      grouped.push(stack.boxes.map((b) => b.quad));
    } else {
      singles.push(stack.boxes[0]!.quad);
    }
  }
  return { stacks: grouped, singles };
}
