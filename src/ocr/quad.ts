/**
 * Quad helpers the OCR pipeline needs beyond what react-native-executorch's
 * `cv` module exports: splitting a line into recognizer-sized segments, and
 * axis-aligned enclosures. Every quad here is ordered TL, TR, BR, BL.
 */
import {
  boundingBoxOfPoints,
  interpolatePoint,
  type BoundingBox,
  type Point,
  type Quad,
} from 'react-native-executorch/cv';

/**
 * Builds the axis-aligned quad (ordered TL, TR, BR, BL) of an `xyxy` box: the
 * corner-ordering counterpart of an axis-aligned bounding box.
 * @param box The `xyxy` box.
 * @returns The four corners, ordered TL, TR, BR, BL.
 */
export function quadFromBounds(box: BoundingBox<'xyxy'>): Quad {
  'worklet';
  return [
    { x: box.xmin, y: box.ymin },
    { x: box.xmax, y: box.ymin },
    { x: box.xmax, y: box.ymax },
    { x: box.xmin, y: box.ymax },
  ];
}

/**
 * Splits an ordered TL, TR, BR, BL quad into `parts` equal vertical bands (each
 * an ordered quad), top to bottom. `parts <= 1` returns the quad unchanged.
 * @param ordered The quad corners ordered TL, TR, BR, BL.
 * @param parts The number of equal vertical bands to split into.
 * @returns The bands as ordered quads, top to bottom.
 */
export function splitTallQuad(ordered: Quad, parts: number): Quad[] {
  'worklet';
  if (parts <= 1) {
    return [ordered];
  }
  const [tl, tr, br, bl] = ordered;
  const out: Quad[] = [];
  for (let i = 0; i < parts; i++) {
    const t0 = i / parts;
    const t1 = (i + 1) / parts;
    out.push([
      interpolatePoint(tl, bl, t0),
      interpolatePoint(tr, br, t0),
      interpolatePoint(tr, br, t1),
      interpolatePoint(tl, bl, t1),
    ]);
  }
  return out;
}

/**
 * Splits an ordered TL, TR, BR, BL quad into `parts` equal horizontal segments
 * (each an ordered quad), left to right. `parts <= 1` returns the quad
 * unchanged.
 * @param ordered The quad corners ordered TL, TR, BR, BL.
 * @param parts The number of equal horizontal segments to split into.
 * @returns The segments as ordered quads, left to right.
 */
export function splitWideQuad(ordered: Quad, parts: number): Quad[] {
  'worklet';
  if (parts <= 1) {
    return [ordered];
  }
  const [tl, tr, br, bl] = ordered;
  const out: Quad[] = [];
  for (let i = 0; i < parts; i++) {
    const t0 = i / parts;
    const t1 = (i + 1) / parts;
    out.push([
      interpolatePoint(tl, tr, t0),
      interpolatePoint(tl, tr, t1),
      interpolatePoint(bl, br, t1),
      interpolatePoint(bl, br, t0),
    ]);
  }
  return out;
}

/**
 * Computes the axis-aligned quad (ordered TL, TR, BR, BL) enclosing a set of
 * quads. Returns a zero quad for empty input.
 * @param quads The quads to enclose.
 * @returns The four enclosing corners, ordered TL, TR, BR, BL.
 */
export function boundingQuadOf(quads: readonly Quad[]): Quad {
  'worklet';
  const all: Point[] = [];
  for (const q of quads) {
    for (const p of q) {
      all.push(p);
    }
  }
  return quadFromBounds(boundingBoxOfPoints(all, 'xyxy'));
}
