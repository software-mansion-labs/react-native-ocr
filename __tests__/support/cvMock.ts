/**
 * Mock of `react-native-executorch/cv`: the pure geometry helpers, copied
 * verbatim from react-native-executorch 0.10.2 (they are plain TypeScript
 * there as well), the types `src/` refers to, and a settable fake for the
 * native DBNet quad decoder.
 */
import type { FakeTensor } from './fakeTensor';

export type BoxMap = Readonly<{
  xyxy: Readonly<{ xmin: number; ymin: number; xmax: number; ymax: number }>;
  xywh: Readonly<{ xmin: number; ymin: number; w: number; h: number }>;
  cxcywh: Readonly<{ cx: number; cy: number; w: number; h: number }>;
}>;

export type BoxFormat = keyof BoxMap;

export type BoundingBox<F extends BoxFormat> = F extends any
  ? { readonly format: F } & BoxMap[F]
  : never;

export type Point = { readonly x: number; readonly y: number };

export type Quad = readonly [Point, Point, Point, Point];

export type ResizeMode = 'stretch' | 'letterbox' | 'crop';

export type ImageFormat = 'rgb' | 'rgba' | 'bgr' | 'bgra' | 'gray';

export type ImageBuffer = {
  readonly data: Uint8Array;
  readonly width: number;
  readonly height: number;
  readonly format: ImageFormat;
  readonly layout: 'hwc';
};

export type NormalizeOptions = {
  readonly alpha?: number | readonly number[];
  readonly beta?: number | readonly number[];
};

export function distance(a: Point, b: Point): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

export function interpolatePoint(a: Point, b: Point, t: number): Point {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

export type ScalePointOptions = {
  readonly from: { readonly width: number; readonly height: number };
  readonly to: { readonly width: number; readonly height: number };
  readonly resizeMode: Exclude<ResizeMode, 'crop'>;
};

export function scalePoint(point: Point, options: ScalePointOptions): Point {
  const { from, to, resizeMode } = options;
  switch (resizeMode) {
    case 'letterbox': {
      const scale = Math.min(from.width / to.width, from.height / to.height);
      const offsetX = (from.width - to.width * scale) / 2.0;
      const offsetY = (from.height - to.height * scale) / 2.0;
      return { x: (point.x - offsetX) / scale, y: (point.y - offsetY) / scale };
    }
    case 'stretch': {
      const scaleX = from.width / to.width;
      const scaleY = from.height / to.height;
      return { x: point.x / scaleX, y: point.y / scaleY };
    }
  }
}

export function boundingBoxOfPoints<F extends BoxFormat>(
  points: readonly Point[],
  format: F
): BoundingBox<F> {
  let xmin = Infinity;
  let ymin = Infinity;
  let xmax = -Infinity;
  let ymax = -Infinity;
  for (const p of points) {
    if (p.x < xmin) xmin = p.x;
    if (p.y < ymin) ymin = p.y;
    if (p.x > xmax) xmax = p.x;
    if (p.y > ymax) ymax = p.y;
  }
  if (points.length === 0) {
    xmin = ymin = xmax = ymax = 0;
  }
  switch (format) {
    case 'xyxy':
      return { format: 'xyxy', xmin, ymin, xmax, ymax } as BoundingBox<F>;
    case 'xywh':
      return { format: 'xywh', xmin, ymin, w: xmax - xmin, h: ymax - ymin } as BoundingBox<F>;
    case 'cxcywh':
      return {
        format: 'cxcywh',
        cx: (xmin + xmax) / 2,
        cy: (ymin + ymax) / 2,
        w: xmax - xmin,
        h: ymax - ymin,
      } as BoundingBox<F>;
    default:
      throw new Error(`boundingBoxOfPoints: unsupported box format '${format}'.`);
  }
}

export function orderQuad(quad: Quad): Quad {
  const sum = quad.map((p) => p.x + p.y);
  const diff = quad.map((p) => p.y - p.x);
  const indexOfMin = (a: number[]) => a.indexOf(Math.min(...a));
  const indexOfMax = (a: number[]) => a.indexOf(Math.max(...a));
  const corners = [indexOfMin(sum), indexOfMin(diff), indexOfMax(sum), indexOfMax(diff)];
  if (new Set(corners).size !== 4) {
    return quad;
  }
  return [quad[corners[0]!]!, quad[corners[1]!]!, quad[corners[2]!]!, quad[corners[3]!]!];
}

export function quadSize(ordered: Quad): { width: number; height: number } {
  const [tl, tr, br, bl] = ordered;
  const width = Math.max(distance(tl, tr), distance(bl, br));
  const height = Math.max(distance(tl, bl), distance(tr, br));
  return { width, height };
}

export type ScaleQuadOptions = {
  readonly from: { readonly width: number; readonly height: number };
  readonly to: { readonly width: number; readonly height: number };
  readonly resizeMode?: Exclude<ResizeMode, 'crop'>;
};

export function scaleQuad(quad: Quad, options: ScaleQuadOptions): Quad {
  const { from, to, resizeMode } = options;
  const map = (p: Point): Point => {
    const m = scalePoint(p, { from, to, resizeMode: resizeMode ?? 'letterbox' });
    return { x: Math.max(0, Math.min(m.x, to.width)), y: Math.max(0, Math.min(m.y, to.height)) };
  };
  return [map(quad[0]), map(quad[1]), map(quad[2]), map(quad[3])];
}

export type DbnetDecodeOptions = {
  readonly binThreshold: number;
  readonly boxThreshold: number;
  readonly unclipRatio: number;
  readonly minBoxSide: number;
  readonly maxCandidates: number;
};

type DbnetFake = (probabilityMap: FakeTensor, options: DbnetDecodeOptions) => Quad[];

let dbnetFake: DbnetFake | null = null;

/**
 * Installs the behavior of `extractDbnetTextQuads` for the current test, or
 * removes it with `null`. The real op is native contour tracing that the
 * package only forwards options to, so tests supply the quads directly.
 */
export function setFakeDbnetTextQuads(fake: DbnetFake | null): void {
  dbnetFake = fake;
}

export function extractDbnetTextQuads(
  probabilityMap: FakeTensor,
  options: DbnetDecodeOptions
): Quad[] {
  if (!dbnetFake) {
    throw new Error('extractDbnetTextQuads: no fake installed; call setFakeDbnetTextQuads first');
  }
  return dbnetFake(probabilityMap, options);
}
