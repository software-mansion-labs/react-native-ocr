/**
 * Mock of `react-native-executorch/cv`: the pure geometry helpers, copied
 * verbatim from react-native-executorch 0.10.2 (they are plain TypeScript
 * there as well), the types `src/` refers to, and a settable fake for the
 * native DBNet quad decoder, and plain-JavaScript versions of the image ops
 * (`resize`, `cvtColor`, `toChannelsFirst`, `normalize`, `rectifyQuad`).
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

// ─── Image ops ───────────────────────────────────────────────────────────────
// Plain-JavaScript stand-ins for the native OpenCV ops. They keep the real
// ops' shape and dtype contracts and produce the geometrically right pixels
// with nearest-neighbor sampling, which is all a pipeline test can assert on.

export type ColorConversionCode = `${'RGB' | 'RGBA' | 'BGR' | 'BGRA' | 'GRAY'}2${
  'RGB' | 'RGBA' | 'BGR' | 'BGRA' | 'GRAY'}`;

export type InterpolationMethod = 'nearest' | 'area' | 'cubic' | 'lanczos' | 'linear';

export type ResizeOptions = {
  readonly mode?: ResizeMode;
  readonly padValue?: number;
  readonly interpolation?: InterpolationMethod;
};

function requireImage(t: FakeTensor, op: string, what: string): [number, number, number] {
  if (t.disposed) throw new Error(`${op}: ${what} has already been disposed`);
  if (t.shape.length !== 3) {
    throw new Error(`${op}: ${what} must be a rank-3 image tensor, got [${t.shape}]`);
  }
  return [t.shape[0]!, t.shape[1]!, t.shape[2]!];
}

function requireDistinct(src: FakeTensor, dst: FakeTensor, op: string): void {
  if (src === dst) throw new Error(`${op}: dst must not alias src`);
  if (src.dtype !== dst.dtype) {
    throw new Error(`${op}: dtype mismatch ('${src.dtype}' -> '${dst.dtype}')`);
  }
}

export function resize(src: FakeTensor, dst: FakeTensor, options?: ResizeOptions): FakeTensor {
  const [srcH, srcW, channels] = requireImage(src, 'resize', 'src');
  const [dstH, dstW, dstChannels] = requireImage(dst, 'resize', 'dst');
  requireDistinct(src, dst, 'resize');
  if (channels !== dstChannels) {
    throw new Error(`resize: channel mismatch (${channels} -> ${dstChannels})`);
  }
  const mode = options?.mode ?? 'stretch';
  if (mode === 'crop') throw new Error("resize: the mock does not implement mode 'crop'");
  const input = src.data as ArrayLike<number>;
  const out = dst.data as { [i: number]: number; fill(value: number): unknown };
  out.fill(options?.padValue ?? 0);

  // Letterbox keeps the aspect ratio and centers the content, the same layout
  // `scalePoint(.., { resizeMode: 'letterbox' })` inverts.
  const scaleX = mode === 'letterbox' ? Math.min(dstW / srcW, dstH / srcH) : dstW / srcW;
  const scaleY = mode === 'letterbox' ? scaleX : dstH / srcH;
  const contentW = Math.min(dstW, Math.max(1, Math.round(srcW * scaleX)));
  const contentH = Math.min(dstH, Math.max(1, Math.round(srcH * scaleY)));
  const offsetX = Math.floor((dstW - contentW) / 2);
  const offsetY = Math.floor((dstH - contentH) / 2);
  for (let y = 0; y < contentH; y++) {
    const sy = Math.min(srcH - 1, Math.floor(((y + 0.5) * srcH) / contentH));
    for (let x = 0; x < contentW; x++) {
      const sx = Math.min(srcW - 1, Math.floor(((x + 0.5) * srcW) / contentW));
      const from = (sy * srcW + sx) * channels;
      const to = ((y + offsetY) * dstW + x + offsetX) * channels;
      for (let c = 0; c < channels; c++) out[to + c] = input[from + c]!;
    }
  }
  return dst;
}

const CHANNEL_ORDER: Record<string, string> = {
  RGB: 'rgb',
  RGBA: 'rgba',
  BGR: 'bgr',
  BGRA: 'bgra',
  GRAY: 'y',
};

export function cvtColor(src: FakeTensor, dst: FakeTensor, code: ColorConversionCode): FakeTensor {
  const [height, width, channels] = requireImage(src, 'cvtColor', 'src');
  const [dstH, dstW, dstChannels] = requireImage(dst, 'cvtColor', 'dst');
  requireDistinct(src, dst, 'cvtColor');
  const [fromName, toName] = code.split('2');
  const from = CHANNEL_ORDER[fromName ?? ''];
  const to = CHANNEL_ORDER[toName ?? ''];
  if (from === undefined || to === undefined) {
    throw new Error(`cvtColor: unsupported conversion code '${code}'`);
  }
  if (channels !== from.length || dstChannels !== to.length || dstH !== height || dstW !== width) {
    throw new Error(`cvtColor: ${code} cannot map [${src.shape}] to [${dst.shape}]`);
  }
  const input = src.data as ArrayLike<number>;
  const out = dst.data as { [i: number]: number };
  for (let i = 0; i < height * width; i++) {
    const pixel: Record<string, number> = { a: 255 };
    for (let c = 0; c < from.length; c++) pixel[from[c]!] = input[i * from.length + c]!;
    if (from === 'y') {
      pixel.r = pixel.g = pixel.b = pixel.y!;
    } else {
      pixel.y = Math.round(0.299 * pixel.r! + 0.587 * pixel.g! + 0.114 * pixel.b!);
    }
    for (let c = 0; c < to.length; c++) out[i * to.length + c] = pixel[to[c]!]!;
  }
  return dst;
}

export function toChannelsFirst(src: FakeTensor, dst: FakeTensor): FakeTensor {
  const [height, width, channels] = requireImage(src, 'toChannelsFirst', 'src');
  requireImage(dst, 'toChannelsFirst', 'dst');
  requireDistinct(src, dst, 'toChannelsFirst');
  if (dst.shape[0] !== channels || dst.shape[1] !== height || dst.shape[2] !== width) {
    throw new Error(`toChannelsFirst: cannot map [${src.shape}] to [${dst.shape}]`);
  }
  const input = src.data as ArrayLike<number>;
  const out = dst.data as { [i: number]: number };
  const plane = height * width;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < channels; c++) out[c * plane + i] = input[i * channels + c]!;
  }
  return dst;
}

export function normalize(
  src: FakeTensor,
  dst: FakeTensor,
  options?: NormalizeOptions
): FakeTensor {
  const [channels, height, width] = requireImage(src, 'normalize', 'src');
  requireImage(dst, 'normalize', 'dst');
  if (src === dst) throw new Error('normalize: dst must not alias src');
  if (dst.shape.join() !== src.shape.join()) {
    throw new Error(`normalize: cannot map [${src.shape}] to [${dst.shape}]`);
  }
  const coefficient = (value: number | readonly number[] | undefined, fallback: number) => {
    if (value === undefined) return new Array<number>(channels).fill(fallback);
    if (typeof value === 'number') return new Array<number>(channels).fill(value);
    if (value.length !== channels) {
      throw new Error(`normalize: expected ${channels} coefficients, got ${value.length}`);
    }
    return [...value];
  };
  const alpha = coefficient(options?.alpha, 1 / 255.0);
  const beta = coefficient(options?.beta, 0.0);
  const input = src.data as ArrayLike<number>;
  const out = dst.data as { [i: number]: number };
  const plane = height * width;
  for (let c = 0; c < channels; c++) {
    for (let i = 0; i < plane; i++) {
      out[c * plane + i] = input[c * plane + i]! * alpha[c]! + beta[c]!;
    }
  }
  return dst;
}

export type RectifyQuadOptions = {
  readonly contentWidth: number;
  readonly align?: 'left' | 'center';
  readonly padValue?: number;
};

/**
 * Samples the quad bilinearly between its corners (no perspective term, which
 * is exact for the parallelograms the tests use) into a `contentWidth` wide
 * block of the canvas, and fills the rest with `padValue`.
 */
export function rectifyQuad(
  src: FakeTensor,
  dst: FakeTensor,
  quad: Quad,
  options: RectifyQuadOptions
): FakeTensor {
  const [srcH, srcW, channels] = requireImage(src, 'rectifyQuad', 'src');
  const [dstH, dstW, dstChannels] = requireImage(dst, 'rectifyQuad', 'dst');
  requireDistinct(src, dst, 'rectifyQuad');
  if (channels !== dstChannels) {
    throw new Error(`rectifyQuad: channel mismatch (${channels} -> ${dstChannels})`);
  }
  const contentWidth = options.contentWidth;
  if (!Number.isInteger(contentWidth) || contentWidth < 1 || contentWidth > dstW) {
    throw new Error(`rectifyQuad: contentWidth (${contentWidth}) must be within [1, ${dstW}]`);
  }
  if (quad.some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y))) {
    throw new Error('rectifyQuad: quad corners must be finite');
  }
  const input = src.data as ArrayLike<number>;
  const out = dst.data as { [i: number]: number; fill(value: number): unknown };
  out.fill(options.padValue ?? 0);
  const offsetX = options.align === 'center' ? Math.floor((dstW - contentWidth) / 2) : 0;
  const [tl, tr, br, bl] = quad;
  for (let y = 0; y < dstH; y++) {
    const v = (y + 0.5) / dstH;
    const left = interpolatePoint(tl, bl, v);
    const right = interpolatePoint(tr, br, v);
    for (let x = 0; x < contentWidth; x++) {
      const p = interpolatePoint(left, right, (x + 0.5) / contentWidth);
      const sx = Math.min(srcW - 1, Math.max(0, Math.floor(p.x)));
      const sy = Math.min(srcH - 1, Math.max(0, Math.floor(p.y)));
      const from = (sy * srcW + sx) * channels;
      const to = (y * dstW + x + offsetX) * channels;
      for (let c = 0; c < channels; c++) out[to + c] = input[from + c]!;
    }
  }
  return dst;
}
