/**
 * The `cv` namespace types from react-native-executorch that `src/` refers to.
 * Types only: nothing in `src/` calls a `cv.*` function yet.
 */

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
