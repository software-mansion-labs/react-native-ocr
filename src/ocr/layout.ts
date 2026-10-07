/**
 * Layout helpers of the OCR pipeline: merging the layout detector's nested
 * regions and assembling the blocks of a result. Pure functions; `createOcr`
 * runs them on the worklet runtime.
 */
import type { ObjectDetection } from 'react-native-executorch';
import type { BoundingBox } from 'react-native-executorch/cv';
import { orderByReadingOrder } from './geometry';
import type { DocumentBlock, OcrDetection } from './types';

/** Layout labels carrying no text (PP-DocLayout names); OCR skips these regions. */
export const DEFAULT_VISUAL_LABELS: readonly string[] = ['image', 'chart', 'seal'];
/** The PP-DocLayout label of a table region. */
export const DEFAULT_TABLE_LABEL = 'table';

// PP-DocLayout's RT-DETR head is set prediction, so it needs no IoU NMS, but
// it still emits nested duplicates (a line's own region inside the paragraph
// region that contains it). Since every region is OCR'd on its own crop, that
// would transcribe the same text twice. Fraction of the smaller region that a
// larger one must cover for the smaller one to be dropped, as PaddleX's
// `layout_merge_bboxes_mode: 'large'` does.
export const REGION_CONTAINMENT = 0.8;

function boxArea(box: BoundingBox<'xyxy'>): number {
  'worklet';
  return Math.max(0, box.xmax - box.xmin) * Math.max(0, box.ymax - box.ymin);
}

/**
 * Fraction of `inner`'s area that `outer` covers (intersection over the
 * smaller box).
 * @param inner The box being tested for coverage.
 * @param outer The box that may cover it.
 * @returns The covered fraction in `[0, 1]`; 0 for an empty `inner`.
 */
export function coveredFraction(inner: BoundingBox<'xyxy'>, outer: BoundingBox<'xyxy'>): number {
  'worklet';
  const w = Math.min(inner.xmax, outer.xmax) - Math.max(inner.xmin, outer.xmin);
  const h = Math.min(inner.ymax, outer.ymax) - Math.max(inner.ymin, outer.ymin);
  const area = boxArea(inner);
  return w <= 0 || h <= 0 || area <= 0 ? 0 : (w * h) / area;
}

/**
 * Drops layout regions already covered by a larger kept one, largest first.
 * Visual regions never swallow anything: a caption inside a figure is still
 * text to read.
 * @category Functions
 * @param regions The layout detector's regions.
 * @param visualLabels Labels carrying no text, which never suppress another region.
 * @returns The surviving regions, in the detector's original order.
 */
export function mergeContainedRegions<L>(
  regions: readonly ObjectDetection<'xyxy', L>[],
  visualLabels: readonly string[]
): ObjectDetection<'xyxy', L>[] {
  'worklet';
  const byArea = regions.map((_, i) => i);
  byArea.sort((a, b) => boxArea(regions[b]!.box) - boxArea(regions[a]!.box));
  const dropped = regions.map(() => false);
  for (let i = 0; i < byArea.length; i++) {
    const outer = byArea[i]!;
    if (dropped[outer] || visualLabels.includes(String(regions[outer]!.label))) {
      continue;
    }
    for (let j = i + 1; j < byArea.length; j++) {
      const inner = byArea[j]!;
      if (
        !dropped[inner] &&
        coveredFraction(regions[inner]!.box, regions[outer]!.box) >= REGION_CONTAINMENT
      ) {
        dropped[inner] = true;
      }
    }
  }
  return regions.filter((_, i) => !dropped[i]);
}

/**
 * Assembles one block: the lines in reading order and their texts joined by
 * newlines.
 * @param regionType The layout label, or `'ungrouped'` for the whole-page block.
 * @param bbox The block's bounds in the result image's pixels.
 * @param layoutConfidence The layout model's confidence (1 for `'ungrouped'`).
 * @param lines The OCR lines inside the block, in any order.
 * @param isTable Whether the region carries the table label.
 * @returns The assembled block.
 */
export function makeBlock<L>(
  regionType: L | 'ungrouped',
  bbox: BoundingBox<'xyxy'>,
  layoutConfidence: number,
  lines: OcrDetection[],
  isTable: boolean
): DocumentBlock<L> {
  'worklet';
  const sorted = orderByReadingOrder(lines);
  return {
    regionType,
    bbox,
    layoutConfidence,
    isTable,
    lines: sorted,
    text: sorted.map((l) => l.text).join('\n'),
  };
}
