/**
 * Input-size resolution against a model's exported dimension domains: the
 * detector's letterbox size and the recognizer's canvas width, both snapped
 * onto whatever the model declares legal.
 */
import type { ConcreteDim, Range } from 'react-native-executorch/schema';

// A dimension whose legal values are sparse (an `enum`) can jump a long way
// from one value to the next, so an image at most this fraction larger than a
// legal value still snaps DOWN to it rather than forcing the next, much bigger
// one. A `range` steps densely, so it always snaps up.
const SNAP_DOWN_TOLERANCE = 0.1;

/**
 * The inclusive `[min, max]` extent of a dimension's domain.
 * @param dim The dimension.
 * @returns Its smallest and largest legal value.
 */
export function dimExtent(dim: ConcreteDim): readonly [number, number] {
  'worklet';
  if (dim.kind === 'constant') {
    return [dim.value, dim.value];
  }
  if (dim.kind === 'range') {
    return [dim.range.min, dim.range.max];
  }
  let min = Infinity;
  let max = -Infinity;
  for (const choice of dim.choices) {
    min = Math.min(min, choice);
    max = Math.max(max, choice);
  }
  return [min, max];
}

/**
 * The smallest value on `range`'s lattice that is `>= size` and satisfies
 * `isLegal`, else the largest legal lattice point. Never shrinks content; only
 * the range max can.
 * @param size The desired size.
 * @param range The dimension's range.
 * @param isLegal Extra predicate a value must satisfy.
 * @returns The snapped size.
 */
export function snapUpRange(
  size: number,
  range: Range,
  isLegal: (value: number) => boolean
): number {
  'worklet';
  const step = Math.max(1, range.step);
  const maxSteps = Math.floor((range.max - range.min) / step);
  for (let k = Math.max(0, Math.ceil((size - range.min) / step)); k <= maxSteps; k++) {
    const target = range.min + k * step;
    if (isLegal(target)) {
      return target;
    }
  }
  // Overflow (or nothing legal above `size`): the largest legal lattice point.
  // `range.max` itself may sit off the lattice, hence the walk down from
  // maxSteps.
  for (let k = maxSteps; k >= 0; k--) {
    const target = range.min + k * step;
    if (isLegal(target)) {
      return target;
    }
  }
  // No legal size anywhere in the range. The load-time contract check rejects
  // this, so returning the largest lattice point here is only a safety net.
  return range.min + maxSteps * step;
}

/**
 * The smallest legal choice `>= size` (modulo the snap-down tolerance when
 * `tolerateShrink`), else the largest legal choice.
 * @param size The desired size.
 * @param choices The dimension's legal values.
 * @param isLegal Extra predicate a value must satisfy.
 * @param tolerateShrink Allow snapping down by up to 10%.
 * @returns The snapped size.
 */
export function snapUpEnum(
  size: number,
  choices: readonly number[],
  isLegal: (value: number) => boolean,
  tolerateShrink: boolean
): number {
  'worklet';
  const legal = choices.filter(isLegal).sort((a, b) => a - b);
  const usable = legal.length > 0 ? legal : [...choices].sort((a, b) => a - b);
  const floor = tolerateShrink ? size / (1 + SNAP_DOWN_TOLERANCE) : size;
  return usable.find((choice) => choice >= floor) ?? usable[usable.length - 1]!;
}

/**
 * The legal size closest above `size` in `dim`'s domain, restricted to the
 * values `isLegal` accepts (the detector's input alignment, the recognizer's
 * CTC timestep lattice). A `constant` domain has a single value, so `size` is
 * ignored.
 * @param size The desired size.
 * @param dim The dimension's domain.
 * @param isLegal Extra predicate a value must satisfy.
 * @param tolerateShrink Allow an `enum` domain to snap down by up to 10%.
 * @returns The snapped size.
 */
export function snapDim(
  size: number,
  dim: ConcreteDim,
  isLegal: (value: number) => boolean,
  tolerateShrink = false
): number {
  'worklet';
  if (dim.kind === 'constant') {
    return dim.value;
  }
  if (dim.kind === 'enum') {
    return snapUpEnum(size, dim.choices, isLegal, tolerateShrink);
  }
  return snapUpRange(size, dim.range, isLegal);
}

/**
 * Whether `dim`'s domain contains any value `isLegal` accepts. Answered by
 * running the very search the run-time path uses, so the two can never
 * disagree: {@link snapDim} walks the whole domain when nothing above the
 * requested size qualifies, and only returns an illegal value when the domain
 * holds none at all.
 * @param dim The dimension's domain.
 * @param isLegal The predicate.
 * @returns `true` when at least one legal value exists.
 */
export function domainAdmits(dim: ConcreteDim, isLegal: (value: number) => boolean): boolean {
  const [min] = dimExtent(dim);
  return isLegal(snapDim(min, dim, isLegal));
}

/**
 * The exported domains of a detector input's height and width.
 */
export type DetectorDims = {
  readonly h: ConcreteDim;
  readonly w: ConcreteDim;
};

/**
 * The largest legal detector input size that avoids upscaling where the
 * domains allow it (a domain whose values all exceed the image must upscale).
 * H and W snap independently: a per-dimension domain is by construction
 * independent of the other, so the legal set is their full grid.
 * @param det The detector's height and width domains.
 * @param imgWidth The image width.
 * @param imgHeight The image height.
 * @param align Both sizes must be multiples of this (the extractor's alignment).
 * @returns The detector input size.
 */
export function resolveDetectorSize(
  det: DetectorDims,
  imgWidth: number,
  imgHeight: number,
  align: number
): { detW: number; detH: number } {
  'worklet';
  const isAligned = (value: number) => {
    'worklet';
    return value % align === 0;
  };
  const scale = Math.min(1, dimExtent(det.h)[1] / imgHeight, dimExtent(det.w)[1] / imgWidth);
  return {
    detW: snapDim(Math.round(imgWidth * scale), det.w, isAligned, true),
    detH: snapDim(Math.round(imgHeight * scale), det.h, isAligned, true),
  };
}

/**
 * The relation between the recognizer's input width and its CTC timestep
 * count, read from the `linear` runtime constraint the model declares over
 * them: `width = pixelsPerStep * timesteps + offset`. SVTR reduces the width by
 * a plain factor (8, 0); the CRNN crops one trailing timestep, so it is affine
 * (4, 4). Inferring a width/timestep ratio would be wrong for it.
 */
export type CtcStride = {
  readonly pixelsPerStep: number;
  readonly offset: number;
};

/**
 * Whether a width sits on the timestep lattice, that is, yields a whole number
 * of CTC timesteps, so the probs tensor can be pre-sized for it.
 * @param width The candidate width.
 * @param stride The recognizer's stride.
 * @returns `true` when `width` is on the lattice.
 */
export function onCtcLattice(width: number, stride: CtcStride): boolean {
  'worklet';
  return (width - stride.offset) % stride.pixelsPerStep === 0;
}

/**
 * The CTC timestep count a legal width produces.
 * @param width A width on the lattice.
 * @param stride The recognizer's stride.
 * @returns The timestep count.
 */
export function ctcTimesteps(width: number, stride: CtcStride): number {
  'worklet';
  return (width - stride.offset) / stride.pixelsPerStep;
}

/**
 * The recognizer input width: the smallest legal width `>=` the desired
 * content, so snapping never squishes (only the domain max can). Restricted to
 * widths sitting on the CTC timestep lattice, so the probs output can always
 * be pre-sized.
 * @param width The recognizer width's domain.
 * @param desiredWidth The content width at recognizer height.
 * @param stride The recognizer's stride.
 * @returns The canvas width.
 */
export function resolveRecWidth(
  width: ConcreteDim,
  desiredWidth: number,
  stride: CtcStride
): number {
  'worklet';
  return snapDim(desiredWidth, width, (value) => {
    'worklet';
    return onCtcLattice(value, stride);
  });
}
