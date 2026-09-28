import type { schema } from 'react-native-executorch';
import {
  ctcTimesteps,
  dimExtent,
  domainAdmits,
  onCtcLattice,
  resolveDetectorSize,
  resolveRecWidth,
  snapDim,
  snapUpEnum,
  snapUpRange,
  type CtcStride,
} from '../../src/ocr/sizing';

const constant = (value: number): schema.ConcreteDim => ({ kind: 'constant', value });
const range = (min: number, max: number, step = 1): schema.ConcreteDim => ({
  kind: 'range',
  range: { min, max, step },
});
const choices = (...values: number[]): schema.ConcreteDim => ({ kind: 'enum', choices: values });

const any = () => true;
const even = (v: number) => v % 2 === 0;
const multipleOf = (n: number) => (v: number) => v % n === 0;

const SVTR: CtcStride = { pixelsPerStep: 8, offset: 0 };
const CRNN: CtcStride = { pixelsPerStep: 4, offset: 4 };

describe('dimExtent', () => {
  it('covers the three domain kinds', () => {
    expect(dimExtent(constant(640))).toEqual([640, 640]);
    expect(dimExtent(range(32, 1280, 32))).toEqual([32, 1280]);
    expect(dimExtent(choices(960, 640, 1280))).toEqual([640, 1280]);
  });
});

describe('snapUpRange', () => {
  const dense = { min: 32, max: 1280, step: 32 };

  it('snaps up to the next lattice point', () => {
    expect(snapUpRange(100, dense, any)).toBe(128);
    expect(snapUpRange(128, dense, any)).toBe(128);
    expect(snapUpRange(1, dense, any)).toBe(32);
  });

  it('skips lattice points the predicate rejects', () => {
    expect(snapUpRange(100, dense, multipleOf(64))).toBe(128);
    expect(snapUpRange(130, dense, multipleOf(64))).toBe(192);
  });

  it('falls back to the largest legal point when the size overflows the range', () => {
    expect(snapUpRange(1300, dense, any)).toBe(1280);
    expect(snapUpRange(1300, dense, multipleOf(3))).toBe(1248);
  });

  it('never returns a max that sits off the lattice', () => {
    expect(snapUpRange(30, { min: 10, max: 25, step: 10 }, any)).toBe(20);
  });
});

describe('snapUpEnum', () => {
  const sizes = [640, 960, 1280];

  it('snaps up to the next choice', () => {
    expect(snapUpEnum(700, sizes, any, false)).toBe(960);
    expect(snapUpEnum(960, sizes, any, false)).toBe(960);
  });

  it('snaps down by up to 10% when tolerated', () => {
    expect(snapUpEnum(700, sizes, any, true)).toBe(640);
    expect(snapUpEnum(704, sizes, any, true)).toBe(640);
    expect(snapUpEnum(705, sizes, any, true)).toBe(960);
  });

  it('caps at the largest choice', () => {
    expect(snapUpEnum(2000, sizes, any, false)).toBe(1280);
  });

  it('prefers legal choices and only falls back when there are none', () => {
    expect(snapUpEnum(100, [641, 960], even, false)).toBe(960);
    expect(snapUpEnum(100, [641, 963], even, false)).toBe(641);
  });
});

describe('snapDim', () => {
  it('ignores the size for a constant domain', () => {
    expect(snapDim(10, constant(640), any)).toBe(640);
    expect(snapDim(10000, constant(640), any)).toBe(640);
  });

  it('delegates to the enum and range searches', () => {
    expect(snapDim(700, choices(640, 960), any)).toBe(960);
    expect(snapDim(700, choices(640, 960), any, true)).toBe(640);
    expect(snapDim(700, range(32, 1280, 32), any)).toBe(704);
  });
});

describe('domainAdmits', () => {
  it('answers whether any legal value exists', () => {
    expect(domainAdmits(range(32, 1280, 32), multipleOf(64))).toBe(true);
    expect(domainAdmits(range(33, 65, 32), even)).toBe(false);
    expect(domainAdmits(choices(641, 963), even)).toBe(false);
    expect(domainAdmits(choices(641, 964), even)).toBe(true);
    expect(domainAdmits(constant(640), even)).toBe(true);
    expect(domainAdmits(constant(641), even)).toBe(false);
  });
});

describe('resolveDetectorSize', () => {
  const dense = { h: range(32, 1280, 32), w: range(32, 1280, 32) };

  it('keeps an image that fits and snaps each side up independently', () => {
    expect(resolveDetectorSize(dense, 1000, 750, 2)).toEqual({ detW: 1024, detH: 768 });
  });

  it('scales a large image down to the domain and never upscales', () => {
    expect(resolveDetectorSize(dense, 2000, 1000, 2)).toEqual({ detW: 1280, detH: 640 });
  });

  it('respects the extractor alignment', () => {
    const unit = { h: range(31, 1281), w: range(31, 1281) };
    expect(resolveDetectorSize(unit, 100, 100, 2)).toEqual({ detW: 100, detH: 100 });
    expect(resolveDetectorSize(unit, 101, 101, 2)).toEqual({ detW: 102, detH: 102 });
    expect(resolveDetectorSize(unit, 101, 101, 1)).toEqual({ detW: 101, detH: 101 });
  });

  it('tolerates a 10% snap-down on sparse domains', () => {
    const sparse = { h: choices(480, 640, 960), w: choices(480, 640, 960) };
    expect(resolveDetectorSize(sparse, 700, 660, 2)).toEqual({ detW: 640, detH: 640 });
    expect(resolveDetectorSize(sparse, 710, 660, 2)).toEqual({ detW: 960, detH: 640 });
  });

  it('returns the fixed size of a constant domain', () => {
    const fixed = { h: constant(640), w: constant(640) };
    expect(resolveDetectorSize(fixed, 3000, 100, 2)).toEqual({ detW: 640, detH: 640 });
  });
});

describe('CTC lattice', () => {
  it('recognizes widths that yield whole timesteps', () => {
    expect(onCtcLattice(320, SVTR)).toBe(true);
    expect(onCtcLattice(324, SVTR)).toBe(false);
    expect(onCtcLattice(100, CRNN)).toBe(true);
    expect(onCtcLattice(102, CRNN)).toBe(false);
  });

  it('counts timesteps per the stride', () => {
    expect(ctcTimesteps(320, SVTR)).toBe(40);
    expect(ctcTimesteps(100, CRNN)).toBe(24);
  });
});

describe('resolveRecWidth', () => {
  it('snaps the content width up onto the lattice', () => {
    const width = range(32, 1600, 4);
    expect(resolveRecWidth(width, 100, CRNN)).toBe(100);
    expect(resolveRecWidth(width, 101, CRNN)).toBe(104);
    expect(resolveRecWidth(width, 101, SVTR)).toBe(104);
    expect(resolveRecWidth(width, 105, SVTR)).toBe(112);
  });

  it('returns a fixed recognizer width unchanged', () => {
    expect(resolveRecWidth(constant(640), 12, SVTR)).toBe(640);
  });

  it('picks the next legal enum width', () => {
    expect(resolveRecWidth(choices(128, 256, 512), 200, SVTR)).toBe(256);
    expect(resolveRecWidth(choices(128, 256, 512), 200, CRNN)).toBe(256);
  });
});
