import { describe, expect, it } from 'vitest';
import { makeTrack } from '../test/fixtures';
import {
  indexAt,
  largestDimension,
  observationAt,
  perFrameSizes,
  poseAt,
  sizeSummary,
} from './geometry';

/*
 * The size an agent's box has AT a frame. The collision check builds each box from the size
 * recorded at that frame, so the scene must too. These tests pin the rule for one track:
 * which size is drawn at a frame, when the scalar is used instead, and what an unusable
 * element does.
 */

// Observed frames with a gap, and not starting at 0, so the vertex index is never the frame
// number and never equal to it by accident: index 0..4 <-> frames 3, 4, 5, 8, 9.
const FRAMES = [3, 4, 5, 8, 9];
const LENGTHS = [4.1, 4.2, 4.3, 4.4, 4.5];
const WIDTHS = [1.91, 1.92, 1.93, 1.94, 1.95];
const SCALAR = { length_m: 9, width_m: 7 }; // deliberately unlike every array element

const perFrame = (over: Parameters<typeof makeTrack>[0] = {}) =>
  makeTrack({ frames: FRAMES, lengths_m: LENGTHS, widths_m: WIDTHS, ...SCALAR, ...over });

describe('indexAt', () => {
  const track = makeTrack({ frames: FRAMES });

  it('is the vertex index of the observed frame, and null in a gap or outside the track', () => {
    expect(FRAMES.map((f) => indexAt(track, f))).toEqual([0, 1, 2, 3, 4]);
    for (const f of [-1, 0, 2, 6, 7, 10]) expect(indexAt(track, f)).toBeNull();
  });

  it('is the index poseAt reads its pose from', () => {
    for (const f of FRAMES) {
      const i = indexAt(track, f)!;
      expect(poseAt(track, f)).toEqual({ x: track.path[i]![0], y: track.path[i]![1], heading: track.headings[i] });
    }
  });
});

describe('perFrameSizes', () => {
  it('returns both arrays when both exist and each is as long as the path', () => {
    expect(perFrameSizes(perFrame())).toEqual({ lengths: LENGTHS, widths: WIDTHS });
  });

  it.each([
    ['both absent', {}],
    ['lengths_m null', { lengths_m: null }],
    ['widths_m null', { widths_m: null }],
    ['lengths_m one short', { lengths_m: LENGTHS.slice(1) }],
    ['lengths_m one long', { lengths_m: [...LENGTHS, 4.6] }],
    ['widths_m one short', { widths_m: WIDTHS.slice(1) }],
    ['widths_m one long', { widths_m: [...WIDTHS, 1.96] }],
    ['lengths_m empty', { lengths_m: [] }],
  ])('returns null when %s', (_name, over) => {
    const base = _name === 'both absent' ? makeTrack({ frames: FRAMES, ...SCALAR }) : perFrame(over);
    expect(perFrameSizes(base)).toBeNull();
  });
});

describe('observationAt: per-frame sizes', () => {
  it('draws the array elements at the vertex index of the frame, for length and width', () => {
    FRAMES.forEach((frame, i) => {
      const o = observationAt(perFrame(), frame)!;
      expect(o.source).toBe('per_frame');
      expect(o.size).toEqual({ length: LENGTHS[i], width: WIDTHS[i] });
    });
  });

  it('takes pose and size from the same vertex', () => {
    const track = perFrame({ headings: [0.1, 0.2, 0.3, 0.4, 0.5] });
    for (const frame of FRAMES) {
      const o = observationAt(track, frame)!;
      expect(o.pose).toEqual(poseAt(track, frame));
      expect(o.pose.heading).toBeCloseTo(0.1 * (indexAt(track, frame)! + 1));
      expect(o.size!.length).toBe(LENGTHS[indexAt(track, frame)!]);
    }
  });

  it('lets the arrays win over scalars that disagree', () => {
    expect(observationAt(perFrame({ length_m: 20, width_m: 20 }), 8)!.size).toEqual({ length: 4.4, width: 1.94 });
  });

  it('draws boxes from the arrays even when the scalars are unusable', () => {
    const o = observationAt(perFrame({ length_m: null, width_m: null }), 5)!;
    expect(o).toMatchObject({ source: 'per_frame', size: { length: 4.3, width: 1.93 } });
  });

  it('is null at a frame the agent was not observed', () => {
    for (const f of [0, 2, 6, 7, 10]) expect(observationAt(perFrame(), f)).toBeNull();
  });
});

describe('observationAt: the scalar fallback', () => {
  it.each([
    ['lengths_m null', { lengths_m: null }],
    ['widths_m null', { widths_m: null }],
    ['lengths_m the wrong length', { lengths_m: LENGTHS.slice(0, 4) }],
    ['widths_m the wrong length', { widths_m: [...WIDTHS, 1.96] }],
  ])('uses the scalar sizes, for both dimensions, when %s', (_name, over) => {
    const track = perFrame(over);
    for (const frame of FRAMES) {
      expect(observationAt(track, frame)).toMatchObject({ source: 'scalar', size: { length: 9, width: 7 } });
    }
  });

  it('is exactly what a track without the fields gives', () => {
    const plain = makeTrack({ frames: FRAMES, length_m: 4.5, width_m: 2 });
    expect(observationAt(plain, 5)).toEqual({ pose: poseAt(plain, 5), size: { length: 4.5, width: 2 }, source: 'scalar' });
  });

  it('is a dot, as before, when the scalars are unusable and there are no arrays', () => {
    for (const bad of [null, undefined, 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const track = makeTrack({ frames: FRAMES, length_m: bad, width_m: 2 });
      expect(observationAt(track, 5)).toMatchObject({ source: 'scalar', size: null });
    }
  });
});

describe('observationAt: an unusable element', () => {
  const BAD = [null, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 0, -1];

  it.each(BAD.map((v) => [String(v), v] as const))(
    'draws a dot, never the scalar, where the length is %s',
    (_name, bad) => {
      const lengths = [...LENGTHS];
      lengths[2] = bad as number;
      const track = perFrame({ lengths_m: lengths });
      expect(observationAt(track, 5)).toMatchObject({ source: 'per_frame', size: null });
      // its neighbours are still boxes
      expect(observationAt(track, 4)!.size).toEqual({ length: 4.2, width: 1.92 });
      expect(observationAt(track, 8)!.size).toEqual({ length: 4.4, width: 1.94 });
    },
  );

  it.each(BAD.map((v) => [String(v), v] as const))('draws a dot where the width is %s', (_name, bad) => {
    const widths = [...WIDTHS];
    widths[3] = bad as number;
    const track = perFrame({ widths_m: widths });
    expect(observationAt(track, 8)).toMatchObject({ source: 'per_frame', size: null });
    expect(observationAt(track, 9)!.size).toEqual({ length: 4.5, width: 1.95 });
  });
});

describe('largestDimension: what the margin must clear at any frame', () => {
  it('is the largest usable per-frame element, wherever it is, not the scalar', () => {
    const lengths = [4.1, 4.2, 30, 4.4, 4.5]; // the large one is at an unobserved-looking middle frame
    expect(largestDimension(perFrame({ lengths_m: lengths }))).toBe(30);
    expect(largestDimension(perFrame({ length_m: 50, width_m: 50 }))).toBe(4.5);
  });

  it('counts a width larger than every length', () => {
    expect(largestDimension(perFrame({ widths_m: [1, 1, 1, 12, 1] }))).toBe(12);
  });

  it('ignores an element that would be drawn as a dot', () => {
    expect(largestDimension(perFrame({ lengths_m: [4.1, 4.2, 99, 4.4, 4.5], widths_m: [1, 1, null, 1, 1] }))).toBe(4.5);
    expect(largestDimension(perFrame({ lengths_m: [4.1, 4.2, Number.NaN, 4.4, 4.5] }))).toBe(4.5);
  });

  it('is the scalar maximum for a track without arrays, and 0 when nothing can be drawn', () => {
    expect(largestDimension(makeTrack({ frames: FRAMES, length_m: 4.5, width_m: 6 }))).toBe(6);
    expect(largestDimension(makeTrack({ frames: FRAMES, length_m: null, width_m: 6 }))).toBe(0);
    expect(largestDimension(perFrame({ lengths_m: [null, null, null, null, null] }))).toBe(0);
  });

  it('falls back to the scalar when the arrays are unusable as a whole', () => {
    expect(largestDimension(perFrame({ lengths_m: LENGTHS.slice(1) }))).toBe(9);
  });
});

describe('sizeSummary: what the titles may say', () => {
  it('is a single size for a scalar track and for none', () => {
    expect(sizeSummary(makeTrack({ frames: FRAMES, length_m: 4.5, width_m: 2 }))).toEqual({ kind: 'scalar', length: 4.5, width: 2 });
    expect(sizeSummary(makeTrack({ frames: FRAMES, length_m: null, width_m: 2 }))).toEqual({ kind: 'none' });
  });

  it('is the min and max of the boxes that can be drawn for a per-frame track', () => {
    expect(sizeSummary(perFrame())).toEqual({
      kind: 'per_frame',
      length: { min: 4.1, max: 4.5 },
      width: { min: 1.91, max: 1.95 },
    });
  });

  it('has equal min and max where a dimension does not vary', () => {
    const s = sizeSummary(perFrame({ lengths_m: [5, 5, 5, 5, 5] }));
    expect(s).toMatchObject({ kind: 'per_frame', length: { min: 5, max: 5 } });
  });

  it('leaves out elements that would be dots, and is none when no box can be drawn', () => {
    expect(sizeSummary(perFrame({ lengths_m: [null, 4.2, 4.3, 4.4, 99], widths_m: [1, 1, 1, 1, null] }))).toEqual({
      kind: 'per_frame',
      length: { min: 4.2, max: 4.4 },
      width: { min: 1, max: 1 },
    });
    expect(sizeSummary(perFrame({ lengths_m: [null, null, null, null, null] }))).toEqual({ kind: 'none' });
  });

  it('uses the scalar when the arrays are unusable as a whole', () => {
    expect(sizeSummary(perFrame({ widths_m: null }))).toEqual({ kind: 'scalar', length: 9, width: 7 });
  });
});
