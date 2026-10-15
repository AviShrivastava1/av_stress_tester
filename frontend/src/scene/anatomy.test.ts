import { describe, expect, it } from 'vitest';
import { footprint, type Point, type TrackGeometry } from './geometry';
import { boxGap, collisionAnatomy, POSITION_PRECISION_M } from './anatomy';

/* A box of the given length and width centred at (x, y), turned by `heading`. */
const box = (x: number, y: number, length = 4, width = 2, heading = 0): Point[] =>
  footprint({ x, y, heading }, length, width);

describe('boxGap', () => {
  it('is the distance between facing sides of two boxes in a row', () => {
    // 4 long, centres 10 apart along x: 10 - 2 - 2 = 6.
    expect(boxGap(box(0, 0), box(10, 0))).toBeCloseTo(6, 12);
  });

  it('is the same from either side', () => {
    expect(boxGap(box(0, 0), box(10, 3, 5, 2, 0.4))).toBeCloseTo(boxGap(box(10, 3, 5, 2, 0.4), box(0, 0)), 12);
  });

  it('is the distance between the nearest corners of boxes set diagonally', () => {
    // 2 x 2 squares: corners (1, 1) and (2, 2) face each other, sqrt(2) apart.
    expect(boxGap(box(0, 0, 2, 2), box(3, 3, 2, 2))).toBeCloseTo(Math.SQRT2, 12);
  });

  it('is the corner-to-side distance when a corner faces a side', () => {
    // Square at the origin; a 45-degree square whose lowest corner is 0.5 above the top side, centred over it.
    const diamond = box(0, 0.5 + 1 + Math.SQRT2, 2, 2, Math.PI / 4);
    expect(boxGap(box(0, 0, 2, 2), diamond)).toBeCloseTo(0.5, 12);
  });

  it('does not change when the whole arrangement is turned', () => {
    const turn = (p: Point, angle: number): Point => [
      p[0] * Math.cos(angle) - p[1] * Math.sin(angle),
      p[0] * Math.sin(angle) + p[1] * Math.cos(angle),
    ];
    const a = box(0, 0, 4.6, 2, 0.3);
    const b = box(7, 2, 3.1, 1.8, 1.1);
    const before = boxGap(a, b);
    expect(before).toBeGreaterThan(0);
    for (const angle of [0.7, 2.1, -1.3]) {
      expect(boxGap(a.map((p) => turn(p, angle)), b.map((p) => turn(p, angle)))).toBeCloseTo(before, 10);
    }
  });

  it.each([
    ['sharing a side', box(0, 0), box(4, 0)],
    ['sharing only a corner', box(0, 0, 2, 2), box(2, 2, 2, 2)],
    ['overlapping', box(0, 0), box(3, 0.5)],
    ['one inside the other', box(0, 0, 6, 6), box(0, 0, 2, 1)],
    ['the same box', box(1, 1), box(1, 1)],
    ['a corner pressed into a side, turned', box(0, 0, 2, 2), box(0, 1 + Math.SQRT2, 2, 2, Math.PI / 4)],
  ])('is 0 for boxes %s', (_name, a, b) => {
    expect(boxGap(a, b)).toBe(0);
  });

  it('reports a small real gap as a gap, not as 0', () => {
    expect(boxGap(box(0, 0), box(4.01, 0))).toBeCloseTo(0.01, 12);
  });

  it('takes a separation of less than a billionth of a metre for rounding, and a larger one for a gap', () => {
    expect(boxGap(box(0, 0), box(4 + 5e-10, 0))).toBe(0);
    expect(boxGap(box(0, 0), box(4 + 5e-9, 0))).toBeCloseTo(5e-9, 12);
  });

  it('sees a separation that only the second box\u2019s sides show', () => {
    // A 2 x 2 square at the origin and a 45-degree square centred at (2.2, 2.2). Seen along x or along y the two overlap, so the
    // first box's own sides cannot tell them apart; along the diagonal they are 4.4 / sqrt(2) - 1 - sqrt(2) apart.
    const expected = 4.4 / Math.SQRT2 - 1 - Math.SQRT2;
    const a = box(0, 0, 2, 2);
    const b = box(2.2, 2.2, 2, 2, Math.PI / 4);
    expect(boxGap(a, b)).toBeCloseTo(expected, 10);
    expect(boxGap(b, a)).toBeCloseTo(expected, 10);
  });
});

/* A track at one place for the frames listed, each with the same box. */
function stationary(frames: number[], at: Point, length = 4, width = 2, heading = 0): TrackGeometry {
  return {
    path: frames.map(() => [at[0], at[1]]),
    timesteps: frames,
    headings: frames.map(() => heading),
    length_m: length,
    width_m: width,
  };
}
const frames = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, i) => from + i);

describe('collisionAnatomy', () => {
  it('measures the gap on the frame before the collision frame', () => {
    // The challenger sits 0.5 above the SDC's box until the collision frame, and on it from then.
    const sdc = stationary(frames(0, 30), [0, 0]);
    const challenger: TrackGeometry = {
      ...stationary(frames(0, 30), [0, 2.5]),
      path: frames(0, 30).map((f) => (f < 25 ? [0, 2.5] : [0, 1])),
    };
    const out = collisionAnatomy(sdc, challenger, 25);
    expect(out).toMatchObject({ kind: 'gap', frame: 24, collisionFrame: 25 });
    if (out.kind === 'gap') expect(out.gap).toBeCloseTo(0.5, 12);
  });

  it('uses the stored size for that frame when there are per-frame sizes', () => {
    const challenger: TrackGeometry = {
      ...stationary(frames(0, 30), [0, 2.5]),
      lengths_m: frames(0, 30).map(() => 4),
      widths_m: frames(0, 30).map((f) => (f === 24 ? 2.6 : 2)), // wider on the frame before only
    };
    const out = collisionAnatomy(stationary(frames(0, 30), [0, 0]), challenger, 25);
    expect(out.kind === 'gap' && out.gap).toBeCloseTo(0.2, 12);
  });

  it('is 0 when the boxes already touch or overlap on the frame before', () => {
    const out = collisionAnatomy(stationary(frames(0, 30), [0, 0]), stationary(frames(0, 30), [0, 1.5]), 25);
    expect(out).toEqual({ kind: 'gap', frame: 24, collisionFrame: 25, gap: 0 });
  });

  it('works for a collision on the last frame the tracks have', () => {
    const out = collisionAnatomy(stationary(frames(0, 90), [0, 0]), stationary(frames(0, 90), [0, 2.5]), 90);
    expect(out.kind === 'gap' && out.frame).toBe(89);
    expect(out.kind === 'gap' && out.gap).toBeCloseTo(0.5, 12);
  });

  it('has no earlier frame for a collision on the first frame', () => {
    const out = collisionAnatomy(stationary(frames(0, 30), [0, 0]), stationary(frames(0, 30), [0, 2.5]), 0);
    expect(out).toEqual({ kind: 'no_earlier_frame', collisionFrame: 0 });
  });

  it('has no earlier frame when one agent is first observed at the collision frame', () => {
    expect(collisionAnatomy(stationary(frames(0, 30), [0, 0]), stationary(frames(25, 30), [0, 2.5]), 25)).toEqual({
      kind: 'no_earlier_frame',
      collisionFrame: 25,
    });
    expect(collisionAnatomy(stationary(frames(25, 30), [0, 0]), stationary(frames(0, 30), [0, 2.5]), 25).kind).toBe(
      'no_earlier_frame',
    );
  });

  it('does not stand in an earlier frame when one agent was not observed on the frame before', () => {
    const challenger = stationary([...frames(0, 23), ...frames(25, 30)], [0, 2.5]); // no frame 24
    expect(collisionAnatomy(stationary(frames(0, 30), [0, 0]), challenger, 25)).toEqual({
      kind: 'not_observed',
      frame: 24,
      collisionFrame: 25,
    });
  });

  it('says so when no box size is stored for the frame before', () => {
    const challenger: TrackGeometry = {
      ...stationary(frames(0, 30), [0, 2.5]),
      lengths_m: frames(0, 30).map((f) => (f === 24 ? null : 4)),
      widths_m: frames(0, 30).map(() => 2),
    };
    expect(collisionAnatomy(stationary(frames(0, 30), [0, 0]), challenger, 25)).toEqual({
      kind: 'no_size',
      frame: 24,
      collisionFrame: 25,
    });
    expect(collisionAnatomy({ ...stationary(frames(0, 30), [0, 0]), length_m: null }, stationary(frames(0, 30), [0, 2.5]), 25).kind).toBe(
      'no_size',
    );
  });

  it('treats a track with no frames as never observed', () => {
    expect(collisionAnatomy(stationary([], [0, 0]), stationary(frames(0, 30), [0, 2.5]), 25).kind).toBe('no_earlier_frame');
  });
});

it('the precision is about a millimetre', () => {
  expect(POSITION_PRECISION_M).toBe(0.001);
});
