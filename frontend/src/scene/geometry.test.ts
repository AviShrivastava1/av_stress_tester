import { describe, expect, it } from 'vitest';
import { makeTrack, range } from '../test/fixtures';
import {
  chooseReferenceFrame,
  footprint,
  frontCentre,
  hasDimensions,
  niceScaleLength,
  padBounds,
  poseAt,
  segments,
  toSvg,
  trackBounds,
  viewBox,
} from './geometry';

describe('handedness', () => {
  it('flips y so world-up is screen-up', () => {
    expect(toSvg([3, 4])).toEqual([3, -4]);
  });

  it('draws a heading of pi/2 pointing up the screen', () => {
    const pose = { x: 0, y: 0, heading: Math.PI / 2 };
    const [sx, sy] = toSvg(frontCentre(pose, 4));
    expect(sx).toBeCloseTo(0);
    expect(sy).toBeCloseTo(-2); // SVG y decreases upward
  });

  it('turns counter-clockwise for positive heading, as the WOMD frame does', () => {
    // heading pi/4 from +x: front is up and to the right on screen
    const [sx, sy] = toSvg(frontCentre({ x: 0, y: 0, heading: Math.PI / 4 }, 2));
    expect(sx).toBeGreaterThan(0);
    expect(sy).toBeLessThan(0);
  });
});

describe('footprint', () => {
  it('is an axis-aligned rectangle of the given size at heading 0', () => {
    const corners = footprint({ x: 10, y: 5, heading: 0 }, 4, 2);
    expect(corners).toEqual([
      [12, 6],
      [8, 6],
      [8, 4],
      [12, 4],
    ]);
  });

  it('swaps its extent when rotated a quarter turn', () => {
    const corners = footprint({ x: 0, y: 0, heading: Math.PI / 2 }, 4, 2);
    const xs = corners.map((c) => c[0]);
    const ys = corners.map((c) => c[1]);
    expect(Math.max(...xs) - Math.min(...xs)).toBeCloseTo(2);
    expect(Math.max(...ys) - Math.min(...ys)).toBeCloseTo(4);
  });
});

describe('hasDimensions', () => {
  it('needs both dimensions finite and positive', () => {
    expect(hasDimensions(makeTrack())).toBe(true);
    for (const bad of [null, undefined, 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(hasDimensions(makeTrack({ length_m: bad }))).toBe(false);
      expect(hasDimensions(makeTrack({ width_m: bad }))).toBe(false);
    }
  });
});

describe('segments', () => {
  it('breaks the path at a validity gap instead of bridging it', () => {
    const runs = segments(makeTrack({ frames: [0, 1, 2, 5, 6] }));
    expect(runs.map((r) => r.length)).toEqual([3, 2]);
    expect(runs[0]!.at(-1)).toEqual([2, 0]);
    expect(runs[1]![0]).toEqual([5, 0]);
  });

  it('keeps an isolated observation as its own one-point run', () => {
    expect(segments(makeTrack({ frames: [0, 1, 4, 7, 8] })).map((r) => r.length)).toEqual([2, 1, 2]);
  });

  it('returns no runs for an empty track', () => {
    expect(segments(makeTrack({ frames: [] }))).toEqual([]);
  });
});

describe('poseAt', () => {
  const track = makeTrack({ frames: [0, 1, 2, 5, 6], heading: 0 });

  it('returns the exact observed pose', () => {
    expect(poseAt(track, 5)).toEqual({ x: 5, y: 0, heading: 0 });
  });

  it('does not interpolate into a gap or extrapolate past the ends', () => {
    expect(poseAt(track, 3)).toBeNull();
    expect(poseAt(track, -1)).toBeNull();
    expect(poseAt(track, 7)).toBeNull();
  });
});

describe('bounds', () => {
  it('covers every vertex of every track', () => {
    const b = trackBounds([makeTrack({ frames: [0, 1, 2] }), makeTrack({ frames: [0, 1], origin: [-5, 10] })]);
    expect(b).toEqual({ minX: -5, minY: 0, maxX: 2, maxY: 10 });
  });

  it('is null when there is nothing to bound', () => {
    expect(trackBounds([])).toBeNull();
    expect(trackBounds([makeTrack({ frames: [] })])).toBeNull();
  });

  it('never produces a zero-size view for a stationary agent', () => {
    const b = padBounds({ minX: 3, minY: 3, maxX: 3, maxY: 3 }, 0, 20);
    expect(b.maxX - b.minX).toBe(20);
    expect(b.maxY - b.minY).toBe(20);
  });

  it('maps world bounds to a y-flipped viewBox', () => {
    expect(viewBox({ minX: -1, minY: 2, maxX: 5, maxY: 10 })).toEqual({ x: -1, y: -10, width: 6, height: 8 });
  });
});

describe('niceScaleLength', () => {
  it.each([
    [100, 20],
    [37, 5],
    [1000, 200],
    [12, 2],
  ])('span %d m -> %d m bar', (span, bar) => {
    expect(niceScaleLength(span)).toBe(bar);
  });
});

describe('chooseReferenceFrame', () => {
  const sdc = makeTrack({ frames: range(0, 41) });
  const lateChallenger = makeTrack({ frames: range(20, 41) });

  it('uses the collision frame when there is one', () => {
    expect(chooseReferenceFrame(25, [sdc, lateChallenger])).toEqual({ kind: 'collision', frame: 25 });
  });

  it('treats a collision at frame 0 as a collision, not as absent', () => {
    expect(chooseReferenceFrame(0, [sdc])).toEqual({ kind: 'collision', frame: 0 });
  });

  it('picks the earliest frame at which every focused track is observed', () => {
    expect(chooseReferenceFrame(null, [sdc, lateChallenger])).toEqual({ kind: 'all_observed', frame: 20 });
  });

  it('reports the fallback as a fallback when no frame is shared', () => {
    const early = makeTrack({ frames: range(5, 10) });
    const late = makeTrack({ frames: range(20, 30) });
    expect(chooseReferenceFrame(null, [late, early])).toEqual({ kind: 'earliest_observed', frame: 5 });
  });

  it('ignores tracks with no observations when choosing', () => {
    expect(chooseReferenceFrame(null, [makeTrack({ frames: [] }), lateChallenger])).toEqual({
      kind: 'all_observed',
      frame: 20,
    });
  });

  it('has nothing to choose from an empty focus set', () => {
    expect(chooseReferenceFrame(null, [])).toEqual({ kind: 'none' });
  });
});
