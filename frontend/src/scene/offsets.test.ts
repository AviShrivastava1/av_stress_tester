import { describe, expect, it } from 'vitest';
import type { TrackGeometry } from './geometry';
import { frameOffsets, summariseOffsets } from './offsets';

/*
 * The distance between the logged challenger and the perturbed challenger on each frame where both were observed
 * (scene/offsets.ts). Positions only, in metres, taken from the two tracks the API serves.
 */

function track(frames: number[], at: (frame: number) => [number, number]): TrackGeometry {
  return { path: frames.map(at), timesteps: frames, headings: frames.map(() => 0) };
}
const range = (a: number, b: number) => Array.from({ length: b - a + 1 }, (_, i) => a + i);

describe('frameOffsets', () => {
  it('is the straight-line distance between the two positions on each shared frame', () => {
    const logged = track(range(0, 3), (f) => [f, 0]);
    const perturbed = track(range(0, 3), (f) => [f + 3, 4]); // 3 along and 4 across: a 3-4-5 triangle on every frame
    expect(frameOffsets(logged, perturbed)).toEqual([0, 1, 2, 3].map((frame) => ({ frame, metres: 5 })));
  });

  it('is 0 where the two are on the same spot', () => {
    const a = track(range(0, 2), () => [10, -4]);
    expect(frameOffsets(a, a).map((o) => o.metres)).toEqual([0, 0, 0]);
  });

  it('is the same from either side', () => {
    const a = track(range(0, 4), (f) => [f * 2, f]);
    const b = track(range(0, 4), (f) => [f * 2 + 1, f * 3]);
    expect(frameOffsets(a, b)).toEqual(frameOffsets(b, a));
  });

  it('counts only the frames both were observed on, and never stands in a neighbouring frame', () => {
    const logged = track([0, 1, 2, 5, 6], (f) => [f, 0]);
    const perturbed = track([1, 2, 3, 4, 5], (f) => [f, 1]);
    expect(frameOffsets(logged, perturbed)).toEqual([
      { frame: 1, metres: 1 },
      { frame: 2, metres: 1 },
      { frame: 5, metres: 1 },
    ]);
  });

  it('is empty when the two share no frame, or one has none', () => {
    expect(frameOffsets(track([0, 1], () => [0, 0]), track([5, 6], () => [0, 0]))).toEqual([]);
    expect(frameOffsets(track([], () => [0, 0]), track([0, 1], () => [0, 0]))).toEqual([]);
  });

  it('keeps the frames in order, whatever the order of the tracks’ own frames', () => {
    const offsets = frameOffsets(track(range(3, 9), (f) => [0, f]), track(range(0, 6), (f) => [0, f * 2]));
    expect(offsets.map((o) => o.frame)).toEqual([3, 4, 5, 6]);
    expect(offsets.map((o) => o.metres)).toEqual([3, 4, 5, 6]);
  });
});

describe('summariseOffsets', () => {
  const offsets = [
    { frame: 4, metres: 0.5 },
    { frame: 5, metres: 2 },
    { frame: 6, metres: 2 },
    { frame: 7, metres: 1.25 },
  ];

  it('names the offset on the collision frame, the largest, the first and the last', () => {
    expect(summariseOffsets(offsets, 6)).toEqual({
      count: 4,
      first: { frame: 4, metres: 0.5 },
      last: { frame: 7, metres: 1.25 },
      largest: { frame: 5, metres: 2 }, // the earlier frame of a tie
      atCollision: { frame: 6, metres: 2 },
    });
  });

  it('has no collision entry when there is no collision frame, or it is not a shared frame', () => {
    expect(summariseOffsets(offsets, null).atCollision).toBeNull();
    expect(summariseOffsets(offsets, 9).atCollision).toBeNull();
    expect(summariseOffsets(offsets, 3).atCollision).toBeNull();
  });

  it('has nothing to name when there are no offsets', () => {
    expect(summariseOffsets([], 5)).toEqual({ count: 0, first: null, last: null, largest: null, atCollision: null });
  });

  it('copes with a single offset', () => {
    const one = [{ frame: 9, metres: 3 }];
    expect(summariseOffsets(one, 9)).toEqual({ count: 1, first: one[0], last: one[0], largest: one[0], atCollision: one[0] });
  });
});
