import { describe, expect, it } from 'vitest';
import { makeTrack, range } from '../test/fixtures';
import { advance, clampFrame, frameRange, isStatic, MAX_TICK_MS, startFrame } from './playback';

const r = { first: 0, last: 90 };
const playing = (frame: number, carryMs = 0) => ({ frame, carryMs, playing: true });

describe('frameRange', () => {
  it('spans the union of every track, late starters included', () => {
    expect(frameRange([makeTrack({ frames: range(20, 41) }), makeTrack({ frames: range(5, 12) })])).toEqual({
      first: 5,
      last: 40,
    });
  });

  it('ignores tracks with no frames, and is null when none has any', () => {
    expect(frameRange([makeTrack({ frames: [] }), makeTrack({ frames: [7] })])).toEqual({ first: 7, last: 7 });
    expect(frameRange([makeTrack({ frames: [] })])).toBeNull();
    expect(frameRange([])).toBeNull();
  });

  it('treats zero or one frame as static', () => {
    expect(isStatic(null)).toBe(true);
    expect(isStatic({ first: 7, last: 7 })).toBe(true);
    expect(isStatic({ first: 7, last: 8 })).toBe(false);
  });
});

describe('advance', () => {
  it('steps whole frames at the rate and carries the remainder', () => {
    // 20 frames/s = 50 ms per frame: 80 ms is one frame with 30 ms carried, and
    // the carried 30 ms plus the next 90 ms is two more with 20 ms carried.
    expect(advance(playing(10), 80, 20, r)).toEqual({ frame: 11, carryMs: 30, playing: true });
    expect(advance(playing(11, 30), 90, 20, r)).toEqual({ frame: 13, carryMs: 20, playing: true });
  });

  it('never produces a fractional frame', () => {
    const next = advance(playing(0), 37, 30, r);
    expect(Number.isInteger(next.frame)).toBe(true);
  });

  it('runs faster at a higher rate', () => {
    expect(advance(playing(0), 100, 20, r).frame).toBe(2);
    expect(advance(playing(0), 100, 5, r).frame).toBe(0);
  });

  it('caps a single tick, so a backgrounded tab does not jump the playhead', () => {
    expect(advance(playing(0), 60_000, 30, r).frame).toBe(Math.floor((MAX_TICK_MS * 30) / 1000));
  });

  it('ignores negative time', () => {
    expect(advance(playing(5), -500, 10, r)).toEqual(playing(5));
  });

  it('stops on the last frame', () => {
    expect(advance(playing(89), 100, 30, r)).toEqual({ frame: 90, carryMs: 0, playing: false });
  });

  it('does nothing while paused', () => {
    const paused = { frame: 3, carryMs: 0, playing: false };
    expect(advance(paused, 1000, 10, r)).toBe(paused);
  });
});

describe('startFrame and clampFrame', () => {
  it('restarts from the first frame when play is pressed at the end', () => {
    expect(startFrame(90, r)).toBe(0);
    expect(startFrame(40, r)).toBe(40);
  });

  it('keeps a seek inside the range and on a whole frame', () => {
    expect(clampFrame(-3, r)).toBe(0);
    expect(clampFrame(120, r)).toBe(90);
    expect(clampFrame(12.6, r)).toBe(13);
  });
});
