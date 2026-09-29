/**
 * Pure playback clock. No React, no timers — the hook in usePlayback.ts drives it
 * from requestAnimationFrame, and the tests drive it with plain numbers.
 *
 * WHOLE FRAMES ONLY. The playhead is always an integer frame, and each drawn pose
 * comes from `poseAt` unchanged, so every state on screen is a state that was
 * observed. Interpolating between frames is not done even where it would not
 * bridge a validity gap: check_collision_trajectory is the project's "arbiter of
 * truth: approximate to search, exact to decide" (Block 3 v6, Concept 9), and it
 * decides on whole frames only. An in-between image would show a collision-geometry
 * state the pipeline never evaluated. The cost is stepped motion at low rates.
 */
import type { TrackGeometry } from './geometry';

export interface FrameRange {
  first: number;
  last: number;
}

/** Playback rates offered, in frames per second. Never labelled as seconds. */
export const RATES = [5, 10, 20, 30] as const;
export type Rate = (typeof RATES)[number];
export const DEFAULT_RATE: Rate = 10;

/**
 * The most time one tick may add. A backgrounded tab stops requestAnimationFrame;
 * without a cap, the first tick after returning would jump the playhead by however
 * long the tab was away.
 */
export const MAX_TICK_MS = 100;

/** First to last frame observed by ANY track; null when no track has a frame. */
export function frameRange(tracks: TrackGeometry[]): FrameRange | null {
  let first = Infinity;
  let last = -Infinity;
  for (const t of tracks) {
    if (t.timesteps.length === 0) continue;
    first = Math.min(first, t.timesteps[0]!);
    last = Math.max(last, t.timesteps[t.timesteps.length - 1]!);
  }
  return Number.isFinite(first) ? { first, last } : null;
}

/** Nothing to move through: zero or one frame. */
export function isStatic(range: FrameRange | null): boolean {
  return range === null || range.first === range.last;
}

export function clampFrame(frame: number, range: FrameRange): number {
  return Math.min(range.last, Math.max(range.first, Math.round(frame)));
}

export interface Clock {
  frame: number;
  /** Elapsed time not yet spent on a whole frame. */
  carryMs: number;
  playing: boolean;
}

/** Where playback starts when play is pressed: the current frame, or the first if at the end. */
export function startFrame(frame: number, range: FrameRange): number {
  return frame >= range.last ? range.first : clampFrame(frame, range);
}

/**
 * Move the clock forward by `dtMs` at `rate` frames per second. Whole frames only;
 * the remainder carries into the next tick. Stops, and stays, on the last frame.
 */
export function advance(clock: Clock, dtMs: number, rate: number, range: FrameRange): Clock {
  if (!clock.playing) return clock;
  const dt = Math.min(Math.max(dtMs, 0), MAX_TICK_MS);
  const stepMs = 1000 / rate;
  const elapsed = clock.carryMs + dt;
  const steps = Math.floor(elapsed / stepMs);
  const frame = clock.frame + steps;
  if (frame >= range.last) return { frame: range.last, carryMs: 0, playing: false };
  return { frame, carryMs: elapsed - steps * stepMs, playing: true };
}
