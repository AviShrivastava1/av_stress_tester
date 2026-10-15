import { indexAt, type TrackGeometry } from './geometry';

/*
 * The distance between the logged challenger and the perturbed challenger, frame by frame. Both tracks come from GET /perturbed
 * (`baseline`, the challenger as logged, and `perturbed`, the track simulated from the stored perturbation); only their
 * positions are used, in the planar metres of the data. A frame counts only where BOTH tracks were observed: a frame with no
 * vertex in one of them is skipped, never filled from the frame beside it (the same rule as scene/geometry.ts poseAt).
 * Nothing is simulated here: both tracks are stored.
 */

export interface FrameOffset {
  frame: number;
  metres: number;
}

export function frameOffsets(logged: TrackGeometry, perturbed: TrackGeometry): FrameOffset[] {
  const offsets: FrameOffset[] = [];
  for (let i = 0; i < logged.timesteps.length; i++) {
    const frame = logged.timesteps[i]!;
    const j = indexAt(perturbed, frame);
    if (j === null) continue;
    const a = logged.path[i]!;
    const b = perturbed.path[j]!;
    offsets.push({ frame, metres: Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!) });
  }
  return offsets.sort((x, y) => x.frame - y.frame);
}

export interface OffsetSummary {
  count: number;
  first: FrameOffset | null;
  last: FrameOffset | null;
  /** The largest offset; on a tie, the earlier frame. */
  largest: FrameOffset | null;
  /** The offset on the collision frame, when that is a frame both tracks share. */
  atCollision: FrameOffset | null;
}

export function summariseOffsets(offsets: readonly FrameOffset[], collisionFrame: number | null): OffsetSummary {
  let largest: FrameOffset | null = null;
  for (const o of offsets) if (largest === null || o.metres > largest.metres) largest = o;
  return {
    count: offsets.length,
    first: offsets[0] ?? null,
    last: offsets[offsets.length - 1] ?? null,
    largest,
    atCollision: collisionFrame === null ? null : (offsets.find((o) => o.frame === collisionFrame) ?? null),
  };
}
