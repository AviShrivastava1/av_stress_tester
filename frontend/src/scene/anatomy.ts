/**
 * The gap between the two footprints on the frame before first contact, computed here from the stored boxes.
 *
 * Nothing in this file is stored by the search. The stored data is each agent's position and heading at each observed frame
 * and its box size (the per-frame length and width where the export wrote them, otherwise the scalar size at the first
 * observed frame); `observationAt` (scene/geometry.ts) reads those exactly as the scene draws them, and `footprint` turns
 * each into the oriented rectangle the collision check built from the same numbers (src/danger/collision_detector.py
 * get_corners). The gap is the shortest distance between the two rectangles, or 0 when they touch or overlap.
 *
 * WHY THE FRAME BEFORE. `collision_timestep` is the first frame at which the check reported contact (scenario_scores
 * .collision_timestep, the first colliding t). The frame before it is the last frame the check found no contact, so the
 * gap there is how far apart the footprints were at that point. The frames are whole numbers a tenth of a second apart
 * (scene/playback.ts), so "before" is exactly `collisionFrame - 1`; a frame at which an agent was not observed is not
 * replaced by an earlier one.
 *
 * HOW PRECISE. Contact is decided on the stored positions, which are written with repr(float(...)) and so carry full binary64
 * precision, but the positions themselves are WOMD measurements; the app states contact "up to about a millimetre at the
 * coordinates in this data" (pages/detail/ScenePanel.tsx). A gap below that is reported as below that precision, not as a
 * number: POSITION_PRECISION_M.
 *
 * NO OVERLAP POLYGON. When the boxes touch or overlap the gap is 0 and nothing else is computed: how deep one box sits in
 * the other would be a claim finer than the data.
 */
import { footprint, observationAt, type Point, type TrackGeometry } from './geometry';

/** About a millimetre: the precision to which contact is decided on the stored positions. */
export const POSITION_PRECISION_M = 0.001;

/** Separation smaller than this is rounding error in the arithmetic, not a gap. */
const EPSILON = 1e-9;

function sub(a: Point, b: Point): Point {
  return [a[0] - b[0], a[1] - b[1]];
}

function dot(a: Point, b: Point): number {
  return a[0] * b[0] + a[1] * b[1];
}

/** Distance from point p to the segment ab. */
function distanceToSegment(p: Point, a: Point, b: Point): number {
  const ab = sub(b, a);
  const ap = sub(p, a);
  const lengthSquared = dot(ab, ab);
  const t = lengthSquared === 0 ? 0 : Math.max(0, Math.min(1, dot(ap, ab) / lengthSquared));
  return Math.hypot(p[0] - (a[0] + t * ab[0]), p[1] - (a[1] + t * ab[1]));
}

/** Shadow of a polygon on an axis: its smallest and largest projection. */
function shadow(polygon: readonly Point[], axis: Point): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of polygon) {
    const d = dot(p, axis);
    lo = Math.min(lo, d);
    hi = Math.max(hi, d);
  }
  return [lo, hi];
}

/** True when no edge normal of either polygon separates them: they touch or overlap (separating-axis test, convex shapes). */
function touchOrOverlap(a: readonly Point[], b: readonly Point[]): boolean {
  for (const polygon of [a, b]) {
    for (let i = 0; i < polygon.length; i++) {
      const edge = sub(polygon[(i + 1) % polygon.length]!, polygon[i]!);
      const length = Math.hypot(edge[0], edge[1]);
      if (length === 0) continue;
      const axis: Point = [-edge[1] / length, edge[0] / length];
      const [aLo, aHi] = shadow(a, axis);
      const [bLo, bHi] = shadow(b, axis);
      if (aLo - bHi > EPSILON || bLo - aHi > EPSILON) return false;
    }
  }
  return true;
}

/**
 * The shortest distance between two convex polygons, in the units of their coordinates; 0 when they touch or overlap.
 * Two convex polygons that do not touch are closest at a vertex of one and a point on an edge of the other, so the
 * minimum over every vertex-to-edge pair, both ways, is exact.
 */
export function boxGap(a: readonly Point[], b: readonly Point[]): number {
  if (touchOrOverlap(a, b)) return 0;
  let best = Infinity;
  for (const [from, to] of [[a, b], [b, a]] as const) {
    for (const p of from) {
      for (let i = 0; i < to.length; i++) {
        best = Math.min(best, distanceToSegment(p, to[i]!, to[(i + 1) % to.length]!));
      }
    }
  }
  return best;
}

export type Anatomy =
  /** Both boxes could be drawn on the frame before; `gap` is metres, 0 when they already touch or overlap there. */
  | { kind: 'gap'; frame: number; collisionFrame: number; gap: number }
  /** One of the two agents is first observed at the collision frame (or later), so no earlier frame has both. */
  | { kind: 'no_earlier_frame'; collisionFrame: number }
  /** Not both agents were observed on the frame before the collision frame. */
  | { kind: 'not_observed'; frame: number; collisionFrame: number }
  /** Both were observed, but no box size is stored for one of them on that frame. */
  | { kind: 'no_size'; frame: number; collisionFrame: number };

/** What can be said about the colliding pair on the frame before `collisionFrame`. */
export function collisionAnatomy(
  sdc: TrackGeometry,
  challenger: TrackGeometry,
  collisionFrame: number,
): Anatomy {
  const firstObserved = (t: TrackGeometry) => (t.timesteps.length === 0 ? Infinity : t.timesteps[0]!);
  if (firstObserved(sdc) >= collisionFrame || firstObserved(challenger) >= collisionFrame) {
    return { kind: 'no_earlier_frame', collisionFrame };
  }
  const frame = collisionFrame - 1;
  const first = observationAt(sdc, frame);
  const second = observationAt(challenger, frame);
  if (first === null || second === null) return { kind: 'not_observed', frame, collisionFrame };
  if (first.size === null || second.size === null) return { kind: 'no_size', frame, collisionFrame };
  const gap = boxGap(
    footprint(first.pose, first.size.length, first.size.width),
    footprint(second.pose, second.size.length, second.size.width),
  );
  return { kind: 'gap', frame, collisionFrame, gap };
}
