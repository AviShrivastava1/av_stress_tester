/**
 * Pure scene geometry. No React, no DOM — everything the scene view draws is decided
 * here, so it can be tested directly and reused unchanged by playback.
 *
 * Coordinates are WOMD's local planar frame in metres: y points UP and headings are
 * radians counter-clockwise from +x. SVG's y points DOWN. `toSvg` is the one place
 * that flips it; drawing world coordinates straight into SVG would mirror the scene,
 * and every left turn would render as a right turn.
 */

export type Point = readonly [number, number];

/** The fields of an AgentTrack that geometry needs. */
export interface TrackGeometry {
  path: number[][];
  timesteps: number[];
  headings: number[];
  length_m?: number | null;
  width_m?: number | null;
}

export interface Pose {
  x: number;
  y: number;
  heading: number;
}

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** World (y-up) to SVG user space (y-down). */
export function toSvg([x, y]: Point): Point {
  return [x, -y];
}

function pointAt(track: TrackGeometry, i: number): Point {
  const p = track.path[i]!;
  return [p[0]!, p[1]!];
}

/**
 * The track as runs of consecutive frames. A validity gap starts a new run and is
 * never bridged: the exporter stores only observed frames and a line across the gap
 * would draw a position nobody observed.
 */
export function segments(track: TrackGeometry): Point[][] {
  const runs: Point[][] = [];
  let current: Point[] = [];
  for (let i = 0; i < track.path.length; i++) {
    if (i > 0 && track.timesteps[i]! - track.timesteps[i - 1]! !== 1) {
      runs.push(current);
      current = [];
    }
    current.push(pointAt(track, i));
  }
  if (current.length > 0) runs.push(current);
  return runs;
}

/**
 * Where the agent was AT `frame`, or null if it was not observed then. Exact lookup,
 * no interpolation: `timesteps` is ascending, so a binary search suffices.
 */
export function poseAt(track: TrackGeometry, frame: number): Pose | null {
  let lo = 0;
  let hi = track.timesteps.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const t = track.timesteps[mid]!;
    if (t === frame) {
      const [x, y] = pointAt(track, mid);
      return { x, y, heading: track.headings[mid]! };
    }
    if (t < frame) lo = mid + 1;
    else hi = mid - 1;
  }
  return null;
}

function usable(v: number | null | undefined): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v > 0;
}

/**
 * Both dimensions present, finite and positive. Anything else draws as a dot: a box
 * with a guessed or zero side would assert a size nobody measured.
 */
export function hasDimensions(track: TrackGeometry): boolean {
  return usable(track.length_m) && usable(track.width_m);
}

/** The rotated rectangle's corners in world coordinates, front-left first. */
export function footprint(pose: Pose, length: number, width: number): Point[] {
  const c = Math.cos(pose.heading);
  const s = Math.sin(pose.heading);
  const hl = length / 2;
  const hw = width / 2;
  const corner = (dl: number, dw: number): Point => [
    pose.x + dl * c - dw * s,
    pose.y + dl * s + dw * c,
  ];
  return [corner(hl, hw), corner(-hl, hw), corner(-hl, -hw), corner(hl, -hw)];
}

/** The midpoint of the footprint's front edge — where the heading tick ends. */
export function frontCentre(pose: Pose, length: number): Point {
  return [
    pose.x + (length / 2) * Math.cos(pose.heading),
    pose.y + (length / 2) * Math.sin(pose.heading),
  ];
}

/** Bounds of every vertex of every track, in world coordinates; null if empty. */
export function trackBounds(tracks: TrackGeometry[]): Bounds | null {
  let b: Bounds | null = null;
  for (const t of tracks) {
    for (let i = 0; i < t.path.length; i++) {
      const [x, y] = pointAt(t, i);
      if (b === null) b = { minX: x, minY: y, maxX: x, maxY: y };
      else {
        b.minX = Math.min(b.minX, x);
        b.minY = Math.min(b.minY, y);
        b.maxX = Math.max(b.maxX, x);
        b.maxY = Math.max(b.maxY, y);
      }
    }
  }
  return b;
}

/**
 * Grow bounds so footprints at the edge are not clipped and a stationary agent does
 * not produce a zero-size view. `minSpan` is metres.
 */
export function padBounds(b: Bounds, margin: number, minSpan = 20): Bounds {
  const cx = (b.minX + b.maxX) / 2;
  const cy = (b.minY + b.maxY) / 2;
  const halfW = Math.max(b.maxX - b.minX + 2 * margin, minSpan) / 2;
  const halfH = Math.max(b.maxY - b.minY + 2 * margin, minSpan) / 2;
  return { minX: cx - halfW, minY: cy - halfH, maxX: cx + halfW, maxY: cy + halfH };
}

/** SVG viewBox for world bounds, after the y-flip. */
export function viewBox(b: Bounds): { x: number; y: number; width: number; height: number } {
  return { x: b.minX, y: -b.maxY, width: b.maxX - b.minX, height: b.maxY - b.minY };
}

/** A 1/2/5 x 10^k length close to a fifth of the span, for the scale bar. */
export function niceScaleLength(span: number): number {
  const target = span / 5;
  if (!(target > 0) || !Number.isFinite(target)) return 1;
  const pow = 10 ** Math.floor(Math.log10(target));
  const m = target / pow;
  const step = m >= 5 ? 5 : m >= 2 ? 2 : 1;
  return step * pow;
}

export type ReferenceFrame =
  | { kind: 'collision'; frame: number }
  | { kind: 'all_observed'; frame: number }
  | { kind: 'earliest_observed'; frame: number }
  | { kind: 'none' };

/**
 * The single frame the static view draws footprints at.
 *
 *   1. The collision frame, when the search found one.
 *   2. Otherwise the earliest frame at which EVERY track in the focus set is observed
 *      — so a late-starting challenger is not missing from the first view.
 *   3. Otherwise (no frame is shared by all of them) the earliest frame at which ANY
 *      is observed, reported as such rather than passed off as case 2.
 */
export function chooseReferenceFrame(
  collisionFrame: number | null | undefined,
  focus: TrackGeometry[],
): ReferenceFrame {
  if (collisionFrame !== null && collisionFrame !== undefined) {
    return { kind: 'collision', frame: collisionFrame };
  }
  const nonEmpty = focus.filter((t) => t.timesteps.length > 0);
  if (nonEmpty.length === 0) return { kind: 'none' };

  const [first, ...rest] = nonEmpty;
  const others = rest.map((t) => new Set(t.timesteps));
  for (const t of first!.timesteps) {
    if (others.every((s) => s.has(t))) return { kind: 'all_observed', frame: t };
  }
  const earliest = Math.min(...nonEmpty.map((t) => t.timesteps[0]!));
  return { kind: 'earliest_observed', frame: earliest };
}
