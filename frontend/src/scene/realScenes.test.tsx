import { cleanup, render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { AgentTrack } from '../api/client';
import fixtureFile from '../test/realCollisionFrames.json';
import { footprint, observationAt, type Point } from './geometry';
import type { DrawnTrack } from './sceneModel';
import { SceneView } from './SceneView';

/*
 * The real scenes behind the box-size finding.
 *
 * realCollisionFrames.json holds, for each of the 18 scenarios that have a perturbed path, the SDC
 * and the perturbed challenger around the collision frame, exactly as the API returned them (its
 * _provenance says how). The search's collision check builds each box from the size recorded at that
 * frame and asks Shapely's `intersects`, which counts touching as a collision and has no tolerance
 * (src/danger/collision_detector.py: `return poly_a.intersects(poly_b)`).
 *
 * The geometry below is this file's own, independent of the app: a separating-axis test for overlap and an
 * exact point-to-segment distance for boxes that are apart. The boxes themselves come from the app's
 * own observationAt + footprint, so the code path under test is the code path drawn.
 */

interface Entry {
  scenario_id: string;
  collision_timestep: number;
  sdc: AgentTrack;
  perturbed: AgentTrack;
}
const FIXTURE = fixtureFile as unknown as { _provenance: Record<string, string>; scenarios: Entry[] };

// The nine whose boxes, drawn at the scalar sizes, do not touch at the collision frame (the review's
// check 5: "intersect under (ii) only"). Pinned from the review, not computed here.
const SCALAR_GAP = new Set([
  '38c703d6cb4da5ce', '6725f3566168ce34', '78afd0d26d19ef41', '85333d34425536d9', '888d2ac2b622078b',
  '91a2a6ad0c50ce55', 'a6bf1ade33cb6e93', 'b1e5a34520fe6043', 'b262200be0d2b7f0',
]);

// ── an independent separating-axis test and polygon distance ────────────────────────────────

type Poly = readonly (readonly [number, number])[];

function normals(p: Poly): [number, number][] {
  return p.map((a, i) => {
    const b = p[(i + 1) % p.length]!;
    const ex = b[0] - a[0];
    const ey = b[1] - a[1];
    const n = Math.hypot(ex, ey);
    return [-ey / n, ex / n];
  });
}

/** Largest gap between the boxes' shadows over every edge normal: > 0 apart, <= 0 touching or overlapping. */
function separation(a: Poly, b: Poly): number {
  let best = -Infinity;
  for (const [nx, ny] of [...normals(a), ...normals(b)]) {
    const pa = a.map(([x, y]) => nx! * x + ny! * y);
    const pb = b.map(([x, y]) => nx! * x + ny! * y);
    best = Math.max(best, Math.min(...pa) - Math.max(...pb), Math.min(...pb) - Math.max(...pa));
  }
  return best;
}

function pointToSegment(p: readonly [number, number], a: readonly [number, number], b: readonly [number, number]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

/** Exact distance between two boxes that do not overlap: the nearest vertex-to-edge pair. */
function distance(a: Poly, b: Poly): number {
  let best = Infinity;
  for (const [P, Q] of [[a, b], [b, a]] as const) {
    for (const v of P) {
      for (let i = 0; i < Q.length; i++) best = Math.min(best, pointToSegment(v, Q[i]!, Q[(i + 1) % Q.length]!));
    }
  }
  return best;
}

describe('the separating-axis test used below', () => {
  const square = (x: number, y: number): Poly => [[x + 1, y + 1], [x, y + 1], [x, y], [x + 1, y]];

  it('is positive for boxes apart, zero for boxes sharing an edge, negative for overlap', () => {
    expect(separation(square(0, 0), square(3, 0))).toBeCloseTo(2);
    expect(separation(square(0, 0), square(1, 0))).toBeCloseTo(0);
    expect(separation(square(0, 0), square(0.75, 0.5))).toBeCloseTo(-0.25);
  });

  it('is positive for boxes apart on a diagonal, and the exact distance is the corner-to-corner one', () => {
    expect(separation(square(0, 0), square(2, 2))).toBeGreaterThan(0);
    expect(distance(square(0, 0), square(2, 2))).toBeCloseTo(Math.SQRT2);
    expect(distance(square(0, 0), square(3, 0))).toBeCloseTo(2);
  });

  it('works for rotated boxes', () => {
    const c = Math.cos(Math.PI / 4);
    // a diamond centred at (2, 0) with half-diagonal 2c = 1.414: its left tip (0.586, 0) is inside the unit square
    const diamond: Poly = [[2 + 2 * c, 0], [2, 2 * c], [2 - 2 * c, 0], [2, -2 * c]];
    expect(separation(square(0, 0), diamond)).toBeLessThan(0);
    expect(separation(square(0, 0), diamond.map(([x, y]) => [x + 3, y] as [number, number]))).toBeGreaterThan(0);
  });
});

// ── the fixture ────────────────────────────────────────────────────────────────────────

describe('the captured fixture', () => {
  it('states where it came from', () => {
    const p = FIXTURE._provenance;
    expect(p.dump_sha256).toBe('84edda7ed5051ce0ee87996f5ca1e2383c2ebc5520e0316ffdfe8c7b01a4bd1d');
    expect(p.dump_produced_by_commit).toMatch(/62ab1a6/);
    expect(p.served_by_api_commit).toMatch(/^[0-9a-f]{40}$/);
    expect(p.values).toMatch(/Verbatim/);
    expect(p.database).toMatch(/av_rerun_dump_check/);
  });

  it('holds all 18 scenarios, each collision frame observed by both tracks, every array aligned with its path', () => {
    expect(FIXTURE.scenarios).toHaveLength(18);
    for (const e of FIXTURE.scenarios) {
      for (const t of [e.sdc, e.perturbed]) {
        expect(t.timesteps).toContain(e.collision_timestep);
        expect(t.lengths_m).toHaveLength(t.path.length);
        expect(t.widths_m).toHaveLength(t.path.length);
        expect(t.headings).toHaveLength(t.path.length);
      }
    }
    for (const id of SCALAR_GAP) expect(FIXTURE.scenarios.map((e) => e.scenario_id)).toContain(id);
  });
});

// ── the regression ────────────────────────────────────────────────────────────────────

const withoutArrays = (t: AgentTrack): AgentTrack => ({ ...t, lengths_m: null, widths_m: null });

function boxes(e: Entry, perFrame: boolean): { sdc: Poly; challenger: Poly } {
  const frame = e.collision_timestep;
  const box = (t: AgentTrack): Poly => {
    const o = observationAt(perFrame ? t : withoutArrays(t), frame)!;
    expect(o.size, `${e.scenario_id}: a box is drawn`).not.toBeNull();
    expect(o.source).toBe(perFrame ? 'per_frame' : 'scalar');
    return footprint(o.pose, o.size!.length, o.size!.width);
  };
  return { sdc: box(e.sdc), challenger: box(e.perturbed) };
}

describe('the boxes drawn at the collision frame', () => {
  const nine = FIXTURE.scenarios.filter((e) => SCALAR_GAP.has(e.scenario_id));
  const control = FIXTURE.scenarios.filter((e) => !SCALAR_GAP.has(e.scenario_id));

  it.each(nine.map((e) => [e.scenario_id, e] as const))(
    '%s: at the scalar sizes the boxes are apart (the bug)',
    (_id, e) => {
      const { sdc, challenger } = boxes(e, false);
      expect(separation(sdc, challenger)).toBeGreaterThan(0);
      expect(distance(sdc, challenger)).toBeGreaterThan(0);
    },
  );

  it('the nine are exactly the scenarios whose scalar boxes are apart', () => {
    const apart = FIXTURE.scenarios
      .filter((e) => { const b = boxes(e, false); return separation(b.sdc, b.challenger) > 0; })
      .map((e) => e.scenario_id)
      .sort();
    expect(apart).toEqual([...SCALAR_GAP].sort());
  });

  it.each(FIXTURE.scenarios.map((e) => [e.scenario_id, e] as const))(
    '%s: at the per-frame sizes the boxes overlap, which the verifier counts as a collision',
    (_id, e) => {
      const { sdc, challenger } = boxes(e, true);
      expect(separation(sdc, challenger)).toBeLessThan(0);
    },
  );

  it.each(control.map((e) => [e.scenario_id, e] as const))(
    '%s (control): the scalar boxes already overlapped, and still do',
    (_id, e) => {
      expect(separation(boxes(e, false).sdc, boxes(e, false).challenger)).toBeLessThan(0);
      expect(separation(boxes(e, true).sdc, boxes(e, true).challenger)).toBeLessThan(0);
    },
  );

  it('the per-frame sizes at those frames are not the scalars, for the challenger of every one of the nine', () => {
    for (const e of nine) {
      const o = observationAt(e.perturbed, e.collision_timestep)!;
      expect([o.size!.length, o.size!.width]).not.toEqual([e.perturbed.length_m, e.perturbed.width_m]);
    }
  });
});

// ── the component, at 888d2ac2b622078b ────────────────────────────────────────────────

describe('SceneView at the collision frame of 888d2ac2b622078b', () => {
  const entry = FIXTURE.scenarios.find((e) => e.scenario_id === '888d2ac2b622078b')!;
  const drawn: DrawnTrack[] = [
    { key: 'sdc', role: 'sdc', track: entry.sdc },
    { key: 'challenger-perturbed', role: 'challenger_perturbed', track: entry.perturbed },
  ];
  const parse = (points: string): [number, number][] =>
    points.split(' ').map((p) => p.split(',').map(Number) as [number, number]);

  it('draws the two colliding polygons overlapping in SVG space', () => {
    render(<SceneView drawn={drawn} focus={drawn} frame={entry.collision_timestep} collisionFrame={entry.collision_timestep} />);
    const polygons = [...document.querySelectorAll('[data-colliding="true"] polygon')].map((p) =>
      parse(p.getAttribute('points')!),
    );
    expect(polygons).toHaveLength(2);
    expect(separation(polygons[0]!, polygons[1]!)).toBeLessThan(0);
  });

  it('would have drawn them apart at the scalar sizes', () => {
    const scalar = drawn.map((d) => ({ ...d, track: withoutArrays(d.track) }));
    cleanup();
    render(<SceneView drawn={scalar} focus={scalar} frame={entry.collision_timestep} collisionFrame={entry.collision_timestep} />);
    const polygons = [...document.querySelectorAll('[data-colliding="true"] polygon')].map((p) =>
      parse(p.getAttribute('points')!),
    );
    expect(separation(polygons[0]!, polygons[1]!)).toBeGreaterThan(0);
  });

  it('draws the polygons at the world positions the boxes() above computed, flipped into SVG space', () => {
    cleanup();
    render(<SceneView drawn={drawn} focus={drawn} frame={entry.collision_timestep} collisionFrame={entry.collision_timestep} />);
    const drawnPolys = [...document.querySelectorAll('[data-colliding="true"] polygon')].map((p) => parse(p.getAttribute('points')!));
    const { sdc, challenger } = boxes(entry, true);
    const flip = (poly: Poly): Point[] => poly.map(([x, y]) => [x, -y] as Point);
    expect(drawnPolys[0]).toEqual(flip(sdc));
    expect(drawnPolys[1]).toEqual(flip(challenger));
  });
});
