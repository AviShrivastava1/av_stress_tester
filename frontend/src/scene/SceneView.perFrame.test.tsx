import { cleanup, render } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { makeTrack } from '../test/fixtures';
import * as geometry from './geometry';
import { footprint, frontCentre, poseAt, toSvg } from './geometry';
import type { DrawnTrack, Role } from './sceneModel';
import { SceneView } from './SceneView';

// `segments` is called only by the static layer, so its call count measures how often it renders.
vi.mock('./geometry', async (importOriginal) => {
  const real = await importOriginal<typeof import('./geometry')>();
  return { ...real, segments: vi.fn(real.segments) };
});

/*
 * What the scene draws at a frame: the box at the size recorded for THAT frame.
 *
 * Expected polygons and ticks are built here with the pure geometry functions the existing tests
 * already trust (footprint, frontCentre, toSvg), from the numbers each test sets, and compared with
 * what the component put in the SVG.
 */

// Frames with a gap and not starting at 0: vertex index i <-> frame FRAMES[i], never equal.
const FRAMES = [3, 4, 5, 8, 9];
const LENGTHS = [4.1, 4.2, 4.3, 4.4, 4.5];
const WIDTHS = [1.91, 1.92, 1.93, 1.94, 1.95];
const FRAME = 8; // vertex index 3
const I = 3;

const sizes = (over: Parameters<typeof makeTrack>[0] = {}) =>
  makeTrack({
    agent_idx: 0, is_sdc: true, frames: FRAMES, heading: 0.3, origin: [100, 50],
    lengths_m: LENGTHS, widths_m: WIDTHS, length_m: 9, width_m: 7, ...over,
  });

const drawnOf = (track: ReturnType<typeof makeTrack>, role: Role = 'sdc'): DrawnTrack[] => [
  { key: `k-${role}`, role, track },
];

function view(drawn: DrawnTrack[], frame: number | null = FRAME, collisionFrame: number | null = null) {
  return render(<SceneView drawn={drawn} focus={drawn} frame={frame} collisionFrame={collisionFrame} />);
}

const pointsOf = (corners: readonly (readonly [number, number])[]) => corners.map((p) => toSvg(p).join(',')).join(' ');
const polygon = (role = 'sdc') =>
  document.querySelector(`[data-footprint="${role}"] polygon`)?.getAttribute('points');
const tick = (role = 'sdc') => {
  const el = document.querySelector(`[data-footprint="${role}"] line[data-heading-tick]`)!;
  const n = (a: string) => Number(el.getAttribute(a));
  return { x1: n('x1'), y1: n('y1'), x2: n('x2'), y2: n('y2') };
};
const footprintTitle = (role = 'sdc') => document.querySelector(`[data-footprint="${role}"] > title`)?.textContent;
const staticTitle = (role = 'sdc') => document.querySelector(`g.track[data-role="${role}"] > title`)?.textContent;

describe('SceneView: the box at a frame uses the size recorded at that frame', () => {
  it('draws the polygon from the array elements at the pose’s vertex, not from the scalars', () => {
    const track = sizes();
    view(drawnOf(track));
    const pose = poseAt(track, FRAME)!;
    expect(polygon()).toBe(pointsOf(footprint(pose, LENGTHS[I]!, WIDTHS[I]!)));
    expect(polygon()).not.toBe(pointsOf(footprint(pose, 9, 7)));
  });

  it('draws it at every observed frame, including those after a gap', () => {
    const track = sizes();
    FRAMES.forEach((frame, i) => {
      cleanup();
      view(drawnOf(track), frame);
      expect(polygon()).toBe(pointsOf(footprint(poseAt(track, frame)!, LENGTHS[i]!, WIDTHS[i]!)));
    });
  });

  it('ends the heading tick at the front of the per-frame length', () => {
    const track = sizes();
    view(drawnOf(track));
    const pose = poseAt(track, FRAME)!;
    const [x2, y2] = toSvg(frontCentre(pose, LENGTHS[I]!));
    const [x1, y1] = toSvg([pose.x, pose.y]);
    expect(tick()).toEqual({ x1, y1, x2, y2 });
    const [sx, sy] = toSvg(frontCentre(pose, 9));
    expect([tick().x2, tick().y2]).not.toEqual([sx, sy]);
  });

  it('keeps the colliding marking, ghost styling and roles', () => {
    const sdc = sizes();
    const perturbed = sizes({ agent_idx: 1, is_sdc: false, origin: [100, 53] });
    const logged = sizes({ agent_idx: 1, is_sdc: false, origin: [100, 55] });
    const drawn: DrawnTrack[] = [
      { key: 'a', role: 'sdc', track: sdc },
      { key: 'b', role: 'challenger_logged', track: logged },
      { key: 'c', role: 'challenger_perturbed', track: perturbed },
    ];
    view(drawn, FRAME, FRAME);
    const roles = [...document.querySelectorAll<SVGElement>('[data-footprint]')].map((el) => ({
      role: el.dataset.footprint, colliding: el.dataset.colliding === 'true', ghost: el.classList.contains('ghost'),
    }));
    expect(roles).toEqual([
      { role: 'sdc', colliding: true, ghost: false },
      { role: 'challenger_logged', colliding: false, ghost: true },
      { role: 'challenger_perturbed', colliding: true, ghost: false },
    ]);
  });
});

describe('SceneView: when the per-frame sizes are not used', () => {
  const scalarPolygon = (track: ReturnType<typeof makeTrack>) =>
    pointsOf(footprint(poseAt(track, FRAME)!, track.length_m!, track.width_m!));

  it.each([
    ['lengths_m null', { lengths_m: null }],
    ['widths_m null', { widths_m: null }],
    ['lengths_m the wrong length', { lengths_m: LENGTHS.slice(1) }],
    ['widths_m the wrong length', { widths_m: [...WIDTHS, 2] }],
  ])('draws the scalar box, for both dimensions, when %s', (_name, over) => {
    const track = sizes(over);
    view(drawnOf(track));
    expect(polygon()).toBe(scalarPolygon(track));
    const pose = poseAt(track, FRAME)!;
    expect([tick().x2, tick().y2]).toEqual(toSvg(frontCentre(pose, 9)));
  });

  it('draws a track without the fields exactly as before', () => {
    const track = makeTrack({ agent_idx: 0, is_sdc: true, frames: FRAMES, origin: [100, 50], heading: 0.3 });
    view(drawnOf(track));
    expect(polygon()).toBe(pointsOf(footprint(poseAt(track, FRAME)!, 4.5, 2)));
    expect(document.querySelector('[data-footprint="sdc"]')).toHaveAttribute('data-shape', 'box');
  });

  it('draws a dot, as before, for a track with no usable size at all', () => {
    view(drawnOf(makeTrack({ is_sdc: true, frames: FRAMES, length_m: null, width_m: null })));
    expect(document.querySelector('[data-footprint="sdc"]')).toHaveAttribute('data-shape', 'dot');
  });
});

describe('SceneView: an unusable element at the drawn frame', () => {
  it.each([null, Number.NaN, 0, -2, Number.POSITIVE_INFINITY])(
    'draws a dot, not a box of any size, when the length is %s',
    (bad) => {
      const lengths = [...LENGTHS];
      lengths[I] = bad as number;
      view(drawnOf(sizes({ lengths_m: lengths })));
      const el = document.querySelector('[data-footprint="sdc"]')!;
      expect(el).toHaveAttribute('data-shape', 'dot');
      expect(el.querySelector('polygon')).toBeNull();
      expect(el.querySelector('line[data-heading-tick]')).toBeNull();
    },
  );

  it('draws a dot when the width is unusable, and a box at the neighbouring frames', () => {
    const widths = [...WIDTHS];
    widths[I] = null as unknown as number;
    const track = sizes({ widths_m: widths });
    view(drawnOf(track), FRAME);
    expect(document.querySelector('[data-footprint="sdc"]')).toHaveAttribute('data-shape', 'dot');
    cleanup();
    view(drawnOf(track), 9);
    expect(polygon()).toBe(pointsOf(footprint(poseAt(track, 9)!, LENGTHS[4]!, WIDTHS[4]!)));
  });
});

describe('SceneView: titles state what is drawn', () => {
  it('names the size drawn at the frame on the footprint', () => {
    view(drawnOf(sizes()), FRAME);
    expect(footprintTitle()).toBe('SDC · agent 0 · vehicle · 4.40 × 1.94 m at frame 8');
    cleanup();
    view(drawnOf(sizes()), 9);
    expect(footprintTitle()).toBe('SDC · agent 0 · vehicle · 4.50 × 1.95 m at frame 9');
  });

  it('says the size was not recorded at a frame drawn as a dot', () => {
    const lengths = [...LENGTHS];
    lengths[I] = null as unknown as number;
    view(drawnOf(sizes({ lengths_m: lengths })), FRAME);
    expect(footprintTitle()).toBe('SDC · agent 0 · vehicle · size not recorded at frame 8');
  });

  it('gives a range on the path when the sizes vary, never one size', () => {
    view(drawnOf(sizes()));
    expect(staticTitle()).toBe('SDC · agent 0 · vehicle · length 4.10–4.50 m, width 1.91–1.95 m');
  });

  it('gives a single size on the path only where a dimension does not vary', () => {
    view(drawnOf(sizes({ lengths_m: [5, 5, 5, 5, 5] })));
    expect(staticTitle()).toBe('SDC · agent 0 · vehicle · length 5.00 m, width 1.91–1.95 m');
    cleanup();
    view(drawnOf(sizes({ lengths_m: [5, 5, 5, 5, 5], widths_m: [2, 2, 2, 2, 2] })));
    expect(staticTitle()).toBe('SDC · agent 0 · vehicle · 5.00 × 2.00 m');
  });

  it('keeps today’s strings for a track drawn from scalars', () => {
    const plain = makeTrack({ agent_idx: 0, is_sdc: true, frames: FRAMES });
    view(drawnOf(plain), FRAME);
    expect(staticTitle()).toBe('SDC · agent 0 · vehicle · 4.5 × 2.0 m');
    expect(footprintTitle()).toBe('SDC · agent 0 · vehicle · 4.5 × 2.0 m');
    cleanup();
    view(drawnOf(makeTrack({ is_sdc: true, frames: FRAMES, length_m: null })), FRAME);
    expect(staticTitle()).toBe('SDC · agent 0 · vehicle · dimensions unknown');
    expect(footprintTitle()).toBe('SDC · agent 0 · vehicle · dimensions unknown');
  });

  it('says dimensions unknown when no element pair can be drawn', () => {
    view(drawnOf(sizes({ lengths_m: [null, null, null, null, null] })));
    expect(staticTitle()).toBe('SDC · agent 0 · vehicle · dimensions unknown');
  });
});

describe('SceneView: the margin clears the largest box the track can show', () => {
  const viewBoxWidth = () => Number(document.querySelector('svg.scene')!.getAttribute('viewBox')!.split(' ')[2]);

  it('grows for a large element at a frame that is not drawn', () => {
    const track = sizes({ lengths_m: [4.1, 4.2, 30, 4.4, 4.5] }); // 30 m at frame 5; frame 3 is drawn
    view(drawnOf(track), 3);
    const xs = track.path.map((p) => p[0]!);
    const rawWidth = Math.max(...xs) - Math.min(...xs);
    expect(viewBoxWidth()).toBeGreaterThanOrEqual(rawWidth + 2 * 30);
  });

  it('does not grow for a scalar that the arrays override', () => {
    view(drawnOf(sizes({ length_m: 80, width_m: 80 })), 3);
    const xs = sizes().path.map((p) => p[0]!);
    expect(viewBoxWidth()).toBeLessThan(Math.max(...xs) - Math.min(...xs) + 2 * 80);
  });
});

describe('SceneView: playback re-renders only the footprint layer', () => {
  it('does not render the static layer again as the frame changes, but does when the framing changes', () => {
    const drawn = drawnOf(sizes());
    const focusA = drawn;
    const { rerender } = render(<SceneView drawn={drawn} focus={focusA} frame={3} collisionFrame={null} />);
    const spy = vi.mocked(geometry.segments);
    const settled = spy.mock.calls.length;
    for (const frame of [4, 5, 8, 9, 3]) {
      rerender(<SceneView drawn={drawn} focus={focusA} frame={frame} collisionFrame={null} />);
    }
    expect(spy.mock.calls.length).toBe(settled);

    rerender(<SceneView drawn={drawn} focus={[...drawn]} frame={3} collisionFrame={null} />);
    expect(spy.mock.calls.length).toBeGreaterThan(settled);
  });
});
