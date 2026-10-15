import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { AgentTrack } from '../api/client';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * "Gap before first contact" (pages/detail/CollisionAnatomy.tsx, scene/anatomy.ts): how far apart the SDC's box and the
 * perturbed challenger's box were on the frame before the collision frame, worked out in the browser from the stored boxes.
 *
 * The boxes here are 4 long and 2 wide and do not turn, so every gap is a distance along y worked out by hand: the SDC's
 * box spans y from -1 to 1 at the origin, and the challenger's box is centred at y with half its width either side.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });
const FRAMES = range(0, 91);

const sdc = makeTrack({
  agent_idx: 0,
  is_sdc: true,
  frames: FRAMES,
  path: FRAMES.map(() => [0, 0]),
  headings: FRAMES.map(() => 0),
});

interface Scene {
  collisionFrame: number | null;
  /** The perturbed challenger's y by frame. */
  y: (frame: number) => number;
  frames?: number[];
  widths?: (frame: number) => number | null;
  sdcFrames?: number[];
  withPerturbed?: boolean;
}

function challengerTrack({ y, frames = FRAMES, widths }: Scene): AgentTrack {
  return makeTrack({
    agent_idx: 1,
    frames,
    path: frames.map((f) => [0, y(f)]),
    headings: frames.map(() => 0),
    ...(widths ? { lengths_m: frames.map(() => 4), widths_m: frames.map(widths) } : {}),
  });
}

async function open(scene: Scene) {
  const own = scene.sdcFrames
    ? makeTrack({
        agent_idx: 0,
        is_sdc: true,
        frames: scene.sdcFrames,
        path: scene.sdcFrames.map(() => [0, 0]),
        headings: scene.sdcFrames.map(() => 0),
      })
    : sdc;
  const challenger = challengerTrack(scene);
  const logged = makeTrack({ agent_idx: 1, frames: FRAMES, origin: [0, 40] });
  const perturbed =
    scene.withPerturbed === false
      ? makePerturbed()
      : makePerturbed({ target_idx: 1, collision_timestep: scene.collisionFrame, baseline: logged, perturbed: challenger });
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([own, logged]));
    if (req.path.endsWith('/perturbed')) return ok(perturbed);
    return ok(makeDetail({ stress_tested: true, stress_outcome: 'collision_found', collision_timestep: scene.collisionFrame }));
  });
  renderApp(`/scenarios/${ID}`);
  await screen.findByTestId('frame-readout');
}

const anatomy = () => screen.queryByRole('region', { name: 'Gap before first contact' });
const text = () => anatomy()?.textContent ?? '';
const DERIVED = /Worked out in this browser from the stored positions and box sizes of the SDC and the perturbed challenger; the search does not store it\./;
const PRECISION = /Contact is decided at the precision of the stored positions, up to about a millimetre at the coordinates in this data, so no finer figure is claimed\./;

describe('the gap on the frame before first contact', () => {
  it('is measured on the frame before the collision frame, not on it', async () => {
    // 0.5 clear of the SDC's box until frame 25, inside it from frame 25: the collision frame.
    await open({ collisionFrame: 25, y: (f) => (f < 25 ? 2.5 : 1) });
    expect(text()).toContain('The boxes were 0.500 m apart on frame 24, the frame before first contact on frame 25.');
  });

  it('says it is derived in the browser from the stored boxes, and how precise contact is', async () => {
    await open({ collisionFrame: 25, y: (f) => (f < 25 ? 2.5 : 1) });
    expect(text()).toMatch(DERIVED);
    expect(text()).toMatch(PRECISION);
    expect(screen.getByRole('heading', { name: 'Gap before first contact', level: 3 })).toBeInTheDocument();
  });

  it('uses the box size stored for that frame', async () => {
    // The scalar width is 2, which would give 0.5; the width stored for frame 24 is 2.6, which gives 0.2.
    await open({ collisionFrame: 25, y: () => 2.5, widths: (f) => (f === 24 ? 2.6 : 2) });
    expect(text()).toContain('The boxes were 0.200 m apart on frame 24');
  });

  it('gives no figure for a gap below the precision of the positions', async () => {
    await open({ collisionFrame: 25, y: (f) => (f < 25 ? 2.0004 : 1) });
    expect(text()).toContain('The boxes were less than a millimetre apart on frame 24');
    expect(text()).not.toMatch(/0\.000/);
  });

  it('says the boxes already touch when they do on the frame before, and draws no overlap', async () => {
    await open({ collisionFrame: 25, y: () => 1.5 });
    expect(text()).toContain(
      'The boxes already touch or overlap on frame 24, as drawn from the stored sizes, although first contact is reported on frame 25.',
    );
    expect(text()).toMatch(DERIVED);
    expect(document.querySelector('.scene [data-overlap], .scene .overlap')).toBeNull();
  });

  it('works for a collision on the last recorded frame', async () => {
    await open({ collisionFrame: 90, y: (f) => (f < 90 ? 2.5 : 1) });
    expect(text()).toContain('The boxes were 0.500 m apart on frame 89, the frame before first contact on frame 90.');
  });

  it('says there is nothing to measure for a collision on the first frame', async () => {
    await open({ collisionFrame: 0, y: () => 1 });
    expect(text()).toContain('The SDC or the perturbed challenger is not observed before frame 0, so there is no earlier frame to measure.');
    expect(text()).not.toMatch(DERIVED);
    expect(text()).not.toMatch(/ m apart/);
  });

  it('says so when one agent first appears on the collision frame', async () => {
    await open({ collisionFrame: 30, y: () => 1, frames: range(30, 91) });
    expect(text()).toContain('is not observed before frame 30, so there is no earlier frame to measure.');
  });

  it('does not use an earlier frame when one agent was not observed on the frame before', async () => {
    await open({ collisionFrame: 25, y: () => 2.5, frames: [...range(0, 24), ...range(25, 91)] });
    expect(text()).toContain('were not both observed on frame 24, so there is no gap to measure there.');
    expect(text()).not.toMatch(/ m apart/);
  });

  it('says so when no box size is stored for that frame', async () => {
    await open({ collisionFrame: 25, y: () => 2.5, widths: (f) => (f === 24 ? null : 2) });
    expect(text()).toContain('No box size is stored for both agents on frame 24, so the gap is not shown.');
  });
});

describe('where it is shown', () => {
  it('is not tied to the frame on screen', async () => {
    await open({ collisionFrame: 25, y: (f) => (f < 25 ? 2.5 : 1) });
    const before = text();
    expect(before).toContain('0.500 m apart on frame 24');
    fireEvent.change(screen.getByRole('slider', { name: 'Frame' }), { target: { value: '70' } });
    expect(screen.getByTestId('frame-readout')).toHaveTextContent('frame 70');
    expect(text()).toBe(before);
  });

  it('is absent when there is no collision frame', async () => {
    await open({ collisionFrame: null, y: () => 2.5, withPerturbed: false });
    expect(anatomy()).toBeNull();
    expect(screen.queryByText(/Gap before first contact/)).not.toBeInTheDocument();
  });

  it('is absent when the perturbed path is not drawn, even though the score has a collision frame', async () => {
    await open({ collisionFrame: 25, y: () => 2.5, withPerturbed: false });
    expect(anatomy()).toBeNull();
  });

  it('has no number of its own in its words: the digits come from the data', async () => {
    await open({ collisionFrame: 31, y: (f) => (f < 31 ? 3.25 : 1) });
    expect(text()).toContain('The boxes were 1.250 m apart on frame 30, the frame before first contact on frame 31.');
    expect(text().replace(/1\.250|\b30\b|\b31\b/g, '')).not.toMatch(/\d/);
  });
});
