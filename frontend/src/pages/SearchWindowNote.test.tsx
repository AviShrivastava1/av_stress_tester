import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { AgentTrack } from '../api/client';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The search window, on the detail page. The search simulates and checks only the recorded frames, so a collision
 * stored at the recording's last frame is a collision found there and nowhere later. When that is the case a line
 * under "Positions at frame N, the first colliding frame." says so, and in no other case.
 *
 * "The last frame of the recording" is the largest timestep over every track /trajectories returned: not the focused
 * tracks (the SDC and the challenger) and not the challenger's own last observed frame.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });
const NOTE = 'This is the last recorded frame; the search does not look past it.';
const FIRST = /, the first colliding frame\.$/;

interface Scene {
  collision: number;
  sdcFrames?: number[];
  challengerFrames?: number[];
  others?: AgentTrack[];
  trajectories?: Reply;
}

function open({ collision, sdcFrames = range(0, 91), challengerFrames = range(0, 91), others = [], trajectories }: Scene) {
  const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: sdcFrames, origin: [-20, 0] });
  const logged = makeTrack({ agent_idx: 1, frames: challengerFrames, origin: [0, -28], heading: Math.PI / 2 });
  const perturbed = makeTrack({ agent_idx: 1, frames: challengerFrames, origin: [1, -28], heading: Math.PI / 2 });
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return trajectories ?? ok(makeTrajectories([sdc, logged, ...others]));
    if (req.path.endsWith('/perturbed')) {
      return ok(makePerturbed({ target_idx: 1, collision_timestep: collision, min_perturbation: 0.02, baseline: logged, perturbed }));
    }
    return ok(makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.02, collision_timestep: collision }));
  });
  return renderApp(`/scenarios/${ID}`);
}

const referenceNote = () => screen.findByText(FIRST);
const windowNote = () => screen.queryByText(NOTE);

describe('the search-window line on the detail page', () => {
  it('is shown, visibly, directly under the first-colliding-frame note, when the collision is at the last frame', async () => {
    open({ collision: 90, others: [makeTrack({ agent_idx: 2, frames: range(0, 91), origin: [8, 8] })] });
    const first = await referenceNote();
    const line = await screen.findByText(NOTE);
    expect(line).toBeVisible();
    expect(first.nextElementSibling).toBe(line);
    expect(line.tagName).toBe('P');
    expect(line).toHaveClass('note');
    expect(NOTE).not.toMatch(/\d/);
  });

  it('is absent one frame earlier', async () => {
    open({ collision: 89 });
    await referenceNote();
    expect(windowNote()).not.toBeInTheDocument();
  });

  it('is absent when the collision is at the challenger\'s last observed frame but the recording runs on', async () => {
    open({ collision: 20, challengerFrames: range(0, 21) });
    await referenceNote();
    expect(windowNote()).not.toBeInTheDocument();
  });

  it('is absent when the challenger\'s observations end early (observed to 20, collision at 15, recording to 90)', async () => {
    open({ collision: 15, challengerFrames: range(0, 21) });
    await referenceNote();
    expect(windowNote()).not.toBeInTheDocument();
  });

  it('is absent when the focused tracks end at the collision frame but another agent is recorded to a later frame', async () => {
    open({ collision: 60, sdcFrames: range(0, 61), challengerFrames: range(0, 61), others: [makeTrack({ agent_idx: 2, frames: range(0, 91), origin: [8, 8] })] });
    await referenceNote();
    expect(windowNote()).not.toBeInTheDocument();
  });

  it('is absent when another agent is recorded past the collision frame, even at the focused tracks\' last frame', async () => {
    open({ collision: 90, others: [makeTrack({ agent_idx: 2, frames: range(0, 95), origin: [8, 8] })] });
    await referenceNote();
    expect(windowNote()).not.toBeInTheDocument();
  });

  it('is absent when /trajectories has no geometry', async () => {
    open({ collision: 90, trajectories: ok(makeTrajectories([])) });
    await screen.findByText(/Only the challenger is drawn: no other agent geometry\./);
    expect(windowNote()).not.toBeInTheDocument();
  });

  it('is absent when /trajectories fails', async () => {
    open({ collision: 90, trajectories: { status: 500, body: { detail: 'boom' } } });
    expect((await screen.findAllByRole('alert')).length).toBeGreaterThan(0);
    expect(windowNote()).not.toBeInTheDocument();
  });

  it('goes with the first-colliding-frame note once the playhead has moved', async () => {
    open({ collision: 90 });
    await screen.findByText(NOTE);
    const frame = screen.getByLabelText(/^Scene\./);
    fireEvent.keyDown(frame, { key: 'ArrowLeft' });
    expect(screen.queryByText(FIRST)).not.toBeInTheDocument();
    expect(windowNote()).not.toBeInTheDocument();
  });
});
