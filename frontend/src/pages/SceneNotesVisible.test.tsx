import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * Every note under the scene that says something is missing, failed, loading or approximate
 * is VISIBLE, not merely in the document.
 *
 * `toBeInTheDocument` passes for text inside a collapsed element; `toBeVisible` does not. These
 * notes were once at risk of being folded into a collapsed <details>, where a viewer would not
 * see that a perturbed path is absent or that a request failed. The notes that other test files
 * already find (the opening-frame notes, "Loading the perturbed path…", "could not be loaded", the
 * box-size notes) carry their own toBeVisible assertions there; this file covers the rest.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
const challengerLogged = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -28], heading: Math.PI / 2 });
const challengerPerturbed = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 });

function open(replies: { detail?: Reply; trajectories?: Reply; perturbed?: Reply }) {
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return replies.trajectories ?? ok(makeTrajectories([sdc, challengerLogged]));
    if (req.path.endsWith('/perturbed')) return replies.perturbed ?? ok(makePerturbed());
    return replies.detail ?? ok(makeDetail());
  });
  return renderApp(`/scenarios/${ID}`);
}

describe('notes that say a perturbed path is missing', () => {
  it.each([
    [
      'the search found no collision',
      makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'no_collision_found', no_collision_found: true }),
      'The search found no collision, so there is no perturbed path to draw.',
    ],
    [
      'a pass reached the scenario but stored no result',
      makeDetail({ stress_tested: false, stress_attempted: true, last_attempt_outcome: 'replay_infeasible' }),
      'A stress-test pass reached this scenario but stored no result, so there is no perturbed path.',
    ],
    [
      'the scenario was never stress-tested',
      makeDetail(),
      'Not stress-tested yet, so there is no perturbed path.',
    ],
  ])('is visible when %s', async (_name, detail, text) => {
    open({ detail: ok(detail) });
    expect(await screen.findByText(text)).toBeVisible();
  });
});

describe('notes that say geometry is missing', () => {
  it('says only the challenger is drawn when no other agent geometry exists, and says it visibly', async () => {
    open({
      detail: ok(makeDetail({ stress_tested: true, stress_outcome: 'collision_found', collision_timestep: 25 })),
      trajectories: ok(makeTrajectories([])),
      perturbed: ok(
        makePerturbed({ target_idx: 1, collision_timestep: 25, baseline: challengerLogged, perturbed: challengerPerturbed }),
      ),
    });
    expect(await screen.findByText(/Only the challenger is drawn: no other agent geometry\./)).toBeVisible();
  });
});
