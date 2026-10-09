import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The detail page's header, its result overview and its panel headings.
 *
 * The overview carries what the stored search found (outcome, minimum perturbation, collision
 * frame, challengers searched); the Summary panel beside the replay keeps the scenario's own
 * numbers. Nothing here claims a collision, a perturbed path or a comparison unless the
 * response carries one.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
const challengerLogged = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -28], heading: Math.PI / 2 });
const challengerPerturbed = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 });

const COLLISION_DETAIL = makeDetail({
  stress_tested: true,
  stress_attempted: true,
  stress_outcome: 'collision_found',
  last_attempt_outcome: 'collision_found',
  min_perturbation: 0.123456,
  collision_timestep: 25,
  challengers_searched: 3,
  challengers_total: 20,
});
const COLLISION_PERTURBED = makePerturbed({
  target_idx: 1,
  delta: [0.02, -0.01, 0.009, -0.011],
  delta_labels: ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'],
  min_perturbation: 0.123456,
  collision_timestep: 25,
  baseline: challengerLogged,
  perturbed: challengerPerturbed,
});

const INTRO = 'Compare the recorded motion with the smallest collision-producing replay found by the search.';
const TOOLTIP = 'The search perturbs one heuristically chosen challenger, not every agent.';

interface Replies {
  detail?: Reply;
  trajectories?: Reply;
  perturbed?: Reply;
}
function open(replies: Replies = {}) {
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return replies.trajectories ?? ok(makeTrajectories([]));
    if (req.path.endsWith('/perturbed')) return replies.perturbed ?? ok(makePerturbed());
    return replies.detail ?? ok(makeDetail());
  });
  return renderApp(`/scenarios/${ID}`);
}
const collisionScenario = () =>
  open({
    detail: ok(COLLISION_DETAIL),
    trajectories: ok(makeTrajectories([sdc, challengerLogged])),
    perturbed: ok(COLLISION_PERTURBED),
  });
/** A stored result whose perturbed path was never exported: the response has no perturbed track. */
const resultWithoutPath = () =>
  open({
    detail: ok(COLLISION_DETAIL),
    trajectories: ok(makeTrajectories([sdc, challengerLogged])),
    perturbed: ok({ ...COLLISION_PERTURBED, baseline: null, perturbed: null }),
  });

const overview = () => screen.findByRole('region', { name: 'Search result' });
const dtLabels = () => [...document.querySelectorAll('.facts dt')].map((el) => el.textContent);

describe('the result overview', () => {
  it('carries the stored result: outcome, minimum perturbation, collision frame, challengers searched', async () => {
    collisionScenario();
    const region = await overview();
    expect(region).toBeVisible();
    expect(within(region).getByText('Search outcome')).toBeInTheDocument();
    expect(within(region).getByText('collision found')).toBeInTheDocument();
    expect(within(region).getByText('Min perturbation')).toBeInTheDocument();
    expect(within(region).getByText('0.123')).toHaveAttribute('title', '0.123456');
    expect(within(region).getByText('Collision frame').nextElementSibling).toHaveTextContent('frame 25');
    expect(within(region).getByText('Challengers searched').nextElementSibling).toHaveTextContent('3 of 20');
  });

  it('keeps the tooltip that says only one challenger is searched', async () => {
    collisionScenario();
    const region = await overview();
    expect(within(region).getByText('Challengers searched')).toHaveAttribute('title', TOOLTIP);
  });

  it('shows a dash for each figure the response does not have, and says the scenario was not tested', async () => {
    open({ detail: ok(makeDetail()) });
    const region = await overview();
    expect(within(region).getByText('not tested')).toBeInTheDocument();
    expect(within(region).getAllByText('—')).toHaveLength(3);
    expect(within(region).queryByText(/frame \d/)).not.toBeInTheDocument();
  });

  it('says "Min perturbation" once on the page, and the Summary panel no longer repeats the result rows', async () => {
    collisionScenario();
    await overview();
    expect(screen.getAllByText('Min perturbation')).toHaveLength(1);
    const labels = dtLabels();
    for (const moved of ['Outcome', 'Min perturbation', 'Collision frame', 'Challengers searched']) {
      expect(labels).not.toContain(moved);
    }
    expect(labels).toEqual(expect.arrayContaining(['Fragility', 'Min TTC', 'Min PET', 'Challenger', 'Method', 'Agents', 'Shard']));
  });

  it('appears only once the scenario has loaded', async () => {
    open({ detail: { status: 500, body: { detail: 'boom' } } });
    expect((await screen.findAllByRole('alert')).length).toBeGreaterThan(0);
    expect(screen.queryByRole('region', { name: 'Search result' })).not.toBeInTheDocument();
  });
});

describe('the header', () => {
  it('keeps the eyebrow "SCENARIO REPLAY" (it is shown for scenarios that have no collision too)', async () => {
    open();
    expect(await screen.findByText('SCENARIO REPLAY')).toBeInTheDocument();
    expect(screen.queryByText(/FAILURE/)).not.toBeInTheDocument();
  });

  it('introduces the comparison only when a perturbed track is drawn', async () => {
    collisionScenario();
    const intro = await screen.findByText(INTRO);
    expect(intro).toBeVisible();
    expect(screen.getByText('Recorded scene · simulated challenger')).toBeVisible();
  });

  it.each([
    ['a stored result whose perturbed path was not exported', resultWithoutPath],
    ['a scenario with nothing stored', () => open({ trajectories: ok(makeTrajectories([sdc])) })],
  ])('makes no claim of a comparison for %s', async (_name, openIt) => {
    openIt();
    await screen.findByRole('region', { name: 'Search result' });
    await screen.findByTestId('frame-readout');
    expect(screen.queryByText(INTRO)).not.toBeInTheDocument();
    expect(screen.queryByText(/simulated challenger/)).not.toBeInTheDocument();
    expect(screen.getByText('Recorded scene')).toBeInTheDocument();
  });
});

describe('the panel headings', () => {
  const heading = (name: string) => screen.findByRole('heading', { name, level: 2 });

  it('names the replay panel "Replay", with a kicker that says what it compares', async () => {
    collisionScenario();
    await screen.findByTestId('frame-readout'); // the scene has loaded, so what the page claims is settled
    await heading('Replay');
    expect(screen.getByText('RECORDED VS PERTURBED, FRAME BY FRAME')).toBeVisible();
    expect(screen.queryByRole('heading', { name: /^(Scene|Collision replay)$/ })).not.toBeInTheDocument();
  });

  it.each([
    ['a stored result without a perturbed path', resultWithoutPath],
    ['a scenario with nothing stored', () => open({ trajectories: ok(makeTrajectories([sdc])) })],
  ])('says "RECORDED SCENE", not a comparison or a collision, for %s', async (_name, openIt) => {
    openIt();
    await screen.findByTestId('frame-readout'); // loaded: not the first render, where this would hold trivially
    await heading('Replay');
    expect(screen.getByText('RECORDED SCENE')).toBeInTheDocument();
    expect(screen.queryByText(/RECORDED VS PERTURBED/)).not.toBeInTheDocument();
    expect(screen.queryByText(/collision replay/i)).not.toBeInTheDocument();
  });

  it('names the summary panel "Run details" and the perturbation panel "Perturbation" under "SEARCH RESULT"', async () => {
    collisionScenario();
    await heading('Run details');
    expect(screen.queryByRole('heading', { name: 'Summary' })).not.toBeInTheDocument();
    expect(screen.getByText('RESULT CONTEXT')).toBeInTheDocument();
    await heading('Perturbation');
    expect(screen.getByText('SEARCH RESULT')).toBeInTheDocument();
    expect(screen.queryByText('SEARCH PARAMETERS')).not.toBeInTheDocument();
  });
});
