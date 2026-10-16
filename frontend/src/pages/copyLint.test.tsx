import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { HealthResponse, StatsResponse } from '../api/client';
import { makeDetail, makePage, makePerturbed, makeRow, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * A lint over the words the site is not allowed to say, applied to the rendered text of every page
 * that carries project copy: the list, the Method page, the Corpus page, and the detail page in each
 * of its states.
 *
 * The words are matched as WHOLE words, so "delivery" is not "live" and "safety" is not "safe":
 *   live        nothing on this site is computed live: the search runs offline and the API reads stored rows
 *   safety      the project is not a safety certificate
 *   end-to-end  the offline stages are run by hand in Colab; the site does not automate them
 *   adversarial nowhere: the search is an optimiser for a perturbation that produces contact, not an opposing agent
 *               (src/optimization/scipy_optimizer.py optimize_scenario). The brand subtitle that once said it is gone,
 *               so nothing is exempt
 *
 * "safe" is forbidden on the list, the Method page and the detail page, but NOT on /stats: the Corpus
 * page lists the server's `robustly_safe` count under the label ROBUSTLY_SAFE.label ("robustly safe",
 * frontend/src/domain/outcomes.ts), which is the server's own reserved field and is worded once there.
 * The matching guard for the list and the detail page is the /\bsafe\b/i assertion in
 * ScenarioListPage.test.tsx.
 *
 * Only visible text is linted (`title` attributes are not). The negation "not a safety certificate" is
 * visible text on /stats (frontend/src/domain/outcomes.ts, `describeOutcome`), which is why ALLOWED_DENIALS
 * exists.
 */

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

const ok = (body: unknown): Reply => ({ status: 200, body });

/**
 * Phrases that contain a banned word only to DENY the claim. There is exactly one, on `main` since before
 * this change: the no-collision definition (frontend/src/domain/outcomes.ts, `describeOutcome`), shown on
 * /stats, says "This is a statement about the search, not a safety certificate." A lint that forbade
 * that sentence would push the site toward not saying it.
 */
const ALLOWED_DENIALS = ['not a safety certificate'];

const BANNED: [string, RegExp][] = [
  ['live', /\blive\b/i],
  ['safety', /\bsafety\b/i],
  ['end-to-end', /\bend-to-end\b/i],
  ['adversarial', /\badversarial\b/i],
];
const SAFE = /\bsafe\b/i;

function violations(text: string, { allowSafe = false } = {}): string[] {
  let scrubbed = text;
  for (const phrase of ALLOWED_DENIALS) scrubbed = scrubbed.split(phrase).join(' ');
  const hits = BANNED.filter(([, re]) => re.test(scrubbed)).map(([word]) => word);
  if (!allowSafe && SAFE.test(scrubbed)) hits.push('safe');
  return hits;
}

/**
 * The visible text of an element with a space between every text node. `textContent` alone glues
 * neighbouring elements together ("...refinement" + "End-to-end ..." reads "refinementEnd-to-end ..."),
 * which removes the word boundary a whole-word match needs and hides a banned word at the start of a
 * paragraph or heading. (The one-line /\bsafe\b/ guard in ScenarioListPage.test.tsx uses plain textContent,
 * as specified, and has that blind spot; the "safe" check here does not.)
 */
function visibleText(root: Element): string {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const parts: string[] = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) parts.push(node.textContent ?? '');
  return parts.join(' ');
}
const pageText = () => visibleText(document.querySelector('.page')!);

describe('the matcher itself', () => {
  it.each([
    ['Explore live results', ['live']],
    ['A safety analysis', ['safety']],
    ['an END-TO-END system', ['end-to-end']],
    ['Adversarial search', ['adversarial']],
    ['never presented as safe', ['safe']],
    ['Delivery is alive and well, a delivered result, unsafe to say', []],
    ['safety and safe', ['safety', 'safe']],
    ['This is a statement about the search, not a safety certificate.', []],
    ['a safety certificate', ['safety']],
    ['Adversarial scenario search', ['adversarial']],
    ['An adversarial pipeline', ['adversarial']],
    ['Recorded scene · adversarial challenger', ['adversarial']],
    ['Adversarial scenario search, and an adversarial reading of it', ['adversarial']],
    ['proves it is safe: not a safety certificate', ['safe']],
  ])('%s -> %j', (text, expected) => {
    expect(violations(text)).toEqual(expected);
  });

  it('sees a banned word that starts an element right after another element, which textContent glues shut', () => {
    document.body.innerHTML = '<ul><li>refinement</li></ul><p>End-to-end Differential Evolution</p><h2>Adversarial pipeline</h2><p>safe</p>';
    expect(document.body.textContent).not.toMatch(/\bend-to-end\b/i); // the blind spot
    expect(violations(visibleText(document.body))).toEqual(['end-to-end', 'adversarial', 'safe']);
    document.body.innerHTML = '';
  });

  it('exempts no phrase for "adversarial": the only exempt phrase is the denial of a safety certificate', () => {
    expect(violations('Adversarial scenario search')).toEqual(['adversarial']);
    expect(violations('adversarial')).toEqual(['adversarial']);
    expect(violations('This is a statement about the search, not a safety certificate.')).toEqual([]);
    expect(ALLOWED_DENIALS).toEqual(['not a safety certificate']);
  });

  it('allows "safe" only when told to', () => {
    expect(violations('robustly safe', { allowSafe: true })).toEqual([]);
    expect(violations('robustly safe')).toEqual(['safe']);
  });
});

describe('the words the project cannot back do not appear', () => {
  const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
  const logged = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -28], heading: Math.PI / 2 });
  const perturbedTrack = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 });
  const collisionDetail = makeDetail({
    stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', last_attempt_outcome: 'collision_found',
    min_perturbation: 0.123, collision_timestep: 25, challengers_searched: 3, challengers_total: 20,
  });
  const collisionPerturbed = makePerturbed({
    target_idx: 1, delta: [0.02, -0.01, 0.009, -0.011], collision_timestep: 25, min_perturbation: 0.123,
    delta_labels: ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'],
    baseline: logged, perturbed: perturbedTrack,
  });

  function detailServer(replies: { detail?: unknown; trajectories?: unknown; perturbed?: unknown }) {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/trajectories')) return ok(replies.trajectories ?? makeTrajectories([]));
      if (req.path.endsWith('/perturbed')) return ok(replies.perturbed ?? makePerturbed());
      return ok(replies.detail ?? makeDetail());
    });
  }

  it('on the scenario list', async () => {
    stubFetch(() => ok(makePage([
      makeRow({ scenario_id: 'synthetic-0001', stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.02 }),
      makeRow({ scenario_id: 'synthetic-0002' }),
      makeRow({ scenario_id: 'synthetic-0003', stress_tested: true, stress_attempted: true, stress_outcome: 'no_collision_found', no_collision_found: true }),
    ], null)));
    renderApp('/');
    await screen.findByRole('link', { name: 'synthetic-0003' });
    expect(pageText().length).toBeGreaterThan(500); // the hero, the guide and the table are all there
    expect(violations(pageText())).toEqual([]);
  });

  it.each([
    ['a request that is filed', { status: 201, body: { request_id: 'r-1', scenario_id: 'synthetic-0001', seed: 3, state: 'filed' } }, /Nothing has run/],
    ['a request that was filed before', { status: 200, body: { request_id: 'r-1', scenario_id: 'synthetic-0001', seed: 3, state: 'existing' } }, /Already filed/],
    ['a limit reached', { status: 429, body: { detail: 'x' }, headers: { 'Retry-After': '3600' } }, /reached a limit/],
    ['a full queue', { status: 503, body: { detail: 'x' }, headers: { 'Retry-After': '3600' } }, /cannot be filed right now/],
    ['an unknown scenario', { status: 404, body: { detail: 'x' } }, /not in the stored results/],
    ['a server error', { status: 500, body: { detail: 'boom' } }, /not known/],
  ] as const)('on the request form, with %s', async (_name, reply, expected) => {
    vi.stubEnv('VITE_SEARCH_REQUESTS', 'true');
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path === '/search-requests') return reply as unknown as Reply;
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ target_idx: 1, collision_timestep: 25, baseline: logged }));
      return ok(makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.1, collision_timestep: 25 }));
    });
    renderApp('/scenarios/synthetic-0001');
    const user = userEvent.setup();
    expect(violations(pageText())).toEqual([]); // before anything is filed
    await user.type(await screen.findByRole('textbox', { name: 'Seed' }), '3');
    await user.click(screen.getByRole('button', { name: 'File request' }));
    await screen.findByText(expected);
    expect(pageText()).toContain('Request a search');
    expect(violations(pageText())).toEqual([]);
  });

  it('on a scenario page in the whole-scene framing, with the caption about the other agents', async () => {
    const parked = makeTrack({ agent_idx: 2, frames: range(0, 91), origin: [10, 10] });
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged, parked]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ target_idx: 1, collision_timestep: 25, baseline: logged, perturbed: makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 }) }));
      return ok(makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.1, collision_timestep: 25 }));
    });
    renderApp('/scenarios/synthetic-0001');
    await userEvent.click(await screen.findByRole('button', { name: 'Whole scene' }));
    await screen.findByText(/drawn as logged/);
    expect(violations(pageText())).toEqual([]);
  });

  it.each([
    ['a breakdown that adds up', { ttc_weight: 0.6, pet_weight: 0.4, ttc_signal: 0.5, pet_signal: 0.25, ttc_term: 0.3, pet_term: 0.1, recomputed_score: 0.4, ttc_floor_s: 0.5, pet_floor_s: 0.25, ttc_none: false, pet_none: false, ttc_at_floor: false, pet_at_floor: false, reproduces_stored_score: true }],
    ['a breakdown that does not add up', { ...{ ttc_weight: 0.6, pet_weight: 0.4, ttc_signal: 0.5, pet_signal: 0.25, ttc_term: 0.3, pet_term: 0.1, recomputed_score: 0.4, ttc_floor_s: 0.5, pet_floor_s: 0.25, ttc_none: false, pet_none: false, ttc_at_floor: false, pet_at_floor: false, reproduces_stored_score: true }, reproduces_stored_score: false }],
    ['no breakdown', null],
  ])('on the fragility breakdown in Run details, with %s', async (_name, breakdown) => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
      return ok(makeDetail({ min_ttc: 2, min_pet: 4, fragility_score: 0.4, fragility_breakdown: breakdown }));
    });
    renderApp('/scenarios/synthetic-0001');
    await screen.findByRole('group', { name: 'Fragility, by term' });
    expect(pageText()).toContain('Fragility, by term');
    expect(violations(pageText())).toEqual([]);
  });

  it('on the walkthrough, with all three scenes shown', async () => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged]));
      return ok(makePage([
        makeRow({ scenario_id: 'synthetic-0001', stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.02, collision_timestep: 25 }),
        makeRow({ scenario_id: 'synthetic-0002', stress_attempted: true, last_attempt_outcome: 'replay_infeasible', n_agents: 5 }),
      ], null));
    });
    renderApp('/walkthrough');
    await screen.findByText(/has the smallest stored perturbation/);
    expect(pageText().length).toBeGreaterThan(500); // the heading, the lede and the three scenes are all there
    expect(violations(pageText())).toEqual([]);
  });

  it.each([
    ['two scenarios of the same model', ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'], ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)']],
    ['two scenarios of different models', ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'], ['initial vx (m/s)', 'initial vy (m/s)', 'x-acceleration bias (m/s^2)', 'y-acceleration bias (m/s^2)']],
    ['two scenarios with the same names in a different order', ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'], ['steering bias (rad)', 'acceleration bias (m/s^2)', 'initial heading (rad)', 'initial speed (m/s)']],
  ])('on the comparison page for %s', async (_name, first, second) => {
    stubFetch((req: RecordedRequest): Reply => {
      const id = decodeURIComponent(req.path.split('/')[2] ?? '');
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ scenario_id: id, delta: [1, 2, 3, 4], delta_labels: id.endsWith('1') ? first : second, target_idx: 1 }));
      return ok(makeDetail({ scenario_id: id, stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.02, collision_timestep: 25 }));
    });
    renderApp('/compare?a=synthetic-0001&b=synthetic-0002');
    await screen.findByText(/Min perturbation is a weighted norm/);
    await waitFor(() => expect(screen.getAllByText(/initial (speed|vx)/).length).toBeGreaterThan(0));
    expect(pageText().length).toBeGreaterThan(500);
    expect(violations(pageText())).toEqual([]);
  });

  it('on the Method page', async () => {
    stubFetch(() => ok(makePage([], null)));
    renderApp('/method');
    await screen.findByRole('heading', { level: 1, name: 'A stored replay is shown only with the run it came from.' });
    expect(pageText().length).toBeGreaterThan(500); // the whole page is there: the hero, the steps and the limits
    expect(violations(pageText())).toEqual([]);
  });

  it('on the Corpus page, where only "safe" is allowed (the ROBUSTLY_SAFE label)', async () => {
    const stats: StatsResponse = {
      total_scenarios: 100, stress_tested: 18, collisions_found: 18, no_collision_found: 0,
      replay_infeasible: 2, heading_blend_singularity: 0, no_challenger: 0, stress_errors: 0,
      robustly_safe: 0, with_geometry: 20, fragility_min: 0.5, fragility_max: 4.5, fragility_mean: 2,
    };
    const health: HealthResponse = { status: 'ok', database: 'connected', postgis: '3.6' };
    stubFetch((req: RecordedRequest) => ok(req.path === '/health' ? health : stats));
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Stored results' });
    expect(violations(pageText(), { allowSafe: true })).toEqual([]);
    // the one place "safe" is expected: the server's reserved field, under its own label
    expect(pageText()).toMatch(/robustly safe/i);
  });

  it('on the Corpus page when the health answer is "degraded"', async () => {
    const stats: StatsResponse = {
      total_scenarios: 100, stress_tested: 18, collisions_found: 18, no_collision_found: 0,
      replay_infeasible: 2, heading_blend_singularity: 0, no_challenger: 0, stress_errors: 0,
      robustly_safe: 0, with_geometry: 20, fragility_min: 0.5, fragility_max: 4.5, fragility_mean: 2,
    };
    const health: HealthResponse = { status: 'degraded', database: 'connected', postgis: null };
    stubFetch((req: RecordedRequest) => ok(req.path === '/health' ? health : stats));
    renderApp('/stats');
    await screen.findByText('not available');
    expect(pageText()).toMatch(/degraded/);
    expect(violations(pageText(), { allowSafe: true })).toEqual([]);
  });

  it.each([
    ['a collision with a perturbed path', () => detailServer({
      detail: collisionDetail, trajectories: makeTrajectories([sdc, logged]), perturbed: collisionPerturbed })],
    ['a stored result whose perturbed path was not exported', () => detailServer({
      detail: collisionDetail, trajectories: makeTrajectories([sdc, logged]),
      perturbed: { ...collisionPerturbed, baseline: null, perturbed: null } })],
    ['a scenario with geometry and nothing stored', () => detailServer({ trajectories: makeTrajectories([sdc]) })],
    ['a scenario whose only attempt was refused', () => detailServer({
      detail: makeDetail({ stress_attempted: true, last_attempt_outcome: 'replay_infeasible' }),
      trajectories: makeTrajectories([sdc]) })],
    ['a search that found no collision', () => detailServer({
      detail: makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'no_collision_found', no_collision_found: true }),
      trajectories: makeTrajectories([sdc]) })],
    ['a scenario with no geometry', () => detailServer({})],
  ])('on the scenario page for %s', async (_name, open) => {
    open();
    renderApp('/scenarios/synthetic-0001');
    await screen.findByRole('region', { name: 'Search result' });
    await screen.findByText(/No geometry to draw|frame \d+ of/);
    expect(violations(pageText())).toEqual([]);
  });
});
