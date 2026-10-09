import { screen } from '@testing-library/react';
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
 *   adversarial only the one backed subtitle may say it (added with the chrome diff, see ALLOWED_SUBTITLES)
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
});

const ok = (body: unknown): Reply => ({ status: 200, body });

/** The one sentence that may say "adversarial", once the chrome diff adds it. Empty until then. */
const ALLOWED_SUBTITLES: string[] = [];

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
  for (const phrase of [...ALLOWED_SUBTITLES, ...ALLOWED_DENIALS]) scrubbed = scrubbed.split(phrase).join(' ');
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

  it('on the Method page', async () => {
    stubFetch(() => ok(makePage([], null)));
    renderApp('/method');
    await screen.findByRole('heading', { level: 1, name: /stored, replayable result/ });
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
