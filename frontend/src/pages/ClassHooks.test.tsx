import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { makeDetail, makePage, makePerturbed, makeRow, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { readFrontendFile } from '../test/readFrontendFile';
import { renderApp } from '../test/renderApp';
import { cssRules } from '../test/cssRules';
import type { HealthResponse, StatsResponse } from '../api/client';

/*
 * Every class the pages render has a rule in the stylesheet. A class with no rule is a hook for nothing: it reads as if
 * it did something, and it outlives the layout it was written for (the detail page once carried two). The classes that
 * have no rule on purpose are named here, with the reason.
 */

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

const ok = (body: unknown): Reply => ({ status: 200, body });

/** Classes with no rule, each for a stated reason. */
const NO_RULE_ON_PURPOSE: Record<string, string> = {
  'hero-eyebrow': 'on the list hero eyebrow since before this series; StyleTokens.test.ts finds the eyebrow by it',
  'path-point': 'a single-point path in the scene (SceneView.tsx): it takes its fill and stroke from the track classes around it',
};

const styled = new Set(
  cssRules(readFrontendFile('src/styles.css')).flatMap((r) => [...r.selector.matchAll(/\.([A-Za-z][\w-]*)/g)].map((m) => m[1]!)),
);

function renderedClasses(): string[] {
  const found = new Set<string>();
  for (const el of document.querySelectorAll('[class]')) {
    for (const name of (el.getAttribute('class') ?? '').split(/\s+/)) if (name) found.add(name);
  }
  return [...found].sort();
}
const unstyled = () => renderedClasses().filter((name) => !styled.has(name) && !(name in NO_RULE_ON_PURPOSE));

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
const logged = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -28], heading: Math.PI / 2 });
const perturbed = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 });
const STATS: StatsResponse = {
  total_scenarios: 100, stress_tested: 18, collisions_found: 18, no_collision_found: 0, replay_infeasible: 2,
  heading_blend_singularity: 0, no_challenger: 0, stress_errors: 0, robustly_safe: 0, with_geometry: 20,
  fragility_min: 0.5, fragility_max: 4.5, fragility_mean: 2,
};
const HEALTH: HealthResponse = { status: 'ok', database: 'connected', postgis: '3.6' };

describe('every class a page renders has a rule', () => {
  it('on the list', async () => {
    stubFetch(() => ok(makePage([makeRow({ scenario_id: 'synthetic-0001', stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.02 }), makeRow({ scenario_id: 'synthetic-0002' })], null)));
    renderApp('/');
    await screen.findByRole('link', { name: 'synthetic-0002' });
    expect(renderedClasses().length).toBeGreaterThan(30); // the page is there
    expect(unstyled()).toEqual([]);
  });

  it('on a scenario page with a collision and a perturbed path', async () => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ target_idx: 1, collision_timestep: 25, min_perturbation: 0.1, baseline: logged, perturbed }));
      return ok(makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', last_attempt_outcome: 'collision_found', min_perturbation: 0.1, collision_timestep: 25, challengers_searched: 3, challengers_total: 20 }));
    });
    renderApp('/scenarios/synthetic-0001');
    await screen.findByTestId('frame-readout');
    await screen.findByRole('group', { name: 'Search outcome' });
    expect(renderedClasses().length).toBeGreaterThan(30);
    expect(unstyled()).toEqual([]);
  });

  it('on a scenario page with the request form, with a request filed', async () => {
    vi.stubEnv('VITE_SEARCH_REQUESTS', 'true');
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path === '/search-requests') return { status: 201, body: { request_id: 'r-1', scenario_id: 'synthetic-0001', seed: 3, state: 'filed' } };
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ target_idx: 1, collision_timestep: 25, baseline: logged, perturbed }));
      return ok(makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.1, collision_timestep: 25 }));
    });
    renderApp('/scenarios/synthetic-0001');
    const user = userEvent.setup();
    await user.type(await screen.findByRole('textbox', { name: 'Seed' }), '3');
    await user.click(screen.getByRole('button', { name: 'File request' }));
    await screen.findByText(/Nothing has run/);
    expect(document.querySelector('.search-request')).not.toBeNull();
    expect(unstyled()).toEqual([]);
  });

  it('on a scenario page with the fragility breakdown', async () => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ target_idx: 1, collision_timestep: 25, baseline: logged, perturbed }));
      return ok(makeDetail({ min_ttc: 0, min_pet: -2, fragility_score: 100, fragility_breakdown: { ttc_weight: 0.6, pet_weight: 0.4, ttc_signal: 0.5, pet_signal: 0.25, ttc_term: 0.3, pet_term: 0.1, recomputed_score: 0.4, ttc_floor_s: 0.5, pet_floor_s: 0.25, ttc_none: false, pet_none: false, ttc_at_floor: false, pet_at_floor: false, reproduces_stored_score: true } }));
    });
    renderApp('/scenarios/synthetic-0001');
    await screen.findByRole('group', { name: 'Fragility, by term' });
    expect(document.querySelector('.fragility-table')).not.toBeNull();
    expect(unstyled()).toEqual([]);
  });

  it('on a scenario page with the provenance panel opened', async () => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/provenance')) return ok({ scenario_id: 'synthetic-0001', stress_run_id: 'r', scene_fingerprint: 's', sdc_idx: 1, scored_at: '2026-09-01T10:15:30Z', stress_attempted_at: null, stress_tested_at: '2026-09-02T00:00:00Z', search: { replay_model_version: 'v', collision_geometry_version: 'g', delta_parameterization: 'bicycle', challenger_selection: 'n', selected_stage: 'de', de_popsize: 15, de_maxiter: 200, de_tol: 0.001, de_seed: 1, de_n_iter: 3, de_n_eval: 99, de_candidate_source: 'de_winner', de_archive_covered_all_evaluations: true, refine_n_iters: null, bounds: [[-1, 1], [-1, 1]], max_baseline_drift: 1.5, max_speed_step: 2, speed_step_hold_s: 0.5, held_speed_excess: 0.1, a_max: 12, heading_speed_floor: 0.5, heading_transition_width: 0.05, target_min_speed: 3, frames_observed: 91, frames_below_heading_floor: 0, frames_in_heading_transition_band: 0, target_has_interior_gap: false } });
      if (req.path.endsWith('/trace')) return ok({ scenario_id: 'synthetic-0001', stress_run_id: 'r', trace: [{ generation: 1, n_eval_cumulative: 30, best_weighted_norm: null }, { generation: 2, n_eval_cumulative: 60, best_weighted_norm: 0.5 }, { generation: 3, n_eval_cumulative: 90, best_weighted_norm: 0.25 }], max_abs_delta_difference: 0, tolerance: 1e-6, recorded_at: '2026-10-10T10:00:00Z' });
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ target_idx: 1, collision_timestep: 25, baseline: logged, perturbed }));
      return ok(makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.1, collision_timestep: 25 }));
    });
    renderApp('/scenarios/synthetic-0001');
    await userEvent.click(await screen.findByText('Provenance: where this result came from'));
    await screen.findByText('Scene fingerprint');
    await screen.findByRole('group', { name: 'How the search closed in' });
    expect(document.querySelector('.provenance-bounds')).not.toBeNull();
    expect(unstyled()).toEqual([]);
  });

  it('on a scenario page with the other searches opened', async () => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/other-searches')) return ok({ scenario_id: 'synthetic-0001', stress_run_id: 'r', stored_delta_parameterization: 'bicycle', truncated: true, searches: [{ search_run_id: '0123456789abcdef', kind: 'seed', challenger_rank: null, target_idx: 1, seed: 3, popsize: 15, maxiter: 200, tol: 0.001, outcome: 'collision_found', min_perturbation: 0.25, collision_timestep: 40, delta: [0.5, 0.01, -0.2, 0.003], delta_parameterization: 'bicycle', delta_labels: ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'], n_iter: 12, n_eval: 900 }, { ...{ search_run_id: '0123456789abcdef', kind: 'seed', challenger_rank: null, target_idx: 1, seed: 3, popsize: 15, maxiter: 200, tol: 0.001, outcome: 'collision_found', min_perturbation: 0.25, collision_timestep: 40, delta: [0.5, 0.01, -0.2, 0.003], delta_parameterization: 'bicycle', delta_labels: ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'], n_iter: 12, n_eval: 900 }, search_run_id: 'fedcba9876543210', kind: 'budget', delta_parameterization: 'linear', delta_labels: null }] });
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ target_idx: 1, collision_timestep: 25, baseline: logged, perturbed }));
      return ok(makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.1, collision_timestep: 25 }));
    });
    renderApp('/scenarios/synthetic-0001');
    await userEvent.click(await screen.findByText('Other searches of this scenario'));
    await screen.findByRole('table', { name: 'Other searches of this scenario' });
    await userEvent.click(screen.getByText('Perturbation components'));
    expect(document.querySelector('.other-searches-table')).not.toBeNull();
    expect(screen.getByText(/Not every recorded search is listed here/)).toBeInTheDocument();
    expect(document.querySelector('.other-searches-components h4')).not.toBeNull();
    expect(unstyled()).toEqual([]);
  });

  it('on the Method page', async () => {
    stubFetch(() => ok(makePage([], null)));
    renderApp('/method');
    await screen.findByRole('heading', { level: 1, name: 'A stored replay is shown only with the run it came from.' });
    expect(unstyled()).toEqual([]);
  });

  it('on the Corpus page', async () => {
    stubFetch((req: RecordedRequest) => ok(req.path === '/health' ? HEALTH : STATS));
    renderApp('/stats');
    await screen.findByText('API status');
    expect(unstyled()).toEqual([]);
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
    expect(screen.getAllByRole('link', { name: /^Open scenario/ })).toHaveLength(3);
    expect(unstyled()).toEqual([]);
  });

  it('on the comparison page, with both scenarios shown and their components lined up', async () => {
    const labels = ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'];
    stubFetch((req: RecordedRequest): Reply => {
      const id = decodeURIComponent(req.path.split('/')[2] ?? '');
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ scenario_id: id, delta: [1, 2, 3, 4], delta_labels: labels, target_idx: 1 }));
      return ok(makeDetail({ scenario_id: id, stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.02, collision_timestep: 25 }));
    });
    renderApp('/compare?a=synthetic-0001&b=synthetic-0002');
    await screen.findByRole('table', { name: 'Perturbation components, side by side' });
    expect(unstyled()).toEqual([]);
  });

  it('on the page for an unknown address', async () => {
    stubFetch(() => ok(makePage([], null)));
    renderApp('/nowhere');
    await screen.findByRole('heading', { name: 'Page not found' });
    expect(unstyled()).toEqual([]);
  });

  it('the classes allowed to have no rule are still rendered somewhere, so the list does not outlive them', async () => {
    stubFetch(() => ok(makePage([makeRow({ scenario_id: 'synthetic-0001' })], null)));
    renderApp('/');
    await screen.findByRole('link', { name: 'synthetic-0001' });
    expect(renderedClasses()).toContain('hero-eyebrow');
  });
});
