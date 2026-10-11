import { screen } from '@testing-library/react';
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
