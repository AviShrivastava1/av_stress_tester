import { screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { makePage, makeRow } from '../test/fixtures';
import { stubFetch, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The top of the scenario list: what the project is, in words each of which the repository backs, and the
 * way down to the stored results. Nothing here claims the site computes anything live or certifies anything.
 */

afterEach(() => {
  vi.unstubAllGlobals();
});

const ok = (body: unknown): Reply => ({ status: 200, body });

const HERO = {
  eyebrow: 'AUTONOMOUS-VEHICLE SCENARIO STRESS-TESTING',
  h1: 'How small a change turns a recorded driving scene into a collision?',
  lede:
    'This project searches recorded driving scenes for the smallest model-constrained change that produces ' +
    'geometric contact, then lets you inspect each stored result frame by frame.',
} as const;

const STACK = ['Python', 'SciPy', 'PyTorch', 'PostGIS', 'FastAPI', 'React'];

const PIPELINE = [
  ['Rank', 'SDC-restricted TTC + PET'],
  ['Perturb', 'Bounded 4-D kinematic search'],
  ['Verify', 'Replay gates + oriented boxes'],
  ['Publish', 'PostGIS → API → browser replay'],
] as const;

const CARDS = [
  ['Kinematic replay', 'Bicycle and linear motion models, with explicit fidelity refusals.'],
  [
    'Global search, optional refinement',
    'Differential Evolution, optional gradient refinement, float64 oriented-box verification.',
  ],
  ['Provenance stored with each result', 'Scene identity, run provenance, read-only production API.'],
] as const;

const INTRO =
  'Open a scene to replay it. Where the search found a collision, the replay compares the recorded challenger ' +
  'with the smallest collision-producing perturbation it found.';

const GUIDE =
  'Fragility ranks the recorded scene; it is not a collision probability. Min perturbation is the smallest ' +
  'weighted perturbation found by the finite search, not a proven global minimum. TTC is time-to-collision; ' +
  'PET is post-encroachment time. “None” means no finite value in that model. A negative PET means two agents ' +
  'occupied the same connected conflict zone at overlapping times.';

async function openList() {
  const requests = stubFetch(() =>
    ok(makePage([makeRow({ scenario_id: 'synthetic-0001' }), makeRow({ scenario_id: 'synthetic-0002' })], null)),
  );
  renderApp('/');
  await screen.findByRole('link', { name: 'synthetic-0001' });
  return requests;
}

describe('the hero', () => {
  it('has one h1, which asks the question the project answers, under an eyebrow and a lede', async () => {
    await openList();
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(HERO.h1);
    expect(screen.getByText(HERO.eyebrow)).toBeVisible();
    expect(screen.getByText(HERO.lede)).toBeVisible();
  });

  it('no longer has the old heading or eyebrow', async () => {
    await openList();
    expect(screen.queryByRole('heading', { name: 'Scenarios', level: 1 })).not.toBeInTheDocument();
    expect(screen.queryByText('EXPLORE THE EDGE CASES')).not.toBeInTheDocument();
    expect(screen.queryByText(/Stress-test the edge cases/)).not.toBeInTheDocument();
  });

  it('has an "Explore the results" link that lands on the explorer', async () => {
    await openList();
    const link = screen.getByRole('link', { name: 'Explore the results' });
    expect(link).toHaveAttribute('href', '#scenario-explorer');
    const target = document.getElementById('scenario-explorer');
    expect(target).not.toBeNull();
    expect(within(target!).getByRole('heading', { level: 2, name: 'Scenario explorer' })).toBeVisible();
    expect(within(target!).getByRole('link', { name: 'synthetic-0001' })).toBeVisible();
  });

  it('links to the Method page', async () => {
    await openList();
    const link = screen.getByRole('link', { name: /How the system works/ });
    expect(link).toHaveAttribute('href', '/method');
    expect(link).toHaveTextContent('How the system works →');
  });

  it('lists exactly the six technologies the repository uses', async () => {
    await openList();
    const items = within(screen.getByLabelText('Technology stack')).getAllByRole('listitem');
    expect(items.map((li) => li.textContent)).toEqual(STACK);
  });

  it('shows the four-stage pipeline and says its results are stored, not live', async () => {
    await openList();
    const panel = screen.getByLabelText('Analysis pipeline');
    expect(within(panel).getByText('offline search · stored results')).toBeVisible();
    const items = within(panel).getAllByRole('listitem');
    expect(items.map((li) => [li.querySelector('strong')!.textContent, li.querySelector('small')!.textContent])).toEqual(
      PIPELINE.map((p) => [...p]),
    );
    expect(within(panel).getByText('A stored result you can replay, with the settings that produced it.')).toBeVisible();
  });

  it('has three highlight cards, in order', async () => {
    await openList();
    const cards = within(screen.getByLabelText('Project highlights')).getAllByRole('article');
    expect(cards.map((c) => [c.querySelector('strong')!.textContent, c.querySelector('p')!.textContent])).toEqual(
      CARDS.map((c) => [...c]),
    );
  });
});

describe('the explorer under it', () => {
  it('is headed "STORED RESULTS" and "Scenario explorer", with the new introduction', async () => {
    await openList();
    expect(screen.getByText('STORED RESULTS')).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Scenario explorer', level: 2 })).toBeVisible();
    expect(screen.getByText(INTRO)).toBeVisible();
    expect(screen.queryByText(/Explore recorded driving scenes/)).not.toBeInTheDocument();
  });

  it('words the metric guide as decided: "Min perturbation", a finite search, and the PET sentence', async () => {
    await openList();
    expect(screen.getByText(GUIDE)).toBeInTheDocument();
    expect(screen.queryByText(/Stored rows should be re-scored/)).not.toBeInTheDocument();
    expect(screen.queryByText(/component-aware/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Min change/)).not.toBeInTheDocument();
  });

  it('still shows the ranking caption and the table, unchanged by this diff', async () => {
    await openList();
    expect(screen.getByText(/Ranked by fragility score, most fragile first\. The order comes from the server/)).toBeVisible();
    expect(screen.getByRole('columnheader', { name: 'Min perturbation' })).toBeInTheDocument();
  });
});
