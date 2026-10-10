import { screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { makePage, makeRow } from '../test/fixtures';
import { stubFetch, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The top of the scenario list: what the project is, in words each of which the repository backs, and the way down to
 * the stored results. It must not read as if a visitor can run a search here: the search ran offline, and the page
 * shows stored results. The note that says so, and that "smallest" means the smallest found, is visible with no click.
 */

afterEach(() => {
  vi.unstubAllGlobals();
});

const ok = (body: unknown): Reply => ({ status: 200, body });

const HERO = {
  eyebrow: 'COUNTERFACTUAL MOTION SEARCH',
  h1: 'Find the smallest change that breaks the scene.',
  lede: 'Search recorded AV motion. Replay the counterfactual. Inspect every frame.',
  note:
    'Smallest found, not proven: one challenger per scenario, a finite search. Searches ran offline; this site shows ' +
    'stored results. Scenarios with equal fragility are listed by scenario ID, which says nothing about which is more fragile.',
  illustration: 'ILLUSTRATION · not measured data',
} as const;

const SIGNATURE = [
  ['Search', 'Differential Evolution'],
  ['Replay', 'bicycle / linear kinematics'],
  ['Contact', 'float64 oriented boxes'],
  ['Storage', 'PostGIS, read-only API'],
] as const;

const GUIDE =
  'Fragility ranks the recorded scene; it is not a collision probability. Min perturbation is the smallest ' +
  'weighted perturbation found within a finite search budget, not a proven global minimum. TTC is ' +
  'time-to-collision; PET is post-encroachment time. “None” means no finite value in that model. Negative PET ' +
  'marks overlapping occupancy of one connected conflict zone.';

const withResult = (id: string, min: number) =>
  makeRow({ scenario_id: id, stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: min });

async function openList(rows = [withResult('synthetic-0001', 0.02), makeRow({ scenario_id: 'synthetic-0002' })]) {
  const requests = stubFetch(() => ok(makePage(rows, null)));
  renderApp('/');
  await screen.findByRole('link', { name: rows[0]!.scenario_id });
  return requests;
}
const hero = () => document.querySelector('section.project-hero') as HTMLElement;

describe('the hero', () => {
  it('has one h1 under an eyebrow, and a lede, all visible', async () => {
    await openList();
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(HERO.h1);
    expect(screen.getByText(HERO.eyebrow)).toBeVisible();
    expect(screen.getByText(HERO.lede)).toBeVisible();
  });

  it('says "counterfactual" in the eyebrow and the lede, in the markup and not hidden', async () => {
    await openList();
    expect(within(hero()).getByText(HERO.eyebrow)).toBeVisible();
    expect(within(hero()).getByText(HERO.lede)).toBeVisible();
    expect(hero().textContent!.match(/counterfactual/gi)).toHaveLength(2);
  });

  it('does not bring back the old hero, the stack list, the pipeline panel or the highlight cards', async () => {
    await openList();
    expect(screen.queryByText(/How small a change turns/)).not.toBeInTheDocument();
    expect(screen.queryByText('AUTONOMOUS-VEHICLE SCENARIO STRESS-TESTING')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Technology stack')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Analysis pipeline')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Project highlights')).not.toBeInTheDocument();
    expect(screen.queryByText('offline search · stored results')).not.toBeInTheDocument();
    expect(screen.queryByText(/Open a scene to replay it/)).not.toBeInTheDocument();
  });

  it('has "Explore scenarios" and "Inspect the system", whose names are the words alone', async () => {
    await openList();
    const explore = screen.getByRole('link', { name: 'Explore scenarios' });
    const method = screen.getByRole('link', { name: 'Inspect the system' });
    expect(explore).toHaveAttribute('href', '#scenario-explorer');
    expect(method).toHaveAttribute('href', '/method');
    // the arrows are aria-hidden spans in the markup, not generated content
    expect(explore.querySelector('span[aria-hidden="true"]')).toHaveTextContent('↓');
    expect(method.querySelector('span[aria-hidden="true"]')).toHaveTextContent('↗');
  });

  it('says in a visible note that the smallest change is the smallest found, that the search ran offline, and what a tie means', async () => {
    await openList();
    const note = hero().querySelector('.hero-note') as HTMLElement;
    expect(note).not.toBeNull();
    expect(note).toBeVisible();
    expect(note).toHaveTextContent(HERO.note);
    expect(note.closest('details')).toBeNull(); // no click needed
    for (const word of ['not proven', 'offline', 'stored', 'scenario ID', 'finite search']) expect(note.textContent).toContain(word);
  });

  it('offers no search in the hero: no field, no button, only links', async () => {
    await openList();
    const inHero = within(hero());
    expect(inHero.queryByRole('searchbox')).not.toBeInTheDocument();
    expect(inHero.queryByRole('textbox')).not.toBeInTheDocument();
    expect(inHero.queryByRole('button')).not.toBeInTheDocument();
    expect(hero().querySelector('form, input, select, textarea')).toBeNull();
  });

  it('labels the drawing "ILLUSTRATION · not measured data", beside the drawing, and hides the drawing from screen readers', async () => {
    await openList();
    const figure = hero().querySelector('figure.hero-visual')!;
    const caption = figure.querySelector('figcaption')!;
    expect(caption).toHaveTextContent(HERO.illustration);
    expect(caption).toBeVisible();
    const svg = figure.querySelector('svg.hero-trace')!;
    expect(svg).toHaveAttribute('aria-hidden', 'true');
    expect([...svg.querySelectorAll('text')].map((t) => t.textContent)).toEqual(['RECORDED / SDC', 'PERTURBED / CHALLENGER', 'CONTACT']);
  });

  it('keeps the CONTACT label next to a mark: a dot, a ring and a leader line to the label', async () => {
    await openList();
    const svg = hero().querySelector('svg.hero-trace')!;
    expect(svg.querySelector('circle.trace-contact')).not.toBeNull();
    expect(svg.querySelector('circle.trace-contact-pulse')).not.toBeNull();
    expect(svg.querySelector('line.trace-leader')).not.toBeNull();
  });

  it('shows four signals: search, replay, contact, storage', async () => {
    await openList();
    const signature = screen.getByLabelText('System components');
    const items = [...signature.querySelectorAll(':scope > div')];
    expect(items.map((item) => [item.querySelector('dt')!.textContent, item.querySelector('dd')!.textContent])).toEqual(
      SIGNATURE.map((item) => [...item]),
    );
  });
});

describe('the stored replay link', () => {
  it('opens the first loaded row that has a stored perturbation, and says that is what it is', async () => {
    await openList([makeRow({ scenario_id: 'synthetic-0001' }), withResult('synthetic-0002', 0.02), withResult('synthetic-0003', 0.5)]);
    const link = screen.getByRole('link', { name: 'Open stored replay for scenario synthetic-0002' });
    expect(link).toHaveAttribute('href', '/scenarios/synthetic-0002');
    expect(link).toHaveTextContent('Stored replay · row 02');
    expect(link).toHaveTextContent('Min perturbation 0.020');
  });

  it('is not called "featured", and the row number is the row\'s # in the table', async () => {
    await openList([makeRow({ scenario_id: 'synthetic-0001' }), withResult('synthetic-0002', 0.02)]);
    expect(screen.queryByText(/featured/i)).not.toBeInTheDocument();
    expect(document.querySelector('.hero-replay')!.getAttribute('aria-label')).not.toMatch(/featured/i);
    const row = screen.getAllByRole('link', { name: 'synthetic-0002' })[0]!.closest('tr')!;
    expect(within(row).getAllByRole('cell')[0]).toHaveTextContent('2');
  });

  it('does not appear when no loaded row has a stored perturbation', async () => {
    await openList([makeRow({ scenario_id: 'synthetic-empty' })]);
    expect(screen.queryByRole('link', { name: /Open stored replay/ })).not.toBeInTheDocument();
  });
});

describe('the index under it', () => {
  it('is the target of the hero action, headed "RANKED RESULT SET" and "Scenario index", labelled Waymo Open Dataset', async () => {
    await openList();
    const target = document.getElementById('scenario-explorer');
    expect(target).not.toBeNull();
    expect(within(target!).getByText('RANKED RESULT SET')).toBeVisible();
    expect(within(target!).getByRole('heading', { level: 2, name: 'Scenario index' })).toBeVisible();
    expect(target!.querySelector('.dataset-label')).toHaveTextContent(/^Waymo Open Dataset$/);
    expect(within(target!).getByRole('link', { name: 'synthetic-0001' })).toBeVisible();
    expect(screen.queryByText('Scenario explorer')).not.toBeInTheDocument();
    expect(screen.queryByText('STORED RESULTS')).not.toBeInTheDocument();
  });

  it('keeps the full caption visible with no click: ties, and "not a proven minimum"', async () => {
    await openList();
    const caption = document.querySelector('p.caption') as HTMLElement;
    expect(caption).toBeVisible();
    expect(caption.closest('details')).toBeNull();
    expect(caption).toHaveTextContent(/Equal scores are tie-broken by scenario ID; position within a tie does not mean one scenario is more fragile than another/);
    expect(caption).toHaveTextContent(/not a proven minimum/);
  });

  it('words the metric definitions as decided, and does not bring back the old phrases', async () => {
    await openList();
    expect(screen.getByText('Metric definitions')).toBeVisible();
    expect(screen.queryByText('How to read the results')).not.toBeInTheDocument();
    expect(screen.getByText(GUIDE)).toBeInTheDocument();
    expect(screen.queryByText(/Stored rows should be re-scored/)).not.toBeInTheDocument();
    expect(screen.queryByText(/component-aware/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Min change/)).not.toBeInTheDocument();
  });

  it('keeps the structured table, named by its order', async () => {
    await openList();
    expect(screen.getByRole('columnheader', { name: 'Min perturbation' })).toBeInTheDocument();
    expect(screen.getByRole('table')).toHaveAccessibleName('Scenarios ordered by fragility score, highest first');
  });

  it('names each table link by the scenario ID alone, and the lookup button "Open"', async () => {
    await openList([withResult('synthetic-0001', 0.02), makeRow({ scenario_id: 'synthetic-0002' }), makeRow({ scenario_id: 'synthetic-0003' })]);
    const table = screen.getByRole('table');
    const links = within(table).getAllByRole('link');
    expect(links.map((a) => a.textContent)).toEqual(['synthetic-0001', 'synthetic-0002', 'synthetic-0003']);
    for (const a of links) expect(a).toHaveAccessibleName(a.textContent!);
    expect(screen.getByRole('button', { name: 'Open' })).toBeInTheDocument();
  });
});
