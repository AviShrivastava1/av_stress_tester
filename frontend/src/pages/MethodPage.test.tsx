import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { makePage } from '../test/fixtures';
import { stubFetch, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The Method page: how a recorded scene becomes a stored, replayable result, and what the output
 * does not mean. It is static: it must not call the API.
 */

afterEach(() => {
  vi.unstubAllGlobals();
});

const ok = (body: unknown): Reply => ({ status: 200, body });

const STEPS = [
  {
    title: 'Ingest recorded motion',
    body: 'Parse masked agent trajectories from WOMD and preserve the local metric coordinate frame.',
    meta: 'WOMD · NumPy · validity masks',
  },
  {
    title: 'Rank scenes by fragility',
    body: 'Measure SDC-restricted TTC and component-aware PET, then rank the corpus by a fragility score.',
    meta: 'TTC · PET · deterministic ranking',
  },
  {
    title: 'Search a constrained replay',
    body:
      'Perturb one selected challenger in a four-dimensional, model-specific space using Differential ' +
      'Evolution and optional autograd refinement.',
    meta: 'SciPy · optional PyTorch · kinematic models',
  },
  {
    title: 'Verify before publishing',
    body:
      'Refuse unfaithful baselines, check contact with oriented boxes, and bind every result to its scene ' +
      'and replay provenance.',
    meta: 'Replay gates · float64 geometry · hashes',
  },
  {
    title: 'Serve a stored replay',
    body:
      'Export geometry once, keep the production API read-only, and let the browser compare logged and ' +
      'perturbed paths frame by frame.',
    meta: 'PostGIS · FastAPI · React',
  },
] as const;

const PRINCIPLES = [
  {
    title: 'Failure is a first-class result',
    body:
      'Replay drift, speed discontinuities, heading singularities, stale exports, and ordinary search ' +
      'misses stay distinct from one another.',
  },
  {
    title: 'The online path stays lightweight',
    body:
      'Waymo parsing and optimization run offline. The deployed API reads Postgres/PostGIS and never needs ' +
      'a shard or TensorFlow at request time.',
  },
  {
    title: 'Claims carry their provenance',
    body:
      'Scene fingerprints, SDC identity, replay settings, search budgets, and run IDs travel with stored ' +
      'results and exported geometry.',
  },
] as const;

const LIMITS = [
  'The minimum perturbation is the best result found within a finite search budget, not a proven global minimum.',
  'Each run searches one heuristically selected challenger, not every possible agent interaction.',
  'A collision is an oriented-box intersection at stored coordinate precision, not a forecast of real-world impact.',
  'The browser draws each box at the size recorded for that frame, which is the size the collision check used. ' +
    'Data exported before per-frame sizes were stored is drawn at one size per agent, and the note under the ' +
    'scene says so.',
] as const;

async function openMethod() {
  const requests = stubFetch(() => ok(makePage([], null)));
  renderApp('/method');
  // The page itself must be there, so the "nothing is hidden / no request / no banned word" checks are not vacuous.
  const h1 = await screen.findByRole('heading', { level: 1, name: /stored, replayable result/ });
  return { requests, h1 };
}

describe('the Method page', () => {
  it('is a page of its own, with one h1, an eyebrow and a lede', async () => {
    const { h1 } = await openMethod();
    expect(h1).toHaveTextContent('From a recorded scene to a stored, replayable result.');
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1);
    expect(screen.getByText('SYSTEM DESIGN')).toBeVisible();
    expect(
      screen.getByText(
        'This is not a collision-probability model. It asks a narrower question: under a bounded kinematic ' +
          "replay, how small a change to one agent's motion produces geometric contact?",
      ),
    ).toBeVisible();
    expect(screen.queryByText('Page not found')).not.toBeInTheDocument();
  });

  it('makes no API request', async () => {
    const { requests } = await openMethod();
    expect(requests).toEqual([]);
  });

  it('lists the five pipeline steps in order, each with its sentence and its tools', async () => {
    await openMethod();
    const list = screen.getByLabelText('System pipeline');
    const steps = within(list).getAllByRole('article');
    expect(steps).toHaveLength(STEPS.length);
    STEPS.forEach((step, i) => {
      const el = steps[i]!;
      expect(within(el).getByRole('heading', { level: 2 })).toHaveTextContent(step.title);
      expect(within(el).getByText(step.body)).toBeVisible();
      expect(within(el).getByText(step.meta)).toBeVisible();
      expect(el).toHaveTextContent(String(i + 1).padStart(2, '0'));
    });
  });

  it('says "Serve a stored replay" and does not call the ranking "transparent"', async () => {
    await openMethod();
    expect(screen.getByRole('heading', { name: 'Rank scenes by fragility', level: 2 })).toBeVisible();
    expect(screen.queryByRole('heading', { name: 'Rank dangerous scenes' })).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Serve a stored replay', level: 2 })).toBeVisible();
    expect(screen.queryByRole('heading', { name: 'Serve an auditable replay' })).not.toBeInTheDocument();
    expect(screen.queryByText(/transparent/i)).not.toBeInTheDocument();
  });

  it('gives the three engineering principles', async () => {
    await openMethod();
    expect(screen.getByText('ENGINEERING PRINCIPLES')).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Built to expose assumptions, not hide them.', level: 2 })).toBeVisible();
    for (const p of PRINCIPLES) {
      expect(screen.getByRole('heading', { name: p.title, level: 3 })).toBeVisible();
      expect(screen.getByText(p.body)).toBeVisible();
    }
  });

  it('states what the output does not mean, in four visible bullets, the fourth describing what the browser does now', async () => {
    await openMethod();
    expect(screen.getByText('INTERPRETATION')).toBeVisible();
    const section = screen.getByRole('heading', { name: 'What the output does—and does not—mean', level: 2 }).closest('section')!;
    const bullets = within(section).getAllByRole('listitem');
    expect(bullets.map((li) => li.textContent)).toEqual([...LIMITS]);
    for (const li of bullets) expect(li).toBeVisible();
    const fourth = bullets[3]!.textContent!;
    expect(fourth).not.toMatch(/first-observed|first observed/);
    expect(fourth).not.toMatch(/currently draws/);
  });

  it('hides nothing in a collapsed element', async () => {
    await openMethod();
    expect(document.querySelectorAll('main details')).toHaveLength(0);
  });

  it('uses none of the words the project cannot back: live, safe, safety, end-to-end, adversarial', async () => {
    await openMethod();
    const text = document.querySelector('main')!.textContent!;
    expect(text).not.toMatch(/\b(live|safe|safety|end-to-end|adversarial)\b/i);
  });
});

describe('the navigation to it', () => {
  it('has a "Method" link between "Scenarios" and "Corpus", active on the Method page only', async () => {
    await openMethod();
    const nav = screen.getByRole('navigation', { name: 'Main' });
    const links = within(nav).getAllByRole('link');
    expect(links.map((a) => a.textContent)).toEqual(['Scenarios', 'Method', 'Corpus']);
    expect(within(nav).getByRole('link', { name: 'Method' })).toHaveAttribute('href', '/method');
    expect(within(nav).getByRole('link', { name: 'Method' })).toHaveClass('active');
    expect(within(nav).getByRole('link', { name: 'Scenarios' })).not.toHaveClass('active');
    expect(within(nav).getByRole('link', { name: 'Corpus' })).not.toHaveClass('active');
  });

  it('opens the page from the list without leaving the app', async () => {
    stubFetch(() => ok(makePage([], null)));
    renderApp('/');
    const nav = await screen.findByRole('navigation', { name: 'Main' });
    await userEvent.click(within(nav).getByRole('link', { name: 'Method' }));
    expect(await screen.findByRole('heading', { level: 1, name: /stored, replayable result/ })).toBeVisible();
    expect(within(screen.getByRole('navigation', { name: 'Main' })).getByRole('link', { name: 'Method' })).toHaveClass('active');
  });
});
