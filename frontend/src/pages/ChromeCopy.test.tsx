import { screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { HealthResponse, StatsResponse } from '../api/client';
import { WAYMO_ATTRIBUTION } from '../components/Footer';
import { makePage } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The words around every page (the brand subtitle and the footer) and at the top of the Corpus page.
 * Their wording is pinned here; the words the site may not say are pinned in copyLint.test.tsx.
 */

afterEach(() => {
  vi.unstubAllGlobals();
});

const ok = (body: unknown): Reply => ({ status: 200, body });

const STATS: StatsResponse = {
  total_scenarios: 100, stress_tested: 18, collisions_found: 18, no_collision_found: 0,
  replay_infeasible: 2, heading_blend_singularity: 0, no_challenger: 0, stress_errors: 0,
  robustly_safe: 0, with_geometry: 20, fragility_min: 0.5, fragility_max: 4.5, fragility_mean: 2,
};
const HEALTH: HealthResponse = { status: 'ok', database: 'connected', postgis: '3.6' };

async function openCorpus() {
  stubFetch((req: RecordedRequest) => ok(req.path === '/health' ? HEALTH : STATS));
  renderApp('/stats');
  return screen.findByRole('heading', { name: 'Corpus', level: 1 });
}

describe('the brand subtitle', () => {
  it.each(['/', '/method'])('says "Adversarial scenario search" on %s', async (path) => {
    stubFetch(() => ok(makePage([], null)));
    renderApp(path);
    await screen.findByRole('navigation', { name: 'Main' });
    const subtitle = document.querySelector('.brand-subtitle')!;
    expect(subtitle).toHaveTextContent(/^Adversarial scenario search$/);
    expect(document.body).not.toHaveTextContent('Autonomous driving · scenario analysis');
  });

  it('is on the Corpus page too', async () => {
    await openCorpus();
    expect(document.querySelector('.brand-subtitle')).toHaveTextContent(/^Adversarial scenario search$/);
  });
});

describe('the footer', () => {
  async function footer() {
    stubFetch(() => ok(makePage([], null)));
    renderApp('/method');
    await screen.findByRole('navigation', { name: 'Main' });
    return document.querySelector('footer.site-footer') as HTMLElement;
  }

  it('names the project and says what it is, above the Waymo attribution', async () => {
    const el = await footer();
    const name = within(el).getByText('AV Scenario Stress-Tester');
    const what = within(el).getByText(
      'Independent engineering project · scenario stress-testing on Waymo Open Motion data',
    );
    const attribution = within(el).getByText(WAYMO_ATTRIBUTION);
    expect(name).toBeVisible();
    expect(what).toBeVisible();
    const before = (a: Node, b: Node) => Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
    expect(before(name, what)).toBe(true);
    expect(before(what, attribution)).toBe(true);
  });

  it('keeps the attribution as its own paragraph, equal to the constant exactly', async () => {
    const el = await footer();
    const paragraphs = [...el.querySelectorAll('p')];
    const last = paragraphs[paragraphs.length - 1]!;
    expect(last.textContent).toBe(WAYMO_ATTRIBUTION);
    // and it is the only paragraph that carries it, so it was not folded into the project line
    expect(paragraphs.filter((p) => p.textContent!.includes('Waymo Dataset License Agreement'))).toHaveLength(1);
  });
});

describe('the Corpus page header', () => {
  it('keeps "DATASET OVERVIEW" and the h1 "Corpus"', async () => {
    const h1 = await openCorpus();
    expect(h1).toHaveTextContent(/^Corpus$/);
    expect(screen.getByText('DATASET OVERVIEW')).toBeVisible();
    expect(within(screen.getByRole('navigation', { name: 'Main' })).getByRole('link', { name: 'Corpus' })).toHaveClass('active');
  });

  it('says which four things the page shows', async () => {
    await openCorpus();
    expect(
      screen.getByText("The scored sample, the stored search results, the latest attempts, and the service's health."),
    ).toBeVisible();
  });

  it('says the counts are the API\'s own and may not partition the corpus', async () => {
    await openCorpus();
    expect(
      screen.getByText(
        'Counts are shown exactly as the API reports them. They represent different fields and are not assumed to partition the corpus.',
      ),
    ).toBeVisible();
  });

  it('no longer says "Nothing on this page is summed or charted"', async () => {
    await openCorpus();
    expect(screen.queryByText(/Nothing on this page is summed or charted/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Counts as the API reports them\./)).not.toBeInTheDocument();
  });
});
