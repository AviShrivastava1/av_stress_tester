import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RouterProvider, createMemoryRouter } from 'react-router';
import { describe, expect, it, vi } from 'vitest';
import { routes } from '../App';
import type { StatsResponse } from '../api/client';
import { makePage, makeRow } from '../test/fixtures';
import type { RecordedRequest, Reply } from '../test/fetchStub';

/*
 * The outcome filter and the sort on the list (pages/ScenarioListPage.tsx, domain/listView.ts). They live in the URL beside
 * `tested` and `limit`, so a reload or a shared link gives the same view, and each change is a history entry, so Back undoes
 * it. The API filters on one thing only (stress_tested_only), so the page loads every remaining page before it applies a
 * filter or a sort, and says how many rows are loaded of how many exist while it does. It never shows a partly loaded list
 * as if it were the whole one.
 *
 * Five rows over three pages of two (page size 2), the API's order being the ranking:
 *   r1 collision found, 0.30   r2 replay infeasible, none   r3 collision found, 0.05   r4 not tested   r5 collision found, 0.20
 */

const ROWS = [
  makeRow({ scenario_id: 'r1', min_perturbation: 0.3, stress_outcome: 'collision_found', stress_tested: true, fragility_score: 5 }),
  makeRow({ scenario_id: 'r2', stress_outcome: 'replay_infeasible', stress_tested: true, fragility_score: 4 }),
  makeRow({ scenario_id: 'r3', min_perturbation: 0.05, stress_outcome: 'collision_found', stress_tested: true, fragility_score: 3 }),
  makeRow({ scenario_id: 'r4', fragility_score: 2 }),
  makeRow({ scenario_id: 'r5', min_perturbation: 0.2, stress_outcome: 'collision_found', stress_tested: true, fragility_score: 1 }),
];
const PAGE_OF: Record<string, [typeof ROWS, string | null]> = {
  first: [ROWS.slice(0, 2), 'c1'],
  c1: [ROWS.slice(2, 4), 'c2'],
  c2: [ROWS.slice(4), null],
};
const STATS = {
  total_scenarios: 5,
  stress_tested: 4,
  collisions_found: 3,
  no_collision_found: 0,
  replay_infeasible: 1,
  heading_blend_singularity: 0,
  no_challenger: 0,
  stress_errors: 0,
  robustly_safe: 0,
  with_geometry: 0,
  fragility_min: 1,
  fragility_max: 5,
  fragility_mean: 3,
} satisfies StatsResponse;

interface Options {
  /** The first page holds one row instead of two. */
  singleFirstRow?: boolean;
  /** Cursors whose page is held until `release(cursor)`. */
  gated?: string[];
  /** Cursors whose page fails with a 500 until `fixed.add(cursor)`. */
  failing?: string[];
  stats?: 'ok' | 'fail';
}

function installServer({ singleFirstRow = false, gated = [], failing = [], stats = 'ok' }: Options = {}) {
  const requests: RecordedRequest[] = [];
  const waiting = new Map<string, () => void>();
  const fixed = new Set<string>();

  function reply(req: RecordedRequest): Reply {
    if (req.path === '/stats') {
      return stats === 'ok' ? { status: 200, body: STATS } : { status: 500, body: { detail: 'stats down' } };
    }
    const cursor = req.query.get('cursor') ?? 'first';
    if (failing.includes(cursor) && !fixed.has(cursor)) return { status: 500, body: { detail: 'page failed' } };
    const [items, next] = PAGE_OF[cursor]!;
    return { status: 200, body: makePage(singleFirstRow && cursor === 'first' ? items.slice(0, 1) : items, next) };
  }

  vi.stubGlobal(
    'fetch',
    (input: Request) =>
      new Promise<Response>((resolve) => {
        const url = new URL(input.url);
        const req: RecordedRequest = { path: url.pathname, query: url.searchParams };
        requests.push(req);
        const respond = () => {
          const { status, body } = reply(req);
          resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }));
        };
        const cursor = req.query.get('cursor');
        if (req.path === '/scenarios' && cursor !== null && gated.includes(cursor) && !fixed.has(cursor)) {
          waiting.set(cursor, respond);
        } else {
          respond();
        }
      }),
  );
  return {
    requests,
    pagesRequested: () => requests.filter((r) => r.path === '/scenarios').map((r) => r.query.get('cursor') ?? 'first'),
    statsRequested: () => requests.filter((r) => r.path === '/stats').length,
    release: (cursor: string) => {
      fixed.add(cursor);
      waiting.get(cursor)?.();
      waiting.delete(cursor);
    },
    fix: (cursor: string) => fixed.add(cursor),
  };
}

function renderAt(url: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter(routes, { initialEntries: [url] });
  const view = render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return { router, ...view };
}

const outcomeSelect = () => screen.getByRole('combobox', { name: 'Outcome' }) as HTMLSelectElement;
const sortSelect = () => screen.getByRole('combobox', { name: 'Sort' }) as HTMLSelectElement;
const bodyRows = () => screen.getAllByRole('row').slice(1);
const ids = () => bodyRows().map((row) => within(row).getAllByRole('cell')[1]!.textContent);
const ranks = () => bodyRows().map((row) => within(row).getAllByRole('cell')[0]!.textContent);
const search = (router: ReturnType<typeof renderAt>['router']) => router.state.location.search;
const progress = () => document.querySelector('.status[role="status"]')?.textContent ?? null;
const viewNote = () => document.querySelector('.view-note')?.textContent ?? null;

describe('the view comes from the URL', () => {
  it('applies the outcome and the sort named in the address', async () => {
    installServer();
    renderAt('/?outcome=collision_found&sort=perturbation');
    await waitFor(() => expect(ids()).toEqual(['r3', 'r5', 'r1']));
    expect(ranks()).toEqual(['3', '5', '1']);
    expect(outcomeSelect().value).toBe('collision_found');
    expect(sortSelect().value).toBe('perturbation');
  });

  it('ignores a value it does not know, and keeps a valid one beside it', async () => {
    const server = installServer();
    renderAt('/?outcome=bogus&sort=bogus');
    await screen.findByRole('table');
    expect(ids()).toEqual(['r1', 'r2']);
    expect(outcomeSelect().value).toBe('');
    expect(sortSelect().value).toBe('fragility');
    // An ignored value is no view at all: the other pages are not fetched and neither are the counts.
    expect(server.pagesRequested()).toEqual(['first']);
    expect(server.statsRequested()).toBe(0);
  });

  it('keeps the valid sort when the outcome is unknown', async () => {
    installServer();
    renderAt('/?outcome=bogus&sort=id');
    await waitFor(() => expect(ids()).toEqual(['r1', 'r2', 'r3', 'r4', 'r5']));
    expect(sortSelect().value).toBe('id');
    expect(outcomeSelect().value).toBe('');
  });

  it('offers every outcome and the three sorts', async () => {
    installServer();
    renderAt('/');
    await screen.findByRole('table');
    expect(within(outcomeSelect()).getAllByRole('option').map((o) => o.textContent)).toEqual([
      'All outcomes',
      'collision found',
      'no collision found',
      'replay infeasible',
      'heading-blend singularity',
      'no challenger',
      'error',
      'outcome not recorded',
      'no stored result',
      'not tested',
      'unrecognized',
    ]);
    expect(within(sortSelect()).getAllByRole('option').map((o) => o.textContent)).toEqual([
      'Fragility, highest first',
      'Min perturbation, smallest first',
      'Scenario ID',
    ]);
  });
});

describe('the view goes to the URL, and Back undoes it', () => {
  it('writes each choice as a query value and drops it when it is the default', async () => {
    installServer();
    const { router } = renderAt('/');
    await screen.findByRole('table');

    await userEvent.selectOptions(outcomeSelect(), 'collision_found');
    expect(search(router)).toBe('?outcome=collision_found');
    await userEvent.selectOptions(sortSelect(), 'perturbation');
    expect(search(router)).toBe('?outcome=collision_found&sort=perturbation');
    await userEvent.selectOptions(outcomeSelect(), '');
    expect(search(router)).toBe('?sort=perturbation');
    await userEvent.selectOptions(sortSelect(), 'fragility');
    expect(search(router)).toBe('');
  });

  it('keeps tested and limit when the view changes', async () => {
    installServer();
    const { router } = renderAt('/?tested=1&limit=2');
    await screen.findByRole('table');
    await userEvent.selectOptions(outcomeSelect(), 'collision_found');
    expect(new URLSearchParams(search(router)).toString()).toBe('tested=1&limit=2&outcome=collision_found');
  });

  it('Back goes to the view before, and the table and the controls follow', async () => {
    installServer();
    const { router } = renderAt('/');
    await screen.findByRole('table');
    await userEvent.selectOptions(outcomeSelect(), 'collision_found');
    await userEvent.selectOptions(sortSelect(), 'perturbation');
    await waitFor(() => expect(ids()).toEqual(['r3', 'r5', 'r1']));

    await act(async () => {
      await router.navigate(-1);
    });
    expect(search(router)).toBe('?outcome=collision_found');
    await waitFor(() => expect(ids()).toEqual(['r1', 'r3', 'r5']));
    expect(sortSelect().value).toBe('fragility');
    expect(outcomeSelect().value).toBe('collision_found');

    await act(async () => {
      await router.navigate(-1);
    });
    expect(search(router)).toBe('');
    await waitFor(() => expect(ids()).toEqual(['r1', 'r2', 'r3', 'r4', 'r5']));
  });

  it('a reload of the address gives the same view', async () => {
    installServer();
    const first = renderAt('/');
    await screen.findByRole('table');
    await userEvent.selectOptions(outcomeSelect(), 'collision_found');
    await userEvent.selectOptions(sortSelect(), 'perturbation');
    await waitFor(() => expect(ids()).toEqual(['r3', 'r5', 'r1']));
    const address = `/${search(first.router)}`;
    const note = viewNote();
    first.unmount();

    renderAt(address);
    await waitFor(() => expect(ids()).toEqual(['r3', 'r5', 'r1']));
    expect(viewNote()).toBe(note);
    expect(outcomeSelect().value).toBe('collision_found');
    expect(sortSelect().value).toBe('perturbation');
  });
});

describe('a partly loaded list is never shown as the whole one', () => {
  it('loads the rest, says how many rows are loaded of how many exist, and shows no table meanwhile', async () => {
    const server = installServer({ gated: ['c1', 'c2'] });
    renderAt('/?outcome=collision_found');

    await waitFor(() =>
      expect(progress()).toBe('Loading every row so the filter covers all of them: 2 of 5 loaded.'),
    );
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'r1' })).not.toBeInTheDocument(); // no row link (the hero's link is named for its aria-label)
    expect(server.pagesRequested()).toEqual(['first', 'c1']);

    await act(async () => server.release('c1'));
    await waitFor(() =>
      expect(progress()).toBe('Loading every row so the filter covers all of them: 4 of 5 loaded.'),
    );
    expect(screen.queryByRole('table')).not.toBeInTheDocument();

    await act(async () => server.release('c2'));
    await waitFor(() => expect(ids()).toEqual(['r1', 'r3', 'r5']));
    expect(progress()).toBeNull();
    expect(server.pagesRequested()).toEqual(['first', 'c1', 'c2']);
  });

  it('names what is being applied: a sort, or a filter and a sort', async () => {
    installServer({ gated: ['c1'] });
    const { unmount } = renderAt('/?sort=id');
    await waitFor(() => expect(progress()).toBe('Loading every row so the sort covers all of them: 2 of 5 loaded.'));
    unmount();

    installServer({ gated: ['c1'] });
    renderAt('/?sort=id&outcome=error');
    await waitFor(() =>
      expect(progress()).toBe('Loading every row so the filter and sort covers all of them: 2 of 5 loaded.'),
    );
  });

  it('counts against the tested-only total when that filter is on', async () => {
    installServer({ gated: ['c1'] });
    renderAt('/?tested=1&outcome=collision_found');
    await waitFor(() =>
      expect(progress()).toBe('Loading every row so the filter covers all of them: 2 of 4 loaded.'),
    );
  });

  it('without the total, says how many are loaded so far, not a made-up total', async () => {
    installServer({ gated: ['c1'], stats: 'fail' });
    renderAt('/?outcome=collision_found');
    await waitFor(() =>
      expect(progress()).toBe('Loading every row so the filter covers all of them: 2 rows loaded so far.'),
    );
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('a single loaded row is "1 row", not "1 rows"', async () => {
    installServer({ singleFirstRow: true, gated: ['c1'], stats: 'fail' });
    renderAt('/?outcome=collision_found');
    await waitFor(() =>
      expect(progress()).toBe('Loading every row so the filter covers all of them: 1 row loaded so far.'),
    );
  });

  it('if a page cannot be loaded, says the view covers only the loaded rows, and Load more retries', async () => {
    const server = installServer({ failing: ['c1'] });
    renderAt('/?outcome=collision_found');

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Not every row could be loaded: this view covers only 2 of 5 rows.');
    expect(alert).toHaveTextContent('500: page failed');
    expect(ids()).toEqual(['r1']); // the loaded rows that match, under the warning
    expect(viewNote()).toBeNull(); // not the "every row has been loaded" note
    expect(server.pagesRequested()).toEqual(['first', 'c1']); // no retry loop

    server.fix('c1');
    await userEvent.click(screen.getByRole('button', { name: 'Load more' }));
    await waitFor(() => expect(ids()).toEqual(['r1', 'r3', 'r5']));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('applies no view and loads nothing more when the view is the default', async () => {
    const server = installServer();
    renderAt('/');
    await screen.findByRole('table');
    expect(ids()).toEqual(['r1', 'r2']);
    expect(screen.getByRole('button', { name: 'Load more' })).toBeInTheDocument();
    expect(server.pagesRequested()).toEqual(['first']);
    expect(server.statsRequested()).toBe(0);
  });
});

describe('once every row is loaded', () => {
  it('says what is shown and where the numbers come from', async () => {
    installServer();
    renderAt('/?outcome=collision_found&sort=perturbation');
    await waitFor(() => expect(ids()).toEqual(['r3', 'r5', 'r1']));
    expect(viewNote()).toBe(
      'Showing 3 of 5 rows with the outcome “collision found”, sorted by min perturbation, smallest first. ' +
        'Filtering and sorting happen in this browser, after every row has been loaded. ' +
        'The # column is each row’s place in the fragility ranking.',
    );
    expect(screen.getByText(/end of list/)).toHaveTextContent('3 of 5 shown');
  });

  it('describes a filter alone and a sort alone', async () => {
    installServer();
    const { unmount } = renderAt('/?outcome=not_tested');
    await waitFor(() => expect(ids()).toEqual(['r4']));
    expect(viewNote()).toMatch(/^Showing 1 of 5 rows with the outcome “not tested”\. Filtering/);
    unmount();

    renderAt('/?sort=id');
    await waitFor(() => expect(ids()).toHaveLength(5));
    expect(viewNote()).toMatch(/^All 5 rows, sorted by scenario ID\. Filtering/);
  });

  it('keeps each row’s place in the fragility ranking in the # column', async () => {
    installServer();
    renderAt('/?outcome=collision_found');
    await waitFor(() => expect(ranks()).toEqual(['1', '3', '5']));
  });

  it('names the table by its order for assistive technology', async () => {
    installServer();
    const { unmount } = renderAt('/');
    await screen.findByRole('table');
    expect(screen.getByRole('table')).toHaveAccessibleName('Scenarios ordered by fragility score, highest first');
    unmount();

    installServer();
    renderAt('/?sort=id');
    await waitFor(() => expect(ids()).toHaveLength(5));
    expect(screen.getByRole('table')).toHaveAccessibleName('Scenarios ordered by scenario ID');
  });

  it('says so when no row has the outcome', async () => {
    installServer();
    renderAt('/?outcome=error');
    expect(await screen.findByText('None of the 5 rows has the outcome “error”.')).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('changing the view afterwards does not fetch anything again', async () => {
    const server = installServer();
    renderAt('/?sort=id');
    await waitFor(() => expect(ids()).toHaveLength(5));
    const before = server.requests.length;
    await userEvent.selectOptions(outcomeSelect(), 'collision_found');
    await userEvent.selectOptions(sortSelect(), 'perturbation');
    await waitFor(() => expect(ids()).toEqual(['r3', 'r5', 'r1']));
    expect(server.requests.length).toBe(before);
  });
});
