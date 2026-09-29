import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { makePage, makeRow } from '../test/fixtures';
import { stubFetch } from '../test/fetchStub';
import { renderApp as renderListPage } from '../test/renderApp';

// Contains every character base64 can produce that a URL would mangle if the client
// built or re-encoded the cursor itself.
const OPAQUE_CURSOR = 'eyJmIjogMS4zNDc1LCAiaWQiOiAic3luIn0+/==';

function renderedIds(): string[] {
  const [, ...bodyRows] = screen.getAllByRole('row');
  return bodyRows.map((row) => within(row).getAllByRole('cell')[1]!.textContent ?? '');
}

function scenariosOnly<T extends { path: string }>(requests: T[]): T[] {
  return requests.filter((r) => r.path === '/scenarios');
}

describe('ScenarioListPage — pagination', () => {
  it('asks for the first page with no cursor', async () => {
    const requests = stubFetch(() => ({ status: 200, body: makePage([makeRow()], null) }));
    renderListPage();

    await screen.findByText('synthetic-0001');
    const [first] = scenariosOnly(requests);
    expect(first!.query.has('cursor')).toBe(false);
    expect(first!.query.get('stress_tested_only')).toBe('false');
  });

  it('passes next_cursor back verbatim and appends the next page', async () => {
    const requests = stubFetch(({ query }) =>
      query.has('cursor')
        ? { status: 200, body: makePage([makeRow({ scenario_id: 'synthetic-b' })], null) }
        : { status: 200, body: makePage([makeRow({ scenario_id: 'synthetic-a' })], OPAQUE_CURSOR) },
    );
    renderListPage();

    await screen.findByText('synthetic-a');
    await userEvent.click(screen.getByRole('button', { name: 'Load more' }));
    await screen.findByText('synthetic-b');

    const second = scenariosOnly(requests)[1]!;
    expect(second.query.get('cursor')).toBe(OPAQUE_CURSOR);
    expect(renderedIds()).toEqual(['synthetic-a', 'synthetic-b']);
    expect(screen.getByText(/end of list/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load more' })).not.toBeInTheDocument();
  });

  it('offers no further page when next_cursor is null', async () => {
    stubFetch(() => ({ status: 200, body: makePage([makeRow()], null) }));
    renderListPage();

    await screen.findByText('synthetic-0001');
    expect(screen.queryByRole('button', { name: 'Load more' })).not.toBeInTheDocument();
    expect(screen.getByText(/end of list/)).toBeInTheDocument();
  });

  it('renders rows in server order, even when that is not fragility order', async () => {
    // Deliberately out of order: if the page ever re-sorts, this fails.
    stubFetch(() => ({
      status: 200,
      body: makePage(
        [
          makeRow({ scenario_id: 'synthetic-low', fragility_score: 0.5 }),
          makeRow({ scenario_id: 'synthetic-high', fragility_score: 3.0 }),
          makeRow({ scenario_id: 'synthetic-mid', fragility_score: 1.0 }),
        ],
        null,
      ),
    }));
    renderListPage();

    await screen.findByText('synthetic-low');
    expect(renderedIds()).toEqual(['synthetic-low', 'synthetic-high', 'synthetic-mid']);
  });

  it('forwards a valid ?limit= and drops an invalid one', async () => {
    const requests = stubFetch(() => ({ status: 200, body: makePage([makeRow()], null) }));
    const { unmount } = renderListPage('/?limit=2');
    await screen.findByText('synthetic-0001');
    expect(scenariosOnly(requests).at(-1)!.query.get('limit')).toBe('2');
    unmount();

    for (const bad of ['0', '201', 'abc', '2.5']) {
      const { unmount: done } = renderListPage(`/?limit=${bad}`);
      await screen.findByText('synthetic-0001');
      expect(scenariosOnly(requests).at(-1)!.query.has('limit')).toBe(false);
      done();
    }
  });
});

describe('ScenarioListPage — stress_tested_only', () => {
  it('restarts from the first page, with no cursor, when toggled', async () => {
    const requests = stubFetch(({ query }) => {
      if (query.get('stress_tested_only') === 'true') {
        return { status: 200, body: makePage([makeRow({ scenario_id: 'synthetic-tested' })], null) };
      }
      return query.has('cursor')
        ? { status: 200, body: makePage([makeRow({ scenario_id: 'synthetic-b' })], 'second-cursor') }
        : { status: 200, body: makePage([makeRow({ scenario_id: 'synthetic-a' })], OPAQUE_CURSOR) };
    });
    renderListPage();

    await screen.findByText('synthetic-a');
    await userEvent.click(screen.getByRole('button', { name: 'Load more' }));
    await screen.findByText('synthetic-b');

    await userEvent.click(screen.getByRole('checkbox', { name: 'Stress-tested only' }));
    await screen.findByText('synthetic-tested');

    const filtered = scenariosOnly(requests).filter(
      (r) => r.query.get('stress_tested_only') === 'true',
    );
    expect(filtered).toHaveLength(1);
    expect(filtered[0]!.query.has('cursor')).toBe(false);
    expect(renderedIds()).toEqual(['synthetic-tested']);
  });

  it('reads the filter from the URL', async () => {
    const requests = stubFetch(() => ({ status: 200, body: makePage([makeRow()], null) }));
    renderListPage('/?tested=1');

    await screen.findByText('synthetic-0001');
    expect(scenariosOnly(requests)[0]!.query.get('stress_tested_only')).toBe('true');
    expect(screen.getByRole('checkbox', { name: 'Stress-tested only' })).toBeChecked();
  });
});

describe('ScenarioListPage — values', () => {
  it('shows the 999 sentinel as "none", never as a number', async () => {
    stubFetch(() => ({
      status: 200,
      body: makePage([makeRow({ min_ttc: 999.0, min_pet: 999.0 })], null),
    }));
    renderListPage();

    await screen.findByText('synthetic-0001');
    expect(screen.getAllByText('none')).toHaveLength(2);
    expect(screen.queryByText(/999/)).not.toBeInTheDocument();
  });

  it('shows a negative PET as a real value, marked as overlap', async () => {
    stubFetch(() => ({ status: 200, body: makePage([makeRow({ min_pet: -0.3 })], null) }));
    renderListPage();

    await screen.findByText('synthetic-0001');
    expect(screen.getByText(/-0\.30 s/)).toBeInTheDocument();
    expect(screen.getByText('overlap')).toBeInTheDocument();
  });

  it('never presents a null min_perturbation as safe', async () => {
    stubFetch(() => ({
      status: 200,
      body: makePage(
        [
          makeRow({ scenario_id: 'synthetic-untested' }),
          makeRow({
            scenario_id: 'synthetic-clean',
            stress_tested: true,
            stress_attempted: true,
            stress_outcome: 'no_collision_found',
            last_attempt_outcome: 'no_collision_found',
            no_collision_found: true,
          }),
        ],
        null,
      ),
    }));
    renderListPage();

    await screen.findByText('synthetic-untested');
    expect(screen.getAllByText('not tested')).toHaveLength(2); // perturbation cell + outcome cell
    expect(screen.getByText('no collision found')).toBeInTheDocument();
    expect(screen.queryByText(/safe/i)).not.toBeInTheDocument();
  });
});

describe('ScenarioListPage — states', () => {
  it('explains an empty corpus', async () => {
    stubFetch(() => ({ status: 200, body: makePage([], null) }));
    renderListPage();
    expect(await screen.findByText('No scenarios have been scored yet.')).toBeInTheDocument();
  });

  it('explains an empty filtered list differently', async () => {
    stubFetch(() => ({ status: 200, body: makePage([], null) }));
    renderListPage('/?tested=1');
    expect(await screen.findByText(/No scenarios have a stored stress-test result yet/)).toBeInTheDocument();
  });

  it('reports a 503 as busy, not broken', async () => {
    stubFetch(() => ({
      status: 503,
      body: { detail: 'All database connections are busy. Retry shortly.' },
      headers: { 'Retry-After': '1' },
    }));
    renderListPage();
    expect(await screen.findByRole('alert')).toHaveTextContent(/The API is busy/);
  });

  it('reports an unreachable API', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('Failed to fetch');
    });
    renderListPage();
    expect(await screen.findByRole('alert')).toHaveTextContent(/Could not reach the API/);
  });

  it('keeps loaded rows when a later page fails', async () => {
    stubFetch(({ query }) =>
      query.has('cursor')
        ? { status: 500, body: { detail: 'boom' } }
        : { status: 200, body: makePage([makeRow({ scenario_id: 'synthetic-a' })], OPAQUE_CURSOR) },
    );
    renderListPage();

    await screen.findByText('synthetic-a');
    await userEvent.click(screen.getByRole('button', { name: 'Load more' }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('500: boom'));
    expect(renderedIds()).toEqual(['synthetic-a']);
  });
});
