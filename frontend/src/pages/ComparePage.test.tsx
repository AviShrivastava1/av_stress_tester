import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { routes } from '../App';
import { makeDetail, makePerturbed, makeTrajectories } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';
import { cssRules } from '../test/cssRules';
import { readFrontendFile } from '../test/readFrontendFile';

/*
 * The comparison page, /compare?a=<id>&b=<id> (pages/ComparePage.tsx), first slice: two scenarios' summaries side by side and, where
 * it is honest to, their stored perturbation components lined up. No scenes. It refuses to line up two perturbation vectors unless
 * the API gave each the SAME list of component names and units (`delta_labels`): a bicycle vector and a linear vector have
 * different components, so a heading in radians never sits in a row with a velocity in metres per second. The address carries both
 * ids, so a link to it shows the same page.
 */

const ok = (body: unknown): Reply => ({ status: 200, body });
const BICYCLE = ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'];
const LINEAR = ['initial vx (m/s)', 'initial vy (m/s)', 'x-acceleration bias (m/s^2)', 'y-acceleration bias (m/s^2)'];

interface Scenario {
  detail?: Partial<ReturnType<typeof makeDetail>>;
  delta?: number[] | null;
  labels?: string[] | null;
  status?: number;
}
const DATA: Record<string, Scenario> = {
  sa: { detail: { fragility_score: 2.5, min_ttc: 0.5, min_pet: -1.2, n_agents: 40, min_perturbation: 0.0226, collision_timestep: 25, challengers_searched: 1, challengers_total: 39, stress_method: 'de', stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', last_attempt_outcome: 'collision_found' }, delta: [1.5, 0.25, -0.5, 0.125], labels: BICYCLE },
  sb: { detail: { fragility_score: 1.75, min_ttc: 999, min_pet: 0.4, n_agents: 12, min_perturbation: 0.5, collision_timestep: 31, challengers_searched: 1, challengers_total: 11, stress_method: 'de+autograd', stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', last_attempt_outcome: 'collision_found' }, delta: [2.5, 0.5, 0.75, 0.0625], labels: BICYCLE },
  sl: { detail: { fragility_score: 0.9, n_agents: 8, min_perturbation: 0.3, collision_timestep: 12, stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', last_attempt_outcome: 'collision_found' }, delta: [0.1, 0.2, 0.3, 0.4], labels: LINEAR },
  su: { detail: { fragility_score: 1.1, stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', last_attempt_outcome: 'collision_found', min_perturbation: 0.2, collision_timestep: 9 }, delta: [9, 8, 7, 6], labels: null },
  sn: { detail: { fragility_score: 0.2, stress_tested: true, stress_attempted: true, stress_outcome: 'no_collision_found', last_attempt_outcome: 'no_collision_found' }, delta: null, labels: null },
  sp: { detail: { fragility_score: 1.9, stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.1, collision_timestep: 5 }, delta: [1, 2, 3, 4], labels: [...BICYCLE].reverse() }, // the same four names, in another order
  sx: { status: 404 },
  se: { status: 500 },
};

function serve(extra: Record<string, Scenario> = {}) {
  const all = { ...DATA, ...extra };
  const requests: RecordedRequest[] = [];
  stubFetch((req: RecordedRequest): Reply => {
    requests.push(req);
    const id = decodeURIComponent(req.path.split('/')[2] ?? '');
    const data = all[id];
    if (!data) return { status: 404, body: { detail: `Unknown scenario: ${id}` } };
    if (data.status !== undefined) return { status: data.status, body: { detail: data.status === 404 ? `Unknown scenario: ${id}` : 'boom' } };
    if (req.path.endsWith('/perturbed')) {
      return ok(makePerturbed({ scenario_id: id, delta: data.delta ?? null, delta_labels: data.labels ?? null, target_idx: 3, collision_timestep: data.detail?.collision_timestep ?? null }));
    }
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([], id));
    return ok(makeDetail({ scenario_id: id, ...data.detail }));
  });
  return requests;
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

const column = (id: string) => screen.getByRole('region', { name: `Scenario ${id}` });
const summary = () => screen.getByRole('table', { name: 'Summary, side by side' });
const row = (label: string) => within(summary()).getByRole('row', { name: new RegExp(`^${label}`) });
const cells = (label: string) => within(row(label)).getAllByRole('cell').map((c) => c.textContent);
const REFUSED_DIFFERENT_MODELS =
  'These two were searched with different kinematic models, so the components of their perturbation vectors differ in meaning and ' +
  'units. They are not lined up here; each is shown on its own.';
const REFUSED_UNNAMED = 'At least one of these has no parameterization stored, so its components cannot be named or given units. None are lined up here.';

describe('two scenarios of the same model', () => {
  it('shows their summaries side by side', async () => {
    serve();
    renderAt('/compare?a=sa&b=sb');
    await screen.findByRole('table', { name: 'Summary, side by side' });
    expect(within(summary()).getAllByRole('columnheader').map((h) => h.textContent)).toEqual(['', 'sa', 'sb']);
    expect(cells('Fragility')).toEqual(['2.500', '1.750']);
    expect(cells('Min TTC')).toEqual(['0.50 s', 'none']);
    expect(cells('Min PET')).toEqual(['-1.20 soverlap', '0.40 s']);
    expect(cells('Min perturbation')).toEqual(['0.023', '0.500']);
    expect(cells('Collision frame')).toEqual(['frame 25', 'frame 31']);
    expect(cells('Challengers searched')).toEqual(['1 of 39', '1 of 11']);
    expect(cells('Agents')).toEqual(['40', '12']);
    expect(cells('Method')).toEqual(['de', 'de+autograd']);
    expect(within(row('Search result')).getAllByText('collision found')).toHaveLength(2);
  });

  it('does not say they are the same scenario when they are two', async () => {
    serve();
    renderAt('/compare?a=sa&b=sb');
    await screen.findByRole('table', { name: 'Summary, side by side' });
    expect(screen.queryByText('Both are the same scenario.')).not.toBeInTheDocument();
  });

  it('names each scenario and links to its own page', async () => {
    serve();
    renderAt('/compare?a=sa&b=sb');
    await screen.findByRole('table', { name: 'Summary, side by side' });
    expect(within(summary()).getByRole('link', { name: 'sa' })).toHaveAttribute('href', '/scenarios/sa');
    expect(within(summary()).getByRole('link', { name: 'sb' })).toHaveAttribute('href', '/scenarios/sb');
  });

  it('lines up the perturbation components row by row, named, with their units', async () => {
    serve();
    renderAt('/compare?a=sa&b=sb');
    const table = await screen.findByRole('table', { name: 'Perturbation components, side by side' });
    const rows = within(table).getAllByRole('row').slice(1);
    expect(rows.map((r) => within(r).getByRole('rowheader').textContent)).toEqual(BICYCLE);
    expect(rows.map((r) => within(r).getAllByRole('cell').map((c) => c.textContent))).toEqual([
      ['1.5000', '2.5000'],
      ['0.2500', '0.5000'],
      ['-0.5000', '0.7500'],
      ['0.1250', '0.0625'],
    ]);
    expect(screen.queryByText(REFUSED_DIFFERENT_MODELS)).not.toBeInTheDocument();
  });

  it('says Min perturbation is shown, not compared', async () => {
    serve();
    renderAt('/compare?a=sa&b=sb');
    await screen.findByRole('table', { name: 'Summary, side by side' });
    expect(document.body).toHaveTextContent(
      'Min perturbation is a weighted norm over components of mixed units. The two figures are shown side by side; nothing here subtracts or divides them.',
    );
  });
});

describe('where it refuses to line the components up', () => {
  it('refuses a bicycle vector against a linear one, and shows each on its own', async () => {
    serve();
    renderAt('/compare?a=sa&b=sl');
    await screen.findByText(REFUSED_DIFFERENT_MODELS);
    expect(screen.queryByRole('table', { name: 'Perturbation components, side by side' })).not.toBeInTheDocument();
    for (const [id, labels] of [['sa', BICYCLE], ['sl', LINEAR]] as const) {
      const own = within(column(id)).getByRole('table', { name: `Perturbation components of ${id}` });
      expect(within(own).getAllByRole('rowheader').map((h) => h.textContent)).toEqual(labels);
    }
    // The summary rows that are about the scenario, not its vector, still line up.
    expect(cells('Fragility')).toEqual(['2.500', '0.900']);
  });

  it('refuses the same four names in a different order: lined up only when the lists are identical', async () => {
    serve();
    renderAt('/compare?a=sa&b=sp');
    await screen.findByText(REFUSED_DIFFERENT_MODELS);
    expect(screen.queryByRole('table', { name: 'Perturbation components, side by side' })).not.toBeInTheDocument();
  });

  it('refuses when a vector has no parameterization, and does not name or give units to its components', async () => {
    serve();
    renderAt('/compare?a=sa&b=su');
    await screen.findByText(REFUSED_UNNAMED);
    expect(screen.queryByRole('table', { name: 'Perturbation components, side by side' })).not.toBeInTheDocument();
    const own = within(column('su')).getByRole('table', { name: 'Perturbation components of su' });
    expect(within(own).getAllByRole('rowheader').map((h) => h.textContent)).toEqual(['component 1', 'component 2', 'component 3', 'component 4']);
    expect(within(column('su')).getByText('Parameterization unknown: its components are not named and carry no units.')).toBeInTheDocument();
  });

  it('refuses when one has no stored perturbation, says which, and shows the other alone', async () => {
    serve();
    renderAt('/compare?a=sa&b=sn');
    await screen.findByText('sn has no stored perturbation.');
    expect(screen.queryByRole('table', { name: 'Perturbation components, side by side' })).not.toBeInTheDocument();
    expect(within(column('sa')).getByRole('table', { name: 'Perturbation components of sa' })).toBeInTheDocument();
    // A value that is missing is a dash, never a zero, in the summary beside it.
    await screen.findByRole('table', { name: 'Summary, side by side' });
    expect(cells('Min perturbation')).toEqual(['0.023', '—']);
    expect(cells('Collision frame')).toEqual(['frame 25', '—']);
    expect(cells('Challengers searched')).toEqual(['1 of 39', '—']);
    expect(cells('Agents')).toEqual(['40', '4']); // the fixture's default row has 4 agents
    expect(cells('Method')).toEqual(['de', '—']);
  });

  it('never lines up a scenario with itself as if it were a comparison, and says so', async () => {
    serve();
    renderAt('/compare?a=sa&b=sa');
    await screen.findByText('Both are the same scenario.');
  });
});

describe('what it does when a scenario cannot be shown', () => {
  it('says an unknown id is unknown, in its own column, and still shows the other', async () => {
    serve();
    renderAt('/compare?a=sx&b=sa');
    expect(await within(column('sx')).findByText('No scenario with ID sx exists.')).toBeInTheDocument();
    expect(await within(column('sa')).findByRole('link', { name: 'sa' })).toBeInTheDocument();
    expect(screen.queryByRole('table', { name: 'Perturbation components, side by side' })).not.toBeInTheDocument();
  });

  it('shows a failed request with a retry that works', async () => {
    const data: Record<string, Scenario> = { ...DATA };
    serve();
    renderAt('/compare?a=se&b=sa');
    const alert = await within(column('se')).findByRole('alert');
    expect(alert).toHaveTextContent('500: boom');
    stubFetch((req: RecordedRequest): Reply => {
      const id = decodeURIComponent(req.path.split('/')[2] ?? '');
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ scenario_id: id, delta: null }));
      return ok(makeDetail({ scenario_id: id, fragility_score: 3 }));
    });
    await userEvent.click(within(column('se')).getByRole('button', { name: 'Try again' }));
    await waitFor(() => expect(within(column('se')).queryByRole('alert')).not.toBeInTheDocument());
    expect(data.se).toBeDefined();
  });
});

describe('the address and the fields', () => {
  it('asks for two ids when there are none, and takes them into the address', async () => {
    serve();
    const { router } = renderAt('/compare');
    expect(await screen.findByText('Enter two scenario IDs to compare them.')).toBeInTheDocument();
    const a = screen.getByRole('textbox', { name: 'First scenario ID' });
    const b = screen.getByRole('textbox', { name: 'Second scenario ID' });
    const submit = screen.getByRole('button', { name: 'Compare' });
    expect(submit).toBeDisabled();
    await userEvent.type(a, ' sa ');
    expect(submit).toBeDisabled(); // one is not enough
    await userEvent.type(b, 'sb');
    expect(submit).toBeEnabled();
    await userEvent.click(submit);
    expect(router.state.location.search).toBe('?a=sa&b=sb');
    await screen.findByRole('table', { name: 'Summary, side by side' });
  });

  it('fills the fields from the address, so a shared link shows the same page', async () => {
    serve();
    renderAt('/compare?a=sa&b=sb');
    await screen.findByRole('table', { name: 'Summary, side by side' });
    expect((screen.getByRole('textbox', { name: 'First scenario ID' }) as HTMLInputElement).value).toBe('sa');
    expect((screen.getByRole('textbox', { name: 'Second scenario ID' }) as HTMLInputElement).value).toBe('sb');
  });

  it('shows the one it has and asks for the other', async () => {
    const requests = serve();
    renderAt('/compare?a=sa');
    expect(await screen.findByText('Enter a second scenario ID to compare it with sa.')).toBeInTheDocument();
    await within(column('sa')).findByRole('link', { name: 'sa' });
    expect(screen.queryByRole('table', { name: 'Perturbation components, side by side' })).not.toBeInTheDocument();
    expect(requests.every((r) => r.path.includes('/scenarios/sa'))).toBe(true);
  });

  it('a change of ids is a history entry, so Back returns to the earlier pair', async () => {
    serve();
    const { router } = renderAt('/compare?a=sa&b=sb');
    await screen.findByRole('table', { name: 'Summary, side by side' });
    await userEvent.clear(screen.getByRole('textbox', { name: 'Second scenario ID' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Second scenario ID' }), 'sl');
    await userEvent.click(screen.getByRole('button', { name: 'Compare' }));
    expect(router.state.location.search).toBe('?a=sa&b=sl');
    await act(async () => { await router.navigate(-1); });
    expect(router.state.location.search).toBe('?a=sa&b=sb');
    await waitFor(() => expect((screen.getByRole('textbox', { name: 'Second scenario ID' }) as HTMLInputElement).value).toBe('sb'));
    expect((screen.getByRole('textbox', { name: 'First scenario ID' }) as HTMLInputElement).value).toBe('sa');
    await screen.findByRole('table', { name: 'Perturbation components, side by side' }); // and the page is the earlier pair's
  });
});

describe('what the page is, and is not', () => {
  it('has no scene on it: no replay, no playback', async () => {
    serve();
    renderAt('/compare?a=sa&b=sb');
    await screen.findByRole('table', { name: 'Summary, side by side' });
    expect(document.querySelector('.scene-frame, .scene')).toBeNull();
    expect(screen.queryByRole('slider', { name: 'Frame' })).not.toBeInTheDocument();
  });

  it('asks only for the two scenarios’ summaries and perturbations', async () => {
    const requests = serve();
    renderAt('/compare?a=sa&b=sb');
    await screen.findByRole('table', { name: 'Summary, side by side' });
    await screen.findByRole('table', { name: 'Perturbation components, side by side' });
    expect(requests.some((r) => r.path.endsWith('/trajectories'))).toBe(false);
    expect(new Set(requests.map((r) => r.path))).toEqual(new Set(['/scenarios/sa', '/scenarios/sb', '/scenarios/sa/perturbed', '/scenarios/sb/perturbed']));
  });

  it('is reached from a scenario page, with that scenario already in the first field', async () => {
    serve();
    renderApp('/scenarios/sa');
    const link = await screen.findByRole('link', { name: 'Compare with another scenario' });
    expect(link).toHaveAttribute('href', '/compare?a=sa');
  });
});

describe('the tables on a narrow screen', () => {
  /*
   * jsdom lays nothing out, so the width is checked on the rules instead. Found in a real browser at 390 px: the table was
   * 441 px wide (row heads that never wrap, two 16-character ids as column heads), so the whole page was pushed sideways.
   * The ids may break anywhere, and below 560 px the row heads may wrap and the cells take less padding.
   */
  const rules = cssRules(readFrontendFile('src/styles.css'));
  const declarationsOf = (selector: string, media: string | null) =>
    rules.filter((r) => r.selector === selector && r.media === media).map((r) => r.declarations).join('');

  it('lets a scenario ID in a column head break anywhere', () => {
    expect(declarationsOf('.compare-table a.mono', null)).toContain('overflow-wrap:anywhere');
  });

  it('lets the row heads wrap, and tightens the cells, below 560 px', () => {
    expect(declarationsOf('.compare-table tbody th', '(max-width:560px)')).toContain('white-space:normal');
    expect(declarationsOf('.compare-table th,.compare-table td', '(max-width:560px)')).toContain('padding:8px6px');
  });

  it('keeps the row heads on one line above that width, where there is room', () => {
    expect(declarationsOf('.compare-table tbody th', null)).toContain('white-space:nowrap');
  });
});
