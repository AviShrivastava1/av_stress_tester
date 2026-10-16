import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import type { OtherSearch, OtherSearchesResponse } from '../api/client';
import { makeDetail, makePerturbed, makeTrajectories } from '../test/fixtures';
import { stubFetch, stubFetchDeferred, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * "Other searches of this scenario" (pages/detail/OtherSearches.tsx), read from GET /scenarios/{id}/other-searches. The figures are the
 * response's; the page writes none. It says, before any number, that each is a separate search and none is shown to be the smallest;
 * it is not next to the headline figure; it lists the searches in the order the API gave and marks none; and it shows no number of a
 * search under a model other than the stored result's.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });
const BICYCLE = ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'];
const LINEAR = ['initial vx (m/s)', 'initial vy (m/s)', 'x-acceleration bias (m/s^2)', 'y-acceleration bias (m/s^2)'];

function search(over: Partial<OtherSearch> = {}): OtherSearch {
  return {
    search_run_id: '0123456789abcdef', kind: 'seed', challenger_rank: null, target_idx: 1, seed: 3, popsize: 15, maxiter: 200, tol: 0.001,
    outcome: 'collision_found', min_perturbation: 0.25, collision_timestep: 40, delta: [0.5, 0.01, -0.2, 0.003],
    delta_parameterization: 'bicycle', delta_labels: BICYCLE, n_iter: 12, n_eval: 900, ...over,
  };
}
function body(searches: OtherSearch[], over: Partial<OtherSearchesResponse> = {}): OtherSearchesResponse {
  return { scenario_id: ID, stress_run_id: 'run-abc', stored_delta_parameterization: 'bicycle', searches, truncated: false, ...over };
}

function open(reply: () => Reply = () => ok(body([search()]))) {
  const requests: RecordedRequest[] = [];
  stubFetch((req: RecordedRequest): Reply => {
    requests.push(req);
    if (req.path.endsWith('/other-searches')) return reply();
    if (req.path.endsWith('/provenance')) return ok({ scenario_id: ID, stress_run_id: 'run-abc' });
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
    if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ delta: [1, 2, 3, 4], delta_labels: BICYCLE }));
    return ok(makeDetail({ scenario_id: ID, stress_tested: true, stress_outcome: 'collision_found', min_perturbation: 0.1 }));
  });
  renderApp(`/scenarios/${ID}`);
  return requests;
}
const summary = () => screen.findByText('Other searches of this scenario', { selector: 'summary' });
async function expand() {
  await userEvent.click(await summary());
  return (await summary()).closest('details')!;
}
const rows = (panel: HTMLElement) => within(within(panel).getByRole('table')).getAllByRole('row').slice(1);

describe('next to the convergence trace', () => {
  it('each asks for and shows its own data when both are open on one page', async () => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/other-searches')) return ok(body([search({ search_run_id: 'aaaaaaaaaaaaaaa1', seed: 41 })]));
      if (req.path.endsWith('/trace')) {
        return ok({ scenario_id: ID, stress_run_id: 'run-abc', trace: [{ generation: 1, n_eval_cumulative: 10, best_weighted_norm: 0.5 }], max_abs_delta_difference: 0, tolerance: 1e-6, recorded_at: null });
      }
      if (req.path.endsWith('/provenance')) return ok({ scenario_id: ID, stress_run_id: 'run-abc', search: { delta_parameterization: 'bicycle' } });
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
      return ok(makeDetail({ scenario_id: ID }));
    });
    renderApp(`/scenarios/${ID}`);
    await userEvent.click(await screen.findByText('Provenance: where this result came from'));
    await screen.findByRole('group', { name: 'How the search closed in' });
    const panel = await expand();
    await within(panel).findByRole('table');
    expect(within(panel).getAllByRole('row')).toHaveLength(2);
    expect(within(panel).getByText('41')).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'How the search closed in' })).toHaveTextContent('First collision foundgeneration 1');
  });
});

describe('closed', () => {
  it('draws nothing but its title until it is opened: no note, no table, no word about seeds', async () => {
    open();
    const panel = (await summary()).closest('details')!;
    await screen.findByRole('heading', { name: 'Run details' });
    expect(panel).not.toHaveAttribute('open');
    expect(panel.textContent).toBe('Other searches of this scenario');
  });
});

describe('asking', () => {
  it('asks for nothing until it is opened, and once, for this scenario, when it is', async () => {
    const requests = open();
    await summary();
    await screen.findByRole('heading', { name: 'Run details' });
    expect(requests.some((r) => r.path.endsWith('/other-searches'))).toBe(false);
    const panel = await expand();
    await within(panel).findByRole('table');
    await userEvent.click(await summary());
    await userEvent.click(await summary());
    expect(requests.filter((r) => r.path === `/scenarios/${ID}/other-searches`)).toHaveLength(1);
  });

  it('says it is loading while it waits', async () => {
    const deferred = stubFetchDeferred((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/other-searches')) return ok(body([search()]));
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
      return ok(makeDetail({ scenario_id: ID }));
    });
    renderApp(`/scenarios/${ID}`);
    deferred.releaseAll();
    await expand();
    expect(await screen.findByText('Loading other searches…')).toBeInTheDocument();
    deferred.releaseAll();
    await screen.findByRole('table');
  });

  it('shows an error with a retry when the request fails, and the rest of the page stays', async () => {
    open(() => ({ status: 500, body: { detail: 'boom' } }));
    const panel = await expand();
    expect(await within(panel).findByRole('button', { name: /retry|try again/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Run details' })).toBeInTheDocument();
  });
});

describe('what it says', () => {
  it('says first that each is a separate search, that none is shown to be the smallest, and where the stored result is', async () => {
    open();
    const panel = await expand();
    expect(panel.querySelector('p.note')).toHaveTextContent(
      'Each of these is a separate search of this scenario, run with a seed or a budget of its own. None is shown to be the smallest: ' +
        'each figure is the smallest perturbation that one finite search found, and another seed or budget can find a smaller or a larger ' +
        'one. The stored result and its search are in Provenance above.',
    );
    expect(panel.querySelector('p.note')!.compareDocumentPosition(await within(panel).findByRole('table'))).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  it('is its own section below the Provenance panel, and not in the panels that carry the headline figure', async () => {
    open();
    const panel = await expand();
    const provenance = screen.getByText('Provenance: where this result came from').closest('details')!;
    expect(provenance.compareDocumentPosition(panel)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(panel.closest('.detail-side')).toBeNull();
    expect(screen.getByRole('heading', { name: 'Perturbation' }).closest('section')!.contains(panel)).toBe(false);
    expect(screen.getByRole('heading', { name: 'Run details' }).closest('section')!.contains(panel)).toBe(false);
  });

  it('says there is none recorded, and draws no table, for an empty list', async () => {
    open(() => ok(body([])));
    const panel = await expand();
    expect(await within(panel).findByText('No other search is recorded for this scenario.')).toBeInTheDocument();
    expect(within(panel).queryByRole('table')).toBeNull();
  });
});

describe('the rows', () => {
  it('lists the searches in the order the API gave, not by norm, and marks none', async () => {
    open(() => ok(body([
      search({ search_run_id: 'aaaaaaaaaaaaaaa1', seed: 1, min_perturbation: 0.9 }),
      search({ search_run_id: 'aaaaaaaaaaaaaaa2', seed: 2, min_perturbation: 0.1 }),
      search({ search_run_id: 'aaaaaaaaaaaaaaa3', kind: 'budget', seed: 0, popsize: 10, maxiter: 50, min_perturbation: 0.5 }),
    ])));
    const panel = await expand();
    const r = rows(panel);
    expect(r).toHaveLength(3);
    expect(r.map((x) => within(x).getAllByRole('cell')[0]!.textContent)).toEqual(['1', '2', '0']);
    expect(r[0]).toHaveTextContent('0.900');
    expect(r[1]).toHaveTextContent('0.100');
    for (const x of r) expect(x.className).toBe('');
    expect(within(within(panel).getByRole('table')).queryByText(/smallest|best|lowest|winner/i)).toBeNull();
  });

  it('names the kind, and the configuration of each search from the response', async () => {
    open(() => ok(body([
      search({ kind: 'seed', seed: 7, popsize: 15, maxiter: 200, tol: 0.001 }),
      search({ search_run_id: 'bbbbbbbbbbbbbbbb', kind: 'budget', seed: 0, popsize: 30, maxiter: 400, tol: 0.0001 }),
      search({ search_run_id: 'cccccccccccccccc', kind: 'challenger', challenger_rank: 2, seed: 0 }),
    ])));
    const r = rows(await expand());
    expect(r[0]).toHaveTextContent('Another seed');
    expect(within(r[0]!).getAllByRole('cell').map((c) => c.textContent).slice(0, 4)).toEqual(['7', '15', '200', '0.0010']);
    expect(r[1]).toHaveTextContent('Another budget');
    expect(within(r[1]!).getAllByRole('cell').map((c) => c.textContent).slice(0, 4)).toEqual(['0', '30', '400', '0.0001']);
    expect(r[2]).toHaveTextContent('Another challenger, rank 2');
  });

  it('gives the result in the interface’s own words, with the collision frame, and a dash for no norm', async () => {
    open(() => ok(body([
      search({ outcome: 'collision_found', collision_timestep: 40, min_perturbation: 0.25 }),
      search({ search_run_id: 'dddddddddddddddd', seed: 4, outcome: 'no_collision_found', collision_timestep: null, min_perturbation: null, delta: null }),
    ])));
    const r = rows(await expand());
    expect(r[0]).toHaveTextContent('collision found, frame 40');
    expect(r[0]).toHaveTextContent('0.250');
    expect(r[1]).toHaveTextContent('no collision found');
    expect(within(r[1]!).getAllByRole('cell')[5]).toHaveTextContent('—');
  });

  it('shows no collision frame for a search that found none, whatever frame the row carries', async () => {
    open(() => ok(body([search({ outcome: 'no_collision_found', collision_timestep: 40, min_perturbation: null })])));
    const [r] = rows(await expand());
    expect(r).toHaveTextContent('no collision found');
    expect(r).not.toHaveTextContent('frame');
  });

  it('shows the content hash of each search as its name, so that two searches can be told apart', async () => {
    open(() => ok(body([search({ search_run_id: 'aaaaaaaaaaaaaaa1', seed: 1 }), search({ search_run_id: 'bbbbbbbbbbbbbbb2', seed: 2 })])));
    const [a, b] = rows(await expand());
    expect(within(a!).getByText('aaaaaaaaaaaaaaa1')).toHaveAttribute('title', 'The content hash of this search');
    expect(within(b!).getByText('bbbbbbbbbbbbbbb2')).toBeInTheDocument();
  });

  it('shows what was not recorded as not recorded, never as zero', async () => {
    open(() => ok(body([search({ seed: null, popsize: null, maxiter: null, tol: null, outcome: null, min_perturbation: null, n_eval: null, collision_timestep: null, delta: null })])));
    const [r] = rows(await expand());
    const cells = within(r!).getAllByRole('cell').map((c) => c.textContent);
    expect(cells.slice(0, 5)).toEqual(['Not recorded', 'Not recorded', 'Not recorded', 'Not recorded', 'Not recorded']);
    expect(cells[5]).toBe('—');
    expect(cells[6]).toBe('Not recorded');
  });

  it('gives the full value as the tooltip of a rounded one', async () => {
    open(() => ok(body([search({ min_perturbation: 0.123456789, tol: 0.00012345 })])));
    const [r] = rows(await expand());
    expect(within(r!).getByText('0.123')).toHaveAttribute('title', '0.123456789');
  });
});

describe('the components and the model', () => {
  it('names each component with the labels of that search’s own model, which carry the units', async () => {
    open(() => ok(body([search({ delta: [0.5, 0.01, -0.2, 0.003] })])));
    const panel = await expand();
    await userEvent.click(within(panel).getByText('Perturbation components'));
    const dl = panel.querySelector('dl')!;
    expect([...dl.querySelectorAll('dt')].map((d) => d.textContent)).toEqual(BICYCLE);
    expect([...dl.querySelectorAll('dd')].map((d) => d.textContent)).toEqual(['0.5000', '0.0100', '-0.2000', '0.0030']);
  });

  it('uses another model’s names for a search under that model, when the stored result is under it too', async () => {
    open(() => ok(body([search({ delta_parameterization: 'linear', delta_labels: LINEAR })], { stored_delta_parameterization: 'linear' })));
    const panel = await expand();
    await userEvent.click(within(panel).getByText('Perturbation components'));
    expect([...panel.querySelectorAll('dt')].map((d) => d.textContent)).toEqual(LINEAR);
  });

  it('falls back to numbered components when the labels are not the length of the delta', async () => {
    open(() => ok(body([search({ delta_labels: BICYCLE.slice(0, 3) })])));
    const panel = await expand();
    await userEvent.click(within(panel).getByText('Perturbation components'));
    expect([...panel.querySelectorAll('dt')].map((d) => d.textContent)).toEqual(['component 1', 'component 2', 'component 3', 'component 4']);
  });

  it.each([
    ['another model', search({ delta_parameterization: 'linear', delta_labels: LINEAR }), 'bicycle'],
    ['an unrecorded model', search({ delta_parameterization: null, delta_labels: null }), 'bicycle'],
    ['a stored result under an unrecorded model', search(), null],
  ])('shows no number of a search under %s, and says why', async (_name, s, stored) => {
    open(() => ok(body([s], { stored_delta_parameterization: stored })));
    const panel = await expand();
    const [r] = rows(panel);
    expect(r).toHaveTextContent('Under a different or unrecorded model; its figures are not shown against the stored result’s.');
    expect(r).not.toHaveTextContent('0.250');
    expect(r).not.toHaveTextContent('collision found');
    expect(within(panel).queryByText('Perturbation components')).toBeNull();
    expect(within(r!).getAllByRole('cell')[0]).toHaveTextContent('3'); // the configuration is still shown
  });

  it('does not line up a bicycle search and a linear one in the same list: only the stored result’s model shows numbers', async () => {
    open(() => ok(body([
      search({ search_run_id: 'eeeeeeeeeeeeeee1', seed: 1 }),
      search({ search_run_id: 'eeeeeeeeeeeeeee2', seed: 2, delta_parameterization: 'linear', delta_labels: LINEAR, min_perturbation: 0.77 }),
    ])));
    const panel = await expand();
    const [a, b] = rows(panel);
    expect(a).toHaveTextContent('0.250');
    expect(b).not.toHaveTextContent('0.770');
    expect(within(panel).getAllByText('Perturbation components')).toHaveLength(1);
  });
});

describe('a list the API cut off', () => {
  const SENTENCE = 'Not every recorded search is listed here; those shown are the first in the order above.';

  it('says so in one sentence under the table when the response says it is truncated', async () => {
    open(() => ok(body([search(), search({ search_run_id: 'aaaaaaaaaaaaaaa1', seed: 4 })], { truncated: true })));
    const panel = await expand();
    await within(panel).findByRole('table');
    const sentence = within(panel).getByText(SENTENCE);
    expect(within(panel).getAllByText(SENTENCE)).toHaveLength(1);
    expect(sentence.tagName).toBe('P');
    expect(within(panel).getByRole('table').compareDocumentPosition(sentence)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(panel.querySelector('.other-searches-components')!.compareDocumentPosition(sentence)).toBe(Node.DOCUMENT_POSITION_PRECEDING);
  });

  it('types no number into the sentence', () => {
    expect(SENTENCE).not.toMatch(/\d/);
  });

  it('is absent when the response says it is not truncated', async () => {
    open(() => ok(body([search()], { truncated: false })));
    const panel = await expand();
    await within(panel).findByRole('table');
    expect(within(panel).queryByText(SENTENCE)).toBeNull();
    expect(panel.textContent).not.toMatch(/Not every recorded search/);
  });

  it('is absent when the response does not carry the field, and for an empty list', async () => {
    const { truncated, ...without } = body([search()]);
    expect(truncated).toBe(false);
    open(() => ok(without));
    const panel = await expand();
    await within(panel).findByRole('table');
    expect(panel.textContent).not.toMatch(/Not every recorded search/);
  });

  it('is shown only for the value true: not for text or a number that happens to be truthy', async () => {
    for (const odd of ['yes', 1, 'true']) {
      open(() => ok({ ...body([search()]), truncated: odd } as unknown as OtherSearchesResponse));
      const panel = await expand();
      await within(panel).findByRole('table');
      expect(panel.textContent).not.toMatch(/Not every recorded search/);
      document.body.innerHTML = '';
    }
  });

  it('is not shown for an empty list even if the flag were set', async () => {
    open(() => ok(body([], { truncated: true })));
    const panel = await expand();
    await within(panel).findByText('No other search is recorded for this scenario.');
    expect(panel.textContent).not.toMatch(/Not every recorded search/);
  });
});

describe('waiting for the table', () => {
  it('draws the table only once the response is in', async () => {
    open();
    const panel = await expand();
    await waitFor(() => expect(within(panel).getByRole('table')).toBeInTheDocument());
  });
});
