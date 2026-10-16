import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import type { ProvenanceResponse, SearchRecord, TraceResponse } from '../api/client';
import { makeDetail, makePerturbed, makeTrajectories } from '../test/fixtures';
import { stubFetch, stubFetchDeferred, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The convergence trace in the provenance panel (pages/detail/TraceFigure.tsx), read from GET /scenarios/{id}/trace. The figures are the
 * response's; the page writes none. Nothing is asked for before the panel is opened and the provenance has loaded, and nothing is drawn
 * for a result that has no trace.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });

const SEARCH = {
  replay_model_version: 'v', collision_geometry_version: 'g', delta_parameterization: 'bicycle', challenger_selection: 'n', selected_stage: 'de',
  de_popsize: 15, de_maxiter: 200, de_tol: 0.001, de_seed: 1, de_n_iter: 3, de_n_eval: 90, de_candidate_source: 'de_winner',
  de_archive_covered_all_evaluations: true, refine_n_iters: null, bounds: [[-1, 1], [-1, 1]], max_baseline_drift: 1.5, max_speed_step: 2,
  speed_step_hold_s: 0.5, held_speed_excess: 0.1, a_max: 12, heading_speed_floor: 0.5, heading_transition_width: 0.05, target_min_speed: 3,
  frames_observed: 91, frames_below_heading_floor: 0, frames_in_heading_transition_band: 0, target_has_interior_gap: false,
} as SearchRecord;

const PROV: ProvenanceResponse = {
  scenario_id: ID, stress_run_id: 'run-abc123', scene_fingerprint: 'scene', sdc_idx: 1, scored_at: '2026-09-01T10:15:30Z',
  stress_attempted_at: null, stress_tested_at: '2026-09-02T00:00:00Z', search: SEARCH, run: null, delta_labels: null,
};

const POINTS = [
  { generation: 1, n_eval_cumulative: 30, best_weighted_norm: null },
  { generation: 2, n_eval_cumulative: 60, best_weighted_norm: 0.8 },
  { generation: 3, n_eval_cumulative: 90, best_weighted_norm: 0.2 },
];

function trace(over: Partial<TraceResponse> = {}): TraceResponse {
  return { scenario_id: ID, stress_run_id: 'run-abc123', trace: POINTS, max_abs_delta_difference: 0, tolerance: 1e-6, recorded_at: '2026-10-10T10:00:00Z', ...over };
}

function open(replyTrace: () => Reply = () => ok(trace()), provenance: ProvenanceResponse = PROV) {
  const requests: RecordedRequest[] = [];
  stubFetch((req: RecordedRequest): Reply => {
    requests.push(req);
    if (req.path.endsWith('/provenance')) return ok(provenance);
    if (req.path.endsWith('/trace')) return replyTrace();
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
    if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
    return ok(makeDetail({ scenario_id: ID }));
  });
  renderApp(`/scenarios/${ID}`);
  return requests;
}

async function expand() {
  await userEvent.click(await screen.findByText('Provenance: where this result came from'));
}
const block = () => screen.findByRole('group', { name: 'How the search closed in' });

describe('asking for the trace', () => {
  it('asks for nothing until the panel is opened', async () => {
    const requests = open();
    await screen.findByRole('heading', { name: 'Run details' });
    expect(requests.some((r) => r.path.endsWith('/trace'))).toBe(false);
  });

  it('asks once, for this scenario, once the provenance has loaded', async () => {
    const requests = open();
    await expand();
    await block();
    expect(requests.filter((r) => r.path === `/scenarios/${ID}/trace`)).toHaveLength(1);
  });

  it('does not ask for a result that has no run id', async () => {
    const requests = open(() => ok(trace()), { ...PROV, stress_run_id: null, search: null });
    await expand();
    await screen.findByText('No search provenance is stored for this scenario, so how it was searched is not recorded.');
    expect(requests.some((r) => r.path.endsWith('/trace'))).toBe(false);
  });

  it('does not ask when a search is recorded without a run id either, and says nothing about a trace', async () => {
    const requests = open(() => ok(trace()), { ...PROV, stress_run_id: null });
    await expand();
    await screen.findByText('Seed');
    expect(requests.some((r) => r.path.endsWith('/trace'))).toBe(false);
    expect(screen.queryByText('No convergence trace is recorded for this result.')).toBeNull();
  });

  it('says it is loading, and shows an error with a retry when the request fails without taking the panel down', async () => {
    open(() => ({ status: 500, body: { detail: 'boom' } }));
    await expand();
    await screen.findByText('Run ID');
    expect(await screen.findByRole('button', { name: /retry|try again/i })).toBeInTheDocument();
    expect(screen.getByText('Search budget and seed')).toBeInTheDocument();
  });
});

describe('while it waits', () => {
  it('says it is loading the trace, then shows it', async () => {
    const deferred = stubFetchDeferred((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/provenance')) return ok(PROV);
      if (req.path.endsWith('/trace')) return ok(trace());
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
      return ok(makeDetail({ scenario_id: ID }));
    });
    renderApp(`/scenarios/${ID}`);
    deferred.releaseAll();
    await expand();
    await waitFor(() => expect(deferred.requests.some((r) => r.path.endsWith('/provenance'))).toBe(true));
    deferred.releaseAll(); // the provenance arrives; the trace is asked for and does not
    expect(await screen.findByText('Loading the search trace…')).toBeInTheDocument();
    deferred.releaseAll();
    await block();
    expect(screen.queryByText('Loading the search trace…')).toBeNull();
  });
});

describe('a recorded trace', () => {
  it('says what it is: a re-run of the same seed and budget that ended at the stored perturbation, the DE stage only, and not a proof', async () => {
    open();
    await expand();
    const b = await block();
    expect(b).toHaveTextContent('read from a re-run of this search with the same seed and budget, which ended at the stored perturbation');
    expect(b).toHaveTextContent('the largest difference in any component is 0.000000000');
    expect(b).toHaveTextContent('a trace is shown only when it is below 0.000001');
    expect(b).toHaveTextContent('It covers the differential-evolution stage only, so a perturbation refined afterwards can be smaller than the last point.');
    expect(b).toHaveTextContent('it does not show that a smaller one does not exist');
  });

  it('names the first generation with a collision, the last best and the evaluations, from the response', async () => {
    open();
    await expand();
    const b = await block();
    expect(b).toHaveTextContent('First collision foundgeneration 2, after 60 evaluations, at 0.800');
    expect(b).toHaveTextContent('Best after the last generation (generation 3)0.200');
    expect(b).toHaveTextContent('Evaluations90');
  });

  it('draws only the generations that have a colliding candidate, the largest at the top and the horizontal axis in evaluations', async () => {
    open();
    await expand();
    const b = await block();
    const points = b.querySelector('polyline')!.getAttribute('points')!.split(' ');
    expect(points).toHaveLength(2);
    // evaluations 30 to 90 span x = 4 to 316; the largest norm (0.8) is on the top margin, 0.2 a quarter of the way up
    expect(points[0]).toBe('160.00,6.00');
    expect(points[1]).toBe('316.00,' + (76 - 0.25 * 70).toFixed(2));
    expect(b.querySelector('figcaption')).toHaveTextContent('30 evaluationschart top: 0.80090 evaluations');
    expect(within(b).getByRole('figure')).toHaveAccessibleName(/by evaluations spent: 0\.800 after 60 evaluations, 0\.200 after 90/);
  });

  it('lists every generation in a table that opens on request, with "none yet" before the first collision', async () => {
    open();
    await expand();
    const b = await block();
    const details = b.querySelector('details')!;
    expect(within(details).queryByRole('table')).toBeNull();
    await userEvent.click(within(details).getByText('Best after every generation'));
    const rows = within(await within(details).findByRole('table')).getAllByRole('row').slice(1);
    expect(rows).toHaveLength(3);
    expect(rows[0]).toHaveTextContent('130none yet');
    expect(rows[2]).toHaveTextContent('3900.200');
    await userEvent.click(within(details).getByText('Best after every generation'));
    await waitFor(() => expect(within(details).queryByRole('table')).toBeNull());
  });

  it('draws no line, and says so, when no generation found a collision', async () => {
    open(() => ok(trace({ trace: POINTS.map((p) => ({ ...p, best_weighted_norm: null })) })));
    await expand();
    const b = await block();
    expect(b).toHaveTextContent('First collision foundnone in any generation');
    expect(b).toHaveTextContent('Best after the last generation (generation 3)none found');
    expect(b.querySelector('polyline')).toBeNull();
  });

  it('draws a single point without NaN', async () => {
    open(() => ok(trace({ trace: [{ generation: 1, n_eval_cumulative: 75, best_weighted_norm: 0.5 }] })));
    await expand();
    const b = await block();
    expect(b.querySelector('polyline')!.getAttribute('points')).not.toMatch(/NaN|Infinity/);
  });

  it('shows the difference the response gives, not a made-up zero, and the tolerance the response carries', async () => {
    open(() => ok(trace({ max_abs_delta_difference: 4e-7, tolerance: 2e-6 })));
    await expand();
    const b = await block();
    expect(b).toHaveTextContent('the largest difference in any component is 0.000000400');
    expect(b).toHaveTextContent('a trace is shown only when it is below 0.000002');
  });
});

describe('a result with no trace', () => {
  it.each([
    ['a null trace', trace({ trace: null, max_abs_delta_difference: null, recorded_at: null })],
    ['an empty one', trace({ trace: [] })],
  ])('says so for %s and draws nothing', async (_name, body) => {
    open(() => ok(body));
    await expand();
    expect(await screen.findByText('No convergence trace is recorded for this result.')).toBeInTheDocument();
    expect(screen.queryByRole('group', { name: 'How the search closed in' })).toBeNull();
    expect(document.querySelector('.provenance polyline')).toBeNull();
  });
});
