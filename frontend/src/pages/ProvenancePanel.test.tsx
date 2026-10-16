import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import type { ProvenanceResponse, SearchRecord } from '../api/client';
import { makeDetail, makePerturbed, makeTrajectories } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { readFrontendFile } from '../test/readFrontendFile';
import { renderApp } from '../test/renderApp';
import { formatUtc } from './detail/ProvenancePanel';

/*
 * The provenance panel (pages/detail/ProvenancePanel.tsx): where a stored result came from, read from GET /scenarios/{id}/provenance.
 * It is closed until opened and asks for nothing before that. Every value is the payload's: the fixtures here use numbers that
 * are NOT the production ones (seed 424242, a population of 33) so that a value remembered by the page would show.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });

const LABELS = ['lbl one', 'lbl two', 'lbl three', 'lbl four'];

const SEARCH: SearchRecord = {
  replay_model_version: 'replay-test-v9', collision_geometry_version: 'geometry-test-v9', delta_parameterization: 'linear',
  challenger_selection: 'selection-test', selected_stage: 'de+autograd',
  de_popsize: 33, de_maxiter: 77, de_tol: 0.0042, de_seed: 424242, de_n_iter: 12, de_n_eval: 4321,
  de_candidate_source: 'archive', de_archive_covered_all_evaluations: false, refine_n_iters: 5,
  bounds: [[-1.5, 1.5], [-0.25, 0.25], [-3, 3], [-0.125, 0.125]],
  max_baseline_drift: 2.25, max_speed_step: 1.75, speed_step_hold_s: 0.8, held_speed_excess: 0.375, a_max: 9.5,
  heading_speed_floor: 0.6, heading_transition_width: 0.07, target_min_speed: 2.125,
  frames_observed: 88, frames_below_heading_floor: 0, frames_in_heading_transition_band: 4, target_has_interior_gap: false,
};

function prov(over: Partial<ProvenanceResponse> = {}): ProvenanceResponse {
  return {
    scenario_id: ID, stress_run_id: 'run-abc123', scene_fingerprint: 'scene-def456', sdc_idx: 7,
    scored_at: '2026-09-01T10:15:30Z', stress_attempted_at: '2026-09-03T01:02:03+00:00', stress_tested_at: '2026-09-02T23:59:59Z',
    search: SEARCH, delta_labels: LABELS, ...over,
  };
}

function open(reply: () => Reply = () => ok(prov()), labels: string[] | null = LABELS) {
  const requests: RecordedRequest[] = [];
  stubFetch((req: RecordedRequest): Reply => {
    requests.push(req);
    if (req.path.endsWith('/provenance')) return reply();
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
    if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ delta: [1, 2, 3, 4], delta_labels: labels }));
    return ok(makeDetail({ scenario_id: ID }));
  });
  renderApp(`/scenarios/${ID}`);
  return requests;
}

const summary = () => screen.findByText('Provenance: where this result came from');
async function expand() {
  await userEvent.click(await summary());
  return screen.getByText('Provenance: where this result came from').closest('details')!;
}
const fact = (panel: HTMLElement, label: string) => {
  const dt = within(panel).getByText(label, { selector: 'dt' });
  return dt.nextElementSibling!.textContent;
};

describe('opening the panel', () => {
  it('is closed, and has asked for nothing, until it is opened', async () => {
    const requests = open();
    const panel = (await summary()).closest('details')!;
    await screen.findByRole('heading', { name: 'Run details' });
    expect(panel).not.toHaveAttribute('open');
    expect(requests.some((r) => r.path.endsWith('/provenance'))).toBe(false);
    expect(within(panel).queryByText('Run ID')).not.toBeInTheDocument();
  });

  it('asks once, for this scenario, when it is opened, and not again when closed and reopened', async () => {
    const requests = open();
    await expand();
    await screen.findByText('Run ID');
    await userEvent.click(await summary());
    await userEvent.click(await summary());
    await screen.findByText('Run ID');
    expect(requests.filter((r) => r.path === `/scenarios/${ID}/provenance`)).toHaveLength(1);
  });

  it('says it is loading while it waits', async () => {
    let release: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const requests: RecordedRequest[] = [];
    // a reply that arrives only when released
    const { stubFetchDeferred } = await import('../test/fetchStub');
    const deferred = stubFetchDeferred((req) => {
      requests.push(req);
      if (req.path.endsWith('/provenance')) return ok(prov());
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
      return ok(makeDetail({ scenario_id: ID }));
    });
    renderApp(`/scenarios/${ID}`);
    deferred.releaseAll();
    await expand();
    expect(await screen.findByText('Loading provenance…')).toBeInTheDocument();
    release(); await gate;
    deferred.releaseAll();
    await screen.findByText('Run ID');
  });
});

describe('what it shows', () => {
  it('shows the identity, the times and the model, as the payload has them', async () => {
    open();
    const panel = await expand();
    await screen.findByText('Run ID');
    expect(fact(panel, 'Run ID')).toBe('run-abc123');
    expect(fact(panel, 'Scene fingerprint')).toBe('scene-def456');
    expect(fact(panel, 'SDC index')).toBe('7');
    expect(fact(panel, 'Replay model')).toBe('replay-test-v9');
    expect(fact(panel, 'Collision geometry')).toBe('geometry-test-v9');
    expect(fact(panel, 'Parameterization')).toBe('linear');
    expect(fact(panel, 'Challenger chosen')).toBe('selection-test');
    expect(fact(panel, 'Stage')).toBe('de+autograd');
  });

  it('shows the times in UTC, with the instant itself as the tooltip', async () => {
    open();
    const panel = await expand();
    await screen.findByText('Run ID');
    expect(fact(panel, 'Scored')).toBe('2026-09-01 10:15 UTC');
    expect(fact(panel, 'Last attempted')).toBe('2026-09-03 01:02 UTC');
    expect(fact(panel, 'Result stored')).toBe('2026-09-02 23:59 UTC');
    expect(within(panel).getByText('2026-09-01 10:15 UTC')).toHaveAttribute('title', '2026-09-01T10:15:30Z');
  });

  it('shows the search budget and the seed', async () => {
    open();
    const panel = await expand();
    await screen.findByText('Run ID');
    expect(fact(panel, 'Seed')).toBe('424242');
    expect(fact(panel, 'Population size')).toBe('33');
    expect(fact(panel, 'Maximum iterations')).toBe('77');
    expect(fact(panel, 'Convergence tolerance')).toBe('0.0042');
    expect(fact(panel, 'Iterations run')).toBe('12');
    expect(fact(panel, 'Evaluations')).toBe('4321');
    expect(fact(panel, 'Winning candidate from')).toBe('archive');
    expect(fact(panel, 'Archive covered every evaluation')).toBe('No');
    expect(fact(panel, 'Refinement iterations')).toBe('5');
  });

  it('shows the gates with their units, and the challenger facts', async () => {
    open();
    const panel = await expand();
    await screen.findByText('Run ID');
    expect(fact(panel, 'Baseline drift limit')).toBe('2.2500 m');
    expect(fact(panel, 'Speed step limit')).toBe('1.7500 m/s');
    expect(fact(panel, 'Speed step held for')).toBe('0.8000 s');
    expect(fact(panel, 'Held speed excess')).toBe('0.3750 m/s');
    expect(fact(panel, 'Acceleration limit')).toBe('9.5000 m/s²');
    expect(fact(panel, 'Heading speed floor')).toBe('0.6000 m/s');
    expect(fact(panel, 'Heading transition width')).toBe('0.0700 m/s');
    expect(fact(panel, 'Lowest logged speed')).toBe('2.1250 m/s');
    expect(fact(panel, 'Frames observed')).toBe('88');
    expect(fact(panel, 'Frames in the transition band')).toBe('4');
    expect(fact(panel, 'Has a gap inside its track')).toBe('No');
  });

  it('says what is not stored anywhere', async () => {
    open();
    const panel = await expand();
    await screen.findByText('Run ID');
    expect(panel).toHaveTextContent('Not stored anywhere, so not shown: the version of the code, the versions of the libraries, and a checksum of the shard the scenario was read from.');
  });
});

describe('the bounds', () => {
  it('are named by the perturbation panel’s own labels when there is one per component', async () => {
    open();
    const panel = await expand();
    const table = await within(panel).findByRole('table', { name: 'Bounds of the search' });
    const rows = within(table).getAllByRole('row').slice(1);
    expect(rows.map((r) => [...r.querySelectorAll('th,td')].map((c) => c.textContent))).toEqual([
      ['lbl one', '-1.5000', '1.5000'], ['lbl two', '-0.2500', '0.2500'], ['lbl three', '-3.0000', '3.0000'], ['lbl four', '-0.1250', '0.1250'],
    ]);
  });

  it.each([
    ['no labels', null],
    ['too few labels', ['one', 'two']],
  ])('are "component N" with %s, and carry no names or units', async (_name, labels) => {
    open(() => ok(prov()), labels);
    const panel = await expand();
    const table = await within(panel).findByRole('table', { name: 'Bounds of the search' });
    expect(within(table).getAllByRole('rowheader').map((h) => h.textContent)).toEqual(['component 1', 'component 2', 'component 3', 'component 4']);
  });

  it.each([
    ['the perturbed response is of the other parameterization', ['bicycle a', 'bicycle b', 'bicycle c', 'bicycle d'], prov({ delta_labels: ['linear a', 'linear b', 'linear c', 'linear d'] })],
    ['the provenance has no labels (no parameterization recorded)', LABELS, prov({ delta_labels: null })],
    ['the provenance has no labels at all (the field is absent)', LABELS, (() => { const p = prov(); delete (p as { delta_labels?: unknown }).delta_labels; return p; })()],
    ['the perturbed response has no labels', null, prov()],
    ['the same names come in another order', ['lbl two', 'lbl one', 'lbl three', 'lbl four'], prov()],
    ['one name differs', ['lbl one', 'lbl two', 'lbl three', 'lbl 4'], prov()],
    ['the provenance has one more label than the perturbed response, the first four the same', LABELS, prov({ delta_labels: [...LABELS, 'lbl five'] })],
    ['both agree but there is one more bound than labels', ['lbl one', 'lbl two', 'lbl three'], prov({ delta_labels: ['lbl one', 'lbl two', 'lbl three'] })],
  ])('are "component N", with no names or units, when %s', async (_name, perturbedLabels, provenance) => {
    open(() => ok(provenance), perturbedLabels);
    const panel = await expand();
    const table = await within(panel).findByRole('table', { name: 'Bounds of the search' });
    expect(within(table).getAllByRole('rowheader').map((h) => h.textContent)).toEqual(['component 1', 'component 2', 'component 3', 'component 4']);
    expect(table).not.toHaveTextContent(/lbl|bicycle|linear/);
  });

  it('are named when the two agree exactly, and from the perturbed response’s own labels', async () => {
    open(() => ok(prov({ delta_labels: ['x1', 'x2', 'x3', 'x4'] })), ['x1', 'x2', 'x3', 'x4']);
    const panel = await expand();
    const table = await within(panel).findByRole('table', { name: 'Bounds of the search' });
    expect(within(table).getAllByRole('rowheader').map((h) => h.textContent)).toEqual(['x1', 'x2', 'x3', 'x4']);
  });

  it('say so when they were not recorded', async () => {
    open(() => ok(prov({ search: { ...SEARCH, bounds: null } })));
    const panel = await expand();
    await screen.findByText('Run ID');
    expect(within(panel).queryByRole('table', { name: 'Bounds of the search' })).not.toBeInTheDocument();
    expect(panel).toHaveTextContent('Bounds: not recorded for this result.');
  });
});

describe('what was not recorded', () => {
  it('is "Not recorded for this result", and a zero or a No is a value, not a gap', async () => {
    open(() => ok(prov({
      stress_run_id: null, sdc_idx: 0, scored_at: null,
      search: { ...SEARCH, de_seed: 0, de_popsize: null, de_tol: null, target_has_interior_gap: false, target_min_speed: 0, refine_n_iters: null, replay_model_version: null },
    })));
    const panel = await expand();
    await screen.findByText('Scene fingerprint');
    const missing = 'Not recorded for this result';
    expect(fact(panel, 'Run ID')).toBe(missing);
    expect(fact(panel, 'Scored')).toBe(missing);
    expect(fact(panel, 'Population size')).toBe(missing);
    expect(fact(panel, 'Convergence tolerance')).toBe(missing);
    expect(fact(panel, 'Refinement iterations')).toBe(missing);
    expect(fact(panel, 'Replay model')).toBe(missing);
    expect(fact(panel, 'SDC index')).toBe('0');
    expect(fact(panel, 'Seed')).toBe('0');
    expect(fact(panel, 'Has a gap inside its track')).toBe('No');
    expect(fact(panel, 'Lowest logged speed')).toBe('0.0000 m/s');
  });

  it('with no stored search, shows what is known and says how it was searched is not recorded', async () => {
    open(() => ok(prov({ search: null, stress_run_id: null, scene_fingerprint: null, sdc_idx: null, stress_attempted_at: null, stress_tested_at: null })));
    const panel = await expand();
    await screen.findByText('No search provenance is stored for this scenario, so how it was searched is not recorded.');
    expect(fact(panel, 'Scored')).toBe('2026-09-01 10:15 UTC');
    expect(fact(panel, 'Run ID')).toBe('Not recorded for this result');
    expect(within(panel).queryByText('Seed')).not.toBeInTheDocument();
    expect(within(panel).queryByRole('table')).not.toBeInTheDocument();
  });
});

describe('with a run and a scene on record but no search provenance', () => {
  it('still names the run and the scene correctly, each in its own row', async () => {
    open(() => ok(prov({ search: null })));
    const panel = await expand();
    await screen.findByText('No search provenance is stored for this scenario, so how it was searched is not recorded.');
    expect(fact(panel, 'Run ID')).toBe('run-abc123');
    expect(fact(panel, 'Scene fingerprint')).toBe('scene-def456');
    expect(fact(panel, 'SDC index')).toBe('7');
    expect(fact(panel, 'Result stored')).toBe('2026-09-02 23:59 UTC');
  });
});

describe('when the request fails', () => {
  it('shows the error with a way to try again, and the retry asks again', async () => {
    let n = 0;
    const requests = open(() => (++n === 1 ? { status: 500, body: { detail: 'boom' } } : ok(prov())));
    const panel = await expand();
    expect(await within(panel).findByRole('alert')).toHaveTextContent(/500: boom/);
    await userEvent.click(within(panel).getByRole('button', { name: 'Try again' }));
    await within(panel).findByText('Run ID');
    expect(requests.filter((r) => r.path.endsWith('/provenance'))).toHaveLength(2);
  });
});

describe('formatUtc', () => {
  it('writes an instant in UTC to the minute, whatever offset it came with', () => {
    expect(formatUtc('2026-09-01T10:15:30Z')).toBe('2026-09-01 10:15 UTC');
    expect(formatUtc('2026-09-01T03:15:30-07:00')).toBe('2026-09-01 10:15 UTC');
    expect(formatUtc('2026-09-01T10:15:30.123456+00:00')).toBe('2026-09-01 10:15 UTC');
    expect(formatUtc('2026-12-31T23:59:59Z')).toBe('2026-12-31 23:59 UTC');
  });
  it('shows what is not an instant as it came', () => {
    expect(formatUtc('yesterday')).toBe('yesterday');
    expect(formatUtc('')).toBe('');
  });
});

describe('the page writes no number of its own', () => {
  it('has no figure of the data in the component outside its comments', () => {
    const source = readFrontendFile('src/pages/detail/ProvenancePanel.tsx').replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
    const numbers = [...source.matchAll(/(?<![\w.])\d+(?:\.\d+)?(?![\w.])/g)].map((m) => m[0]);
    for (const n of numbers) expect(['0', '1', '4', '10', '16'], `a number in the source: ${n}`).toContain(n);
  });
});
