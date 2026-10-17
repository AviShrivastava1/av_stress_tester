import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import type { SensitivityResponse, SensitivitySlice } from '../api/client';
import { makeDetail, makePerturbed, makeTrajectories } from '../test/fixtures';
import { stubFetch, stubFetchDeferred, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * "Sensitivity slice of this scenario" (pages/detail/SensitivitySlice.tsx), read from GET /scenarios/{id}/slice. The figures are the
 * response's; the page writes none. It says, before any cell, that it is a slice of a four-dimensional search with two components held,
 * and that a cell without a collision and a finite grid prove nothing; it never says more than the cells do; meaning is carried by words
 * and by the pattern of a cell, not by colour alone; there is a table of the same cells; the stored point is marked; and no figure of a
 * slice under a model other than the stored result's is shown.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });
const BICYCLE = ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'];
const LINEAR = ['initial vx (m/s)', 'initial vy (m/s)', 'x-acceleration bias (m/s^2)', 'y-acceleration bias (m/s^2)'];

// a 3 by 2 grid: x runs -1, 0, 1 (stored value 0); y runs 0.5, 1.5 (stored value 1.5); cells row-major, y outer
function slice(over: Partial<SensitivitySlice> = {}): SensitivitySlice {
  return {
    slice_id: 7, axis_x: 0, axis_y: 2, delta_parameterization: 'bicycle', axis_x_label: BICYCLE[0], axis_y_label: BICYCLE[2], delta_labels: BICYCLE,
    x_values: [-1, 0, 1], y_values: [0.5, 1.5], weighted_norm: [0.9, 0.3, 0.8, 0.7, 0.2, 0.6], collides: [false, true, false, false, true, null],
    held_delta: [0, 0.05, 1.5, -0.02], stored_x_index: 1, stored_y_index: 1, ...over,
  };
}
function body(slices: SensitivitySlice[], over: Partial<SensitivityResponse> = {}): SensitivityResponse {
  return { scenario_id: ID, stress_run_id: 'run-abc', stored_delta_parameterization: 'bicycle', slices, truncated: false, ...over };
}

function open(reply: () => Reply = () => ok(body([slice()]))) {
  const requests: RecordedRequest[] = [];
  stubFetch((req: RecordedRequest): Reply => {
    requests.push(req);
    if (req.path.endsWith('/slice')) return reply();
    if (req.path.endsWith('/other-searches')) return ok({ scenario_id: ID, stress_run_id: 'run-abc', stored_delta_parameterization: 'bicycle', searches: [], truncated: false });
    if (req.path.endsWith('/provenance')) return ok({ scenario_id: ID, stress_run_id: 'run-abc' });
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
    if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ delta: [1, 2, 3, 4], delta_labels: BICYCLE }));
    return ok(makeDetail({ scenario_id: ID, stress_tested: true, stress_outcome: 'collision_found', min_perturbation: 0.1 }));
  });
  renderApp(`/scenarios/${ID}`);
  return requests;
}
const summary = () => screen.findByText('Sensitivity slice of this scenario', { selector: 'summary' });
async function expand() {
  await userEvent.click(await summary());
  return (await summary()).closest('details')!;
}
const NOTE =
  'A slice of a four-dimensional search. Two components of the stored perturbation vary over the grid below; the other two are held at the stored ' +
  'values. Each cell shows whether the replay collides with that pair of values, and the weighted norm of the whole perturbation. A cell with no ' +
  'collision shows only that this one perturbation did not collide: collisions can lie between the cells and in the held components, and a finite ' +
  'grid is not a proof.';
const SLICE_TITLE = / against .*, \d+ by \d+$/;
/** Opens the nth slice of the section (each is a closed details of its own) and returns the section. */
async function openSlice(panel: HTMLElement, index = 0) {
  const summaries = await within(panel).findAllByText(SLICE_TITLE, { selector: 'summary' });
  await userEvent.click(summaries[index]!);
  return panel;
}
const expandOpen = async () => openSlice(await expand());
const cells = (panel: HTMLElement) => [...panel.querySelectorAll('rect.sens-cell')] as SVGRectElement[];

describe('asking', () => {
  it('asks for nothing until it is opened, and once, for this scenario, when it is', async () => {
    const requests = open();
    const panel = (await summary()).closest('details')!;
    await screen.findByRole('heading', { name: 'Run details' });
    expect(panel).not.toHaveAttribute('open');
    expect(panel.textContent).toBe('Sensitivity slice of this scenario');
    expect(requests.some((r) => r.path.endsWith('/slice'))).toBe(false);
    await expand();
    await screen.findByText('A cell with no collision', { exact: false });
    await userEvent.click(await summary());
    await userEvent.click(await summary());
    expect(requests.filter((r) => r.path === `/scenarios/${ID}/slice`)).toHaveLength(1);
  });

  it('says it is loading while it waits', async () => {
    const deferred = stubFetchDeferred((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/slice')) return ok(body([slice()]));
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
      return ok(makeDetail({ scenario_id: ID }));
    });
    renderApp(`/scenarios/${ID}`);
    deferred.releaseAll();
    const panel = await expand();
    expect(await screen.findByText('Loading the slice…')).toBeInTheDocument();
    deferred.releaseAll();
    await openSlice(panel);
    await screen.findByRole('figure');
  });

  it('shows an error with a retry when the request fails, and the rest of the page stays', async () => {
    open(() => ({ status: 500, body: { detail: 'boom' } }));
    const panel = await expand();
    expect(await within(panel).findByRole('button', { name: /retry|try again/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Run details' })).toBeInTheDocument();
  });
});

describe('what it says', () => {
  it('says first that it is a slice of a four-dimensional search with two components held, and what a cell without a collision does not show', async () => {
    open();
    const panel = await expandOpen();
    expect(panel.querySelector('p.note')).toHaveTextContent(NOTE);
    expect(panel.querySelector('p.note')!.compareDocumentPosition(await within(panel).findByRole('figure'))).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  it('is its own section below the Provenance panel and the other searches, not in the panels of the headline figure', async () => {
    open();
    const panel = await expand();
    const provenance = screen.getByText('Provenance: where this result came from').closest('details')!;
    const others = screen.getByText('Other searches of this scenario', { selector: 'summary' }).closest('details')!;
    expect(provenance.compareDocumentPosition(panel)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(others.compareDocumentPosition(panel)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(panel.closest('.detail-side')).toBeNull();
    expect(screen.getByRole('heading', { name: 'Perturbation' }).closest('section')!.contains(panel)).toBe(false);
  });

  it('claims nothing the cells cannot back: no word of safety, regions, boundaries or probability, anywhere in it', async () => {
    open();
    const panel = await expandOpen();
    await userEvent.click(within(panel).getByText('Table of the slice'));
    await within(panel).findByRole('table');
    expect(panel.textContent).not.toMatch(/\b(safe|safety|region|regions|boundary|boundaries|probability|probable|likely|robust)\b/i);
    expect(NOTE).not.toMatch(/\b(safe|safety|region|boundary|probability)\b/i);
  });

  it('says there is none recorded, and draws nothing, for an empty list', async () => {
    open(() => ok(body([])));
    const panel = await expand();
    expect(await within(panel).findByText('No slice is recorded for this scenario.')).toBeInTheDocument();
    expect(within(panel).queryByRole('figure')).toBeNull();
  });
});

describe('the drawing', () => {
  it('draws one cell for every pair of values, in the response’s layout: x along, y up', async () => {
    open();
    const panel = await expandOpen();
    await within(panel).findByRole('figure');
    const c = cells(panel);
    expect(c).toHaveLength(6);
    // the SVG is 3 cells wide and 2 high, 12 units a cell; row 0 (y = 0.5, the smallest) is at the bottom
    expect(c.map((r) => [r.getAttribute('x'), r.getAttribute('y')])).toEqual([['0', '12'], ['12', '12'], ['24', '12'], ['0', '0'], ['12', '0'], ['24', '0']]);
  });

  it('marks a collision by a pattern and a class, a miss by neither, and a cell not recorded by a third look', async () => {
    open();
    const c = cells(await expandOpen());
    expect(c.map((r) => r.getAttribute('class'))).toEqual(['sens-cell sens-miss', 'sens-cell sens-hit', 'sens-cell sens-miss', 'sens-cell sens-miss', 'sens-cell sens-hit', 'sens-cell sens-unknown']);
    expect(c.map((r) => r.getAttribute('fill'))).toEqual([null, 'url(#sens-hatch-7)', null, null, 'url(#sens-hatch-7)', null]);
    expect(document.getElementById('sens-hatch-7')).not.toBeNull();
  });

  it('says in words what each look means', async () => {
    open();
    const panel = await expandOpen();
    expect([...panel.querySelectorAll('.sens-legend li')].map((li) => li.textContent)).toEqual([
      'Hatched cell: the replay collides.', 'Plain cell: it does not.', 'Dotted cell: not recorded.', 'Ring: the stored perturbation.',
    ]);
  });

  it('rings the stored point at its cell, centred on it', async () => {
    open();
    const panel = await expandOpen();
    const ring = panel.querySelector('circle.sens-stored')!;
    // stored cell: x index 1, y index 1 (the top row): centre at (18, 6)
    expect([ring.getAttribute('cx'), ring.getAttribute('cy')]).toEqual(['18', '6']);
  });

  it('rings the stored point wherever it is: a cell that is not on the diagonal of the grid', async () => {
    open(() => ok(body([slice({ stored_x_index: 2, stored_y_index: 0, collides: [false, true, true, false, false, null] })])));
    const panel = await expandOpen();
    const ring = panel.querySelector('circle.sens-stored')!;
    // x index 2 of 3, y index 0 of 2 (the bottom row): centre at (2 * 12 + 6, (2 - 1 - 0) * 12 + 6) = (30, 18)
    expect([ring.getAttribute('cx'), ring.getAttribute('cy')]).toEqual(['30', '18']);
    // and the figure's name reads that cell: row 0, column 2, which collides
    expect(within(panel).getByRole('figure')).toHaveAccessibleName(/where the replay collides$/);
    expect(within(panel).getByRole('figure')).toHaveAccessibleName(/initial speed \(m\/s\) 1\.0000 and acceleration bias \(m\/s\^2\) 0\.5000/);
  });

  it('does not ring anything, and says so, when the stored value is not one of the grid’s values', async () => {
    open(() => ok(body([slice({ stored_x_index: null })])));
    const panel = await expandOpen();
    await within(panel).findByRole('figure');
    expect(panel.querySelector('circle.sens-stored')).toBeNull();
    expect(panel.querySelector('.sens-legend')).toHaveTextContent('The stored value is not one of the grid’s values, so no cell is ringed.');
    expect(within(panel).getByRole('figure')).toHaveAccessibleName(/the stored value is not one of the grid’s values/);
  });

  it('is named for assistive technology with the axes, the count and the verdict at the stored point', async () => {
    open();
    const panel = await expandOpen();
    expect(within(panel).getByRole('figure')).toHaveAccessibleName(
      'Collision verdicts over initial speed (m/s) against acceleration bias (m/s^2): 2 of 6 cells collide; ' +
        'the stored perturbation is at initial speed (m/s) 0.0000 and acceleration bias (m/s^2) 1.5000, where the replay collides',
    );
    expect(panel.querySelector('svg')).toHaveAttribute('aria-hidden', 'true');
  });

  it.each([
    ['does not collide', [false, true, false, false, false, null]],
    ['has no recorded verdict', [false, true, false, false, null, null]],
  ])('says in the name that the stored point %s when its cell says so', async (words, collides) => {
    open(() => ok(body([slice({ collides })])));
    const panel = await expandOpen();
    expect(within(panel).getByRole('figure')).toHaveAccessibleName(new RegExp(`where the replay ${words}$`));
  });

  it('gives the range of each axis, with its unit, under the drawing, and says which way it runs', async () => {
    open();
    const panel = await expandOpen();
    expect(panel.querySelector('figcaption')).toHaveTextContent('initial speed (m/s): -1.0000 to 1.0000 (left to right)');
    expect(panel.querySelector('figcaption')).toHaveTextContent('acceleration bias (m/s^2): 0.5000 to 1.5000 (bottom to top)');
  });

  it('counts the cells that collide and names the held components with their units', async () => {
    open();
    const panel = await expandOpen();
    expect(within(panel).getByText('2 of 6 cells collide.')).toBeInTheDocument();
    expect(panel.querySelector('.sens-held')).toHaveTextContent('Held at the stored values: initial heading (rad) = 0.0500; steering bias (rad) = -0.0200.');
  });
});

describe('twelve slices of the largest size', () => {
  // every ordered pair of components, each 101 by 101: 10 201 cells a slice, 122 412 in all
  const PAIRS = [0, 1, 2, 3].flatMap((a) => [0, 1, 2, 3].filter((b) => b !== a).map((b) => [a, b] as const));
  const axis = [...Array(101).keys()].map((i) => -1 + i / 50);
  const big = (id: number, a: number, b: number): SensitivitySlice =>
    slice({
      slice_id: id, axis_x: a, axis_y: b, axis_x_label: BICYCLE[a]!, axis_y_label: BICYCLE[b]!, x_values: axis, y_values: axis,
      weighted_norm: Array.from({ length: 101 * 101 }, (_, k) => (k % 97) / 100), collides: Array.from({ length: 101 * 101 }, (_, k) => k % 3 === 0),
      held_delta: [0, 0, 0, 0], stored_x_index: 50, stored_y_index: 50,
    });
  const twelve = () => ok(body(PAIRS.map(([a, b], i) => big(i + 1, a, b))));

  it('draws no cell at all until a slice is opened, and names every slice with its axes and its grid size', async () => {
    open(twelve);
    const panel = await expand();
    const summaries = await within(panel).findAllByText(SLICE_TITLE, { selector: 'summary' });
    expect(summaries).toHaveLength(12);
    expect(summaries[0]).toHaveTextContent('initial speed (m/s) against initial heading (rad), 101 by 101');
    expect(summaries[11]).toHaveTextContent('steering bias (rad) against acceleration bias (m/s^2), 101 by 101');
    expect(panel.querySelectorAll('rect.sens-cell')).toHaveLength(0);
    expect(panel.querySelectorAll('svg, figure, .sens-legend, table, circle, details.sens-table, .sens-held')).toHaveLength(0);
    for (const d of panel.querySelectorAll('details.sens-slice')) expect(d).not.toHaveAttribute('open');
  }, 30000);

  it('draws exactly the cells of the slice that is opened, and takes them away when it is closed', async () => {
    open(twelve);
    const panel = await expand();
    const summaries = await within(panel).findAllByText(SLICE_TITLE, { selector: 'summary' });
    await userEvent.click(summaries[3]!);
    await within(panel).findByRole('figure');
    expect(panel.querySelectorAll('rect.sens-cell')).toHaveLength(10201);
    expect(panel.querySelectorAll('figure')).toHaveLength(1);
    expect(summaries[3]!.closest('details')!.querySelectorAll('rect.sens-cell')).toHaveLength(10201);
    for (const i of [0, 1, 2, 4, 11]) expect(summaries[i]!.closest('details')!.querySelectorAll('rect.sens-cell')).toHaveLength(0);
    await userEvent.click(summaries[3]!);
    expect(panel.querySelectorAll('rect.sens-cell')).toHaveLength(0);
    expect(panel.querySelectorAll('figure, .sens-legend, circle')).toHaveLength(0);
  }, 60000);

  it('opens two slices at once only when asked: each draws its own cells and no other', async () => {
    open(twelve);
    const panel = await expand();
    const summaries = await within(panel).findAllByText(SLICE_TITLE, { selector: 'summary' });
    await userEvent.click(summaries[0]!);
    await userEvent.click(summaries[11]!);
    await waitFor(() => expect(panel.querySelectorAll('figure')).toHaveLength(2));
    expect(panel.querySelectorAll('rect.sens-cell')).toHaveLength(2 * 10201);
  }, 60000);

  it('keeps the table’s own details inside the slice: closed until asked, and drawn only while it is open', async () => {
    open(twelve);
    const panel = await expand();
    const summaries = await within(panel).findAllByText(SLICE_TITLE, { selector: 'summary' });
    await userEvent.click(summaries[0]!);
    const slice0 = summaries[0]!.closest('details')!;
    await within(slice0).findByRole('figure');
    expect(slice0.querySelector('details.sens-table')).not.toHaveAttribute('open');
    expect(slice0.querySelector('table')).toBeNull();
    await userEvent.click(within(slice0).getByText('Table of the slice'));
    const table = await within(slice0).findByRole('table');
    expect(within(table).getAllByRole('row')).toHaveLength(102);
    await userEvent.click(within(slice0).getByText('Table of the slice'));
    await waitFor(() => expect(slice0.querySelector('table')).toBeNull());
    await userEvent.click(within(slice0).getByText('Table of the slice'));
    await within(slice0).findByRole('table');
    await userEvent.click(summaries[0]!);                                 // closing the slice takes the table with it
    expect(slice0.querySelector('table')).toBeNull();
    expect(slice0.querySelectorAll('rect.sens-cell')).toHaveLength(0);
  }, 120000);

  it('has the table’s details closed again when the slice is opened a second time', async () => {
    open(twelve);
    const panel = await expand();
    const summaries = await within(panel).findAllByText(SLICE_TITLE, { selector: 'summary' });
    await userEvent.click(summaries[1]!);
    const d = summaries[1]!.closest('details')!;
    await userEvent.click(await within(d).findByText('Table of the slice'));
    await within(d).findByRole('table');
    await userEvent.click(summaries[1]!);
    await userEvent.click(summaries[1]!);
    await within(d).findByRole('figure');
    expect(d.querySelector('table')).toBeNull();
  }, 120000);
});

describe('when the response is fetched again', () => {
  it('draws the cells of the slice that is open from the new data and still draws none for the others', async () => {
    let calls = 0;
    const requests = open(() => {
      calls += 1;
      // the second response is a new object, and the middle slice now collides everywhere: the count text shows the new data was drawn
      return ok(body([slice({ slice_id: 1 }), slice(calls > 1 ? { slice_id: 2, collides: [true, true, true, true, true, true] } : { slice_id: 2 }), slice({ slice_id: 3 })]));
    });
    const panel = await expand();
    await openSlice(panel, 1);
    await within(panel).findByText('2 of 6 cells collide.');
    expect(cells(panel)).toHaveLength(6);
    const slices = [...panel.querySelectorAll('details.sens-slice')];
    expect(slices.map((d) => d.querySelectorAll('rect.sens-cell').length)).toEqual([0, 6, 0]);
    window.dispatchEvent(new Event('visibilitychange'));                // the page is looked at again: a stale query is fetched again
    await within(panel).findByText('6 of 6 cells collide.');
    expect(requests.filter((r) => r.path === `/scenarios/${ID}/slice`)).toHaveLength(2);
    expect(cells(panel)).toHaveLength(6);
    expect(slices.map((d) => d.querySelectorAll('rect.sens-cell').length)).toEqual([0, 6, 0]);
    expect(panel.querySelectorAll('figure')).toHaveLength(1);
    expect([...panel.querySelectorAll('details.sens-slice')].map((d) => d.hasAttribute('open'))).toEqual([false, true, false]);
  });
});

describe('the table of the slice', () => {
  it('is closed and not drawn until it is opened', async () => {
    open();
    const panel = await expandOpen();
    await within(panel).findByRole('figure');
    const details = panel.querySelector('details.sens-table')!;
    expect(details).not.toHaveAttribute('open');
    expect(within(panel).queryByRole('table')).toBeNull();
  });

  it('gives every cell in words, the weighted norm, and “(stored)” on the stored cell only; the top row is the largest y', async () => {
    open();
    const panel = await expandOpen();
    await userEvent.click(within(panel).getByText('Table of the slice'));
    const table = await within(panel).findByRole('table');
    expect(table.querySelector('caption')).toHaveTextContent('acceleration bias (m/s^2) down the side, initial speed (m/s) across the top');
    expect(within(table).getAllByRole('columnheader').map((h) => h.textContent)).toEqual(['acceleration bias (m/s^2) \\ initial speed (m/s)', '-1.0000', '0.0000', '1.0000']);
    const rows = within(table).getAllByRole('row').slice(1);
    expect(rows.map((r) => within(r).getByRole('rowheader').textContent)).toEqual(['1.5000', '0.5000']);
    expect(within(rows[0]!).getAllByRole('cell').map((c) => c.textContent)).toEqual(['no collision, 0.700', 'collision, 0.200 (stored)', 'Not recorded, 0.600']);
    expect(within(rows[1]!).getAllByRole('cell').map((c) => c.textContent)).toEqual(['no collision, 0.900', 'collision, 0.300', 'no collision, 0.800']);
    expect(table.textContent!.match(/\(stored\)/g)).toHaveLength(1);
  });

  it('is drawn again only while it is open', async () => {
    open();
    const panel = await expandOpen();
    await userEvent.click(within(panel).getByText('Table of the slice'));
    await within(panel).findByRole('table');
    await userEvent.click(within(panel).getByText('Table of the slice'));
    expect(within(panel).queryByRole('table')).toBeNull();
  });
});

describe('the model', () => {
  it('names the axes with the labels of that slice’s own model', async () => {
    open(() => ok(body([slice({ delta_parameterization: 'linear', axis_x_label: LINEAR[0], axis_y_label: LINEAR[2], delta_labels: LINEAR })], { stored_delta_parameterization: 'linear' })));
    const panel = await expandOpen();
    expect(panel.querySelector('details.sens-slice > summary')).toHaveTextContent('initial vx (m/s) against x-acceleration bias (m/s^2), 3 by 2');
    expect(panel.querySelector('.sens-held')).toHaveTextContent('initial vy (m/s) = 0.0500; y-acceleration bias (m/s^2) = -0.0200');
  });

  it.each([
    ['another model', slice({ delta_parameterization: 'linear', axis_x_label: LINEAR[0], axis_y_label: LINEAR[2], delta_labels: LINEAR }), 'bicycle'],
    ['an unrecorded model', slice({ delta_parameterization: null, axis_x_label: null, axis_y_label: null, delta_labels: null }), 'bicycle'],
    ['a stored result under an unrecorded model', slice(), null],
  ])('shows no figure of a slice under %s, and says why', async (_name, s, stored) => {
    open(() => ok(body([s], { stored_delta_parameterization: stored })));
    const panel = await expand();
    await within(panel).findByText('Under a different or unrecorded model; its figures are not shown against the stored result’s.');
    expect(within(panel).queryByRole('figure')).toBeNull();
    expect(within(panel).queryByText('Table of the slice')).toBeNull();
    expect(panel.textContent).not.toMatch(/cells collide/);
  });

  it('does not line up two models: the one under the stored result’s model is drawn, the other is not', async () => {
    open(() => ok(body([slice({ slice_id: 1 }), slice({ slice_id: 2, delta_parameterization: 'linear', delta_labels: LINEAR })])));
    const panel = await expandOpen();
    await within(panel).findByRole('figure');
    expect(within(panel).getAllByRole('figure')).toHaveLength(1);
    expect(within(panel).getAllByText(/different or unrecorded model/)).toHaveLength(1);
  });
});

describe('what is not recorded', () => {
  it('says the grid is not recorded when the API sent none, and draws nothing of it', async () => {
    open(() => ok(body([slice({ x_values: null, weighted_norm: null, collides: null })])));
    const panel = await expand();
    expect(await within(panel).findByText('The grid of this slice is not recorded.')).toBeInTheDocument();
    expect(within(panel).queryByRole('figure')).toBeNull();
    expect(within(panel).queryByText('Table of the slice')).toBeNull();
  });

  it.each([
    ['the x values', { x_values: null }],
    ['the y values', { y_values: null }],
    ['the verdicts', { collides: null }],
    ['the first axis', { axis_x: null }],
    ['the second axis', { axis_y: null }],
  ])('says so, in one place and in no details, when only %s are missing', async (_what, over) => {
    open(() => ok(body([slice(over as Partial<SensitivitySlice>)])));
    const panel = await expand();
    expect(await within(panel).findByText('The grid of this slice is not recorded.')).toBeInTheDocument();
    expect(within(panel).getAllByText('The grid of this slice is not recorded.')).toHaveLength(1);
    expect(panel.querySelectorAll('details.sens-slice, svg, figure, table')).toHaveLength(0);
  });

  it('shows a norm that is not recorded as nothing, not as zero, in the table', async () => {
    open(() => ok(body([slice({ weighted_norm: [0.9, null, 0.8, 0.7, 0.2, 0.6] })])));
    const panel = await expandOpen();
    await userEvent.click(within(panel).getByText('Table of the slice'));
    const table = await within(panel).findByRole('table');
    const row = within(table).getAllByRole('row')[2]!;
    expect(within(row).getAllByRole('cell')[1]!.textContent).toBe('collision');
  });
});

describe('next to the other searches', () => {
  it('each shows its own data when both are open on one page', async () => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/slice')) return ok(body([slice()]));
      if (req.path.endsWith('/other-searches')) {
        return ok({ scenario_id: ID, stress_run_id: 'run-abc', stored_delta_parameterization: 'bicycle', truncated: false, searches: [{ search_run_id: '0123456789abcdef', kind: 'seed', seed: 41, popsize: 15, maxiter: 200, tol: 0.001, outcome: 'collision_found', min_perturbation: 0.25, collision_timestep: 40, delta: [0.5, 0.01, -0.2, 0.003], delta_parameterization: 'bicycle', delta_labels: BICYCLE, n_eval: 900 }] });
      }
      if (req.path.endsWith('/provenance')) return ok({ scenario_id: ID, stress_run_id: 'run-abc' });
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
      return ok(makeDetail({ scenario_id: ID }));
    });
    renderApp(`/scenarios/${ID}`);
    await userEvent.click(await screen.findByText('Other searches of this scenario', { selector: 'summary' }));
    await screen.findByRole('table', { name: 'Other searches of this scenario' });
    const panel = await expandOpen();
    await within(panel).findByText('Hatched cell: the replay collides.');
    expect(within(panel).getByRole('figure')).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'Other searches of this scenario' })).toHaveTextContent('41');
  });
});

describe('a list the API cut off', () => {
  const SENTENCE = 'Not every recorded slice is listed here; those shown are the first in the order above.';

  it('says so in one sentence after the slices when the response says it is truncated', async () => {
    open(() => ok(body([slice(), slice({ slice_id: 8 })], { truncated: true })));
    const panel = await expand();
    const summaries = await within(panel).findAllByText(SLICE_TITLE, { selector: 'summary' });
    expect(summaries).toHaveLength(2);
    const sentence = within(panel).getByText(SENTENCE);
    expect(within(panel).getAllByText(SENTENCE)).toHaveLength(1);
    expect(summaries[1]!.closest('details')!.compareDocumentPosition(sentence)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(SENTENCE).not.toMatch(/\d/);
  });

  it('is absent when the response says it is not, when the field is missing, and for a value that is not true', async () => {
    for (const odd of [false, undefined, 'yes', 1]) {
      open(() => ok({ ...body([slice()]), truncated: odd } as unknown as SensitivityResponse));
      const panel = await expandOpen();
      await within(panel).findByRole('figure');
      expect(panel.textContent).not.toMatch(/Not every recorded slice/);
      document.body.innerHTML = '';
    }
  });
});
