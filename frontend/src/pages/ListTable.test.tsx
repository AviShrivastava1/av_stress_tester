import { screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { describeOutcome } from '../domain/outcomes';
import { formatMagnitude } from '../domain/format';
import { makePage, makeRow } from '../test/fixtures';
import { stubFetch, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The ranked table on the scenario list: what its caption promises about the order, and what each header
 * says. The columns are checked by position as well as by name, so a header moved without its cells fails.
 */

afterEach(() => {
  vi.unstubAllGlobals();
});

const ok = (body: unknown): Reply => ({ status: 200, body });

const CAPTION =
  'Ranked by fragility score, most fragile first. Equal scores are tie-broken by scenario ID; position within a ' +
  'tie does not mean one scenario is more fragile than another. Min perturbation is the smallest found by a ' +
  'finite search, not a proven minimum.';

const HEADERS = ['#', 'Scenario', 'Fragility', 'Timing', 'Search result', 'Min perturbation', 'Agents'];

const TIMING_TITLE = 'Minimum TTC and PET over pairs involving the SDC';
const TTC_TITLE = 'Minimum time-to-collision over pairs involving the SDC';
const PET_TITLE = 'Minimum post-encroachment time over pairs involving the SDC';

const COL = { rank: 0, id: 1, fragility: 2, timing: 3, result: 4, perturbation: 5, agents: 6 } as const;

async function openList(rows = [
  makeRow({
    scenario_id: 'synthetic-0001', fragility_score: 3.5, min_ttc: 1.5, min_pet: 0.8, n_agents: 7,
    stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', min_perturbation: 0.02,
  }),
  makeRow({ scenario_id: 'synthetic-0002', fragility_score: 2, min_ttc: 999, min_pet: -0.4 }),
  makeRow({ scenario_id: 'synthetic-0003', fragility_score: 1, min_ttc: null, min_pet: 999 }),
]) {
  stubFetch(() => ok(makePage(rows, null)));
  renderApp('/');
  await screen.findByRole('link', { name: rows[0]!.scenario_id });
  return screen.getByRole('table');
}

function bodyRow(table: HTMLElement, index: number) {
  return within(table).getAllByRole('row')[index + 1]!;
}
const cell = (row: HTMLElement, col: number) => within(row).getAllByRole('cell')[col]!;

describe('the caption', () => {
  it('says how the order is made, and what a tie does and does not mean', async () => {
    await openList();
    expect(screen.getByText(CAPTION)).toBeVisible();
  });

  it('no longer says the order "comes from the server"', async () => {
    await openList();
    expect(screen.queryByText(/The order comes from the server/)).not.toBeInTheDocument();
  });
});

describe('the headers', () => {
  it('are, in order: #, Scenario, Fragility, Timing, Search result, Min perturbation, Agents', async () => {
    const table = await openList();
    const headers = within(table).getAllByRole('columnheader');
    expect(headers.map((h) => h.textContent)).toEqual(HEADERS);
  });

  it('give Timing the title that says which pairs it covers', async () => {
    const table = await openList();
    expect(within(table).getByRole('columnheader', { name: 'Timing' })).toHaveAttribute('title', TIMING_TITLE);
  });

  it.each(['Min change', 'Min TTC', 'Min PET', 'Outcome'])('have no "%s" column any more', async (old) => {
    const table = await openList();
    expect(within(table).queryByRole('columnheader', { name: old })).not.toBeInTheDocument();
  });

  it('are matched by seven cells in every row, in the same order', async () => {
    const table = await openList();
    const first = bodyRow(table, 0);
    expect(within(first).getAllByRole('cell')).toHaveLength(HEADERS.length);
    expect(cell(first, COL.rank)).toHaveTextContent('1');
    expect(cell(first, COL.id)).toHaveTextContent('synthetic-0001');
    expect(cell(first, COL.fragility)).toHaveTextContent('3.500');
    expect(cell(first, COL.result)).toHaveTextContent(describeOutcome('collision_found').label);
    expect(cell(first, COL.perturbation)).toHaveTextContent(formatMagnitude(0.02));
    expect(cell(first, COL.agents)).toHaveTextContent('7');
    // and not swapped: the result badge is not in the perturbation cell, nor the number in the result cell
    expect(cell(first, COL.perturbation)).not.toHaveTextContent(describeOutcome('collision_found').label);
    expect(cell(first, COL.result)).not.toHaveTextContent(formatMagnitude(0.02));
  });
});

describe('the Timing cell', () => {
  it('shows a labelled TTC line and a labelled PET line, each label carrying its own tooltip', async () => {
    const table = await openList();
    const timing = cell(bodyRow(table, 0), COL.timing);
    const lines = timing.querySelectorAll('.timing-line');
    expect(lines).toHaveLength(2);
    const [ttc, pet] = [...lines] as HTMLElement[];
    expect(within(ttc!).getByText('TTC')).toHaveAttribute('title', TTC_TITLE);
    expect(ttc).toHaveTextContent('TTC 1.50 s');
    expect(within(pet!).getByText('PET')).toHaveAttribute('title', PET_TITLE);
    expect(pet).toHaveTextContent('PET 0.80 s');
  });

  it('keeps TimeMetricCell\'s own rendering: "none" with its meaning, "—" for a missing value', async () => {
    const table = await openList();
    const noInteraction = cell(bodyRow(table, 1), COL.timing);
    const none = within(noInteraction).getByText('none');
    expect(none).toHaveAttribute('title', expect.stringContaining('No finite collision time predicted for an SDC pair'));
    expect(noInteraction).not.toHaveTextContent('999');

    const mixed = cell(bodyRow(table, 2), COL.timing);
    expect(within(mixed).getByText('—')).toBeVisible(); // TTC missing
    expect(within(mixed).getByText('none')).toHaveAttribute('title', expect.stringContaining('No shared conflict zone with the SDC'));
  });

  it('keeps the "overlap" tag on a negative PET', async () => {
    const table = await openList();
    const timing = cell(bodyRow(table, 1), COL.timing);
    expect(timing).toHaveTextContent('PET -0.40 s');
    expect(within(timing).getByText('overlap')).toHaveAttribute('title', expect.stringContaining('Negative PET'));
    // no tag on the positive PET of the first row
    expect(within(cell(bodyRow(table, 0), COL.timing)).queryByText('overlap')).not.toBeInTheDocument();
  });
});
