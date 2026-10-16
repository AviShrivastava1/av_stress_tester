import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { FragilityBreakdown } from '../api/client';
import { makeDetail, makePerturbed, makeTrajectories } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { readFrontendFile } from '../test/readFrontendFile';
import { renderApp } from '../test/renderApp';

/*
 * The fragility breakdown under Run details (pages/detail/FragilityBreakdown.tsx): the two terms of the stored score, as the API
 * worked them out from the stored Min TTC and Min PET. The page writes no weight, floor, sentinel or score: every number below is
 * in the payload, so the fixtures here use numbers that are NOT the production ones (weights 0.6/0.4, floors 0.5 and 0.25) to show
 * they are read, not remembered.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });

/** A breakdown for ttc = 2 s and pet = 4 s with weights 0.6 and 0.4: signals 0.5 and 0.25, terms 0.3 and 0.1, score 0.4. */
function terms(over: Partial<FragilityBreakdown> = {}): FragilityBreakdown {
  return {
    ttc_weight: 0.6, pet_weight: 0.4, ttc_signal: 0.5, pet_signal: 0.25, ttc_term: 0.3, pet_term: 0.1, recomputed_score: 0.4,
    ttc_floor_s: 0.5, pet_floor_s: 0.25, ttc_none: false, pet_none: false, ttc_at_floor: false, pet_at_floor: false,
    reproduces_stored_score: true, ...over,
  };
}

async function open(detail: Parameters<typeof makeDetail>[0]) {
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
    if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
    return ok(makeDetail({ scenario_id: ID, ...detail }));
  });
  renderApp(`/scenarios/${ID}`);
  await screen.findByRole('heading', { name: 'Run details' });
}

const group = () => screen.getByRole('group', { name: 'Fragility, by term' });
const rowOf = (name: string) => within(group()).getByRole('row', { name: new RegExp(`^${name}`) });
const cells = (name: string) => within(rowOf(name)).getAllByRole('cell').map((c) => c.textContent);

describe('the breakdown of a stored score', () => {
  it('sits in Run details, under the facts, with a table of the two terms and the score', async () => {
    await open({ min_ttc: 2, min_pet: 4, fragility_score: 0.4, fragility_breakdown: terms() });
    const panel = screen.getByRole('heading', { name: 'Run details' }).closest('section')!;
    expect(panel.contains(group())).toBe(true);
    expect(within(group()).getByRole('heading', { level: 3, name: 'Fragility, by term' })).toBeInTheDocument();
    const table = within(group()).getByRole('table', { name: 'Fragility, by term' });
    expect(within(table).getAllByRole('columnheader').map((h) => h.textContent)).toEqual(['Term', 'Value', 'Signal', 'Weight', 'Part of the score']);
  });

  it('shows each term as the API worked it out: value, signal, weight and part', async () => {
    await open({ min_ttc: 2, min_pet: 4, fragility_score: 0.4, fragility_breakdown: terms() });
    expect(cells('Min TTC')).toEqual(['2.00 s', '0.500', '0.600', '0.300']);
    expect(cells('Min PET')).toEqual(['4.00 s', '0.250', '0.400', '0.100']);
  });

  it('ends with the score, the one Run details shows above, and the parts add up to it', async () => {
    await open({ min_ttc: 2, min_pet: 4, fragility_score: 0.4, fragility_breakdown: terms() });
    const foot = within(group()).getByRole('row', { name: /^Fragility/ });
    expect(within(foot).getByRole('cell')).toHaveTextContent('0.400');
    expect(screen.getAllByText('0.400').length).toBeGreaterThanOrEqual(2); // the fact above and the table's last row
    const parts = [...cells('Min TTC').slice(-1), ...cells('Min PET').slice(-1)].map(Number);
    expect(parts[0]! + parts[1]!).toBeCloseTo(0.4, 12);
  });

  it('says the score ranks the recorded scene only, and the floors it was given', async () => {
    await open({ min_ttc: 2, min_pet: 4, fragility_score: 0.4, fragility_breakdown: terms() });
    expect(group()).toHaveTextContent(
      'The score ranks the recorded scene only; the search result is not part of it. Each signal is 1 divided by the time in seconds, ' +
        'with the time held to at least 0.500 s for TTC and 0.250 s for PET.',
    );
  });

  it('shows a value at the floor with a tag, and no tag otherwise', async () => {
    await open({ min_ttc: 0.1, min_pet: 4, fragility_score: 1, fragility_breakdown: terms({ ttc_at_floor: true, ttc_signal: 2, ttc_term: 1.2, recomputed_score: 1.3 }) });
    expect(within(rowOf('Min TTC')).getByText('at the floor')).toBeInTheDocument();
    expect(within(rowOf('Min PET')).queryByText('at the floor')).not.toBeInTheDocument();
  });

  it('shows an overlap as the PET cell does, and as at the floor', async () => {
    await open({
      min_ttc: 2, min_pet: -3.6, fragility_score: 1,
      fragility_breakdown: terms({ pet_at_floor: true, pet_signal: 4, pet_term: 1.6, recomputed_score: 1.9 }),
    });
    const pet = rowOf('Min PET');
    expect(within(pet).getByText('overlap')).toBeInTheDocument();
    expect(within(pet).getByText('at the floor')).toBeInTheDocument();
    expect(pet).toHaveTextContent('-3.60 s');
  });

  it('shows the "none" sentinel as none, with no signal and a part of zero', async () => {
    await open({ min_ttc: 999, min_pet: 4, fragility_score: 0.1, fragility_breakdown: terms({ ttc_none: true, ttc_signal: 0, ttc_term: 0, recomputed_score: 0.1 }) });
    const ttc = rowOf('Min TTC');
    expect(within(ttc).getByText('none')).toBeInTheDocument();
    expect(within(ttc).queryByText('999.00 s')).not.toBeInTheDocument();
    expect(cells('Min TTC')).toEqual(['none', '—', '0.600', '0.000']);
    expect(within(ttc).queryByText('at the floor')).not.toBeInTheDocument();
  });

  it('shows a PET that is the sentinel as none while the TTC row stays as it is', async () => {
    await open({ min_ttc: 2, min_pet: 999, fragility_score: 0.3, fragility_breakdown: terms({ pet_none: true, pet_signal: 0, pet_term: 0, recomputed_score: 0.3 }) });
    expect(cells('Min PET')).toEqual(['none', '—', '0.400', '0.000']);
    expect(cells('Min TTC')).toEqual(['2.00 s', '0.500', '0.600', '0.300']);
  });

  it('keeps the TTC row and the PET row apart when only one has the sentinel or the floor', async () => {
    await open({ min_ttc: 999, min_pet: 0.1, fragility_score: 0.4, fragility_breakdown: terms({ ttc_none: true, ttc_signal: 0, ttc_term: 0, pet_at_floor: true }) });
    expect(within(rowOf('Min TTC')).queryByText('at the floor')).not.toBeInTheDocument();
    expect(within(rowOf('Min PET')).getByText('at the floor')).toBeInTheDocument();
    expect(cells('Min PET').slice(1)).not.toContain('—');
  });

  it('shows the stored score in the last row, exactly, even when the recomputed one differs in the last digits', async () => {
    await open({ min_ttc: 2, min_pet: 4, fragility_score: 0.4, fragility_breakdown: terms({ recomputed_score: 0.4000000000000001 }) });
    const cell = within(within(group()).getByRole('row', { name: /^Fragility/ })).getByRole('cell');
    expect(cell).toHaveAttribute('title', '0.4');
  });

  it('keeps the exact figures as tooltips', async () => {
    await open({ min_ttc: 2, min_pet: 4, fragility_score: 0.4, fragility_breakdown: terms({ ttc_term: 0.30000000000000004 }) });
    const cell = within(rowOf('Min TTC')).getAllByRole('cell').slice(-1)[0]!;
    expect(cell).toHaveAttribute('title', '0.30000000000000004');
  });
});

describe('when there is nothing honest to show', () => {
  it('says why when Min TTC or Min PET is not stored', async () => {
    await open({ min_ttc: null, min_pet: 4, fragility_breakdown: null });
    expect(group()).toHaveTextContent('No breakdown: Min TTC or Min PET is not stored for this scenario.');
    expect(within(group()).queryByRole('table')).not.toBeInTheDocument();
  });

  it('shows no table, and both numbers, when the terms do not add up to the stored score', async () => {
    await open({ min_ttc: 2, min_pet: 4, fragility_score: 7, fragility_breakdown: terms({ reproduces_stored_score: false }) });
    expect(group()).toHaveTextContent('The stored score is 7.000. Its stored Min TTC and Min PET give 0.400, so the breakdown does not describe it and is not shown.');
    expect(within(group()).queryByRole('table')).not.toBeInTheDocument();
    expect(within(group()).queryByText('Part of the score')).not.toBeInTheDocument();
  });

  it('shows nothing at all when the API did not send the field', async () => {
    await open({ min_ttc: 2, min_pet: 4, fragility_score: 0.4 });
    expect(screen.queryByRole('group', { name: 'Fragility, by term' })).not.toBeInTheDocument();
  });
});

describe('the page writes no weight, floor or sentinel of its own', () => {
  it('has no such number in the component outside its comments', () => {
    const source = readFrontendFile('src/pages/detail/FragilityBreakdown.tsx')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/\/\/[^\n]*/g, '');
    const numbers = [...source.matchAll(/(?<![\w.])\d+(?:\.\d+)?(?![\w.])/g)].map((m) => m[0]);
    // 1 and 3 appear as the markup's level and span; the data's numbers (0.4, 0.7, 0.3, 0.01, 999, 100) do not.
    for (const n of numbers) expect(['1', '3', '4']).toContain(n);
    for (const forbidden of ['0.4', '0.7', '0.3', '0.01', '999', '100', '4/7', '3/7']) expect(source).not.toContain(forbidden);
  });
});
