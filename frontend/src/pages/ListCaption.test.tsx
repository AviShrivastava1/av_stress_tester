import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import type { StatsResponse } from '../api/client';
import { makePage, makeRow } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The caption under the list's controls follows the sort (pages/ScenarioListPage.tsx FRAGILITY_CAPTION and PERTURBATION_CAPTION).
 * Under the fragility sort, the API's own order, it says the list is ranked by fragility and that equal scores are tied by scenario
 * ID. Under the Scenario ID sort it says that order. Under the Min perturbation sort that would be false, so it says the order that sort really has: smallest first, rows with no
 * perturbation after the rest, equal values by scenario ID, and that the figure is a weighted norm over components of mixed units
 * and not comparable between scenarios searched under different models. Each of those claims is checked below against the rows
 * the page actually shows.
 */

const FRAGILITY =
  'Ranked by fragility score, most fragile first. Equal scores are tie-broken by scenario ID; position within a tie does not mean one ' +
  'scenario is more fragile than another. Min perturbation is the smallest found by a finite search, not a proven minimum.';
const PERTURBATION =
  'Sorted by the smallest perturbation found, smallest first. Min perturbation is a weighted norm over components of mixed units, so ' +
  'figures from scenarios searched under different models are not comparable. Scenarios with no perturbation found come after those ' +
  'that have one. Scenarios with equal Min perturbation are listed by scenario ID. Min perturbation is the smallest found by a finite ' +
  'search, not a proven minimum.';

const ID =
  'Sorted by scenario ID, ascending, compared as text, character by character, so “10” comes before “9”. ' +
  'The fragility score does not decide the order.';

const hit = (id: string, perturbation: number, fragility: number) =>
  makeRow({ scenario_id: id, min_perturbation: perturbation, stress_outcome: 'collision_found', stress_tested: true, stress_attempted: true, fragility_score: fragility });
// The API's order (fragility, then id) is not the perturbation order, and two rows tie on perturbation.
const ROWS = [
  hit('d', 0.9, 9),
  makeRow({ scenario_id: 'm', fragility_score: 8 }), // no perturbation found
  hit('b', 0.1, 7),
  hit('a', 0.1, 6), // ties with b; the API puts b first, the id order puts a first
  makeRow({ scenario_id: 'k', stress_outcome: 'replay_infeasible', stress_tested: true, fragility_score: 5 }), // refused: none
  hit('c', 0.4, 4),
];
const STATS = {
  total_scenarios: 6, stress_tested: 5, collisions_found: 4, no_collision_found: 0, replay_infeasible: 1, heading_blend_singularity: 0,
  no_challenger: 0, stress_errors: 0, robustly_safe: 0, with_geometry: 0, fragility_min: 4, fragility_max: 9, fragility_mean: 6,
} satisfies StatsResponse;

function serve() {
  stubFetch((req: RecordedRequest): Reply => ({ status: 200, body: req.path === '/stats' ? STATS : makePage(ROWS, null) }));
}
async function open(url: string) {
  serve();
  renderApp(url);
  await screen.findByRole('link', { name: 'a' });
  await waitFor(() => expect(document.querySelector('p.caption')).not.toBeNull());
}
const caption = () => document.querySelector('p.caption')!.textContent;
const order = () => within(screen.getByRole('table')).getAllByRole('row').slice(1).map((r) => within(r).getAllByRole('link')[0]!.textContent);

describe('the caption under each sort', () => {
  it('under the fragility sort is the fragility caption, word for word, and none of the perturbation sentences', async () => {
    await open('/');
    expect(caption()).toBe(FRAGILITY);
    expect(caption()).not.toMatch(/Sorted by the smallest perturbation|weighted norm|no perturbation found/);
  });

  it('under the Min perturbation sort is the perturbation caption, word for word, and none of the fragility sentences', async () => {
    await open('/?sort=perturbation');
    expect(caption()).toBe(PERTURBATION);
    expect(caption()).not.toMatch(/Ranked by fragility score|Equal scores are tie-broken|most fragile first/);
  });

  it('follows the control: perturbation, back to fragility', async () => {
    await open('/');
    await userEvent.selectOptions(screen.getByLabelText('Sort'), 'perturbation');
    await waitFor(() => expect(caption()).toBe(PERTURBATION));
    await userEvent.selectOptions(screen.getByLabelText('Sort'), 'fragility');
    await waitFor(() => expect(caption()).toBe(FRAGILITY));
  });

  it('is the same under an outcome filter, and is visible without a click', async () => {
    await open('/?sort=perturbation&outcome=collision_found');
    expect(caption()).toBe(PERTURBATION);
    expect(document.querySelector('p.caption')).toBeVisible();
    expect(document.querySelector('p.caption')!.closest('details')).toBeNull();
  });

  it('has no figure of the data in either caption', () => {
    expect(FRAGILITY).not.toMatch(/\d/);
    expect(PERTURBATION).not.toMatch(/\d/);
  });

  it('under the Scenario ID sort is the ID caption, word for word, and none of the other two captions’ sentences', async () => {
    await open('/?sort=id');
    expect(caption()).toBe(ID);
    expect(caption()).not.toMatch(/Ranked by fragility score|most fragile first|Equal scores are tie-broken|finite search|proven minimum/);
    expect(caption()).not.toMatch(/Sorted by the smallest perturbation|weighted norm|no perturbation found/);
  });

  it('follows the control through all three sorts and back', async () => {
    await open('/');
    for (const [value, expected] of [['id', ID], ['perturbation', PERTURBATION], ['id', ID], ['fragility', FRAGILITY]] as const) {
      await userEvent.selectOptions(screen.getByLabelText('Sort'), value);
      await waitFor(() => expect(caption()).toBe(expected));
    }
  });

  it('is the same under an outcome filter on the Scenario ID sort', async () => {
    await open('/?sort=id&outcome=collision_found');
    expect(caption()).toBe(ID);
  });
});

describe('what the perturbation caption says is what the rows do', () => {
  it('smallest first, rows with no perturbation after those that have one, equal values by scenario ID', async () => {
    await open('/?sort=perturbation');
    // 0.1 a, 0.1 b (tied: a before b by id, though the API lists b first), 0.4 c, 0.9 d, then the two with none: k, m by id
    expect(order()).toEqual(['a', 'b', 'c', 'd', 'k', 'm']);
  });

  it('under the Scenario ID sort the rows are in ascending text order, whatever their fragility', async () => {
    await open('/?sort=id');
    expect(order()).toEqual(['a', 'b', 'c', 'd', 'k', 'm']);
  });

  it('and compares as text: “10” comes before “9”, and an upper-case letter before a lower-case one', async () => {
    stubFetch((req: RecordedRequest): Reply => ({ status: 200, body: req.path === '/stats' ? STATS : makePage([hit('9', 0.1, 9), hit('a', 0.1, 8), hit('10', 0.1, 7), hit('B', 0.1, 6)], null) }));
    renderApp('/?sort=id');
    await screen.findByRole('link', { name: '9' });
    expect(order()).toEqual(['10', '9', 'B', 'a']);
  });

  it('and the fragility sort keeps the API’s order, the thing its caption says', async () => {
    await open('/');
    expect(order()).toEqual(['d', 'm', 'b', 'a', 'k', 'c']);
  });
});
