import { describe, expect, it } from 'vitest';
import { makeRow } from '../test/fixtures';
import { OUTCOMES } from './outcomes';
import {
  applyView,
  DEFAULT_SORT,
  isViewActive,
  OUTCOME_FILTERS,
  parseOutcomeFilter,
  parseSort,
  rowOutcome,
  SORTS,
} from './listView';

describe('rowOutcome', () => {
  it.each(OUTCOMES.map((o) => [o]))('a stored %s is that outcome', (outcome) => {
    expect(rowOutcome(makeRow({ stress_outcome: outcome, stress_tested: true }))).toBe(outcome);
  });

  it('a stored result with no recorded outcome is "not_recorded", never inferred', () => {
    expect(rowOutcome(makeRow({ stress_outcome: null, stress_tested: true, min_perturbation: 0.1 }))).toBe('not_recorded');
  });

  it('an attempt with no stored result is "no_stored_result"', () => {
    expect(rowOutcome(makeRow({ stress_outcome: null, stress_tested: false, stress_attempted: true }))).toBe('no_stored_result');
  });

  it('a row no pass has reached is "not_tested"', () => {
    expect(rowOutcome(makeRow())).toBe('not_tested');
  });

  it('an outcome the interface does not know is "unrecognized", not the nearest known one', () => {
    expect(rowOutcome(makeRow({ stress_outcome: 'collision_foundd', stress_tested: true }))).toBe('unrecognized');
  });

  it('the last attempt does not change the filter key', () => {
    expect(rowOutcome(makeRow({ stress_outcome: 'collision_found', stress_tested: true, last_attempt_outcome: 'replay_infeasible' }))).toBe(
      'collision_found',
    );
  });
});

describe('the query values', () => {
  it('every filter option parses to itself', () => {
    for (const { value } of OUTCOME_FILTERS) expect(parseOutcomeFilter(value)).toBe(value);
  });

  it('offers each of the six outcomes and the four other states, once', () => {
    expect(OUTCOME_FILTERS.map((f) => f.value)).toEqual([...OUTCOMES, 'not_recorded', 'no_stored_result', 'not_tested', 'unrecognized']);
  });

  it.each([null, '', 'bogus', 'COLLISION_FOUND', ' collision_found', 'collision_found,error'])('%j is not a filter', (raw) => {
    expect(parseOutcomeFilter(raw)).toBeNull();
  });

  it('every sort parses to itself, and anything else is the default', () => {
    for (const { value } of SORTS) expect(parseSort(value)).toBe(value);
    for (const raw of [null, '', 'bogus', 'ID', 'perturbation,id']) expect(parseSort(raw)).toBe(DEFAULT_SORT);
    expect(DEFAULT_SORT).toBe('fragility');
  });

  it('a view is active when it differs from the API order, and only then', () => {
    expect(isViewActive({ outcome: null, sort: 'fragility' })).toBe(false);
    expect(isViewActive({ outcome: 'error', sort: 'fragility' })).toBe(true);
    expect(isViewActive({ outcome: null, sort: 'id' })).toBe(true);
    expect(isViewActive({ outcome: null, sort: 'perturbation' })).toBe(true);
  });
});

const ROWS = [
  makeRow({ scenario_id: 'e', min_perturbation: 0.5, stress_outcome: 'collision_found', stress_tested: true }),
  makeRow({ scenario_id: 'b', min_perturbation: null, stress_outcome: 'replay_infeasible', stress_tested: true }),
  makeRow({ scenario_id: 'c', min_perturbation: 0.1, stress_outcome: 'collision_found', stress_tested: true }),
  makeRow({ scenario_id: 'a' }),
  makeRow({ scenario_id: 'd', min_perturbation: 0.1, stress_outcome: 'collision_found', stress_tested: true }),
  makeRow({ scenario_id: 'f', min_perturbation: 0, stress_outcome: 'collision_found', stress_tested: true }),
];
const ids = (ranked: { row: { scenario_id: string } }[]) => ranked.map((r) => r.row.scenario_id);

describe('applyView', () => {
  it('the default view is the API order, untouched, with ranks counting from 1', () => {
    const out = applyView(ROWS, { outcome: null, sort: 'fragility' });
    expect(ids(out)).toEqual(['e', 'b', 'c', 'a', 'd', 'f']);
    expect(out.map((r) => r.rank)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(out.map((r) => r.row)).toEqual(ROWS);
  });

  it('an outcome filter keeps matching rows, in API order, with their place in the API order', () => {
    const out = applyView(ROWS, { outcome: 'collision_found', sort: 'fragility' });
    expect(ids(out)).toEqual(['e', 'c', 'd', 'f']);
    expect(out.map((r) => r.rank)).toEqual([1, 3, 5, 6]);
    expect(ids(applyView(ROWS, { outcome: 'not_tested', sort: 'fragility' }))).toEqual(['a']);
    expect(applyView(ROWS, { outcome: 'error', sort: 'fragility' })).toEqual([]);
  });

  it('sorts by min perturbation, smallest first, a missing one last, ties by scenario ID', () => {
    const out = applyView(ROWS, { outcome: null, sort: 'perturbation' });
    // 0 is a stored value and sorts first; the two 0.1 rows tie and go by ID; the rows with none are last, by ID.
    expect(ids(out)).toEqual(['f', 'c', 'd', 'e', 'a', 'b']);
    expect(out.map((r) => r.rank)).toEqual([6, 3, 5, 1, 4, 2]);
  });

  it('sorts by scenario ID by code point', () => {
    const rows = [makeRow({ scenario_id: 'b' }), makeRow({ scenario_id: 'B' }), makeRow({ scenario_id: 'a' }), makeRow({ scenario_id: '10' }), makeRow({ scenario_id: '9' })];
    expect(ids(applyView(rows, { outcome: null, sort: 'id' }))).toEqual(['10', '9', 'B', 'a', 'b']);
  });

  it('filters and sorts together, and does not change its input', () => {
    const before = ROWS.map((r) => r.scenario_id);
    const out = applyView(ROWS, { outcome: 'collision_found', sort: 'perturbation' });
    expect(ids(out)).toEqual(['f', 'c', 'd', 'e']);
    expect(ROWS.map((r) => r.scenario_id)).toEqual(before);
  });

  it('treats null and undefined as the same kind of missing: they tie, and go by scenario ID', () => {
    const undef = makeRow({ scenario_id: 'x', min_perturbation: undefined });
    const none = makeRow({ scenario_id: 'a', min_perturbation: null });
    for (const rows of [[undef, none], [none, undef]]) {
      expect(ids(applyView(rows, { outcome: null, sort: 'perturbation' }))).toEqual(['a', 'x']);
    }
  });

  it('treats an undefined min_perturbation as missing, whichever order the rows arrive in', () => {
    const missing = makeRow({ scenario_id: 'x', min_perturbation: undefined });
    const stored = makeRow({ scenario_id: 'y', min_perturbation: 2 });
    for (const rows of [[missing, stored], [stored, missing]]) {
      expect(ids(applyView(rows, { outcome: null, sort: 'perturbation' }))).toEqual(['y', 'x']);
    }
  });
});
