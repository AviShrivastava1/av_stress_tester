import type { ScenarioSummary } from '../api/client';
import { describeOutcome, OUTCOMES, parseOutcome } from './outcomes';

/*
 * The view of the list that the URL carries: an outcome filter and a sort, applied in the browser to rows the API has
 * already returned. The API filters on one thing only (`stress_tested_only`) and orders by fragility then scenario ID
 * (src/api/routes.py list_scenarios), so everything here runs after every page has been loaded; the list page loads the
 * rest before it applies a view and says so while it does (pages/ScenarioListPage.tsx).
 *
 * What backs each part:
 *
 *   the outcome of a row
 *       the same rule the Search result column uses (components/OutcomeBadge.tsx OutcomeCell, its primary badge): the stored
 *       `stress_outcome` when there is one; otherwise "outcome not recorded" for a row with a stored result,
 *       "no stored result" for one that was attempted, "not tested" for one no pass has reached. A `stress_outcome` outside
 *       the six values the interface knows (domain/outcomes.ts OUTCOMES) is "unrecognized", as the badge shows it. The
 *       separate `last_attempt_outcome` is not a filter key
 *   the labels
 *       the badge labels themselves (describeOutcome, and the three fixed strings in OutcomeCell), so a filter option and the
 *       badge it selects read the same
 *   the sorts
 *       "fragility" is the order the API returned, kept as it came (the ranking); "perturbation" orders by
 *       `min_perturbation`, smallest first, a row without a stored perturbation last; "id" orders by scenario ID by
 *       code point. Ties go by scenario ID. Nothing is sorted by a value the page computes
 *   the rank
 *       each row keeps its place in the order the API returned (1 is the first row), so a filtered or re-sorted table still
 *       shows where each scenario sits in the fragility ranking
 */

export type OutcomeFilter =
  | (typeof OUTCOMES)[number]
  | 'not_recorded'
  | 'no_stored_result'
  | 'not_tested'
  | 'unrecognized';

export const OUTCOME_FILTERS: readonly { value: OutcomeFilter; label: string }[] = [
  ...OUTCOMES.map((value) => ({ value, label: describeOutcome(value).label })),
  { value: 'not_recorded', label: 'outcome not recorded' },
  { value: 'no_stored_result', label: 'no stored result' },
  { value: 'not_tested', label: 'not tested' },
  { value: 'unrecognized', label: 'unrecognized' },
];

export type SortKey = 'fragility' | 'perturbation' | 'id';
export const DEFAULT_SORT: SortKey = 'fragility';

export const SORTS: readonly { value: SortKey; label: string; short: string }[] = [
  { value: 'fragility', label: 'Fragility, highest first', short: 'fragility, highest first' },
  { value: 'perturbation', label: 'Min perturbation, smallest first', short: 'min perturbation, smallest first' },
  { value: 'id', label: 'Scenario ID', short: 'scenario ID' },
];

export interface ListView {
  outcome: OutcomeFilter | null;
  sort: SortKey;
}

/** A filter or sort that is not the API's own order, so that rows must all be loaded first. */
export function isViewActive(view: ListView): boolean {
  return view.outcome !== null || view.sort !== DEFAULT_SORT;
}

export function rowOutcome(
  row: Pick<ScenarioSummary, 'stress_outcome' | 'stress_tested' | 'stress_attempted'>,
): OutcomeFilter {
  if (row.stress_outcome !== null) return parseOutcome(row.stress_outcome) ?? 'unrecognized';
  if (row.stress_tested) return 'not_recorded';
  if (row.stress_attempted) return 'no_stored_result';
  return 'not_tested';
}

/** The filter named by a query value, or null for anything that is not one (an unknown value is ignored). */
export function parseOutcomeFilter(raw: string | null): OutcomeFilter | null {
  return OUTCOME_FILTERS.find((f) => f.value === raw)?.value ?? null;
}

/** The sort named by a query value, or the default for anything that is not one. */
export function parseSort(raw: string | null): SortKey {
  return SORTS.find((s) => s.value === raw)?.value ?? DEFAULT_SORT;
}

export interface RankedRow {
  row: ScenarioSummary;
  /** The row's place in the order the API returned, counting from 1. */
  rank: number;
}

function compareIds(a: ScenarioSummary, b: ScenarioSummary): number {
  return a.scenario_id < b.scenario_id ? -1 : a.scenario_id > b.scenario_id ? 1 : 0;
}

function comparePerturbation(a: ScenarioSummary, b: ScenarioSummary): number {
  const x = a.min_perturbation;
  const y = b.min_perturbation;
  const hasX = x !== null && x !== undefined;
  const hasY = y !== null && y !== undefined;
  if (hasX && hasY && x !== y) return x < y ? -1 : 1;
  if (hasX !== hasY) return hasX ? -1 : 1;
  return compareIds(a, b);
}

/** The rows that match the view, in the view's order, each with its place in the API's order. */
export function applyView(rows: readonly ScenarioSummary[], view: ListView): RankedRow[] {
  const ranked = rows.map((row, i) => ({ row, rank: i + 1 }));
  const kept = view.outcome === null ? ranked : ranked.filter((r) => rowOutcome(r.row) === view.outcome);
  if (view.sort === 'fragility') return kept;
  const compare = view.sort === 'perturbation' ? comparePerturbation : compareIds;
  return [...kept].sort((a, b) => compare(a.row, b.row));
}
