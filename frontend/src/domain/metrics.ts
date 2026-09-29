/**
 * `min_ttc` / `min_pet` carry 999.0 as a sentinel: the SDC is never in a closing
 * pair (TTC) or never shares a conflict zone (PET). It is not a measurement — it
 * must never be formatted as "999.00 s", plotted on a real axis, or averaged in.
 *
 * Exact equality is intentional: the backend writes the literal 999.0, and a real
 * measurement that happened to land near it would still be a measurement.
 */
export const NO_INTERACTION_SENTINEL = 999.0;

export type TimeMetric =
  | { kind: 'missing' }
  | { kind: 'no_interaction' }
  | { kind: 'measured'; seconds: number };

export function readTimeMetric(value: number | null | undefined): TimeMetric {
  if (value === null || value === undefined) return { kind: 'missing' };
  if (value === NO_INTERACTION_SENTINEL) return { kind: 'no_interaction' };
  return { kind: 'measured', seconds: value };
}
