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

/*
 * What a PET reading of zero or less means, and so when the "overlap" tag is shown (components/TimeMetricCell.tsx).
 *
 * Where the number comes from: src/danger/pet_engine.py compute_pet takes, for each visit of one agent to a connected
 * piece of the conflict zone and each visit of the other, max(enter_b - exit_a, enter_a - exit_b) (line 238), takes the
 * smallest over the visit pairs, and multiplies by DT, the 0.1 s between frames (line 237 onward; `return float(pet)` is
 * line 244). Frame numbers are integers, so a visit is a closed run of sampled frames, and the value is exactly 0.0 when
 * one agent's last frame in the zone is the other agent's first frame in it.
 *
 * What the engine says that means: "Per pair, `max(...) <= 0` iff `enter_Bj <= exit_Ai` AND `enter_Ai <= exit_Bj`, which
 * is exactly the condition for those two closed intervals to overlap. So a negative per-pair value means the two visits
 * share at least one frame" (lines 181-184); the Returns paragraph words it as "both agents' boxes intersected the same
 * connected component of the conflict zone during a common frame" (lines 210-212). The engine's docstring says
 * "negative", but its own condition is `<= 0`: the closed intervals also share a frame when the value is exactly zero. So
 * zero is the same fact as a negative value: both agents were in the zone on one common sampled frame. It is not a
 * claim that their boxes touched each other, which the engine says in the same place.
 *
 * compute_min_pet_sdc (line 280) takes the smallest such value over the pairs that involve the SDC, which is the
 * min_pet the API serves; it can therefore be exactly 0.0.
 *
 * The title sentence, clause by clause:
 *   "PET of zero or less"                the condition `max(...) <= 0` above
 *   "the SDC and another agent"          min_pet is the SDC-restricted minimum (compute_min_pet_sdc)
 *   "were both in a conflict zone at the same time"   both visits share a frame (lines 181-184, 210-212)
 *   "on at least one shared sampled frame"            the visits are closed runs of sampled frames; zero is one frame
 */
export const PET_OVERLAP_TITLE =
  'PET of zero or less: the SDC and another agent were both in a conflict zone at the same time, on at least one shared sampled frame.';

/** True when a measured PET marks simultaneous occupancy: zero (also negative zero) or negative. NaN is not. */
export function isPetOverlap(seconds: number): boolean {
  return seconds <= 0;
}
