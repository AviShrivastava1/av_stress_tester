/**
 * The stress-test outcome vocabulary.
 *
 * Hand-written, and deliberately the only hand-written piece of the API contract:
 * the OpenAPI schema types `stress_outcome` and `last_attempt_outcome` as plain
 * `string | null`, so generated types cannot carry the vocabulary. Making them a
 * `Literal` server-side is a separate backend change — until then, a value outside
 * this list must render as visibly unrecognized, never crash and never be coerced
 * into the nearest known value. The backend's `resolve_outcome` passes any truthy
 * outcome string through unchanged and the column has no CHECK constraint, so an
 * unexpected value is possible, not hypothetical.
 *
 * Mirrors the OUTCOME_* constants in src/scoring/db.py.
 */
export const OUTCOMES = [
  'collision_found',
  'no_collision_found',
  'replay_infeasible',
  'heading_blend_singularity',
  'no_challenger',
  'error',
] as const;

export type Outcome = (typeof OUTCOMES)[number];

export function parseOutcome(raw: string): Outcome | null {
  return (OUTCOMES as readonly string[]).includes(raw) ? (raw as Outcome) : null;
}

export type Tone = 'danger' | 'neutral' | 'warning' | 'muted';

/**
 * The server's `robustly_safe`, worded once for every place that shows it. It is
 * read from the server's boolean or count, never inferred from other fields.
 */
export const ROBUSTLY_SAFE = {
  label: 'robustly safe',
  meaning: 'The server reports a search that certifies no collision is reachable within its bounds.',
} as const;

export interface OutcomeDescription {
  label: string;
  tone: Tone;
  meaning: string;
}

function assertNever(value: never): never {
  throw new Error(`unhandled outcome: ${String(value)}`);
}

/**
 * Exhaustive over OUTCOMES: adding a seventh value to the tuple without a case here
 * is a compile error, not a blank badge.
 */
export function describeOutcome(outcome: Outcome): OutcomeDescription {
  switch (outcome) {
    case 'collision_found':
      return {
        label: 'collision found',
        tone: 'danger',
        // "oriented-box intersection": src/danger/collision_detector.py get_corners / check_collision_trajectory.
        // "stored coordinate precision": the corners are float64 (COLLISION_GEOMETRY_VERSION
        // 'oriented-box-float64-v1') and the exported path is written with repr(float(...)), which round-trips
        // binary64 exactly (src/scoring/export_geometry.py), so the check runs on what is stored.
        meaning: 'A perturbation produced an oriented-box intersection at the stored coordinate precision.',
      };
    case 'no_collision_found':
      return {
        label: 'no collision found',
        tone: 'neutral',
        meaning:
          'The search completed and found no collision within its budget and bounds. ' +
          'This is a statement about the search, not a safety certificate.',
      };
    case 'replay_infeasible':
      return {
        label: 'replay infeasible',
        tone: 'warning',
        meaning:
          'No search ran: the zero-perturbation replay was not faithful enough to ' +
          'measure a perturbation against. Not the same as coming back clean.',
      };
    case 'heading_blend_singularity':
      return {
        label: 'heading-blend singularity',
        tone: 'warning',
        meaning:
          'No search ran: the challenger heading sat near a degenerate point where a ' +
          'tiny perturbation could flip it arbitrarily. The replay itself was faithful.',
      };
    case 'no_challenger':
      return {
        label: 'no challenger',
        tone: 'muted',
        meaning: 'No other agent to perturb.',
      };
    case 'error':
      return {
        label: 'error',
        tone: 'warning',
        meaning: 'The stress-test pass raised an error for this scenario.',
      };
    default:
      return assertNever(outcome);
  }
}
