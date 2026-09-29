import { describeOutcome, parseOutcome, type Tone } from '../domain/outcomes';

function Badge({ tone, label, title }: { tone: Tone; label: string; title: string }) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {label}
    </span>
  );
}

/** One outcome string from the API. Anything outside the known six is shown raw. */
export function OutcomeBadge({ outcome }: { outcome: string }) {
  const known = parseOutcome(outcome);
  if (known === null) {
    return (
      <Badge
        tone="warning"
        label={`unrecognized: ${outcome}`}
        title="The API returned an outcome this dashboard does not know. Shown verbatim rather than guessed."
      />
    );
  }
  const { label, tone, meaning } = describeOutcome(known);
  return <Badge tone={tone} label={label} title={meaning} />;
}

export interface OutcomeFields {
  stress_outcome: string | null;
  last_attempt_outcome: string | null;
  stress_tested: boolean;
  stress_attempted: boolean;
  robustly_safe: boolean;
}

/**
 * The stored result, plus the most recent attempt when it concluded differently.
 *
 * `stress_outcome` describes the run that produced the stored result;
 * `last_attempt_outcome` describes the latest pass. They differ exactly when a later
 * pass failed to reproduce an earlier result — which the stored result survives —
 * so both are shown rather than one silently winning (see src/api/models.py).
 *
 * A NULL `stress_outcome` is never back-inferred from other columns: a row with a
 * stored result but no recorded outcome predates the column, and says so.
 */
export function OutcomeCell({ row }: { row: OutcomeFields }) {
  let primary;
  if (row.stress_outcome !== null) {
    primary = <OutcomeBadge outcome={row.stress_outcome} />;
  } else if (row.stress_tested) {
    primary = (
      <Badge
        tone="muted"
        label="outcome not recorded"
        title="A result is stored, but it was written before outcomes were recorded. Not inferred."
      />
    );
  } else if (row.stress_attempted) {
    primary = (
      <Badge tone="muted" label="no stored result" title="Attempted, but no result was stored." />
    );
  } else {
    primary = <Badge tone="muted" label="not tested" title="No stress-test pass has reached this scenario." />;
  }

  const last = row.last_attempt_outcome;
  const diverged = last !== null && last !== row.stress_outcome;

  return (
    <span className="outcome-cell">
      {primary}
      {diverged && (
        <span className="last-attempt">
          last attempt: <OutcomeBadge outcome={last} />
        </span>
      )}
      {/* Straight from the server's boolean — never derived from a NULL min_perturbation. */}
      {row.robustly_safe && (
        <Badge
          tone="neutral"
          label="robustly safe"
          title="The server reports a search that certifies no collision is reachable within its bounds."
        />
      )}
    </span>
  );
}
