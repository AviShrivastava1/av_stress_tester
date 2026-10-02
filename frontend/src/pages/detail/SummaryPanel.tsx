import type { ScenarioDetail } from '../../api/client';
import { OutcomeCell } from '../../components/OutcomeBadge';
import { TimeMetricCell } from '../../components/TimeMetricCell';
import { formatMagnitude } from '../../domain/format';

function Dash() {
  return <span className="muted">—</span>;
}

export function SummaryPanel({ detail, targetIdx }: { detail: ScenarioDetail; targetIdx: number | null }) {
  const searched =
    detail.challengers_searched !== null && detail.challengers_total !== null
      ? `${detail.challengers_searched} of ${detail.challengers_total}`
      : null;

  return (
    <section className="panel">
      <h2>Summary</h2>
      <dl className="facts">
        <dt>Fragility</dt>
        <dd>{detail.fragility_score.toFixed(3)}</dd>
        <dt>Min TTC</dt>
        <dd>
          <TimeMetricCell value={detail.min_ttc ?? null} metric="ttc" />
        </dd>
        <dt>Min PET</dt>
        <dd>
          <TimeMetricCell value={detail.min_pet ?? null} metric="pet" />
        </dd>
        <dt>Outcome</dt>
        <dd>
          <OutcomeCell row={detail} />
        </dd>
        <dt>Min perturbation</dt>
        <dd>{detail.min_perturbation != null ? <span title={String(detail.min_perturbation)}>{formatMagnitude(detail.min_perturbation)}</span> : <Dash />}</dd>
        <dt>Collision frame</dt>
        <dd>{detail.collision_timestep != null ? `frame ${detail.collision_timestep}` : <Dash />}</dd>
        <dt>Challenger</dt>
        <dd>{targetIdx !== null ? `agent ${targetIdx}` : <Dash />}</dd>
        <dt title="The search perturbs one heuristically chosen challenger, not every agent.">
          Challengers searched
        </dt>
        <dd>{searched ?? <Dash />}</dd>
        <dt>Method</dt>
        <dd>{detail.stress_method ?? <Dash />}</dd>
        <dt>Agents</dt>
        <dd>{detail.n_agents ?? <Dash />}</dd>
        <dt>Shard</dt>
        <dd className="mono">{detail.shard ?? <Dash />}</dd>
      </dl>
    </section>
  );
}
