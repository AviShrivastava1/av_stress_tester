import type { ScenarioDetail } from '../../api/client';
import { TimeMetricCell } from '../../components/TimeMetricCell';
import { ReplayOffsetBlock } from './ReplayOffsetBlock';

function Dash() {
  return <span className="muted">—</span>;
}

export function SummaryPanel({ detail, targetIdx }: { detail: ScenarioDetail; targetIdx: number | null }) {
  return (
    <section className="panel">
      <p className="panel-kicker">RESULT CONTEXT</p>
      <h2>Run details</h2>
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
        <ReplayOffsetBlock detail={detail} />
        <dt>Challenger</dt>
        <dd>{targetIdx !== null ? `agent ${targetIdx}` : <Dash />}</dd>
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
