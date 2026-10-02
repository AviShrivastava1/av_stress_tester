import { Link, useLocation, useParams } from 'react-router';
import { isNotFound } from '../api/errors';
import { usePerturbed, useScenarioDetail, useTrajectories } from '../api/scenarios';
import { PerturbationPanel } from './detail/PerturbationPanel';
import { ScenePanel } from './detail/ScenePanel';
import { SummaryPanel } from './detail/SummaryPanel';
import { RequestError } from '../components/RequestError';
import type { ListReturnState } from './ScenarioListPage';

function backTarget(state: unknown): string {
  if (typeof state === 'object' && state !== null && 'listSearch' in state) {
    const search = (state as ListReturnState).listSearch;
    if (typeof search === 'string' && (search === '' || search.startsWith('?'))) return `/${search}`;
  }
  return '/';
}

export function ScenarioDetailPage() {
  const { scenarioId = '' } = useParams();
  const location = useLocation();

  // Three independent queries, all started on this render: none waits for another.
  const detail = useScenarioDetail(scenarioId);
  const trajectories = useTrajectories(scenarioId);
  const perturbed = usePerturbed(scenarioId);

  const back = (
    <Link to={backTarget(location.state)} className="back-link">
      ← All scenarios
    </Link>
  );

  if (detail.isError && isNotFound(detail.error)) {
    return (
      <section>
        {back}
        <h1 className="detail-title">Unknown scenario</h1>
        <p className="status" role="alert">
          No scenario with ID <span className="mono">{scenarioId}</span> exists.
        </p>
      </section>
    );
  }

  return (
    <section>
      {back}
      <div className="detail-heading"><div><p className="eyebrow">SCENARIO REPLAY</p><h1 className="detail-title mono">{scenarioId}</h1></div>
        <span className="dataset-label">Recorded scene + simulated challenger</span></div>
      <div className="detail-grid">
        <div className="detail-side">
          {detail.isPending ? (
            <section className="panel">
              <p className="status" role="status">Loading scenario…</p>
            </section>
          ) : detail.isError ? (
            <section className="panel">
              <RequestError error={detail.error} retry={() => detail.refetch()} retrying={detail.isFetching} />
            </section>
          ) : (
            <SummaryPanel detail={detail.data} targetIdx={perturbed.data?.target_idx ?? null} />
          )}
          <PerturbationPanel perturbed={perturbed} />
        </div>
        <ScenePanel key={scenarioId} trajectories={trajectories} perturbed={perturbed} detail={detail.data} />
      </div>
    </section>
  );
}
