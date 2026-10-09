import { Link, useLocation, useParams } from 'react-router';
import type { ScenarioDetail } from '../api/client';
import { isNotFound } from '../api/errors';
import { usePerturbed, useScenarioDetail, useTrajectories } from '../api/scenarios';
import { PerturbationPanel } from './detail/PerturbationPanel';
import { ScenePanel } from './detail/ScenePanel';
import { SummaryPanel } from './detail/SummaryPanel';
import { OutcomeCell } from '../components/OutcomeBadge';
import { RequestError } from '../components/RequestError';
import { formatMagnitude } from '../domain/format';
import { perturbedTrackDrawn } from '../scene/sceneModel';
import type { ListReturnState } from './ScenarioListPage';

function Dash() {
  return <span className="muted">—</span>;
}

/**
 * What the stored search found, above the replay it describes: the outcome, the smallest
 * perturbation, the collision frame and how many challengers were searched. The scenario's own
 * numbers (fragility, TTC, PET, replay offset) stay in the Run details panel.
 */
function ResultOverview({ detail }: { detail: ScenarioDetail }) {
  const searched =
    detail.challengers_searched !== null && detail.challengers_total !== null
      ? `${detail.challengers_searched} of ${detail.challengers_total}`
      : null;
  return (
    <section className="result-overview" aria-label="Search result">
      <div className="result-primary">
        <span>Search outcome</span>
        <OutcomeCell row={detail} />
      </div>
      <div>
        <span>Min perturbation</span>
        <strong>
          {detail.min_perturbation != null ? (
            <span title={String(detail.min_perturbation)}>{formatMagnitude(detail.min_perturbation)}</span>
          ) : (
            <Dash />
          )}
        </strong>
      </div>
      <div>
        <span>Collision frame</span>
        <strong>{detail.collision_timestep != null ? `frame ${detail.collision_timestep}` : <Dash />}</strong>
      </div>
      <div>
        <span title="The search perturbs one heuristically chosen challenger, not every agent.">
          Challengers searched
        </span>
        <strong>{searched ?? <Dash />}</strong>
      </div>
    </section>
  );
}

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

  // The comparison is only claimed when a perturbed track is on screen.
  const comparing = perturbedTrackDrawn(trajectories.data?.agents, perturbed.data);

  return (
    <section>
      {back}
      <div className="detail-heading">
        <div>
          <p className="eyebrow">SCENARIO REPLAY</p>
          <h1 className="detail-title mono">{scenarioId}</h1>
          {comparing && (
            <p className="detail-intro">
              Compare the recorded motion with the smallest collision-producing replay found by the search.
            </p>
          )}
        </div>
        <span className="dataset-label">
          {comparing ? 'Recorded scene · simulated challenger' : 'Recorded scene'}
        </span>
      </div>
      {detail.data && <ResultOverview detail={detail.data} />}
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
