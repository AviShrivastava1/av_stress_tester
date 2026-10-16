import { Link, useLocation, useParams } from 'react-router';
import type { ScenarioDetail } from '../api/client';
import { isNotFound } from '../api/errors';
import { usePerturbed, useScenarioDetail, useTrajectories } from '../api/scenarios';
import { BriefButton } from './detail/BriefButton';
import { buildBrief } from './detail/brief';
import { PerturbationPanel } from './detail/PerturbationPanel';
import { OtherSearches } from './detail/OtherSearches';
import { ProvenancePanel } from './detail/ProvenancePanel';
import { RequestSearchPanel } from './detail/RequestSearchPanel';
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
 * What the stored search found, under the replay it describes: the smallest perturbation, the
 * collision frame and how many challengers were searched. The outcome itself is in the page header,
 * and the scenario's own numbers (fragility, TTC, PET, replay offset) stay in the Run details panel.
 */
function ResultOverview({ detail }: { detail: ScenarioDetail }) {
  const searched =
    detail.challengers_searched !== null && detail.challengers_total !== null
      ? `${detail.challengers_searched} of ${detail.challengers_total}`
      : null;
  return (
    <section className="result-overview" aria-label="Search result">
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

/*
 * The words in the detail header. What backs each one:
 *
 *   "logged / counterfactual"
 *       shown only when perturbedTrackDrawn (scene/sceneModel.ts) is true: the logged track (the recorded Waymo
 *       motion, GET /scenarios/{id}/trajectories) is drawn beside the track simulated from the stored perturbation
 *       (perturbed_paths, GET /scenarios/{id}/perturbed; the replay is src/physics/simulator.py ForwardSimulator).
 *       "counterfactual" names that simulated run; it is not a prediction
 *   "logged scene"
 *       no perturbed track is drawn, so only the recording is
 *   the outcome group under it (label "Search outcome", which the result overview used for the same cell)
 *       OutcomeCell (components/OutcomeBadge.tsx): the stored stress_outcome and the last attempt, both from
 *       scenario_scores via GET /scenarios/{id}. It sits in the header so that a refusal or a missing result is on
 *       the first screen, above the replay it qualifies
 */
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
        <div className="detail-heading-id">
          <p className="eyebrow">SCENARIO REPLAY</p>
          <h1 className="detail-title mono">{scenarioId}</h1>
        </div>
        <div className="detail-heading-meta">
          <span className="dataset-label">{comparing ? 'logged / counterfactual' : 'logged scene'}</span>
          {detail.data && (
            <div role="group" aria-label="Search outcome">
              <OutcomeCell row={detail.data} />
            </div>
          )}
        </div>
      </div>
      <ScenePanel key={scenarioId} trajectories={trajectories} perturbed={perturbed} detail={detail.data} />
      {detail.data && <ResultOverview detail={detail.data} />}
      {detail.data && (
        <BriefButton
          build={() =>
            buildBrief({
              origin: window.location.origin,
              scenarioId,
              detail: detail.data!,
              perturbed: perturbed.data,
              agents: trajectories.data?.agents,
            })
          }
        >
          <Link className="text-link" to={`/compare?a=${encodeURIComponent(scenarioId)}`}>
            Compare with another scenario
          </Link>
        </BriefButton>
      )}
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
      <ProvenancePanel scenarioId={scenarioId} labels={perturbed.data?.delta_labels} />
      <OtherSearches scenarioId={scenarioId} />
      <RequestSearchPanel scenarioId={scenarioId} />
    </section>
  );
}
