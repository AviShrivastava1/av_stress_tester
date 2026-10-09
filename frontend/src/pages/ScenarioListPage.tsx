import { useState, type FormEvent } from 'react';
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router';
import type { ScenarioSummary } from '../api/client';
import { describeError } from '../api/errors';
import { useScenarioPages } from '../api/scenarios';
import { OutcomeCell } from '../components/OutcomeBadge';
import { TimeMetricCell } from '../components/TimeMetricCell';
import { RequestError } from '../components/RequestError';
import { formatMagnitude } from '../domain/format';

// Mirrors the server's bounds on `limit` (Query(ge=1, le=max_page_size)).
const MIN_LIMIT = 1;
const MAX_LIMIT = 200;

/** `?limit=` if it is a whole number the server will accept; otherwise the server default. */
function parseLimit(raw: string | null): number | undefined {
  if (raw === null || !/^\d+$/.test(raw)) return undefined;
  const n = Number(raw);
  return n >= MIN_LIMIT && n <= MAX_LIMIT ? n : undefined;
}

/** Carried to the detail page so its back link restores this list's filters. */
export interface ListReturnState {
  listSearch: string;
}

/*
 * The words at the top of the list. Every sentence describes code that exists in this repository, and
 * none claims the site computes anything live or certifies anything. What backs each one:
 *
 *   eyebrow "AUTONOMOUS-VEHICLE SCENARIO STRESS-TESTING"
 *       the scenes are Waymo Open Motion Dataset scenarios (src/data/parser.py ScenarioParser); "stress-testing"
 *       is src/scoring/batch_scorer.py stress_test_scenarios
 *   h1 "How small a change turns a recorded driving scene into a collision?"
 *       the question scenario_scores.min_perturbation answers: the smallest perturbation found by
 *       src/optimization/scipy_optimizer.py optimize_scenario that makes check_collision_trajectory
 *       (src/danger/collision_detector.py) report contact
 *   lede "...smallest model-constrained change that produces geometric contact, then lets you inspect each stored
 *         result frame by frame"
 *       model-constrained: the bicycle and linear replay models (src/physics/bicycle_model.py, linear_model.py)
 *       via src/optimization/perturbation_space.py PerturbationSpace; geometric contact: collision_detector.py;
 *       stored result, frame by frame: scenario_scores / perturbed_paths, replayed by scene/SceneView.tsx
 *       and scene/PlaybackControls.tsx
 *   "Explore the results" / "How the system works": the explorer below, and the /method route (pages/MethodPage.tsx)
 *   technology stack: Python (.python-version), SciPy (scipy_optimizer.py, differential_evolution), PyTorch
 *       (src/optimization/autograd_optimizer.py `import torch`, requirements.txt; the refinement is optional), PostGIS
 *       (src/scoring/export_geometry.py), FastAPI (src/api/main.py), React (frontend/package.json)
 *   pipeline
 *       Rank          src/danger/ttc_engine.py (SDC-restricted, db.py documents it), src/danger/pet_engine.py
 *       Perturb       PerturbationSpace: four dimensions per model (perturbation_space.py), search in scipy_optimizer.py
 *       Verify        perturbation_space.ReplayFidelityError (the replay gates) and collision_detector.get_corners
 *                     (oriented boxes)
 *       Publish       src/scoring/export_geometry.py export_shard_geometry -> PostGIS -> src/api/routes.py -> this app
 *       "offline search · stored results"  the API has no endpoint that runs the optimizer (src/api/__init__.py)
 *       "a stored result you can replay, with the settings that produced it"  scenario_scores.search_provenance
 *                     (src/scoring/db.py) and the replay in scene/SceneView.tsx
 *   highlight cards
 *       Kinematic replay      src/physics/bicycle_model.py and linear_model.py; the fidelity refusals are
 *                             ReplayFidelityError and the status replay_infeasible (batch_scorer._stress_one)
 *       Global search, optional refinement
 *                             scipy_optimizer.optimize_scenario (Differential Evolution); autograd_optimizer.refine_scenario
 *                             runs only when _stress_one(use_autograd=True); collision_detector.get_corners promotes to float64
 *       Provenance stored with each result
 *                             db.compute_scene_fingerprint, db.compute_stress_run_id, the search_provenance column;
 *                             the read-only role av_api_ro (DEPLOY.md step 2) and an API with no write route (src/api/routes.py)
 *   explorer eyebrow "STORED RESULTS" and its introduction
 *       rows read from scenario_scores by GET /scenarios (routes.list_scenarios); a collision's replay compares
 *       the logged track with the perturbed one from perturbed_paths (export_perturbed_path); "the smallest ...
 *       it found" is min_perturbation
 *   metric guide
 *       fragility: src/danger/danger_score.py; "not a collision probability": README.md; "weighted": the weighted
 *       norm in optimize_scenario (and the note in pages/detail/ReplayOffsetBlock.tsx); "finite search": the
 *       Differential Evolution budget recorded in search_provenance (de_popsize, de_maxiter); TTC and PET:
 *       ttc_engine.py, pet_engine.py; "None": no finite value (components/TimeMetricCell.tsx); negative PET:
 *       pet_engine._conflict_components (two agents inside the same connected conflict zone at overlapping times)
 */
const HERO_STACK = ['Python', 'SciPy', 'PyTorch', 'PostGIS', 'FastAPI', 'React'] as const;
const HERO_PIPELINE = [
  ['Rank', 'SDC-restricted TTC + PET'],
  ['Perturb', 'Bounded 4-D kinematic search'],
  ['Verify', 'Replay gates + oriented boxes'],
  ['Publish', 'PostGIS → API → browser replay'],
] as const;
const HERO_CARDS = [
  ['Kinematic replay', 'Bicycle and linear motion models, with explicit fidelity refusals.'],
  ['Global search, optional refinement', 'Differential Evolution, optional gradient refinement, float64 oriented-box verification.'],
  ['Provenance stored with each result', 'Scene identity, run provenance, read-only production API.'],
] as const;

export function scenarioPath(scenarioId: string): string {
  return `/scenarios/${encodeURIComponent(scenarioId)}`;
}

function PerturbationCell({ row }: { row: ScenarioSummary }) {
  if (row.min_perturbation !== null && row.min_perturbation !== undefined) {
    return <span title={String(row.min_perturbation)}>{formatMagnitude(row.min_perturbation)}</span>;
  }
  // NULL is never read as "safe" — the outcome column says what actually happened.
  if (!row.stress_tested && !row.stress_attempted) {
    return <span className="muted">not tested</span>;
  }
  return (
    <span className="muted" title="No stored perturbation. See the outcome column.">
      —
    </span>
  );
}

export function ScenarioListPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [scenarioId, setScenarioId] = useState('');
  const stressTestedOnly = searchParams.get('tested') === '1';
  const limit = parseLimit(searchParams.get('limit'));

  const query = useScenarioPages({ stressTestedOnly, limit });

  function setStressTestedOnly(next: boolean) {
    setSearchParams(
      (prev) => {
        const params = new URLSearchParams(prev);
        if (next) params.set('tested', '1');
        else params.delete('tested');
        return params;
      },
      { replace: true },
    );
  }

  // Concatenated in the order the pages arrived. No sort, no dedupe, no filter.
  const rows = query.data?.pages.flatMap((page) => page.items) ?? [];

  function openScenario(event: FormEvent) {
    event.preventDefault();
    if (scenarioId.trim()) void navigate(scenarioPath(scenarioId.trim()), {
      state: { listSearch: location.search } satisfies ListReturnState,
    });
  }

  return (
    <>
      <section className="project-hero">
        <div className="hero-copy">
          <p className="eyebrow hero-eyebrow">AUTONOMOUS-VEHICLE SCENARIO STRESS-TESTING</p>
          <h1>How small a change turns a recorded driving scene into a collision?</h1>
          <p className="hero-lede">
            This project searches recorded driving scenes for the smallest model-constrained change that produces
            geometric contact, then lets you inspect each stored result frame by frame.
          </p>
          <div className="hero-actions">
            <a className="button-link button-primary" href="#scenario-explorer">Explore the results</a>
            <Link className="text-link" to="/method">How the system works <span aria-hidden="true">→</span></Link>
          </div>
          <ul className="stack-list" role="list" aria-label="Technology stack">
            {HERO_STACK.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>

        <div className="hero-system" role="group" aria-label="Analysis pipeline">
          <div className="hero-system-head">
            <span>Analysis pipeline</span>
            <span className="system-status">offline search · stored results</span>
          </div>
          <ol>
            {HERO_PIPELINE.map(([name, what], i) => (
              <li key={name}><span>{String(i + 1).padStart(2, '0')}</span><div><strong>{name}</strong><small>{what}</small></div></li>
            ))}
          </ol>
          <p className="hero-system-result"><span aria-hidden="true">↳</span> A stored result you can replay, with the settings that produced it.</p>
        </div>
      </section>

      <section className="proof-strip" aria-label="Project highlights">
        {HERO_CARDS.map(([title, body], i) => (
          <article key={title}><span>{String(i + 1).padStart(2, '0')}</span><div><strong>{title}</strong><p>{body}</p></div></article>
        ))}
      </section>

      <section id="scenario-explorer" className="explorer-section">
      <div className="list-header">
        <div><p className="eyebrow">STORED RESULTS</p><h2 className="section-title">Scenario explorer</h2></div>
        <span className="dataset-label">Waymo Open Dataset</span>
      </div>
      <p className="page-intro">
        Open a scene to replay it. Where the search found a collision, the replay compares the recorded challenger
        with the smallest collision-producing perturbation it found.
      </p>
      <div className="list-tools">
        <form className="scenario-lookup" onSubmit={openScenario}>
          <label className="sr-only" htmlFor="scenario-lookup">Open by scenario ID</label>
          <input id="scenario-lookup" type="search" placeholder="Open by scenario ID…" maxLength={256}
            value={scenarioId} onChange={(e) => setScenarioId(e.target.value)} />
          <button type="submit" disabled={!scenarioId.trim()}>Open <span aria-hidden="true">↗</span></button>
        </form>
        <label className="toggle">
          <input
            type="checkbox"
            checked={stressTestedOnly}
            onChange={(e) => setStressTestedOnly(e.target.checked)}
          />
          Stress-tested only
        </label>
      </div>
      <p className="caption">
        Ranked by fragility score, most fragile first. The order comes from the server
        and is not re-sorted here.
      </p>
      <details className="metric-guide"><summary>How to read the results</summary>
        <p>Fragility ranks the recorded scene; it is not a collision probability. Min perturbation is the smallest weighted perturbation found by the finite search, not a proven global minimum. TTC is time-to-collision; PET is post-encroachment time. “None” means no finite value in that model. A negative PET means two agents occupied the same connected conflict zone at overlapping times.</p>
      </details>

      {query.isPending ? (
        <p className="status" role="status">
          Loading scenarios…
        </p>
      ) : query.isError && rows.length === 0 ? (
        <RequestError error={query.error} retry={() => query.refetch()} retrying={query.isFetching} />
      ) : rows.length === 0 ? (
        <p className="status" role="status">
          {stressTestedOnly
            ? 'No scenarios have a stored stress-test result yet. Turn off the filter to see every scored scenario.'
            : 'No scenarios have been scored yet.'}
        </p>
      ) : (
        <>
          <div className="table-scroll" role="region" aria-label="Scenario results" tabIndex={0}>
            <table className="scenario-table">
              <thead>
                <tr>
                  <th scope="col" className="num">#</th>
                  <th scope="col">Scenario</th>
                  <th scope="col" className="num">Fragility</th>
                  <th scope="col" className="num" title="Minimum time-to-collision over pairs involving the SDC">
                    Min TTC
                  </th>
                  <th scope="col" className="num" title="Minimum post-encroachment time over pairs involving the SDC">
                    Min PET
                  </th>
                  <th scope="col" className="num">Min perturbation</th>
                  <th scope="col">Outcome</th>
                  <th scope="col" className="num">Agents</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={row.scenario_id}>
                    <td className="num muted">{i + 1}</td>
                    <td className="mono">
                      <Link
                        to={scenarioPath(row.scenario_id)}
                        state={{ listSearch: location.search } satisfies ListReturnState}
                      >
                        {row.scenario_id}
                      </Link>
                    </td>
                    <td className="num">{row.fragility_score.toFixed(3)}</td>
                    <td className="num">
                      <TimeMetricCell value={row.min_ttc ?? null} metric="ttc" />
                    </td>
                    <td className="num">
                      <TimeMetricCell value={row.min_pet ?? null} metric="pet" />
                    </td>
                    <td className="num">
                      <PerturbationCell row={row} />
                    </td>
                    <td>
                      <OutcomeCell row={row} />
                    </td>
                    <td className="num">{row.n_agents ?? <span className="muted">—</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="list-footer">
            <span className="muted">
              {rows.length} shown{query.hasNextPage ? '' : ' · end of list'}
            </span>
            {query.isFetchNextPageError && (
              <span className="status-error" role="alert">
                {describeError(query.error)}
              </span>
            )}
            {query.hasNextPage && (
              <button
                type="button"
                onClick={() => void query.fetchNextPage()}
                disabled={query.isFetchingNextPage}
              >
                {query.isFetchingNextPage ? 'Loading…' : 'Load more'}
              </button>
            )}
          </div>
        </>
      )}
      </section>
    </>
  );
}
