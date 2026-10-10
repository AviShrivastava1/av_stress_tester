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
 * The words at the top of the list. Every sentence describes code that exists in this repository, and none claims that
 * a visitor can run a search here or that anything is certified. What backs each one:
 *
 *   eyebrow "COUNTERFACTUAL MOTION SEARCH"
 *       "search": src/optimization/scipy_optimizer.py optimize_scenario looks, offline, for a perturbation of recorded
 *       motion that produces contact. "counterfactual": the replay is the track simulated from the stored perturbation
 *       (perturbed_paths, GET /scenarios/{id}/perturbed; src/physics/simulator.py ForwardSimulator), drawn beside the
 *       logged one as on the detail page
 *   h1 "Find the smallest change that breaks the scene."
 *       the question scenario_scores.min_perturbation answers: the smallest perturbation optimize_scenario found that
 *       makes src/danger/collision_detector.py check_collision_trajectory report contact. "Smallest" means the smallest
 *       found, which the note under the actions says in as many words
 *   lede "Search recorded AV motion. Replay the counterfactual. Inspect every frame."
 *       recorded: Waymo Open Motion scenarios (src/data/parser.py ScenarioParser). "Search" is the offline search above,
 *       not something this page runs (the note says so). "Replay" and "Inspect every frame": scene/SceneView.tsx and
 *       scene/PlaybackControls.tsx replay a stored row frame by frame
 *   actions "Explore scenarios" and "Inspect the system"
 *       the explorer below (#scenario-explorer) and the /method route (pages/MethodPage.tsx). The arrows are
 *       aria-hidden spans in the markup, so each link's name is its words alone; the stylesheet generates no text
 *   note "Smallest found, not proven: one challenger per scenario, a finite search."
 *       the first two limits on the Method page (MethodPage.tsx LIMITS); src/optimization/scipy_optimizer.py
 *       optimize_scenario, whose Differential Evolution budget is recorded in search_provenance; the detail page's
 *       "Challengers searched" (challengers_searched of challengers_total): the search perturbs one heuristically
 *       chosen challenger (src/optimization/selection.py)
 *   note "Searches ran offline; this site shows stored results."
 *       src/api/routes.py has GET routes only and no endpoint runs the optimizer (src/api/__init__.py); the rows are
 *       scenario_scores and perturbed_paths read from PostGIS
 *   note "Scenarios with equal fragility are listed by scenario ID, which says nothing about which is more fragile."
 *       the second ORDER BY key in routes.list_scenarios is scenario_id, and ranker.rank_scenarios breaks ties on
 *       str(scenario_id); a tie is equal fragility_score, so the ID order carries no information about fragility
 *   figure caption "ILLUSTRATION · not measured data", and the drawing (HeroTrace)
 *       the drawing is hand-written paths and fixed labels; nothing in it is read from a scenario or a stored result.
 *       Its labels "RECORDED / SDC", "PERTURBED / CHALLENGER" and "CONTACT" name the roles the scene legend uses
 *       (pages/detail/ScenePanel.tsx ROLE_LEGEND: "SDC (self-driving car)", "Challenger, as logged", "Challenger,
 *       perturbed"); CONTACT marks, with a leader line, where the two drawn paths cross
 *   stored replay link "Stored replay · row 01", "Min perturbation 0.022", and its aria-label "Open stored replay for
 *   scenario <ID>"
 *       the first loaded row that has a stored perturbation: rows.findIndex(row => row.min_perturbation != null). It is
 *       not a curated or best pick: the rows come in the order GET /scenarios returns them (fragility_score descending,
 *       then scenario_id), and "row 01" is that row's # in the table below. "Min perturbation" is the table's own
 *       header for scenario_scores.min_perturbation, shown with the table's formatMagnitude
 *   SYSTEM_SIGNATURE (the four signals under the hero, a list named "System components", as the Method page names its own)
 *       Search "Differential Evolution"        src/optimization/scipy_optimizer.py, `differential_evolution` in optimize_scenario
 *       Replay "bicycle / linear kinematics"   src/physics/bicycle_model.py, src/physics/linear_model.py, chosen by
 *                                              src/physics/simulator.py ForwardSimulator.step (vehicles, then pedestrians and cyclists)
 *       Contact "float64 oriented boxes"       src/danger/collision_detector.py get_corners (np.float64), COLLISION_GEOMETRY_VERSION
 *       Storage "PostGIS, read-only API"       src/scoring/export_geometry.py writes the geometry to PostGIS; the API role is
 *                                              read-only (DEPLOY.md step 2, av_api_ro) and src/api/routes.py has GET routes only
 *   explorer eyebrow "RANKED RESULT SET" and heading "Scenario index"
 *       the rows of GET /scenarios (routes.list_scenarios), ranked by fragility_score descending
 *   label "Waymo Open Dataset"
 *       the scenes are Waymo Open Motion Dataset scenarios, which is part of the Waymo Open Dataset (src/data/parser.py
 *       ScenarioParser reads them). It is the name this label had before, and the name the licence attribution uses
 *       (components/Footer.tsx WAYMO_ATTRIBUTION)
 *   "Metric definitions"
 *       fragility: src/danger/danger_score.py; "not a collision probability": README.md; "weighted": the weighted norm in
 *       optimize_scenario (and the note in pages/detail/ReplayOffsetBlock.tsx); "within a finite search budget": the
 *       Differential Evolution budget recorded in search_provenance (de_popsize, de_maxiter), so "not a proven global
 *       minimum"; TTC and PET: src/danger/ttc_engine.py, src/danger/pet_engine.py; "None": no finite value
 *       (components/TimeMetricCell.tsx); "Negative PET marks overlapping occupancy of one connected conflict zone":
 *       pet_engine._conflict_components (two agents inside the same connected conflict zone at overlapping times)
 */
const SYSTEM_SIGNATURE = [
  ['Search', 'Differential Evolution'],
  ['Replay', 'bicycle / linear kinematics'],
  ['Contact', 'float64 oriented boxes'],
  ['Storage', 'PostGIS, read-only API'],
] as const;
const DATASET_LABEL = 'Waymo Open Dataset';

/*
 * The words on the ranked table. What backs each one:
 *
 *   "Ranked by fragility score, most fragile first"
 *       GET /scenarios (src/api/routes.py list_scenarios) orders by fragility_score descending, and
 *       ranker.rank_scenarios (src/scoring/ranker.py) sorts the same way, so rank 1 is the most fragile; this page
 *       renders the rows in the order the server returns them and does not sort
 *   "Equal scores are tie-broken by scenario ID"
 *       the second ORDER BY key in list_scenarios is scenario_id, and the cursor predicate there
 *       (fragility_score = the cursor's score AND scenario_id after the cursor's id) pages through ties in the
 *       same order; ranker.rank_scenarios breaks ties on str(scenario_id) as well. The table shows the score to three
 *       decimals but the tie is on the stored value, so two rows can look equal and not be tied
 *   "position within a tie does not mean one scenario is more fragile than another"
 *       a tie is equal fragility_score, so the scenario_id order carries no information about fragility
 *   "Min perturbation is the smallest found by a finite search, not a proven minimum"
 *       the Differential Evolution budget in search_provenance (scipy_optimizer.optimize_scenario)
 *   the table's screen-reader caption "Scenarios ordered by fragility score, highest first"
 *       the same ordering as the visible caption's first sentence above (GET /scenarios, list_scenarios, fragility_score
 *       descending). It is `sr-only`, so it names the table for assistive technology and takes no space; the visible
 *       caption above the table carries the tie caveat and the "not a proven minimum" sentence
 *   header "Timing" and its title "Minimum TTC and PET over pairs involving the SDC"
 *       min_ttc and min_pet are the values danger_score.score_scenario stores from compute_min_ttc_sdc
 *       (src/danger/ttc_engine.py) and compute_min_pet_sdc (src/danger/pet_engine.py), which restrict to pairs
 *       that involve the SDC; the two labels' own titles say time-to-collision and post-encroachment time
 *   header "Search result"
 *       the cell is OutcomeCell (components/OutcomeBadge.tsx): the stored stress_outcome, or the last attempt
 *   headers "Min perturbation", "Agents"
 *       scenario_scores.min_perturbation and scenario_scores.n_agents (the latter was already so labelled)
 */
const TTC_TITLE = 'Minimum time-to-collision over pairs involving the SDC';
const PET_TITLE = 'Minimum post-encroachment time over pairs involving the SDC';

/** A hand-drawn illustration of the idea, not a plot of any scenario (see the comment at the top of the file). */
function HeroTrace() {
  return (
    <svg className="hero-trace" viewBox="0 0 720 560" aria-hidden="true">
      <defs>
        <pattern id="trace-grid" width="48" height="48" patternUnits="userSpaceOnUse">
          <path d="M 48 0 L 0 0 0 48" className="trace-grid-line" />
        </pattern>
      </defs>
      <rect width="720" height="560" fill="url(#trace-grid)" />
      <path className="trace-guide" d="M92 516 C174 432 224 340 318 302 S520 254 646 76" />
      <path className="trace-path trace-path-sdc" pathLength="1" d="M86 520 C182 418 238 354 326 304 S514 254 650 70" />
      <path className="trace-path trace-path-counterfactual" pathLength="1" d="M634 500 C552 438 492 370 428 310 S324 220 208 82" />
      <circle className="trace-contact-pulse" cx="371" cy="283" r="30" />
      <line className="trace-leader" x1="371" y1="250" x2="371" y2="192" />
      <circle className="trace-contact" cx="371" cy="283" r="5" />
      <g className="trace-agent trace-agent-sdc" transform="translate(337 296) rotate(-31)">
        <rect x="-17" y="-8" width="34" height="16" rx="3" />
        <line x1="4" y1="0" x2="18" y2="0" />
      </g>
      <g className="trace-agent trace-agent-counterfactual" transform="translate(393 302) rotate(41)">
        <rect x="-17" y="-8" width="34" height="16" rx="3" />
        <line x1="4" y1="0" x2="18" y2="0" />
      </g>
      <text x="96" y="548">RECORDED / SDC</text>
      <text x="690" y="522" textAnchor="end">PERTURBED / CHALLENGER</text>
      <text className="trace-contact-label" x="382" y="198">CONTACT</text>
    </svg>
  );
}

export function scenarioPath(scenarioId: string): string {
  return `/scenarios/${encodeURIComponent(scenarioId)}`;
}

function PerturbationCell({ row }: { row: ScenarioSummary }) {
  if (row.min_perturbation !== null && row.min_perturbation !== undefined) {
    return <span title={String(row.min_perturbation)}>{formatMagnitude(row.min_perturbation)}</span>;
  }
  // NULL is never read as "safe" — the Search result column says what actually happened.
  if (!row.stress_tested && !row.stress_attempted) {
    return <span className="muted">not tested</span>;
  }
  return (
    <span className="muted" title="No stored perturbation. See the Search result column.">
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
  // The stored replay link: the first loaded row with a stored perturbation. In the hosted data every such row also has a
  // stored perturbed path (GET /scenarios/{id}/perturbed returned both tracks for all 18 on 2026-10-10), but nothing
  // enforces that. perturbed_paths is written by a separate export step (src/scoring/export_geometry.py
  // export_perturbed_path), and src/api/routes.py get_perturbed returns null paths when none is stored for the row's
  // stress_run_id and scene_fingerprint. A list row has no field that says whether a path exists, so this predicate cannot
  // check it. If the path is missing, the detail page shows the stored result and the recorded scene without a comparison,
  // as for any stored result whose path was not exported (ScenePanel, perturbedTrackDrawn).
  const replayIndex = rows.findIndex((row) => row.min_perturbation != null);
  const replayRow = replayIndex >= 0 ? rows[replayIndex]! : null;

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
          <p className="eyebrow hero-eyebrow">COUNTERFACTUAL MOTION SEARCH</p>
          <h1>Find the smallest change that breaks the scene.</h1>
          <p className="hero-lede">
            Search recorded AV motion. Replay the counterfactual. Inspect every frame.
          </p>
          <div className="hero-actions">
            <a className="button-primary" href="#scenario-explorer">Explore scenarios <span aria-hidden="true">↓</span></a>
            <Link className="text-link" to="/method">Inspect the system <span aria-hidden="true">↗</span></Link>
          </div>
          <p className="hero-note">
            Smallest found, not proven: one challenger per scenario, a finite search. Searches ran offline; this site shows
            stored results. Scenarios with equal fragility are listed by scenario ID, which says nothing about which is more fragile.
          </p>
        </div>

        <figure className="hero-visual">
          <figcaption className="trace-label">ILLUSTRATION · not measured data</figcaption>
          <HeroTrace />
          {replayRow && (
            <Link
              className="hero-replay"
              to={scenarioPath(replayRow.scenario_id)}
              state={{ listSearch: location.search } satisfies ListReturnState}
              aria-label={`Open stored replay for scenario ${replayRow.scenario_id}`}
            >
              <span>Stored replay · row {String(replayIndex + 1).padStart(2, '0')}</span>
              <strong className="mono">{replayRow.scenario_id}</strong>
              <span>Min perturbation {formatMagnitude(replayRow.min_perturbation!)}</span>
            </Link>
          )}
        </figure>

        <dl className="system-signature" aria-label="System components">
          {SYSTEM_SIGNATURE.map(([term, value]) => (
            <div key={term}><dt>{term}</dt><dd>{value}</dd></div>
          ))}
        </dl>
      </section>

      <section id="scenario-explorer" className="explorer-section">
      <div className="list-header">
        <div><p className="eyebrow">RANKED RESULT SET</p><h2 className="section-title">Scenario index</h2></div>
        <span className="dataset-label">{DATASET_LABEL}</span>
      </div>
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
        Ranked by fragility score, most fragile first. Equal scores are tie-broken by scenario ID; position within
        a tie does not mean one scenario is more fragile than another. Min perturbation is the smallest found by a
        finite search, not a proven minimum.
      </p>
      <details className="metric-guide"><summary>Metric definitions</summary>
        <p>Fragility ranks the recorded scene; it is not a collision probability. Min perturbation is the smallest weighted perturbation found within a finite search budget, not a proven global minimum. TTC is time-to-collision; PET is post-encroachment time. “None” means no finite value in that model. Negative PET marks overlapping occupancy of one connected conflict zone.</p>
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
              <caption className="sr-only">Scenarios ordered by fragility score, highest first</caption>
              <thead>
                <tr>
                  <th scope="col" className="num">#</th>
                  <th scope="col">Scenario</th>
                  <th scope="col" className="num">Fragility</th>
                  <th scope="col" className="num timing-col" title="Minimum TTC and PET over pairs involving the SDC">Timing</th>
                  <th scope="col">Search result</th>
                  <th scope="col" className="num">Min perturbation</th>
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
                    <td className="num timing-col">
                      <div className="timing-line">
                        <span className="timing-label" title={TTC_TITLE}>TTC</span>{' '}
                        <TimeMetricCell value={row.min_ttc ?? null} metric="ttc" />
                      </div>
                      <div className="timing-line">
                        <span className="timing-label" title={PET_TITLE}>PET</span>{' '}
                        <TimeMetricCell value={row.min_pet ?? null} metric="pet" />
                      </div>
                    </td>
                    <td>
                      <OutcomeCell row={row} />
                    </td>
                    <td className="num">
                      <PerturbationCell row={row} />
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
