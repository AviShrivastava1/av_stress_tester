import type { ReactNode } from 'react';
import type { HealthResponse, StatsResponse } from '../api/client';
import { useHealth, useStats } from '../api/corpus';
import { RequestError } from '../components/RequestError';
import { describeOutcome, ROBUSTLY_SAFE } from '../domain/outcomes';

/*
 * Every number on this page is a count the API reports, shown as reported. None is
 * summed, divided or charted: the counts come from different columns (see
 * src/api/routes.py's stats handler) and do not partition the corpus, so a total, a
 * percentage or a pie would assert a relationship the data does not have.
 *
 * The words on this page, and what backs each one:
 *   eyebrow "CORPUS / STATE"
 *       the page is the state of the scored corpus: what GET /stats and GET /health returned (api/corpus.ts useStats, useHealth)
 *   "Stored run state, exactly as reported by the API."
 *       the four panels below: CorpusPanel (scenario_scores rows), StoredResultsPanel (stress_tested_at and stress_outcome),
 *       LatestAttemptsPanel (last_attempt_outcome), and the health panel (GET /health). "Exactly as reported": count()
 *       below only adds thousands separators to the number the API sent, and no value on the page is summed, divided or
 *       differenced; the fragility figures are not counts and are rounded to three decimals
 *   "Fields overlap; these counts are not a partition."
 *       src/api/routes.py stats() computes each count with its own count(*) FILTER (WHERE ...) over different columns
 *       (stress_tested_at, stress_outcome, last_attempt_outcome), and with_geometry from scenario_agents, so one scenario
 *       can be in several counts (it can be stress-tested and also have a latest attempt) and the counts do not add up
 *   label "Waymo Open Dataset"
 *       the scenes are Waymo Open Motion Dataset scenarios, which is part of the Waymo Open Dataset (src/data/parser.py
 *       ScenarioParser reads them); the same label as the list page, and the name the licence attribution uses
 *       (components/Footer.tsx WAYMO_ATTRIBUTION)
 *   "Scored scenes."  (total_scenarios)
 *       stats() counts every row of scenario_scores: count(*) over the table
 *   "Scenarios with a stored result."  (stress_tested)
 *       stats() counts the rows where stress_tested_at IS NOT NULL
 *   "Scenarios with agent geometry exported. This includes geometry the scene view may decline to draw because it no
 *    longer matches the stored score."  (with_geometry)
 *       stats() counts DISTINCT scenario_id in scenario_agents. The scene view (routes.get_trajectories) serves a scenario's
 *       agents only where scenario_agents.scene_fingerprint matches the scored row's, and the SDC must match too (the
 *       "THE SDC MUST MATCH TOO" guard), so an exported scene can still be refused; the count includes it
 *   caption "From each scenario's stored stress-test result."  (Stored results)
 *       StoredResultsPanel's counts are over scenario_scores.stress_tested_at, min_perturbation and stress_outcome
 *   caption "The most recent stress-test pass for each scenario."  (Latest attempts)
 *       LatestAttemptsPanel's counts are over scenario_scores.last_attempt_outcome, the latest pass, not the stored result
 *   There are no hard-coded counts in the copy: every digit on the page is a value the API sent (a test checks).
 */

function count(n: number): string {
  return n.toLocaleString('en-US');
}

function fragility(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : v.toFixed(3);
}

function Stat({ id, label, value, children }: { id: string; label: string; value: string; children?: ReactNode }) {
  return (
    <div className="stat" data-stat={id}>
      <dt>{label}</dt>
      <dd>{value}</dd>
      {children && <dd className="stat-def">{children}</dd>}
    </div>
  );
}

function CorpusPanel({ stats }: { stats: StatsResponse }) {
  return (
    <section className="panel">
      <h2>Corpus</h2>
      <dl className="stat-list">
        <Stat id="total_scenarios" label="Scenarios" value={count(stats.total_scenarios)}>
          Scored scenes.
        </Stat>
        <Stat id="with_geometry" label="With exported geometry" value={count(stats.with_geometry)}>
          Scenarios with agent geometry exported. This includes geometry the scene view may decline to
          draw because it no longer matches the stored score.
        </Stat>
        <Stat id="fragility_min" label="Fragility, lowest" value={fragility(stats.fragility_min)} />
        <Stat id="fragility_mean" label="Fragility, mean" value={fragility(stats.fragility_mean)} />
        <Stat id="fragility_max" label="Fragility, highest" value={fragility(stats.fragility_max)} />
      </dl>
    </section>
  );
}

function StoredResultsPanel({ stats }: { stats: StatsResponse }) {
  const noCollision = describeOutcome('no_collision_found');
  return (
    <section className="panel">
      <h2>Stored results</h2>
      <p className="caption">From each scenario&apos;s stored stress-test result.</p>
      <dl className="stat-list">
        <Stat id="stress_tested" label="Stress-tested" value={count(stats.stress_tested)}>
          Scenarios with a stored result.
        </Stat>
        <Stat
          id="collisions_found"
          label="Stored results with a perturbation (collision found)"
          value={count(stats.collisions_found)}
        >
          Counted from the stored perturbation, which the pipeline keeps only for a verified collision, so
          results stored before outcomes were recorded are included.
        </Stat>
        <Stat id="no_collision_found" label={noCollision.label} value={count(stats.no_collision_found)}>
          {noCollision.meaning}
        </Stat>
        <Stat id="robustly_safe" label={ROBUSTLY_SAFE.label} value={count(stats.robustly_safe)}>
          {ROBUSTLY_SAFE.meaning}
        </Stat>
      </dl>
      <p className="note">
        These are the API&apos;s counts as reported. They are not guaranteed to add up to the
        stress-tested total.
      </p>
    </section>
  );
}

const LATEST_ATTEMPTS = [
  ['replay_infeasible', 'replay_infeasible'],
  ['heading_blend_singularity', 'heading_blend_singularity'],
  ['no_challenger', 'no_challenger'],
  ['stress_errors', 'error'],
] as const;

function LatestAttemptsPanel({ stats }: { stats: StatsResponse }) {
  return (
    <section className="panel">
      <h2>Latest attempts</h2>
      <p className="caption">The most recent stress-test pass for each scenario.</p>
      <dl className="stat-list">
        {LATEST_ATTEMPTS.map(([field, outcome]) => {
          const { label, meaning } = describeOutcome(outcome);
          return (
            <Stat key={field} id={field} label={label} value={count(stats[field])}>
              {meaning}
            </Stat>
          );
        })}
      </dl>
      <p className="note">
        A scenario counted here can also hold a stored result from an earlier pass. Latest attempts that
        did or did not find a collision are not counted separately by the API.
      </p>
    </section>
  );
}

/*
 * The words in the service health panel, and what backs each. The panel prints what GET /health sent, and the status is a
 * plain string in the contract (frontend/openapi.json HealthResponse: type string), not an enum.
 *
 *   status "ok"          printed as it is: the API said the database and PostGIS both answered (src/api/routes.py health)
 *   status "degraded"    the API's word for "the database answered and PostGIS did not": printed as a warning badge, the
 *                        tone of the PostGIS row beside it, so it is neither plain "ok" nor shown as the service being
 *                        down (a down service is the request error with a retry button). The answer is still HTTP 200
 *   any other status     printed as it was sent, with no badge: a value this page does not know is not given a tone
 *   PostGIS "not available" and "Degraded: PostGIS did not answer. The scores endpoints do not call it and keep working; the
 *   geometry endpoints do call it."
 *                        `postgis` is null when the version call failed (the extension is missing, or the call raised).
 *                        The scores endpoints (src/api/routes.py list_scenarios, get_scenario, stats) run no PostGIS
 *                        function; get_trajectories and get_perturbed read paths through ST_AsGeoJSON and ST_DumpPoints
 */
const DEGRADED = 'degraded';

function HealthPanelBody({ health }: { health: HealthResponse }) {
  return (
    <dl className="stat-list">
      <div className="stat" data-stat="status">
        <dt>API status</dt>
        <dd>{health.status === DEGRADED ? <span className="badge badge-warning">{health.status}</span> : health.status}</dd>
      </div>
      <Stat id="database" label="Database" value={health.database} />
      {health.postgis !== null && health.postgis !== undefined ? (
        <Stat id="postgis" label="PostGIS" value={health.postgis} />
      ) : (
        <div className="stat" data-stat="postgis">
          <dt>PostGIS</dt>
          <dd>
            <span className="badge badge-warning">not available</span>
          </dd>
          <dd className="stat-def">
            Degraded: PostGIS did not answer. The scores endpoints do not call it and keep working; the geometry endpoints
            do call it.
          </dd>
        </div>
      )}
    </dl>
  );
}

export function StatsPage() {
  // Independent queries, both started on this render.
  const stats = useStats();
  const health = useHealth();

  return (
    <section className="corpus-page">
      <div className="list-header">
        <div><p className="eyebrow">CORPUS / STATE</p><h1>Corpus</h1></div>
        <span className="dataset-label">Waymo Open Dataset</span>
      </div>
      <p className="page-intro">Stored run state, exactly as reported by the API.</p>
      <p className="caption">Fields overlap; these counts are not a partition.</p>

      <div className="stats-grid">
        {stats.isPending ? (
          <section className="panel">
            <p className="status" role="status">Loading corpus counts…</p>
          </section>
        ) : stats.isError ? (
          <section className="panel">
            <RequestError error={stats.error} retry={() => stats.refetch()} retrying={stats.isFetching} />
          </section>
        ) : (
          <>
            <CorpusPanel stats={stats.data} />
            <StoredResultsPanel stats={stats.data} />
            <LatestAttemptsPanel stats={stats.data} />
          </>
        )}

        <section className="panel">
          <h2>Service health</h2>
          {health.isPending ? (
            <p className="status" role="status">Checking…</p>
          ) : health.isError ? (
            <RequestError error={health.error} retry={() => health.refetch()} retrying={health.isFetching} />
          ) : (
            <HealthPanelBody health={health.data} />
          )}
        </section>
      </div>
    </section>
  );
}
