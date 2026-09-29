import type { ReactNode } from 'react';
import type { HealthResponse, StatsResponse } from '../api/client';
import { useHealth, useStats } from '../api/corpus';
import { describeError } from '../api/errors';
import { describeOutcome, ROBUSTLY_SAFE } from '../domain/outcomes';

/*
 * Every number on this page is a count the API reports, shown as reported. None is
 * summed, divided or charted: the counts come from different columns (see
 * src/api/routes.py's stats handler) and do not partition the corpus, so a total, a
 * percentage or a pie would assert a relationship the data does not have.
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
          Every scored scenario.
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
          Scenarios holding a stored stress-test result.
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

function HealthPanelBody({ health }: { health: HealthResponse }) {
  return (
    <dl className="stat-list">
      <Stat id="status" label="API status" value={health.status} />
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
            Degraded: the scores endpoints work, but the geometry endpoints return empty.
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
    <section>
      <div className="list-header">
        <h1>Corpus</h1>
      </div>
      <p className="caption">Counts as the API reports them. Nothing on this page is summed or charted.</p>

      <div className="stats-grid">
        {stats.isPending ? (
          <section className="panel">
            <p className="status" role="status">Loading corpus counts…</p>
          </section>
        ) : stats.isError ? (
          <section className="panel">
            <p className="status status-error" role="alert">{describeError(stats.error)}</p>
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
            <p className="status status-error" role="alert">{describeError(health.error)}</p>
          ) : (
            <HealthPanelBody health={health.data} />
          )}
        </section>
      </div>
    </section>
  );
}
