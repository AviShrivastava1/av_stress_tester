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
    <section>
      <div className="list-header">
        <div><p className="eyebrow">EXPLORE THE EDGE CASES</p><h1>Scenarios</h1></div>
        <span className="dataset-label">Waymo Open Dataset</span>
      </div>
      <p className="page-intro">Explore recorded driving scenes and the small changes that lead to a collision.</p>
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
        <p>Fragility ranks the recorded scene; it is not a collision probability. Min perturbation is the smallest weighted change found by the search, not a proven global minimum. TTC is time-to-collision; PET is post-encroachment time. “None” means no finite value in that model. A negative PET records overlapping occupancy of a conflict zone.</p>
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
  );
}
