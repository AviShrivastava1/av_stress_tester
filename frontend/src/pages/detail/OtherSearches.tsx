import { Fragment, useState } from 'react';
import type { OtherSearch, OtherSearchesResponse } from '../../api/client';
import { RequestError } from '../../components/RequestError';
import { useOtherSearches } from '../../api/scenarios';
import { formatMagnitude } from '../../domain/format';
import { describeOutcome, parseOutcome } from '../../domain/outcomes';

/*
 * The words in "Other searches of this scenario", and what backs each. Every value is read from GET /scenarios/{id}/other-searches
 * (src/api/routes.py get_other_searches; src/api/other_searches.py is the allow-list; src/api/models.py OtherSearch and
 * OtherSearchesResponse describe each field); none is written here. A value the API sends as null is "Not recorded". The section is
 * closed until opened and asks for nothing before that; its note is drawn once it is opened (a closed page carries no text about seeds). It is not next to the headline figure (the Min perturbation of the
 * Perturbation panel and Run details): it sits below the Provenance panel, and says what it is before it shows a number.
 *
 *   "Each of these is a separate search of this scenario, run with a seed, a budget or a challenger agent of its own. None is shown to be the smallest:"
 *       each row is a search_runs row of kind seed, budget or challenger (src/scoring/other_searches.py records them, run by hand):
 *       another run of Differential Evolution on the same scene, with another seed or budget (search_provenance records both for
 *       the stored result, shown in the Provenance panel), or against another challenger agent with the stored seed and budget
 *       (run_challenger_searches). The API lists them in a fixed order (seeds, budgets, challengers, then by seed or budget, a
 *       challenger by its rank), not by norm, and marks none; this page does not sort or mark them either
 *   "each figure is the smallest perturbation that one finite search found, and another seed or budget can find a smaller or a larger one."
 *       the same statement as Method limit 1 (a finite stochastic search; the stored result is one such search)
 *   "A challenger’s rank is its place by distance from the SDC, not by its result, and a figure from another challenger agent is a
 *    perturbation of a different agent, not set beside the stored result’s."
 *       challenger_rank is the place of the agent in perturbation_space.rank_challengers (the ranking the pipeline picks its one
 *       challenger from: the smallest distance between centers over the timesteps both are valid; 1 is the nearest); target_idx is
 *       the agent that search perturbed. The weighted norm of a perturbation of one agent is not a smaller or larger version of the
 *       norm of another agent's (the agent's own bounds and, for a pedestrian, its own model apply), so nothing here sets one beside
 *       the other, and a challenger's figure carries the agent it is for
 *   "The stored result and its search are in Provenance above."
 *       the stored result is not a row of this list (the API leaves out any row whose id is the stored stress_run_id)
 *   the kind: "Another seed", "Another budget", "Another challenger"
 *       search_runs.kind 'seed' / 'budget' / 'challenger'. A challenger row also gives its rank (challenger_rank, 1 = nearest) and,
 *       under its weighted norm, the agent that figure is for ("for agent N", target_idx: the index in the scene's agent arrays)
 *   the configuration columns: Seed, Population size, Maximum iterations, Convergence tolerance
 *       seed, popsize, maxiter, tol of optimize_scenario (tol is the convergence tolerance on the population spread: no unit)
 *   "Result"
 *       outcome through the interface's own outcome words (domain/outcomes.ts describeOutcome), and the collision frame
 *       (collision_timestep) when there is one
 *   "Weighted norm" and "Evaluations"
 *       min_perturbation (a weighted norm over components of mixed units, the same quantity as the Min perturbation figure) and n_eval
 *   the components, "Perturbation components"
 *       delta, named with the labels the API derives from that search's OWN delta_parameterization (delta_labels), which carry the
 *       units; a component is never given a unit its model does not have
 *   "Under a different or unrecorded model; its figures are not shown against the stored result's."
 *       the search's delta_parameterization is null, or is not the stored result's (stored_delta_parameterization): the components
 *       mean different things under different kinematic models, so no number of such a search is shown, and none is compared
 *   "Not every recorded search is listed here; those shown are the first in the order above."
 *       OtherSearchesResponse.truncated: true only when the API found more matching searches than it lists (its limit is
 *       OtherSearchesResponse's MAX_SEARCHES, src/api/other_searches.py; the query asks for one more than the limit and `truncated` is
 *       that one being there). The listed ones are the first in the API's fixed order, the order of the table. No number is typed
 *       here; absent when `truncated` is false or not sent
 *   "No other search is recorded for this scenario."
 *       searches is empty: none recorded, the tables are not in this database, or the API role cannot read them. Never read as a result
 */

const NOT_RECORDED = 'Not recorded';

const KIND_LABEL: Record<string, string> = { seed: 'Another seed', budget: 'Another budget', challenger: 'Another challenger' };

function outcomeText(s: OtherSearch): string {
  const outcome = s.outcome === null || s.outcome === undefined ? null : parseOutcome(s.outcome);
  if (outcome === null) return NOT_RECORDED;
  const label = describeOutcome(outcome).label;
  const frame = s.collision_timestep;
  return outcome === 'collision_found' && frame !== null && frame !== undefined ? `${label}, frame ${frame}` : label;
}

const num = (value: number | null | undefined, digits = 3) => (value === null || value === undefined ? NOT_RECORDED : formatMagnitude(value, digits));
const whole = (value: number | null | undefined) => (value === null || value === undefined ? NOT_RECORDED : String(value));

function comparable(s: OtherSearch, stored: string | null | undefined): boolean {
  return s.delta_parameterization !== null && s.delta_parameterization !== undefined && s.delta_parameterization === stored;
}

function Components({ s }: { s: OtherSearch }) {
  const delta = s.delta;
  const labels = s.delta_labels;
  if (delta === null || delta === undefined) return <span className="muted">{NOT_RECORDED}</span>;
  return (
    <details>
      <summary>Perturbation components</summary>
      <dl className="facts">
        {delta.map((value, i) => (
          <Fragment key={i}>
            <dt>{labels !== null && labels !== undefined && labels.length === delta.length ? labels[i] : `component ${i + 1}`}</dt>
            <dd title={String(value)}>{formatMagnitude(value, 4)}</dd>
          </Fragment>
        ))}
      </dl>
    </details>
  );
}

function Body({ data }: { data: OtherSearchesResponse }) {
  const searches = data.searches ?? [];
  if (searches.length === 0) return <p className="note">No other search is recorded for this scenario.</p>;
  return (
    <>
      <table className="delta-table other-searches-table" aria-label="Other searches of this scenario">
        <thead>
          <tr>
            <th scope="col">Search</th>
            <th scope="col" className="num">Seed</th>
            <th scope="col" className="num">Population size</th>
            <th scope="col" className="num">Maximum iterations</th>
            <th scope="col" className="num">Convergence tolerance</th>
            <th scope="col">Result</th>
            <th scope="col" className="num">Weighted norm</th>
            <th scope="col" className="num">Evaluations</th>
          </tr>
        </thead>
        <tbody>
          {searches.map((s, i) => {
            const shown = comparable(s, data.stored_delta_parameterization);
            return (
              <tr key={s.search_run_id ?? `row-${i}`}>
                <th scope="row">
                  {KIND_LABEL[s.kind] ?? s.kind}
                  {s.kind === 'challenger' && s.challenger_rank !== null && s.challenger_rank !== undefined ? `, rank ${s.challenger_rank}` : ''}
                  {s.search_run_id ? <div className="mono muted" title="The content hash of this search">{s.search_run_id}</div> : null}
                </th>
                <td className="num">{whole(s.seed)}</td>
                <td className="num">{whole(s.popsize)}</td>
                <td className="num">{whole(s.maxiter)}</td>
                <td className="num" title={s.tol === null || s.tol === undefined ? undefined : String(s.tol)}>{num(s.tol, 4)}</td>
                {shown ? (
                  <>
                    <td>{outcomeText(s)}</td>
                    <td className="num" title={s.min_perturbation === null || s.min_perturbation === undefined ? undefined : String(s.min_perturbation)}>
                      {s.min_perturbation === null || s.min_perturbation === undefined ? '—' : num(s.min_perturbation)}
                      {s.kind === 'challenger' && s.target_idx !== null && s.target_idx !== undefined ? (
                        <div className="muted">for agent {s.target_idx}</div>
                      ) : null}
                    </td>
                    <td className="num">{whole(s.n_eval)}</td>
                  </>
                ) : (
                  <td colSpan={3}>Under a different or unrecorded model; its figures are not shown against the stored result’s.</td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      {data.truncated === true && (
        <p className="note">Not every recorded search is listed here; those shown are the first in the order above.</p>
      )}
      <div className="other-searches-components">
        {searches.map((s, i) =>
          comparable(s, data.stored_delta_parameterization) && s.delta ? (
            <div key={s.search_run_id ?? `c-${i}`}>
              <h4>{KIND_LABEL[s.kind] ?? s.kind}{s.seed !== null && s.seed !== undefined && s.kind === 'seed' ? `, seed ${s.seed}` : ''}</h4>
              <Components s={s} />
            </div>
          ) : null,
        )}
      </div>
    </>
  );
}

/** Once opened it stays "opened" for the query's sake: closing and opening again shows what was fetched and asks for nothing. */
export function OtherSearches({ scenarioId }: { scenarioId: string }) {
  const [opened, setOpened] = useState(false);
  const query = useOtherSearches(scenarioId, { enabled: opened });
  return (
    <details className="provenance other-searches" onToggle={(e) => e.currentTarget.open && setOpened(true)}>
      <summary>Other searches of this scenario</summary>
      {opened && (
        <p className="note">
          Each of these is a separate search of this scenario, run with a seed, a budget or a challenger agent of its own. None is shown to
          be the smallest: each figure is the smallest perturbation that one finite search found, and another seed or budget can find a
          smaller or a larger one. A challenger’s rank is its place by distance from the SDC, not by its result, and a figure from another
          challenger agent is a perturbation of a different agent, not set beside the stored result’s. The stored result and its search
          are in Provenance above.
        </p>
      )}
      {opened && query.isPending && <p className="status" role="status">Loading other searches…</p>}
      {opened && query.isError && <RequestError error={query.error} retry={() => query.refetch()} retrying={query.isFetching} />}
      {opened && query.isSuccess && <Body data={query.data} />}
    </details>
  );
}
