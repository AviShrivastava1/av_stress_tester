import { useState } from 'react';
import { RequestError } from '../../components/RequestError';
import { useTrace } from '../../api/scenarios';
import { formatMagnitude } from '../../domain/format';

/*
 * The words under "How the search closed in", and what backs each. The figures are read from GET /scenarios/{id}/trace
 * (src/api/routes.py get_trace; src/api/models.py TraceResponse and TracePoint); no number is written here except the tolerance,
 * which the response carries.
 *
 *   "How the search closed in" and the group's name
 *       one point per generation of Differential Evolution: how many candidates had been evaluated when it ended
 *       (n_eval_cumulative) and the smallest weighted norm of a candidate that exactly collides among all evaluated so far
 *       (best_weighted_norm, the archive of src/optimization/scipy_optimizer.py; null until a collision has been found)
 *   "from a re-run of this search with the same seed and budget"
 *       the trace is not recorded by the stored run: it is a re-run (src/scoring/search_runs.py record_trace_rerun, kind
 *       trace_rerun) whose callback only records. The search is bit for bit the same with and without it (tests/
 *       test_convergence_trace.py), and the API serves the trace only when the re-run's final delta is within `tolerance` of the
 *       stored delta (src/api/trace.py TRACE_DELTA_TOLERANCE), so what is said here is that THAT re-run ended at the stored
 *       perturbation, with the difference the response gives
 *   "It covers the differential-evolution stage only"
 *       a stored result whose stage is 'de+autograd' was refined by gradient descent after DE (batch_scorer._stress_one); the
 *       stored perturbation can then be smaller than the trace's last point
 *   "does not show that a smaller one does not exist"
 *       the search is finite and stochastic (the same limit the Method page states); where the curve goes flat is where this search
 *       stopped finding smaller ones
 *   "First collision found" / "Best after the last generation" / "Evaluations"
 *       the first point whose best_weighted_norm is not null; the last point's best_weighted_norm; the last n_eval_cumulative
 *   "No convergence trace is recorded for this result."
 *       trace is null: none recorded, or the recorded re-run ended somewhere else, or it re-ran a result that is no longer the
 *       stored one. Never shown as a flat line or a zero
 */

const W = 320;
const H = 80;
const PAD = { left: 4, right: 4, top: 6, bottom: 4 };

export function TraceFigure({ scenarioId }: { scenarioId: string }) {
  const query = useTrace(scenarioId);
  const [tableOpen, setTableOpen] = useState(false);
  if (query.isPending) return <p className="status" role="status">Loading the search trace…</p>;
  if (query.isError) return <RequestError error={query.error} retry={() => query.refetch()} retrying={query.isFetching} />;
  const { trace, max_abs_delta_difference: difference, tolerance } = query.data;
  if (!trace || trace.length === 0) return <p className="note">No convergence trace is recorded for this result.</p>;

  const found = trace.filter((p) => p.best_weighted_norm !== null && p.best_weighted_norm !== undefined);
  const first = found[0];
  const last = trace[trace.length - 1]!;
  const evals = [trace[0]!.n_eval_cumulative, last.n_eval_cumulative];
  const x = (n: number) => PAD.left + (evals[1]! === evals[0]! ? 0 : ((n - evals[0]!) / (evals[1]! - evals[0]!)) * (W - PAD.left - PAD.right));
  const top = found.length > 0 ? Math.max(...found.map((p) => p.best_weighted_norm as number)) : 1;
  const y = (norm: number) => H - PAD.bottom - (norm / (top > 0 ? top : 1)) * (H - PAD.top - PAD.bottom);
  const points = found.map((p) => `${x(p.n_eval_cumulative).toFixed(2)},${y(p.best_weighted_norm as number).toFixed(2)}`).join(' ');
  const norm = (value: number) => formatMagnitude(value);

  return (
    <div className="offset-block" role="group" aria-label="How the search closed in">
      <h3>How the search closed in</h3>
      <p className="note">
        After each generation of the differential-evolution search, the smallest perturbation found so far that exactly collides,
        as a weighted norm. It is read from a re-run of this search with the same seed and budget, which ended at the stored
        perturbation (the largest difference in any component is {difference === null || difference === undefined ? 'not recorded' : formatMagnitude(difference, 9)};
        a trace is shown only when it is below {tolerance}). It covers the differential-evolution stage only, so a perturbation
        refined afterwards can be smaller than the last point. Where the curve goes flat is where this search stopped finding
        smaller ones; it does not show that a smaller one does not exist.
      </p>
      <dl className="offset-readouts">
        <div>
          <dt>First collision found</dt>
          <dd>
            {first ? `generation ${first.generation}, after ${first.n_eval_cumulative} evaluations, at ${norm(first.best_weighted_norm as number)}` : 'none in any generation'}
          </dd>
        </div>
        <div>
          <dt>Best after the last generation (generation {last.generation})</dt>
          <dd>{last.best_weighted_norm === null || last.best_weighted_norm === undefined ? 'none found' : norm(last.best_weighted_norm)}</dd>
        </div>
        <div>
          <dt>Evaluations</dt>
          <dd>{last.n_eval_cumulative}</dd>
        </div>
      </dl>
      {found.length > 0 && (
        <figure
          className="offset-chart"
          aria-label={`Smallest colliding weighted norm found so far, by evaluations spent: ${norm(first!.best_weighted_norm as number)} after ${first!.n_eval_cumulative} evaluations, ${
            last.best_weighted_norm === null || last.best_weighted_norm === undefined ? 'none' : norm(last.best_weighted_norm)
          } after ${last.n_eval_cumulative}`}
        >
          <svg className="offset-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
            <line className="offset-axis" x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} />
            <polyline className="offset-line" points={points} />
          </svg>
          <figcaption className="offset-scale">
            <span>{evals[0]} evaluations</span>
            <span>chart top: {norm(top)}</span>
            <span>{evals[1]} evaluations</span>
          </figcaption>
        </figure>
      )}
      <details className="offset-table" onToggle={(e) => setTableOpen(e.currentTarget.open)}>
        <summary>Best after every generation</summary>
        {tableOpen && (
          <table className="delta-table">
            <thead>
              <tr>
                <th scope="col">Generation</th>
                <th scope="col" className="num">Evaluations so far</th>
                <th scope="col" className="num">Smallest colliding weighted norm</th>
              </tr>
            </thead>
            <tbody>
              {trace.map((p) => (
                <tr key={p.generation}>
                  <td>{p.generation}</td>
                  <td className="num">{p.n_eval_cumulative}</td>
                  <td className="num" title={p.best_weighted_norm === null || p.best_weighted_norm === undefined ? undefined : String(p.best_weighted_norm)}>
                    {p.best_weighted_norm === null || p.best_weighted_norm === undefined ? 'none yet' : norm(p.best_weighted_norm)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </details>
    </div>
  );
}
