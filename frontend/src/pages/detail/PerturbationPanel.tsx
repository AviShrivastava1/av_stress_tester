import type { UseQueryResult } from '@tanstack/react-query';
import type { PerturbedResponse } from '../../api/client';
import { RequestError } from '../../components/RequestError';
import { formatMagnitude } from '../../domain/format';

/**
 * The raw perturbation vector. `delta` and `delta_labels` both come from /perturbed,
 * which builds them from one read of the score row, so a delta is never paired with
 * labels from a different read.
 */
export function PerturbationPanel({ perturbed }: { perturbed: UseQueryResult<PerturbedResponse> }) {
  return (
    <section className="panel">
      <p className="panel-kicker">SEARCH RESULT</p>
      <h2>Perturbation</h2>
      <PerturbationBody perturbed={perturbed} />
    </section>
  );
}

function PerturbationBody({ perturbed }: { perturbed: UseQueryResult<PerturbedResponse> }) {
  if (perturbed.isPending) return <p className="status" role="status">Loading perturbation…</p>;
  if (perturbed.isError) {
    return <RequestError error={perturbed.error} retry={() => perturbed.refetch()} retrying={perturbed.isFetching} />;
  }

  const { delta, delta_labels: labels } = perturbed.data;
  if (delta === null || delta === undefined) {
    return <p className="status">No stored perturbation.</p>;
  }

  // Labels are usable only if there is exactly one per component; anything else is
  // as unknown as no labels at all, and units must not be guessed.
  const labelled = labels !== null && labels !== undefined && labels.length === delta.length;

  return (
    <>
      {!labelled && (
        <p className="note" data-testid="parameterization-unknown">
          Parameterization unknown: the API cannot say which kinematic model produced this
          vector, so its components are shown without names or units.
        </p>
      )}
      <table className="delta-table">
        <tbody>
          {delta.map((value, i) => (
            <tr key={i}>
              <th scope="row">{labelled ? labels[i] : `component ${i + 1}`}</th>
              <td className="num" title={String(value)}>{formatMagnitude(value, 4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
