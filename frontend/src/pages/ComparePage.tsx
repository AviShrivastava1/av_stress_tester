import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import { Link, useSearchParams } from 'react-router';
import type { PerturbedResponse, ScenarioDetail } from '../api/client';
import { isNotFound } from '../api/errors';
import { usePerturbed, useScenarioDetail } from '../api/scenarios';
import { OutcomeCell } from '../components/OutcomeBadge';
import { RequestError } from '../components/RequestError';
import { TimeMetricCell } from '../components/TimeMetricCell';
import { formatMagnitude } from '../domain/format';
import { scenarioPath } from './ScenarioListPage';

/*
 * The comparison page, first slice: two scenarios, side by side, from what the API already serves for each. No scenes. The
 * words, and what backs them:
 *
 *   the address /compare?a=<id>&b=<id>
 *       both ids are in the query, so a link shows the same page; changing them is a history entry (Back returns to the pair
 *       before). An id is trimmed and an empty one is ignored
 *   "Enter two scenario IDs to compare them." / "Enter a second scenario ID to compare it with {id}." / "Enter a first
 *   scenario ID to compare it with {id}."
 *       fewer than two ids in the address
 *   "Both are the same scenario."
 *       the two ids are equal
 *   the summary table (name "Summary, side by side")
 *       each row is the field of the same name on the scenario page and the list, shown the way they show it: the Search
 *       result badge (components/OutcomeBadge.tsx), fragility to three places, TTC and PET (components/TimeMetricCell.tsx),
 *       min perturbation (formatMagnitude), the collision frame, "challengers searched" (challengers_searched of
 *       challengers_total), n_agents and stress_method. A missing value is a dash, never zero
 *   "Min perturbation is a weighted norm over components of mixed units. The two figures are shown side by side; nothing here
 *   subtracts or divides them."
 *       the same statement as the Run details panel's replay-offset note (pages/detail/ReplayOffsetBlock.tsx): min_perturbation
 *       is a weighted norm over components of mixed units, and the two scenarios' norms may be over different components
 *   the components table (name "Perturbation components, side by side")
 *       the stored delta of each, from GET /scenarios/{id}/perturbed, lined up row by row ONLY when both have a delta, both have
 *       a delta_labels list with one name per component, and the two lists are IDENTICAL (the same names, the same order). The
 *       names carry the units, and a bicycle vector and a linear vector have different components (src/api/models.py
 *       DELTA_LABELS and LINEAR_DELTA_LABELS: "initial heading (rad)" against "initial vy (m/s)"), so nothing is lined up
 *       otherwise. The same four names in another order are not lined up either: an exact match is the rule
 *   "{id} has no stored perturbation."
 *       delta is null for that scenario (a scenario with no stored result, or a refused one)
 *   "At least one of these has no parameterization stored, so its components cannot be named or given units. None are lined up
 *   here."
 *       a delta whose delta_labels is null, or has a different length (pages/detail/PerturbationPanel.tsx shows such a vector
 *       as "component 1", "component 2" without units, and says the parameterization is unknown)
 *   "These two were searched with different kinematic models, so the components of their perturbation vectors differ in meaning
 *   and units. They are not lined up here; each is shown on its own."
 *       both are named, and the two lists differ
 *   "No scenario with ID {id} exists."
 *       the API's 404 for that id (as the scenario page words it)
 */

const REFUSED_DIFFERENT_MODELS =
  'These two were searched with different kinematic models, so the components of their perturbation vectors differ in meaning and ' +
  'units. They are not lined up here; each is shown on its own.';
const REFUSED_UNNAMED =
  'At least one of these has no parameterization stored, so its components cannot be named or given units. None are lined up here.';

function usePair(id: string) {
  const enabled = id !== '';
  return { id, detail: useScenarioDetail(id, { enabled }), perturbed: usePerturbed(id, { enabled }) };
}
type Pair = ReturnType<typeof usePair>;

/** The delta's component names, only when there is exactly one name for each component. */
function namesOf(p: PerturbedResponse | undefined): string[] | null {
  const delta = p?.delta;
  const labels = p?.delta_labels;
  return delta != null && labels != null && labels.length === delta.length ? labels : null;
}

const same = (x: readonly string[], y: readonly string[]) => x.length === y.length && x.every((name, i) => name === y[i]);

const dash = <span className="muted">—</span>;

function summaryRows(detail: ScenarioDetail): { label: string; value: ReactNode }[] {
  const searched =
    detail.challengers_searched != null && detail.challengers_total != null
      ? `${detail.challengers_searched} of ${detail.challengers_total}`
      : null;
  return [
    { label: 'Search result', value: <OutcomeCell row={detail} /> },
    { label: 'Fragility', value: detail.fragility_score.toFixed(3) },
    { label: 'Min TTC', value: <TimeMetricCell value={detail.min_ttc ?? null} metric="ttc" /> },
    { label: 'Min PET', value: <TimeMetricCell value={detail.min_pet ?? null} metric="pet" /> },
    { label: 'Min perturbation', value: detail.min_perturbation != null ? formatMagnitude(detail.min_perturbation) : dash },
    { label: 'Collision frame', value: detail.collision_timestep != null ? `frame ${detail.collision_timestep}` : dash },
    { label: 'Challengers searched', value: searched ?? dash },
    { label: 'Agents', value: detail.n_agents ?? dash },
    { label: 'Method', value: detail.stress_method ?? dash },
  ];
}

function OwnComponents({ id, perturbed }: { id: string; perturbed: PerturbedResponse }) {
  const delta = perturbed.delta;
  if (delta == null) return <p className="note">{id} has no stored perturbation.</p>;
  const names = namesOf(perturbed);
  return (
    <>
      {names === null && (
        <p className="note">Parameterization unknown: its components are not named and carry no units.</p>
      )}
      <table className="delta-table" aria-label={`Perturbation components of ${id}`}>
        <tbody>
          {delta.map((value, i) => (
            <tr key={i}>
              <th scope="row">{names ? names[i] : `component ${i + 1}`}</th>
              <td className="num" title={String(value)}>{formatMagnitude(value, 4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function Column({ pair, showOwnComponents }: { pair: Pair; showOwnComponents: boolean }) {
  const { id, detail, perturbed } = pair;
  let body: ReactNode = null;
  if (detail.isError && isNotFound(detail.error)) {
    body = <p className="status" role="alert">No scenario with ID {id} exists.</p>;
  } else if (detail.isError) {
    body = <RequestError error={detail.error} retry={() => void Promise.all([detail.refetch(), perturbed.refetch()])} retrying={detail.isFetching} />;
  } else if (perturbed.isError && !isNotFound(perturbed.error)) {
    body = <RequestError error={perturbed.error} retry={() => perturbed.refetch()} retrying={perturbed.isFetching} />;
  } else if (detail.isPending || perturbed.isPending) {
    body = <p className="status" role="status">Loading…</p>;
  } else if (showOwnComponents && perturbed.isSuccess) {
    body = <OwnComponents id={id} perturbed={perturbed.data} />;
  }
  return (
    <section className="compare-col" aria-label={`Scenario ${id}`}>
      <h2>
        <Link className="mono" to={scenarioPath(id)}>{id}</Link>
      </h2>
      {body}
    </section>
  );
}

export function ComparePage() {
  const [params, setParams] = useSearchParams();
  const a = (params.get('a') ?? '').trim();
  const b = (params.get('b') ?? '').trim();
  const [first, setFirst] = useState(a);
  const [second, setSecond] = useState(b);
  // The fields follow the address, so Back and a shared link fill them.
  useEffect(() => {
    setFirst(a);
    setSecond(b);
  }, [a, b]);

  const A = usePair(a);
  const B = usePair(b);

  function submit(event: FormEvent) {
    event.preventDefault();
    const x = first.trim();
    const y = second.trim();
    if (x !== '' && y !== '') setParams({ a: x, b: y }); // a history entry, so Back returns to the pair before
  }

  const both = a !== '' && b !== '';
  const detailsReady = both && A.detail.isSuccess && B.detail.isSuccess;
  const pdA = A.perturbed.isSuccess ? A.perturbed.data : undefined;
  const pdB = B.perturbed.isSuccess ? B.perturbed.data : undefined;
  const namesA = namesOf(pdA);
  const namesB = namesOf(pdB);
  const perturbedReady = detailsReady && pdA !== undefined && pdB !== undefined;
  const missing = perturbedReady ? [a, b].filter((_id, i) => (i === 0 ? pdA : pdB)!.delta == null) : [];
  const lined = perturbedReady && namesA !== null && namesB !== null && same(namesA, namesB); // a missing delta has no names, so it is never lined up
  const refusal =
    perturbedReady && missing.length === 0 && !lined ? (namesA === null || namesB === null ? REFUSED_UNNAMED : REFUSED_DIFFERENT_MODELS) : null;
  // Each shows its own components unless they are lined up in one table below.
  const showOwn = !lined;

  return (
    <section className="compare-page">
      <p className="eyebrow">COMPARE</p>
      <h1>Compare two scenarios</h1>
      <form className="compare-form" onSubmit={submit}>
        <label htmlFor="compare-a">First scenario ID</label>
        <input id="compare-a" type="text" maxLength={256} value={first} onChange={(e) => setFirst(e.target.value)} />
        <label htmlFor="compare-b">Second scenario ID</label>
        <input id="compare-b" type="text" maxLength={256} value={second} onChange={(e) => setSecond(e.target.value)} />
        <button type="submit" disabled={first.trim() === '' || second.trim() === ''}>Compare</button>
      </form>

      {a === '' && b === '' && <p className="status">Enter two scenario IDs to compare them.</p>}
      {a !== '' && b === '' && <p className="status">Enter a second scenario ID to compare it with {a}.</p>}
      {a === '' && b !== '' && <p className="status">Enter a first scenario ID to compare it with {b}.</p>}
      {both && a === b && <p className="note">Both are the same scenario.</p>}

      {(a !== '' || b !== '') && (
        <div className="compare-cols">
          {a !== '' && <Column pair={A} showOwnComponents={showOwn} />}
          {b !== '' && <Column pair={B} showOwnComponents={showOwn} />}
        </div>
      )}

      {detailsReady && (
        <>
          <table className="compare-table" aria-label="Summary, side by side">
            <thead>
              <tr>
                <th scope="col" />
                <th scope="col"><Link className="mono" to={scenarioPath(a)}>{a}</Link></th>
                <th scope="col"><Link className="mono" to={scenarioPath(b)}>{b}</Link></th>
              </tr>
            </thead>
            <tbody>
              {summaryRows(A.detail.data!).map((row, i) => (
                <tr key={row.label}>
                  <th scope="row">{row.label}</th>
                  <td>{row.value}</td>
                  <td>{summaryRows(B.detail.data!)[i]!.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">
            Min perturbation is a weighted norm over components of mixed units. The two figures are shown side by side; nothing
            here subtracts or divides them.
          </p>
        </>
      )}

      {lined && namesA && (
        <table className="compare-table" aria-label="Perturbation components, side by side">
          <thead>
            <tr>
              <th scope="col" />
              <th scope="col" className="mono">{a}</th>
              <th scope="col" className="mono">{b}</th>
            </tr>
          </thead>
          <tbody>
            {namesA.map((name, i) => (
              <tr key={name}>
                <th scope="row">{name}</th>
                <td className="num">{formatMagnitude(pdA!.delta![i]!, 4)}</td>
                <td className="num">{formatMagnitude(pdB!.delta![i]!, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {refusal && <p className="note">{refusal}</p>}
    </section>
  );
}
