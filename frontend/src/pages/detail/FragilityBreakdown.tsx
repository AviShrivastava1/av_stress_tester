import type { ScenarioDetail } from '../../api/client';
import { TimeMetricCell } from '../../components/TimeMetricCell';
import { formatMagnitude } from '../../domain/format';

/*
 * The words under the Fragility figure in Run details, and what backs each. Every number is read from the API: the API works the
 * two terms out from the stored Min TTC and Min PET (src/danger/fragility_terms.py, served as ScenarioDetail.fragility_breakdown)
 * and says whether they add up to the stored score. No weight, floor, sentinel or score is written in this file.
 *
 *   h3 "Fragility, by term" and the table under it
 *       src/danger/danger_score.py compute_danger_score with no perturbation term: each signal is 1 / max(value, floor) and
 *       the 999.0 sentinel gives 0; the fragility score is the sum of the two weighted signals, the weights being the
 *       renormalised ones (0.4/0.7 and 0.3/0.7; the response carries them). The Min TTC and Min PET in the Value column are the
 *       same SDC-restricted figures Run details already shows, in the same cell (components/TimeMetricCell.tsx)
 *   "The score ranks the recorded scene only; the search result is not part of it."
 *       src/scoring/batch_scorer.py _score_one passes min_perturbation=None to score_scenario, and nothing in
 *       src/scoring/db.py recomputes fragility_score afterwards, so the stored score has no perturbation term
 *   "Each signal is 1 divided by the time in seconds, with the time held to at least {floor} s."
 *       signal() in fragility_terms.py; the floors are ttc_floor_s and pet_floor_s in the response
 *   "At the floor" under a signal
 *       ttc_at_floor / pet_at_floor: the value is at or below the floor, a PET of zero or an overlap included, so the signal is
 *       the floor's and stops there. An overlap and a PET of exactly the floor give the same signal
 *   "none" and a part of 0 for a value that is the sentinel
 *       ttc_none / pet_none: the SDC is never in a closing pair (TTC) or shares no conflict zone (PET); nothing is added
 *   "The stored score is {a}. Its stored Min TTC and Min PET give {b}, so the breakdown does not describe it and is not shown."
 *       reproduces_stored_score is false: the response says the terms do not add up to the stored score
 *   "No breakdown: Min TTC or Min PET is not stored for this scenario."
 *       fragility_breakdown is null: either figure is null in scenario_scores
 */

function Dash() {
  return <span className="muted">—</span>;
}

export function FragilityBreakdown({ detail }: { detail: ScenarioDetail }) {
  const b = detail.fragility_breakdown;
  if (b === undefined) return null; // the API did not say; nothing to claim

  if (b === null) {
    return (
      <div className="fragility-terms" role="group" aria-label="Fragility, by term">
        <h3>Fragility, by term</h3>
        <p className="note">No breakdown: Min TTC or Min PET is not stored for this scenario.</p>
      </div>
    );
  }

  if (!b.reproduces_stored_score) {
    return (
      <div className="fragility-terms" role="group" aria-label="Fragility, by term">
        <h3>Fragility, by term</h3>
        <p className="note">
          The stored score is {formatMagnitude(detail.fragility_score)}. Its stored Min TTC and Min PET give{' '}
          {formatMagnitude(b.recomputed_score)}, so the breakdown does not describe it and is not shown.
        </p>
      </div>
    );
  }

  const rows = [
    { key: 'ttc', label: 'Min TTC', value: detail.min_ttc ?? null, metric: 'ttc' as const, signal: b.ttc_signal, weight: b.ttc_weight, term: b.ttc_term, none: b.ttc_none, floor: b.ttc_at_floor },
    { key: 'pet', label: 'Min PET', value: detail.min_pet ?? null, metric: 'pet' as const, signal: b.pet_signal, weight: b.pet_weight, term: b.pet_term, none: b.pet_none, floor: b.pet_at_floor },
  ];

  return (
    <div className="fragility-terms" role="group" aria-label="Fragility, by term">
      <h3>Fragility, by term</h3>
      <p className="note">
        The score ranks the recorded scene only; the search result is not part of it. Each signal is 1 divided by the time in
        seconds, with the time held to at least {formatMagnitude(b.ttc_floor_s)} s for TTC and {formatMagnitude(b.pet_floor_s)} s for PET.
      </p>
      <table className="delta-table fragility-table" aria-label="Fragility, by term">
        <thead>
          <tr>
            <th scope="col">Term</th>
            <th scope="col">Value</th>
            <th scope="col" className="num">Signal</th>
            <th scope="col" className="num">Weight</th>
            <th scope="col" className="num">Part of the score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <th scope="row">{r.label}</th>
              <td>
                <TimeMetricCell value={r.value} metric={r.metric} />
                {r.floor && <span className="tag">at the floor</span>}
              </td>
              <td className="num" title={String(r.signal)}>{r.none ? <Dash /> : formatMagnitude(r.signal)}</td>
              <td className="num" title={String(r.weight)}>{formatMagnitude(r.weight)}</td>
              <td className="num" title={String(r.term)}>{formatMagnitude(r.term)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <th scope="row" colSpan={4}>Fragility</th>
            <td className="num" title={String(detail.fragility_score)}>{formatMagnitude(detail.fragility_score)}</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
