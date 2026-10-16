import { useState, type ReactNode } from 'react';
import type { ProvenanceResponse, SearchRecord } from '../../api/client';
import { RequestError } from '../../components/RequestError';
import { useProvenance } from '../../api/scenarios';
import { formatMagnitude } from '../../domain/format';

/*
 * The words in the provenance panel, and what backs each. Every value is read from GET /scenarios/{id}/provenance
 * (src/api/provenance.py serves an allow-list of the stored search_provenance; src/api/models.py ProvenanceResponse and
 * SearchRecord describe each field); none is written here. A value the API sends as null is "Not recorded for this result": an
 * older result or a key the search did not write, never zero. The panel is closed until opened and asks for nothing before that.
 *
 *   "Run ID" and "Scene fingerprint"
 *       stress_run_id: the hash of the run's identity (inputs and code-contract versions, db.compute_stress_run_id);
 *       scene_fingerprint: the hash of the scene the result was searched against (db.compute_scene_fingerprint)
 *   "SDC index"
 *       sdc_idx: which agent of the scene is the self-driving car, stored with the fingerprint
 *   "Scored", "Last attempted", "Result stored" (shown in UTC)
 *       scored_at (Pass 1), stress_attempted_at (the last stress-test pass that reached it), stress_tested_at (when the stored
 *       result was written): times of the offline runs, not of any request to the API
 *   "Replay model", "Collision geometry", "Parameterization", "Challenger chosen", "Stage"
 *       replay_model_version (src/physics/simulator.py REPLAY_MODEL_VERSION), collision_geometry_version (src/danger/
 *       collision_detector.py), delta_parameterization ('bicycle' or 'linear', the model whose units the components carry),
 *       challenger_selection and selected_stage ('de' or 'de+autograd', which stage's answer is stored)
 *   "Search budget and seed"
 *       de_seed, de_popsize, de_maxiter, de_tol (scipy_optimizer.py optimize_scenario: tol is the convergence tolerance on the
 *       population spread, so it has no unit), de_n_iter and de_n_eval (what the run actually did), de_candidate_source
 *       ('de_winner' or 'archive'), de_archive_covered_all_evaluations, refine_n_iters
 *   "Bounds"
 *       bounds: one [low, high] per component of the perturbation, in the order of delta. They are named only when two sources
 *       agree on which parameterization the components belong to: the labels GET /perturbed served (the Perturbation panel's
 *       own names) and the labels this response derives from the parameterization its own provenance records
 *       (ProvenanceResponse.delta_labels, from the same table, DELTA_LABELS_BY_PARAMETERIZATION in src/api/models.py). GET
 *       /perturbed carries no parameterization to compare, only labels, so the comparison is of the two lists: the same names
 *       in the same order, and one per bound. Otherwise "component N": a bicycle's names must not sit on a linear model's
 *       bounds, and a bound is never given a unit it may not have
 *   "Admission gates"
 *       max_baseline_drift (m, BASELINE_DRIFT_REFUSE_M), max_speed_step (m/s, SPEED_STEP_REFUSE_MPS), speed_step_hold_s (s),
 *       held_speed_excess (m/s, held_speed_excess() in perturbation_space.py), a_max (the acceleration limit, m/s^2),
 *       heading_speed_floor and heading_transition_width (m/s, V_HEADING_MIN and HEADING_TRANSITION_WIDTH)
 *   "The challenger"
 *       target_min_speed (m/s, the logged speed's minimum), frames_observed, frames_below_heading_floor,
 *       frames_in_heading_transition_band, target_has_interior_gap
 *   "Not stored anywhere, so not shown"
 *       the git revision of the code, library versions and the shard file's checksum: nothing in src/scoring records them
 *       (the roadmap inventory's #10 and #20)
 */

const NOT_RECORDED = 'Not recorded for this result';

function Missing() {
  return <span className="muted">{NOT_RECORDED}</span>;
}

/** An ISO instant as "YYYY-MM-DD HH:MM UTC"; anything that is not an instant is shown as it came. */
export function formatUtc(iso: string): string {
  const moment = new Date(iso);
  if (Number.isNaN(moment.getTime())) return iso;
  return `${moment.toISOString().slice(0, 16).replace('T', ' ')} UTC`;
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </>
  );
}

const text = (value: string | null | undefined, mono = false) =>
  value === null || value === undefined ? <Missing /> : mono ? <span className="mono">{value}</span> : value;
const whole = (value: number | null | undefined) =>
  value === null || value === undefined ? <Missing /> : <span title={String(value)}>{value}</span>;
const real = (value: number | null | undefined, unit = '') =>
  value === null || value === undefined ? <Missing /> : <span title={String(value)}>{formatMagnitude(value, 4)}{unit}</span>;
const yesNo = (value: boolean | null | undefined) => (value === null || value === undefined ? <Missing /> : value ? 'Yes' : 'No');
const when = (iso: string | null | undefined) => (iso === null || iso === undefined ? <Missing /> : <span title={iso}>{formatUtc(iso)}</span>);

const sameList = (x: readonly string[], y: readonly string[]) => x.length === y.length && x.every((name, i) => name === y[i]);

function Bounds({ search, labels, recordedLabels }: { search: SearchRecord; labels: string[] | null | undefined; recordedLabels: string[] | null | undefined }) {
  const bounds = search.bounds;
  if (bounds === null || bounds === undefined) return <p className="note">Bounds: {NOT_RECORDED.toLowerCase()}.</p>;
  const named =
    labels !== null && labels !== undefined && recordedLabels !== null && recordedLabels !== undefined &&
    sameList(labels, recordedLabels) && labels.length === bounds.length;
  return (
    <table className="delta-table provenance-bounds" aria-label="Bounds of the search">
      <thead>
        <tr>
          <th scope="col">Component</th>
          <th scope="col" className="num">Low</th>
          <th scope="col" className="num">High</th>
        </tr>
      </thead>
      <tbody>
        {bounds.map(([low, high], i) => (
          <tr key={i}>
            <th scope="row">{named ? labels[i] : `component ${i + 1}`}</th>
            <td className="num" title={String(low)}>{low === undefined ? <Missing /> : formatMagnitude(low, 4)}</td>
            <td className="num" title={String(high)}>{high === undefined ? <Missing /> : formatMagnitude(high, 4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Body({ data, labels }: { data: ProvenanceResponse; labels: string[] | null | undefined }) {
  const s = data.search ?? null;
  if (s === null) {
    return (
      <>
        <dl className="facts">
          <Row label="Run ID">{text(data.stress_run_id, true)}</Row>
          <Row label="Scene fingerprint">{text(data.scene_fingerprint, true)}</Row>
          <Row label="SDC index">{whole(data.sdc_idx)}</Row>
          <Row label="Scored">{when(data.scored_at)}</Row>
          <Row label="Last attempted">{when(data.stress_attempted_at)}</Row>
          <Row label="Result stored">{when(data.stress_tested_at)}</Row>
        </dl>
        <p className="note">No search provenance is stored for this scenario, so how it was searched is not recorded.</p>
      </>
    );
  }
  return (
    <>
      <h3>Identity</h3>
      <dl className="facts">
        <Row label="Run ID">{text(data.stress_run_id, true)}</Row>
        <Row label="Scene fingerprint">{text(data.scene_fingerprint, true)}</Row>
        <Row label="SDC index">{whole(data.sdc_idx)}</Row>
      </dl>
      <h3>When</h3>
      <dl className="facts">
        <Row label="Scored">{when(data.scored_at)}</Row>
        <Row label="Last attempted">{when(data.stress_attempted_at)}</Row>
        <Row label="Result stored">{when(data.stress_tested_at)}</Row>
      </dl>
      <h3>Model and code</h3>
      <dl className="facts">
        <Row label="Replay model">{text(s.replay_model_version, true)}</Row>
        <Row label="Collision geometry">{text(s.collision_geometry_version, true)}</Row>
        <Row label="Parameterization">{text(s.delta_parameterization)}</Row>
        <Row label="Challenger chosen">{text(s.challenger_selection, true)}</Row>
        <Row label="Stage">{text(s.selected_stage)}</Row>
      </dl>
      <h3>Search budget and seed</h3>
      <dl className="facts">
        <Row label="Seed">{whole(s.de_seed)}</Row>
        <Row label="Population size">{whole(s.de_popsize)}</Row>
        <Row label="Maximum iterations">{whole(s.de_maxiter)}</Row>
        <Row label="Convergence tolerance">{real(s.de_tol)}</Row>
        <Row label="Iterations run">{whole(s.de_n_iter)}</Row>
        <Row label="Evaluations">{whole(s.de_n_eval)}</Row>
        <Row label="Winning candidate from">{text(s.de_candidate_source, true)}</Row>
        <Row label="Archive covered every evaluation">{yesNo(s.de_archive_covered_all_evaluations)}</Row>
        <Row label="Refinement iterations">{whole(s.refine_n_iters)}</Row>
      </dl>
      <h3>Bounds</h3>
      <Bounds search={s} labels={labels} recordedLabels={data.delta_labels} />
      <h3>Admission gates</h3>
      <dl className="facts">
        <Row label="Baseline drift limit">{real(s.max_baseline_drift, ' m')}</Row>
        <Row label="Speed step limit">{real(s.max_speed_step, ' m/s')}</Row>
        <Row label="Speed step held for">{real(s.speed_step_hold_s, ' s')}</Row>
        <Row label="Held speed excess">{real(s.held_speed_excess, ' m/s')}</Row>
        <Row label="Acceleration limit">{real(s.a_max, ' m/s²')}</Row>
        <Row label="Heading speed floor">{real(s.heading_speed_floor, ' m/s')}</Row>
        <Row label="Heading transition width">{real(s.heading_transition_width, ' m/s')}</Row>
      </dl>
      <h3>The challenger</h3>
      <dl className="facts">
        <Row label="Lowest logged speed">{real(s.target_min_speed, ' m/s')}</Row>
        <Row label="Frames observed">{whole(s.frames_observed)}</Row>
        <Row label="Frames below the heading floor">{whole(s.frames_below_heading_floor)}</Row>
        <Row label="Frames in the transition band">{whole(s.frames_in_heading_transition_band)}</Row>
        <Row label="Has a gap inside its track">{yesNo(s.target_has_interior_gap)}</Row>
      </dl>
      <p className="note">
        Not stored anywhere, so not shown: the version of the code, the versions of the libraries, and a checksum of the shard
        the scenario was read from.
      </p>
    </>
  );
}

/** Once opened it stays "opened" for the query's sake: closing and opening again shows what was fetched and asks for nothing. */
export function ProvenancePanel({ scenarioId, labels }: { scenarioId: string; labels: string[] | null | undefined }) {
  const [opened, setOpened] = useState(false);
  const provenance = useProvenance(scenarioId, { enabled: opened });
  return (
    <details className="provenance" onToggle={(e) => e.currentTarget.open && setOpened(true)}>
      <summary>Provenance: where this result came from</summary>
      {opened && provenance.isPending && <p className="status" role="status">Loading provenance…</p>}
      {opened && provenance.isError && (
        <RequestError error={provenance.error} retry={() => provenance.refetch()} retrying={provenance.isFetching} />
      )}
      {opened && provenance.isSuccess && <Body data={provenance.data} labels={labels} />}
    </details>
  );
}
