import { useState, type ReactNode } from 'react';
import type { PipelineRun, ProvenanceResponse, SearchRecord } from '../../api/client';
import { RequestError } from '../../components/RequestError';
import { useProvenance } from '../../api/scenarios';
import { formatMagnitude } from '../../domain/format';
import { TraceFigure } from './TraceFigure';

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
 *   "Run manifest" and its rows
 *       ProvenanceResponse.run: the manifest of the run that stored this result (src/scoring/manifest.py writes it, found through
 *       pipeline_run_results by scenario and stress_run_id; src/api/models.py PipelineRun). Kind and status; when it started and
 *       finished (UTC); the git revision (shown as its first characters, the whole one as the tooltip) and whether the working tree
 *       had changes when the run started ("Has changes the revision does not contain": the revision alone then does not reproduce
 *       the code); the Python version and the versions of the five packages; the shard's name and sha256; the counts and the
 *       parameters it reported (labelled "of the run", "in the run" and "per scenario" so that they are not read as the search's own seed
 *       and budget above them, which are the stored result's). Only the keys the API serves through its allow-lists appear (src/api/manifest.py), and only the
 *       counts and parameters that were reported; the command line, host label and notes are never served
 *   "How the search closed in"
 *       the convergence trace of a re-run of the stored search (pages/detail/TraceFigure.tsx, GET /scenarios/{id}/trace). Asked for
 *       only once the provenance has loaded and holds a search with a run id; its words and what backs them are in that file
 *   "Not recorded for this run"
 *       a field of the manifest that is null: the run did not record it (a Colab session that is not a git checkout has no
 *       revision; a shard that was not hashed has no checksum). It is never shown as none or zero
 *   "No run manifest is recorded for this result."
 *       run is null: no run is linked to this result's (scenario, stress_run_id). That is every result stored before manifests
 *       existed, and a result stored by a run that did not record one; none is invented for them
 */

const NOT_RECORDED = 'Not recorded for this result';
const NOT_RECORDED_FOR_RUN = 'Not recorded for this run';

function Missing() {
  return <span className="muted">{NOT_RECORDED}</span>;
}

/** An ISO instant as "YYYY-MM-DD HH:MM UTC"; anything that is not an instant is shown as it came. */
export function formatUtc(iso: string): string {
  const moment = new Date(iso);
  if (Number.isNaN(moment.getTime())) return iso;
  return `${moment.toISOString().slice(0, 16).replace('T', ' ')} UTC`;
}

function MissingForRun() {
  return <span className="muted">{NOT_RECORDED_FOR_RUN}</span>;
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

const COUNT_LABELS: [keyof NonNullable<PipelineRun['counts']>, string][] = [
  ['scenarios', 'Scenarios in the run'], ['scored', 'Scenarios scored'], ['searched', 'Scenarios searched'], ['refused', 'Scenarios refused'], ['errors', 'Errors'],
  ['stored_results', 'Results stored'], ['collisions_found', 'Collisions found'], ['no_collision_found', 'Searches with no collision found'],
];
const PARAMETER_LABELS: [keyof Omit<NonNullable<PipelineRun['parameters']>, 'seeds' | 'de_tol'>, string][] = [
  ['de_seed', 'Seed of the run'], ['de_popsize', 'Population size of the run'], ['de_maxiter', 'Maximum iterations of the run'], ['challengers_k', 'Challengers per scenario'],
];
const PACKAGE_LABELS: [keyof NonNullable<PipelineRun['package_versions']>, string][] = [
  ['python', 'Python'], ['numpy', 'numpy'], ['scipy', 'scipy'], ['shapely', 'shapely'], ['torch', 'torch'], ['psycopg2', 'psycopg2'],
];

const forRun = (value: string | null | undefined, mono = false) =>
  value === null || value === undefined ? <MissingForRun /> : mono ? <span className="mono">{value}</span> : value;

/** The manifest of the run that stored the result, or the plain statement that none is recorded. */
function RunManifest({ run }: { run: PipelineRun | null | undefined }) {
  if (run === null || run === undefined) {
    return <p className="note">No run manifest is recorded for this result.</p>;
  }
  const counts = run.counts ?? null;
  const parameters = run.parameters ?? null;
  const reportedCounts = counts === null ? [] : COUNT_LABELS.filter(([key]) => counts[key] !== null && counts[key] !== undefined);
  const reportedParameters: [string, string][] = [];
  if (parameters !== null) {
    for (const [key, label] of PARAMETER_LABELS) {
      const value = parameters[key];
      if (value !== null && value !== undefined) reportedParameters.push([label, String(value)]);
    }
    if (parameters.de_tol !== null && parameters.de_tol !== undefined) reportedParameters.push(['Convergence tolerance of the run', formatMagnitude(parameters.de_tol, 4)]);
    if (parameters.seeds !== null && parameters.seeds !== undefined) reportedParameters.push(['Seeds of the run', parameters.seeds.join(', ')]);
  }
  return (
    <>
      <h3>Run manifest</h3>
      <dl className="facts">
        <Row label="Run">{forRun(run.kind)}, {forRun(run.status)}</Row>
        <Row label="Started">{when(run.started_at)}</Row>
        <Row label="Finished">{run.finished_at === null || run.finished_at === undefined ? <MissingForRun /> : when(run.finished_at)}</Row>
        <Row label="Code revision">
          {run.git_revision === null || run.git_revision === undefined ? <MissingForRun /> : <span className="mono" title={run.git_revision}>{run.git_revision.slice(0, 12)}</span>}
        </Row>
        <Row label="Working tree">
          {run.git_dirty === null || run.git_dirty === undefined ? <MissingForRun /> : run.git_dirty ? 'Has changes the revision does not contain' : 'Matched the revision'}
        </Row>
        {PACKAGE_LABELS.map(([key, label]) => (
          <Row key={key} label={label}>{forRun(run.package_versions?.[key], true)}</Row>
        ))}
        <Row label="Shard">{forRun(run.shard_name, true)}</Row>
        <Row label="Shard checksum (sha256)">{forRun(run.shard_sha256, true)}</Row>
      </dl>
      <h3>Run parameters and counts</h3>
      {reportedParameters.length === 0 && reportedCounts.length === 0 ? (
        <p className="note">{NOT_RECORDED_FOR_RUN}.</p>
      ) : (
        <dl className="facts">
          {reportedParameters.map(([label, value]) => (
            <Row key={`p-${label}`} label={label}>{value}</Row>
          ))}
          {reportedCounts.map(([key, label]) => (
            <Row key={`c-${key}`} label={label}>{whole(counts?.[key])}</Row>
          ))}
        </dl>
      )}
    </>
  );
}

function Body({ data, labels, scenarioId }: { data: ProvenanceResponse; labels: string[] | null | undefined; scenarioId: string }) {
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
        <RunManifest run={data.run} />
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
      {data.stress_run_id !== null && data.stress_run_id !== undefined && <TraceFigure scenarioId={scenarioId} />}
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
      <RunManifest run={data.run} />
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
      {opened && provenance.isSuccess && <Body data={provenance.data} labels={labels} scenarioId={scenarioId} />}
    </details>
  );
}
