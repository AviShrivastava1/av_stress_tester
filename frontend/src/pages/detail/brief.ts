import type { AgentTrack, PerturbedResponse, ScenarioDetail } from '../../api/client';
import { WAYMO_ATTRIBUTION } from '../../components/Footer';
import { formatMagnitude } from '../../domain/format';
import { OUTCOME_FILTERS, rowOutcome } from '../../domain/listView';
import { isPetOverlap, readTimeMetric } from '../../domain/metrics';
import { describeOutcome, parseOutcome } from '../../domain/outcomes';
import { collisionAnatomy } from '../../scene/anatomy';
import { frameRange } from '../../scene/playback';
import { buildScene, perturbedTrackDrawn, recordingLastFrame, searchWindowNote } from '../../scene/sceneModel';
import { LIMITS } from '../MethodPage';
import { anatomyWords, DERIVED_NOTE } from './CollisionAnatomy';
import { FRAME_INTERVAL_S } from './ScenePanel';

/*
 * The text brief for a scenario: plain text, one line per fact, built from the same values the scenario page shows. Every
 * figure is read from the responses in hand and formatted as the page formats it; none is written here. What backs each line:
 *
 *   "AV Scenario Stress-Tester: scenario {id}" and the address under it
 *       PROJECT_NAME-style heading and the scenario's own route (App.tsx); the address is the page the brief was made on
 *   "Outcome: {label}" and "; last attempt: {label}"
 *       the Search result badge (components/OutcomeBadge.tsx OutcomeCell): the stored outcome, or "outcome not recorded",
 *       "no stored result" or "not tested", and the last attempt when it differs. The labels are the Outcome filter's
 *       (domain/listView.ts). The server's robustly_safe flag is not repeated: the site's copy rules keep that word to the
 *       Corpus page
 *   "Min perturbation: {n} (a weighted norm over components of mixed units; the smallest found by a finite search, not a
 *    proven minimum)"
 *       min_perturbation as the Search result shows it (formatMagnitude). It is a weighted norm over components of mixed
 *       units (pages/detail/ReplayOffsetBlock.tsx) found within a finite search budget (the metric guide on the list), so no
 *       single unit is claimed for it; the components below carry their own
 *   "Collision frame: {n} (frames are {FRAME_INTERVAL_S} s apart)" and "Recording: frames {a} to {b}"
 *       collision_timestep, the interval between frames (ScenePanel FRAME_INTERVAL_S) and the first and last frame of the
 *       tracks /trajectories returned (scene/playback.ts frameRange); the last-frame sentence is searchWindowNote's, when
 *       the scene's collision is on that last frame
 *   "Gap before first contact: …" and the sentence under it
 *       the Gap before first contact block under the replay (pages/detail/CollisionAnatomy.tsx), its wording and its
 *       how-it-was-got sentence, only when that block is shown
 *   "Challenger: agent {n}, {a} of {b} challengers searched"
 *       perturbed.target_idx and challengers_searched / challengers_total as the page shows them
 *   "Perturbation components, as stored:" and one line each
 *       the Perturbation panel: delta with delta_labels, named only when there is exactly one label per component
 *   the Run details lines
 *       fragility_score, min_ttc and min_pet as the table shows them (components/TimeMetricCell.tsx: seconds, "none" for the
 *       999 sentinel, "not recorded" for null, "(overlap)" for a PET of zero or less), the two replay offsets (shown only
 *       for a stored result) and n_agents
 *   "HOW TO READ THIS" and the lines under it
 *       the Method page's limits, each in full (the same array)
 *   "ATTRIBUTION" and the paragraph under it
 *       WAYMO_ATTRIBUTION (components/Footer.tsx), which the licence requires in anything derived from the dataset
 */

export interface BriefInput {
  /** The address the page is served from, for the link back to it. */
  origin: string;
  scenarioId: string;
  detail: ScenarioDetail;
  perturbed: PerturbedResponse | undefined;
  /** The tracks /trajectories returned, when they have loaded. */
  agents: AgentTrack[] | undefined;
}

function labelFor(raw: string): string {
  const known = parseOutcome(raw);
  return known === null ? `unrecognized: ${raw}` : describeOutcome(known).label;
}

function outcomeLine(detail: ScenarioDetail): string {
  const key = rowOutcome(detail);
  const primary =
    key === 'unrecognized'
      ? `unrecognized: ${detail.stress_outcome}`
      : (OUTCOME_FILTERS.find((f) => f.value === key)?.label ?? key);
  const last = detail.last_attempt_outcome;
  const diverged = last !== null && last !== detail.stress_outcome;
  return `Outcome: ${primary}${diverged ? `; last attempt: ${labelFor(last)}` : ''}`;
}

function timeText(value: number | null | undefined, metric: 'ttc' | 'pet'): string {
  const reading = readTimeMetric(value);
  switch (reading.kind) {
    case 'missing':
      return 'not recorded';
    case 'no_interaction':
      return 'none';
    case 'measured':
      return `${reading.seconds.toFixed(2)} s${metric === 'pet' && isPetOverlap(reading.seconds) ? ' (overlap)' : ''}`;
  }
}

const NOT_RECORDED = 'not recorded for this result';

export function buildBrief({ origin, scenarioId, detail, perturbed, agents }: BriefInput): string {
  const lines: string[] = [
    `AV Scenario Stress-Tester: scenario ${scenarioId}`,
    `${origin}/scenarios/${encodeURIComponent(scenarioId)}`,
    '',
    'SEARCH RESULT',
    outcomeLine(detail),
  ];

  if (detail.min_perturbation != null) {
    lines.push(
      `Min perturbation: ${formatMagnitude(detail.min_perturbation)} (a weighted norm over components of mixed units; ` +
        'the smallest found by a finite search, not a proven minimum)',
    );
  }
  if (detail.collision_timestep != null) {
    lines.push(`Collision frame: ${detail.collision_timestep} (frames are ${FRAME_INTERVAL_S} s apart)`);
  }

  // What the replay under the page says about the recording and the pair, only where the page says it.
  const range = agents ? frameRange(agents) : null;
  if (range !== null && agents) {
    lines.push(`Recording: frames ${range.first} to ${range.last}`);
    const drawn = agents && perturbed ? perturbedTrackDrawn(agents, perturbed) : false;
    const collisionFrame = drawn ? (perturbed?.collision_timestep ?? null) : null;
    if (collisionFrame !== null) {
      const note = searchWindowNote({ kind: 'collision', frame: collisionFrame }, recordingLastFrame(agents));
      if (note !== null) lines.push(note);
      const scene = buildScene(agents, perturbed);
      const sdc = scene.find((d) => d.role === 'sdc');
      const challenger = scene.find((d) => d.role === 'challenger_perturbed');
      if (sdc && challenger) {
        const words = anatomyWords(collisionAnatomy(sdc.track, challenger.track, collisionFrame));
        lines.push(`Gap before first contact: ${words.text}`);
        if (words.derived) lines.push(DERIVED_NOTE);
      }
    }
  }

  const target = perturbed?.target_idx;
  const searched =
    detail.challengers_searched != null && detail.challengers_total != null
      ? `${detail.challengers_searched} of ${detail.challengers_total} challengers searched`
      : null;
  if (target != null) lines.push(`Challenger: agent ${target}${searched ? `, ${searched}` : ''}`);
  else if (searched) lines.push(`Challengers searched: ${detail.challengers_searched} of ${detail.challengers_total}`);

  const delta = perturbed?.delta;
  if (delta != null) {
    const labels = perturbed?.delta_labels;
    if (labels != null && labels.length === delta.length) {
      lines.push('Perturbation components, as stored:');
      delta.forEach((value, i) => lines.push(`  ${labels[i]}: ${formatMagnitude(value, 4)}`));
    } else {
      lines.push('Perturbation components: parameterization unknown, so they are not named and carry no units');
    }
  }

  lines.push('', 'RUN DETAILS', `Fragility: ${detail.fragility_score.toFixed(3)}`);
  lines.push(`Min TTC: ${timeText(detail.min_ttc, 'ttc')}`, `Min PET: ${timeText(detail.min_pet, 'pet')}`);
  if (detail.stress_tested) {
    const offset = detail.baseline_offset_at_collision;
    const drift = detail.baseline_replay_error;
    lines.push(
      `Baseline offset at collision: ${
        offset != null
          ? `${formatMagnitude(offset)} m`
          : detail.stress_outcome === 'no_collision_found'
            ? 'none, no collision found'
            : NOT_RECORDED
      }`,
      `Replay drift, whole track (max): ${drift != null ? `${formatMagnitude(drift)} m` : NOT_RECORDED}`,
    );
  }
  if (detail.n_agents != null) lines.push(`Agents: ${detail.n_agents}`);

  lines.push('', 'HOW TO READ THIS', ...LIMITS.map((limit) => `- ${limit}`), '', 'ATTRIBUTION', WAYMO_ATTRIBUTION);
  return lines.join('\n');
}
