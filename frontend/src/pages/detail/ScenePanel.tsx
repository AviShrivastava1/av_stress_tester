import type { UseQueryResult } from '@tanstack/react-query';
import { useMemo, useState, type KeyboardEvent } from 'react';
import type { PerturbedResponse, ScenarioDetail, TrajectoryResponse } from '../../api/client';
import { describeError } from '../../api/errors';
import { describeAgentType, type AgentKind } from '../../domain/agentTypes';
import { chooseReferenceFrame, perFrameSizes, poseAt } from '../../scene/geometry';
import { PlaybackControls } from '../../scene/PlaybackControls';
import { frameRange } from '../../scene/playback';
import {
  buildScene,
  canFocusInteraction,
  focusTracks,
  perturbedPathNote,
  perturbedTrackDrawn,
  referenceFrameNote,
  type DrawnTrack,
  type Focus,
  type Role,
} from '../../scene/sceneModel';
import { SceneView } from '../../scene/SceneView';
import { usePlayback } from '../../scene/usePlayback';
import { RequestError } from '../../components/RequestError';

/**
 * How the colliding pair's boxes are drawn, in three cases. Shown only where the page marks a
 * colliding pair: the perturbed path is drawn AND a collision frame is set.
 *
 * A track exported without per-frame sizes (or whose arrays do not fit its path) is drawn at ONE
 * size, from its first observed frame. The search's exact collision check used each agent's size
 * in every frame, so at the collision frame such boxes may not touch. A track with per-frame
 * sizes is drawn at the size stored for the frame shown, which is the size the check used.
 *
 * The pair is the SDC and the perturbed challenger, whichever of them are drawn.
 */
const BOX_SIZE_NOTE_SCALAR =
  "Boxes are drawn at each agent's size in its first observed frame. The collision check used " +
  "each agent's size in every frame, so at the collision frame the drawn boxes may not touch. " +
  'Contact is decided at the precision of the stored positions, up to about a millimetre at the ' +
  'coordinates in this data.';
const BOX_SIZE_NOTE_PER_FRAME =
  "Boxes are drawn at each agent's stored size for the frame shown, the size the collision check used. " +
  'Contact is decided at the precision of the stored positions, up to about a millimetre at the ' +
  'coordinates in this data.';
const BOX_SIZE_NOTE_MIXED =
  "Some boxes are drawn at the agent's size in its first observed frame, because no per-frame sizes " +
  "are stored for that agent. The collision check used each agent's size in every frame, so at the " +
  'collision frame the drawn boxes may not touch. Contact is decided at the precision of the stored ' +
  'positions, up to about a millimetre at the coordinates in this data.';

function boxSizeNote(drawn: DrawnTrack[]): string {
  const pair = drawn.filter((d) => d.role === 'sdc' || d.role === 'challenger_perturbed');
  const perFrame = pair.filter((d) => perFrameSizes(d.track) !== null).length;
  if (perFrame === 0) return BOX_SIZE_NOTE_SCALAR;
  return perFrame === pair.length ? BOX_SIZE_NOTE_PER_FRAME : BOX_SIZE_NOTE_MIXED;
}

const GEOMETRY_ABSENT_REASON =
  'Either it has not been exported yet, or it was exported from a scene that no longer ' +
  'matches the stored score; the API does not distinguish the two.';
const NO_GEOMETRY = `No geometry to draw. ${GEOMETRY_ABSENT_REASON}`;
const NO_FRAMES = 'Tracks were returned, but none has an observed frame, so there is nothing to draw.';

const ROLE_LEGEND: Record<Exclude<Role, 'other'>, string> = {
  sdc: 'SDC (self-driving car)',
  challenger_logged: 'Challenger, as logged',
  challenger_perturbed: 'Challenger, perturbed',
};

function Legend({ drawn }: { drawn: DrawnTrack[] }) {
  const roles = (Object.keys(ROLE_LEGEND) as (keyof typeof ROLE_LEGEND)[]).filter((r) =>
    drawn.some((d) => d.role === r),
  );
  const kinds = [
    ...new Set(
      drawn.filter((d) => d.role === 'other').map((d) => describeAgentType(d.track.agent_type).kind),
    ),
  ] as AgentKind[];
  return (
    <ul className="legend">
      {roles.map((r) => (
        <li key={r}>
          <span className={`swatch role-${r}`} aria-hidden="true" />
          {ROLE_LEGEND[r]}
        </li>
      ))}
      {kinds.map((k) => (
        <li key={k}>
          <span className={`swatch role-other kind-${k}`} aria-hidden="true" />
          Other agent: {k === 'unknown' ? 'unknown type' : k}
        </li>
      ))}
    </ul>
  );
}

export interface ScenePanelProps {
  trajectories: UseQueryResult<TrajectoryResponse>;
  perturbed: UseQueryResult<PerturbedResponse>;
  detail: ScenarioDetail | undefined;
}

/**
 * The interval between frames in the data: Waymo Open Motion Dataset scenarios are recorded at
 * 10 Hz, the same `DT = 0.1` the backend's replay models and PET use (src/physics/bicycle_model.py,
 * src/danger/pet_engine.py). It is the data's spacing, not a playback rate: playback rates are
 * offered in frames per second (scene/playback.ts) and are not real time.
 */
const FRAME_INTERVAL_S = 0.1;

export function ScenePanel(props: ScenePanelProps) {
  // "Recorded vs perturbed" only when a perturbed track is drawn; otherwise it is just the recording.
  const comparing = perturbedTrackDrawn(props.trajectories.data?.agents, props.perturbed.data);
  return (
    <section className="panel panel-scene">
      <div className="scene-panel-heading">
        <div>
          <p className="panel-kicker">{comparing ? 'RECORDED VS PERTURBED, FRAME BY FRAME' : 'RECORDED SCENE'}</p>
          <h2>Replay</h2>
        </div>
        <span>{FRAME_INTERVAL_S} s / frame</span>
      </div>
      <SceneBody {...props} />
    </section>
  );
}

function SceneBody({ trajectories, perturbed, detail }: ScenePanelProps) {
  const [chosenFocus, setChosenFocus] = useState<Focus | null>(null);

  // Memoized so the scene, its framing and its frame range keep their identity across
  // playback frames; the static layer re-renders only when one of them really changes.
  const agents = trajectories.data?.agents;
  const perturbedData = perturbed.data;
  const drawn = useMemo(() => (agents ? buildScene(agents, perturbedData) : []), [agents, perturbedData]);

  const hasPerturbed = drawn.some((d) => d.role === 'challenger_perturbed');
  const interactionPossible = canFocusInteraction(drawn);
  const defaultFocus: Focus = interactionPossible && hasPerturbed ? 'interaction' : 'scene';
  const focus: Focus = interactionPossible ? (chosenFocus ?? defaultFocus) : 'scene';
  const focused = useMemo(() => focusTracks(drawn, focus), [drawn, focus]);
  const range = useMemo(() => frameRange(drawn.map((d) => d.track)), [drawn]);

  // The collision frame belongs to the perturbed run, so it is the reference only
  // when that run is drawn; without it, frame N shows nothing colliding.
  const collisionFrame = hasPerturbed ? (perturbedData?.collision_timestep ?? null) : null;
  const reference = useMemo(
    () => chooseReferenceFrame(collisionFrame, focused.map((d) => d.track)),
    [collisionFrame, focused],
  );
  const playback = usePlayback(range, reference.kind === 'none' ? null : reference.frame);

  if (trajectories.isPending) return <p className="status" role="status">Loading geometry…</p>;
  if (trajectories.isError) {
    // Includes the API's deliberate 500 for inconsistent geometry: an error, shown as
    // one, never folded into the "nothing exported" empty state.
    return (
      <RequestError context="Scene geometry could not be loaded." error={trajectories.error}
        retry={() => trajectories.refetch()} retrying={trajectories.isFetching} />
    );
  }
  if (drawn.length === 0) return <p className="status">{NO_GEOMETRY}</p>;
  if (range === null) return <p className="status">{NO_FRAMES}</p>;

  const frame = playback.frame;
  const observed = frame === null ? 0 : drawn.filter((d) => poseAt(d.track, frame) !== null).length;

  // The reference note explains the frame the view OPENED on; once the viewer has
  // moved the playhead it no longer describes what is on screen.
  const notes: string[] = playback.touched ? [] : [referenceFrameNote(reference, focused.length)];
  if (trajectories.data.agents.length === 0) {
    notes.push(`Only the challenger is drawn: no other agent geometry. ${GEOMETRY_ABSENT_REASON}`);
  }
  if (perturbed.isPending) notes.push('Loading the perturbed path…');
  else if (perturbed.isError) {
    notes.push(`The perturbed path could not be loaded. ${describeError(perturbed.error)}`);
  } else {
    const note = perturbedPathNote(detail, perturbed.data);
    if (note !== null) notes.push(note);
  }
  // Not tied to the playhead: unlike the reference-frame note above, it describes how every frame is drawn.
  if (collisionFrame !== null) notes.push(boxSizeNote(drawn));

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    // Bound to the scene frame only; the scrubber and buttons sit outside it and keep
    // their own keys.
    if (e.key === ' ') {
      e.preventDefault();
      playback.toggle();
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      playback.step(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      playback.step(1);
    }
  }

  return (
    <>
      <div className="scene-toolbar">
        <div className="segmented" role="group" aria-label="Framing">
          <button
            type="button"
            aria-pressed={focus === 'interaction'}
            disabled={!interactionPossible}
            title={interactionPossible ? undefined : 'Needs both the SDC and the challenger to be drawn.'}
            onClick={() => setChosenFocus('interaction')}
          >
            Challenger &amp; SDC
          </button>
          <button type="button" aria-pressed={focus === 'scene'} onClick={() => setChosenFocus('scene')}>
            Whole scene
          </button>
        </div>
        <span className="muted">Local planar metres</span>
      </div>
      <div
        className="scene-frame"
        tabIndex={0}
        aria-label="Scene. Space plays or pauses; the arrow keys step one frame."
        onKeyDown={onKeyDown}
      >
        <SceneView drawn={drawn} focus={focused} frame={frame} collisionFrame={collisionFrame} />
      </div>
      <PlaybackControls
        range={range}
        playback={playback}
        observed={observed}
        total={drawn.length}
        collisionFrame={collisionFrame}
      />
      <p className="keyboard-hint">Focus the scene, then use <kbd>Space</kbd> to play and <kbd>←</kbd> <kbd>→</kbd> to step.</p>
      <Legend drawn={drawn} />
      {notes.map((n) => (
        <p key={n} className="note">
          {n}
        </p>
      ))}
    </>
  );
}
