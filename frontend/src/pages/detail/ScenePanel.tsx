import type { UseQueryResult } from '@tanstack/react-query';
import { useState } from 'react';
import type { PerturbedResponse, ScenarioDetail, TrajectoryResponse } from '../../api/client';
import { describeError } from '../../api/errors';
import { describeAgentType, type AgentKind } from '../../domain/agentTypes';
import { chooseReferenceFrame } from '../../scene/geometry';
import {
  buildScene,
  canFocusInteraction,
  focusTracks,
  perturbedPathNote,
  referenceFrameNote,
  type DrawnTrack,
  type Focus,
  type Role,
} from '../../scene/sceneModel';
import { SceneView } from '../../scene/SceneView';

const GEOMETRY_ABSENT_REASON =
  'Either it has not been exported yet, or it was exported from a scene that no longer ' +
  'matches the stored score; the API does not distinguish the two.';
const NO_GEOMETRY = `No geometry to draw. ${GEOMETRY_ABSENT_REASON}`;

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

export function ScenePanel(props: ScenePanelProps) {
  return (
    <section className="panel panel-scene">
      <h2>Scene</h2>
      <SceneBody {...props} />
    </section>
  );
}

function SceneBody({ trajectories, perturbed, detail }: ScenePanelProps) {
  const [chosenFocus, setChosenFocus] = useState<Focus | null>(null);

  if (trajectories.isPending) return <p className="status" role="status">Loading geometry…</p>;
  if (trajectories.isError) {
    // Includes the API's deliberate 500 for inconsistent geometry: an error, shown as
    // one, never folded into the "nothing exported" empty state.
    return (
      <p className="status status-error" role="alert">
        Scene geometry could not be loaded. {describeError(trajectories.error)}
      </p>
    );
  }

  const perturbedData = perturbed.data;
  const drawn = buildScene(trajectories.data.agents, perturbedData);
  if (drawn.length === 0) return <p className="status">{NO_GEOMETRY}</p>;

  const hasPerturbed = drawn.some((d) => d.role === 'challenger_perturbed');
  const interactionPossible = canFocusInteraction(drawn);
  const defaultFocus: Focus = interactionPossible && hasPerturbed ? 'interaction' : 'scene';
  const focus: Focus = interactionPossible ? (chosenFocus ?? defaultFocus) : 'scene';
  const focused = focusTracks(drawn, focus);

  // The collision frame belongs to the perturbed run, so it is the reference only
  // when that run is drawn; without it, frame N shows nothing colliding.
  const collisionFrame = hasPerturbed ? (perturbedData?.collision_timestep ?? null) : null;
  const reference = chooseReferenceFrame(
    collisionFrame,
    focused.map((d) => d.track),
  );

  const notes: string[] = [referenceFrameNote(reference, focused.length)];
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
      <div className="scene-frame">
        <SceneView drawn={drawn} focus={focused} reference={reference} />
      </div>
      <Legend drawn={drawn} />
      {notes.map((n) => (
        <p key={n} className="note">
          {n}
        </p>
      ))}
    </>
  );
}
