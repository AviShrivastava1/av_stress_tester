import type { AgentTrack, PerturbedResponse, ScenarioDetail } from '../api/client';
import type { ReferenceFrame } from './geometry';

export type Role = 'other' | 'sdc' | 'challenger_logged' | 'challenger_perturbed';

export interface DrawnTrack {
  key: string;
  role: Role;
  track: AgentTrack;
}

const DRAW_ORDER: Record<Role, number> = {
  other: 0,
  sdc: 1,
  challenger_logged: 2,
  challenger_perturbed: 3,
};

/**
 * Every track the scene draws, bottom layer first.
 *
 * The challenger's logged path comes from `/perturbed`'s `baseline` when present —
 * the same response as the perturbed path it is compared against, so the two are
 * guaranteed to describe the same export. Only when that is null does the
 * `/trajectories` entry for the challenger stand in for it.
 */
export function buildScene(
  agents: AgentTrack[],
  perturbed: PerturbedResponse | undefined,
): DrawnTrack[] {
  const targetIdx = perturbed?.target_idx ?? null;
  const baseline = perturbed?.baseline ?? null;
  const drawn: DrawnTrack[] = [];

  for (const a of agents) {
    const isChallenger = targetIdx !== null && a.agent_idx === targetIdx;
    if (isChallenger && baseline !== null) continue; // drawn from /perturbed below
    drawn.push({
      key: `agent-${a.agent_idx}`,
      role: a.is_sdc ? 'sdc' : isChallenger ? 'challenger_logged' : 'other',
      track: a,
    });
  }
  if (baseline !== null) {
    drawn.push({ key: 'challenger-logged', role: 'challenger_logged', track: baseline });
  }
  if (perturbed?.perturbed) {
    drawn.push({ key: 'challenger-perturbed', role: 'challenger_perturbed', track: perturbed.perturbed });
  }
  return drawn.sort((a, b) => DRAW_ORDER[a.role] - DRAW_ORDER[b.role]);
}

/**
 * Whether the scene draws a perturbed track: the geometry has loaded and the response carries one.
 * The same condition as `buildScene`'s, named once so the page's words ("compare", "simulated
 * challenger") are only used when the comparison is on screen.
 */
export function perturbedTrackDrawn(
  agents: AgentTrack[] | undefined,
  perturbed: PerturbedResponse | undefined,
): boolean {
  return agents !== undefined && Boolean(perturbed?.perturbed);
}

export type Focus = 'interaction' | 'scene';

/** The interaction focus needs both sides of the interaction to be drawable. */
export function canFocusInteraction(drawn: DrawnTrack[]): boolean {
  return (
    drawn.some((d) => d.role === 'sdc') &&
    drawn.some((d) => d.role === 'challenger_logged' || d.role === 'challenger_perturbed')
  );
}

export function focusTracks(drawn: DrawnTrack[], focus: Focus): DrawnTrack[] {
  return focus === 'scene' ? drawn : drawn.filter((d) => d.role !== 'other');
}

/**
 * Why there is no perturbed path, in the terms the score row supports. Null when a
 * perturbed path is drawn, or when the score row has not loaded (no guessing).
 */
export function perturbedPathNote(
  detail: ScenarioDetail | undefined,
  perturbed: PerturbedResponse,
): string | null {
  if (perturbed.perturbed || detail === undefined) return null;
  if (detail.stress_outcome === 'no_collision_found') {
    return 'The search found no collision, so there is no perturbed path to draw.';
  }
  if (detail.stress_tested) {
    return 'A stored result exists, but no perturbed path matching it has been exported.';
  }
  if (detail.stress_attempted) {
    return 'A stress-test pass reached this scenario but stored no result, so there is no perturbed path.';
  }
  return 'Not stress-tested yet, so there is no perturbed path.';
}

/**
 * The last frame of the recording: the largest timestep over EVERY track /trajectories returned. Not the focused tracks,
 * which can end earlier, and not the challenger's own last observed frame, which can end earlier still: the question is
 * where the recording ends, and the search looks at no frame past it. Null when no track has a vertex.
 *
 * Why the search window is the recording: the challenger is re-simulated from its first observed frame to the end of the
 * recording (perturbation_space.py PerturbationSpace.apply, `controls = self.base_controls[self.t0:].copy()` at line 649
 * and `traj = self.simulator.simulate(init, controls, self.dt)` at line 675; simulator.py `T = len(controls) + 1` at
 * line 86; _write_back's `n = min(self.T - t0, traj.shape[0])` at line 774), and the contact check loops over the
 * recording's frames only (collision_detector.py check_collision_trajectory, `for t in range(T):` at line 115;
 * scipy_optimizer.py `for t in range(T):` at line 105). Neither has a special case for the last frame.
 */
export function recordingLastFrame(agents: AgentTrack[]): number | null {
  let last = -Infinity;
  for (const a of agents) {
    if (a.timesteps.length > 0) last = Math.max(last, ...a.timesteps);
  }
  return Number.isFinite(last) ? last : null;
}

export const LAST_FRAME_NOTE = 'This is the last recorded frame; the search does not look past it.';

/**
 * The line under "Positions at frame N, the first colliding frame." when that frame is the recording's last one: the
 * contact was found there, and the search does not look at anything later (see recordingLastFrame for the code). It is
 * shown only for the collision reference, only when its frame equals the recording's last frame, and only when that
 * frame is known.
 */
export function searchWindowNote(ref: ReferenceFrame, lastFrame: number | null): string | null {
  return ref.kind === 'collision' && lastFrame !== null && ref.frame === lastFrame ? LAST_FRAME_NOTE : null;
}

/** States which rule picked the frame. The fallback is never presented as the rule. */
export function referenceFrameNote(ref: ReferenceFrame, focusCount: number): string {
  switch (ref.kind) {
    case 'collision':
      return `Positions at frame ${ref.frame}, the first colliding frame.`;
    case 'all_observed':
      return `Positions at frame ${ref.frame}, the earliest frame where all ${focusCount} focused tracks are observed.`;
    case 'earliest_observed':
      return (
        `Positions at frame ${ref.frame}. No frame has all ${focusCount} focused tracks observed ` +
        'together, so this is the earliest frame at which any of them is observed.'
      );
    case 'none':
      return 'No observed positions to draw.';
  }
}
