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
