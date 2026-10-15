import type { ScenarioSummary } from '../api/client';
import { rowOutcome } from './listView';

/*
 * The three scenes the walkthrough shows, chosen by rule from the rows GET /scenarios returns. No scenario id and no number
 * is written into the page or into this file; change the stored data and the choice changes with it.
 *
 *   a collision
 *       a row whose stored outcome is collision_found (the same outcome the Search result column shows, domain/listView.ts
 *       rowOutcome) with a stored min_perturbation and a stored collision_timestep. The outcome decides it: a row with a
 *       later refused attempt but a stored collision is still a collision
 *   the smallest
 *       the collisions in order of min_perturbation, smallest first, ties by scenario ID (code point). The page walks that
 *       list and takes the first whose collision frame is before the recording's last frame (qualifies), so a collision that
 *       the search only found on the last recorded frame, where it stops looking (scene/sceneModel.ts recordingLastFrame),
 *       is not the one held up
 *   the largest
 *       the collision with the largest min_perturbation, ties by the smaller scenario ID. No condition on the frame
 *   the refused
 *       the first by scenario ID (code point) among rows whose LAST ATTEMPT was refused: last_attempt_outcome is
 *       replay_infeasible. That is how the API itself counts refusals (GET /stats replay_infeasible,
 *       src/api/routes.py: "These describe ATTEMPTS, not stored results, so they count last_attempt_outcome"), and it is
 *       where the hosted data carries them: a refused scenario has no stored outcome. A row refused on its last attempt that
 *       still holds an earlier result is included, and the page says it holds one
 *
 * min_perturbation is compared as the list's column is: the weighted norm the search minimised, which the API serves for
 * every model in the same field.
 */

const compareIds = (a: ScenarioSummary, b: ScenarioSummary) =>
  a.scenario_id < b.scenario_id ? -1 : a.scenario_id > b.scenario_id ? 1 : 0;

export interface Collision extends ScenarioSummary {
  min_perturbation: number;
  collision_timestep: number;
}

export function collisions(rows: readonly ScenarioSummary[]): Collision[] {
  return rows.filter(
    (r): r is Collision =>
      rowOutcome(r) === 'collision_found' &&
      typeof r.min_perturbation === 'number' &&
      typeof r.collision_timestep === 'number',
  );
}

/** The collisions in the order the smallest is looked for: smallest perturbation first, ties by scenario ID. */
export function smallestFirst(rows: readonly ScenarioSummary[]): Collision[] {
  return [...collisions(rows)].sort((a, b) => a.min_perturbation - b.min_perturbation || compareIds(a, b));
}

export function largestCollision(rows: readonly ScenarioSummary[]): Collision | null {
  let best: Collision | null = null;
  for (const c of collisions(rows)) {
    if (best === null || c.min_perturbation > best.min_perturbation || (c.min_perturbation === best.min_perturbation && compareIds(c, best) < 0)) {
      best = c;
    }
  }
  return best;
}

export function firstRefused(rows: readonly ScenarioSummary[]): ScenarioSummary | null {
  const refused = rows.filter((r) => r.last_attempt_outcome === 'replay_infeasible').sort(compareIds);
  return refused[0] ?? null;
}

/** The collision is before the recording's last frame; with no last frame known (no geometry) it cannot be shown to be. */
export function qualifiesAsSmallest(collision: Collision, lastFrame: number | null): boolean {
  return lastFrame !== null && collision.collision_timestep < lastFrame;
}
