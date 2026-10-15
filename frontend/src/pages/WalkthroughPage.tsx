import { useEffect, useState, type ReactNode } from 'react';
import { Link } from 'react-router';
import { useAllScenarios, useTrajectories } from '../api/scenarios';
import { RequestError } from '../components/RequestError';
import type { ScenarioSummary } from '../api/client';
import { formatMagnitude } from '../domain/format';
import { OUTCOME_FILTERS, rowOutcome } from '../domain/listView';
import { describeOutcome } from '../domain/outcomes';
import {
  firstRefused,
  largestCollision,
  qualifiesAsSmallest,
  smallestFirst,
  type Collision,
} from '../domain/walkthrough';
import { recordingLastFrame } from '../scene/sceneModel';
import { scenarioPath } from './ScenarioListPage';

/*
 * The words on the walkthrough, and what backs each. Every id, frame and figure is read from the API and put in by the
 * code below; none is written here. The scenes are chosen by rule (domain/walkthrough.ts).
 *
 *   h1 "Three stored results"
 *       the three sections below: the smallest change before the recording ends, the largest, and a refused replay
 *   "These three scenarios are chosen by rule from the stored results each time the page loads, not by hand."
 *       smallestFirst, largestCollision and firstRefused run on the rows GET /scenarios returned when the page opened
 *   "Every perturbation is the smallest found by a finite search, not a proven minimum."
 *       the same caveat as the list's caption and the Method page's first limits: min_perturbation is the best result of a
 *       Differential Evolution run with a recorded budget (src/optimization/scipy_optimizer.py optimize_scenario)
 *   "Smallest change, before the recording ends" and the sentence under it
 *       the collisions in order of min_perturbation; the first whose collision_timestep is before recordingLastFrame
 *       (scene/sceneModel.ts) of its /trajectories response. That is the frame past which the search does not look, so a
 *       collision on it is left out here: the detail page's line "This is the last recorded frame" would say so. "The
 *       recording runs to frame N" is that last frame
 *   "Checking where each recording ends…"
 *       the last frame is not in the list or in /perturbed, so GET /scenarios/{id}/trajectories is read for one candidate
 *       at a time, smallest perturbation first, until one qualifies. The response is cached under the key the detail page
 *       uses, so opening the scenario does not fetch it again
 *   "Every stored collision is on the last recorded frame, or has no geometry to check that against, so no scene is shown here."
 *       every candidate was read and none qualified: each collision frame equals the last frame, or the response had no
 *       track (a scenario whose geometry was not exported has no last frame to compare with)
 *   "Largest change that still collides" and the sentence under it
 *       largestCollision: the largest stored min_perturbation among the collisions, with no condition on the frame.
 *       "This is the scenario above." when it is the scene chosen for the smallest
 *   "A refused replay" and the sentence under it
 *       firstRefused: the first by scenario ID whose last attempt was refused (last_attempt_outcome replay_infeasible, the
 *       field GET /stats counts); the meaning is describeOutcome's (domain/outcomes.ts); "The scene has N agents" is
 *       n_agents from the row, when the API gave one
 *   "It also has a stored result from an earlier pass: {outcome}."
 *       the row's stress_outcome is not null: the result of the run that produced the stored result, which a later refused
 *       attempt does not remove (src/api/models.py, stress_outcome against last_attempt_outcome); {outcome} is its label
 *   "No stored result has a collision, so there is no scene to show here." and "No scenario is reported as replay infeasible."
 *       the rule found no row: collisions() is empty, firstRefused() is null
 */

type Smallest =
  | { status: 'probing' }
  | { status: 'error'; error: unknown; retry(): void; retrying: boolean }
  | { status: 'none' }
  | { status: 'found'; row: Collision; lastFrame: number };

/** Walks the candidates, smallest perturbation first, one /trajectories request at a time, until one qualifies. */
function useSmallestScene(candidates: Collision[] | null): Smallest {
  const [index, setIndex] = useState(0);
  const current = candidates?.[index];
  const trajectories = useTrajectories(current?.scenario_id ?? '', { enabled: current !== undefined });
  const lastFrame = trajectories.isSuccess ? recordingLastFrame(trajectories.data.agents) : null;
  const qualifies = current !== undefined && trajectories.isSuccess && qualifiesAsSmallest(current, lastFrame);

  useEffect(() => {
    // Idempotent in `index`, so a repeated effect run cannot skip a candidate.
    if (current !== undefined && trajectories.isSuccess && !qualifies) setIndex((i) => (i === index ? i + 1 : i));
  }, [index, current, trajectories.isSuccess, qualifies]);

  if (candidates === null) return { status: 'probing' };
  if (current === undefined) return { status: 'none' };
  if (trajectories.isError) {
    return { status: 'error', error: trajectories.error, retry: () => void trajectories.refetch(), retrying: trajectories.isFetching };
  }
  if (qualifies && lastFrame !== null) return { status: 'found', row: current, lastFrame };
  return { status: 'probing' };
}

function OpenLink({ id }: { id: string }) {
  return (
    <Link className="text-link" to={scenarioPath(id)} aria-label={`Open scenario ${id}`}>
      Open the replay <span aria-hidden="true">↗</span>
    </Link>
  );
}

function Card({ title, id, children }: { title: string; id: string; children: ReactNode }) {
  const headingId = `walk-${id}`;
  return (
    <section className="walk-card" aria-labelledby={headingId}>
      <h2 id={headingId}>{title}</h2>
      {children}
    </section>
  );
}

/** The label of the outcome the stored result was produced with (the same words as the Outcome filter and the badge). */
function storedLabel(row: ScenarioSummary): string {
  return OUTCOME_FILTERS.find((f) => f.value === rowOutcome(row))?.label ?? 'an outcome this interface does not recognize';
}

const NO_COLLISION = 'No stored result has a collision, so there is no scene to show here.';

export function WalkthroughPage() {
  const all = useAllScenarios();
  const candidates = all.rows === null ? null : smallestFirst(all.rows);
  const smallest = useSmallestScene(candidates);

  const largest = all.rows === null ? null : largestCollision(all.rows);
  const refused = all.rows === null ? null : firstRefused(all.rows);

  return (
    <div className="walkthrough">
      <p className="eyebrow">WALKTHROUGH</p>
      <h1>Three stored results</h1>
      <p className="caption">
        These three scenarios are chosen by rule from the stored results each time the page loads, not by hand. Every
        perturbation is the smallest found by a finite search, not a proven minimum.
      </p>

      {all.error !== null ? (
        <RequestError error={all.error} retry={all.retry} retrying={all.retrying} />
      ) : all.rows === null ? (
        <p className="status" role="status">Loading every scenario to choose from…</p>
      ) : (
        <>
          <Card title="Smallest change, before the recording ends" id="smallest">
            {smallest.status === 'probing' && (
              <p className="status" role="status">Checking where each recording ends…</p>
            )}
            {smallest.status === 'error' && (
              <RequestError
                context="A recording could not be read to find where it ends."
                error={smallest.error}
                retry={smallest.retry}
                retrying={smallest.retrying}
              />
            )}
            {smallest.status === 'none' && (
              <p>
                {candidates!.length === 0
                  ? NO_COLLISION
                  : 'Every stored collision is on the last recorded frame, or has no geometry to check that against, so no scene is shown here.'}
              </p>
            )}
            {smallest.status === 'found' && (
              <>
                <p>
                  Scenario <strong className="mono">{smallest.row.scenario_id}</strong> has the smallest stored perturbation,{' '}
                  {formatMagnitude(smallest.row.min_perturbation)}, among the collisions that happen before the last recorded
                  frame. The collision is on frame {smallest.row.collision_timestep}; the recording runs to frame{' '}
                  {smallest.lastFrame}.
                </p>
                <OpenLink id={smallest.row.scenario_id} />
              </>
            )}
          </Card>

          <Card title="Largest change that still collides" id="largest">
            {largest === null ? (
              <p>{NO_COLLISION}</p>
            ) : (
              <>
                <p>
                  Scenario <strong className="mono">{largest.scenario_id}</strong> has the largest stored perturbation,{' '}
                  {formatMagnitude(largest.min_perturbation)}, among all the collisions. The collision is on frame{' '}
                  {largest.collision_timestep}.
                  {smallest.status === 'found' && smallest.row.scenario_id === largest.scenario_id && ' This is the scenario above.'}
                </p>
                <OpenLink id={largest.scenario_id} />
              </>
            )}
          </Card>

          <Card title="A refused replay" id="refused">
            {refused === null ? (
              <p>No scenario is reported as replay infeasible.</p>
            ) : (
              <>
                <p>
                  Scenario <strong className="mono">{refused.scenario_id}</strong> is the first, by scenario ID, that the API
                  reports as replay infeasible. {describeOutcome('replay_infeasible').meaning}
                  {typeof refused.n_agents === 'number' && ` The scene has ${refused.n_agents} agents.`}
                  {refused.stress_outcome !== null && ` It also has a stored result from an earlier pass: ${storedLabel(refused)}.`}
                </p>
                <OpenLink id={refused.scenario_id} />
              </>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
