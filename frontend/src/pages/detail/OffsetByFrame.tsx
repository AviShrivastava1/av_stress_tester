import { useState } from 'react';
import type { PerturbedResponse } from '../../api/client';
import { formatMagnitude } from '../../domain/format';
import { frameOffsets, summariseOffsets } from '../../scene/offsets';

/*
 * The words under the stored perturbation's components, and what backs each. The figures are read from the two tracks GET
 * /perturbed returned and put in by the code below; no number is written here. Nothing is simulated: both tracks are stored.
 *
 *   h3 "Offset from the recording, by frame" and the group's name
 *       the distance between the logged challenger (`baseline`) and the perturbed challenger (`perturbed`) at each frame
 *   "The distance between the logged challenger and the perturbed challenger on each frame where both were observed, taken
 *    from the two stored tracks (their positions only), in metres. Every other agent is as logged."
 *       scene/offsets.ts frameOffsets: Math.hypot of the two positions, only on frames both tracks have a vertex for. The
 *       positions are in the data's planar metres. "Every other agent is as logged": PerturbationSpace.apply rewrites the
 *       target's track only (src/optimization/perturbation_space.py; the same claim as scene/sceneModel.ts)
 *   "The perturbed track is a re-simulation of the challenger, so part of this distance is the replay's own drift from the log,
 *    which Run details reports as the baseline offset."
 *       the perturbed track is the output of the kinematic replay with the stored perturbation applied (PerturbationSpace.apply),
 *       not a recording, and the replay with NO perturbation is already some distance from the log: the Run details panel's
 *       "Baseline offset at collision" and "Replay drift, whole track (max)" (pages/detail/ReplayOffsetBlock.tsx, whose note
 *       says "the zero-perturbation replay of the challenger is this many metres from its logged position"; the fields are
 *       baseline_offset_at_collision and baseline_replay_error in src/api/models.py). The two parts are not separated here:
 *       the drift is stored as one figure at the collision frame and one maximum, not per frame
 *   "At the collision frame (frame N)"
 *       the offset on perturbed.collision_timestep, shown only when that is a frame both tracks share
 *   "Largest, on frame N"
 *       the largest offset over the shared frames (the earlier frame on a tie)
 *   "On the last frame both were observed (frame N)"
 *       the offset on the last shared frame
 *   the chart's name "Offset in metres by frame: from A m on frame a to B m on frame b, largest C m on frame c"
 *       the first, the last and the largest offsets, as the readouts give them; the chart (a figure whose drawing is hidden from
 *       assistive technology, so its name and the table carry the numbers) draws the same values
 *   the line under the chart, "frame {a}", "chart top: {B} m" and "frame {b}"
 *       the first and last shared frames the line runs between, and the offset the top of the chart stands for (the largest);
 *       the bottom of the chart is 0 m. They are HTML under the drawing so that they are the size of the page's other small text
 *   "Offset at every frame"
 *       the table of every shared frame, for a reader who cannot see the chart
 */

const W = 320;
const H = 80;
const PAD = { left: 4, right: 4, top: 6, bottom: 4 };

export function OffsetByFrame({ perturbed }: { perturbed: PerturbedResponse }) {
  // The table is drawn only once it is opened: it has a row for every shared frame, and a closed table is not one a reader can use.
  const [tableOpen, setTableOpen] = useState(false);
  const { baseline, perturbed: counterfactual } = perturbed;
  if (!baseline || !counterfactual) return null;
  const offsets = frameOffsets(baseline, counterfactual);
  if (offsets.length === 0) return null;
  const { first, last, largest, atCollision } = summariseOffsets(offsets, perturbed.collision_timestep ?? null);
  if (!first || !last || !largest) return null;

  const frames = [first.frame, last.frame];
  const x = (frame: number) =>
    PAD.left + (frames[1]! === frames[0]! ? 0 : ((frame - frames[0]!) / (frames[1]! - frames[0]!)) * (W - PAD.left - PAD.right));
  const top = largest.metres > 0 ? largest.metres : 1;
  const y = (metres: number) => H - PAD.bottom - (metres / top) * (H - PAD.top - PAD.bottom);
  const points = offsets.map((o) => `${x(o.frame).toFixed(2)},${y(o.metres).toFixed(2)}`).join(' ');
  const metres = (value: number) => `${formatMagnitude(value)} m`;

  return (
    <div className="offset-block" role="group" aria-label="Offset from the recording, by frame">
      <h3>Offset from the recording, by frame</h3>
      <p className="note">
        The distance between the logged challenger and the perturbed challenger on each frame where both were observed, taken
        from the two stored tracks (their positions only), in metres. Every other agent is as logged. The perturbed track is
        a re-simulation of the challenger, so part of this distance is the replay's own drift from the log, which Run details
        reports as the baseline offset.
      </p>
      <dl className="offset-readouts">
        {atCollision && (
          <div>
            <dt>At the collision frame (frame {atCollision.frame})</dt>
            <dd>{metres(atCollision.metres)}</dd>
          </div>
        )}
        <div>
          <dt>Largest, on frame {largest.frame}</dt>
          <dd>{metres(largest.metres)}</dd>
        </div>
        <div>
          <dt>On the last frame both were observed (frame {last.frame})</dt>
          <dd>{metres(last.metres)}</dd>
        </div>
      </dl>
      <figure
        className="offset-chart"
        aria-label={`Offset in metres by frame: from ${metres(first.metres)} on frame ${first.frame} to ${metres(last.metres)} on frame ${last.frame}, largest ${metres(largest.metres)} on frame ${largest.frame}`}
      >
        <svg className="offset-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
          <line className="offset-axis" x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} />
          <polyline className="offset-line" points={points} />
        </svg>
        <figcaption className="offset-scale">
          <span>frame {first.frame}</span>
          <span>chart top: {metres(largest.metres)}</span>
          <span>frame {last.frame}</span>
        </figcaption>
      </figure>
      <details className="offset-table" onToggle={(e) => setTableOpen(e.currentTarget.open)}>
        <summary>Offset at every frame</summary>
        {tableOpen && (
          <table className="delta-table">
            <thead>
              <tr>
                <th scope="col">Frame</th>
                <th scope="col" className="num">Offset (m)</th>
              </tr>
            </thead>
            <tbody>
              {offsets.map((o) => (
                <tr key={o.frame}>
                  <td>{o.frame}</td>
                  <td className="num" title={String(o.metres)}>{formatMagnitude(o.metres)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </details>
    </div>
  );
}
