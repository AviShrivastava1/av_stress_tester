import type { ScenarioDetail } from '../../api/client';
import { formatMagnitude } from '../../domain/format';

const NOT_RECORDED = 'Not recorded for this result';

/** A distance in metres, with the exact stored value as the tooltip. */
function Metres({ value }: { value: number }) {
  return <span title={String(value)}>{formatMagnitude(value)} m</span>;
}

/**
 * The two replay-drift distances the API reads from the stored search result: how far
 * the zero-perturbation replay of the challenger is from its logged position, at the
 * collision frame and as the largest such distance over the whole track.
 *
 * Shown only for a scenario that has a stored result. A missing figure says so; it is
 * never shown as 0. The offset of a completed search that found no collision is "None",
 * which is a different statement from "not recorded".
 *
 * These are distances in metres. Min perturbation is a weighted norm over components of
 * mixed units, so nothing here divides, subtracts or otherwise compares the two.
 */
export function ReplayOffsetBlock({ detail }: { detail: ScenarioDetail }) {
  if (!detail.stress_tested) return null;

  const offset = detail.baseline_offset_at_collision;
  const drift = detail.baseline_replay_error;

  return (
    <div className="replay-offset" data-testid="replay-offset">
      <dt>Baseline offset at collision</dt>
      <dd>
        {offset != null ? (
          <Metres value={offset} />
        ) : detail.stress_outcome === 'no_collision_found' ? (
          'None, no collision found'
        ) : (
          NOT_RECORDED
        )}
      </dd>
      <dt>Replay drift, whole track (max)</dt>
      <dd>{drift != null ? <Metres value={drift} /> : NOT_RECORDED}</dd>
      <dd className="replay-note">
        Baseline offset: at the collision frame, the zero-perturbation replay of the challenger is
        this many metres from its logged position. Replay drift: the largest such distance, in
        metres, over the whole track. Neither figure is comparable with Min perturbation, which is
        a weighted norm over components of mixed units.
      </dd>
    </div>
  );
}
