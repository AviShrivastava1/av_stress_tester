import { useId } from 'react';
import { isStatic, RATES, type FrameRange, type Rate } from './playback';
import type { Playback } from './usePlayback';

export interface PlaybackControlsProps {
  range: FrameRange;
  playback: Playback;
  /** Tracks observed at the current frame, and tracks drawn. */
  observed: number;
  total: number;
  /** Gated like the footprint marking: null unless the perturbed path is drawn. */
  collisionFrame: number | null;
}

export function PlaybackControls({ range, playback, observed, total, collisionFrame }: PlaybackControlsProps) {
  const ticksId = useId();
  const frozen = isStatic(range);
  const frame = playback.frame ?? range.first;
  const collisionInRange =
    collisionFrame !== null && collisionFrame >= range.first && collisionFrame <= range.last;

  return (
    <div className="playback">
      <div className="playback-row">
        <button
          type="button"
          aria-label="Step back one frame"
          disabled={frozen || frame <= range.first}
          onClick={() => playback.step(-1)}
        >
          |◀︎
        </button>
        <button
          type="button"
          className="play"
          aria-label={playback.playing ? 'Pause' : 'Play'}
          disabled={frozen}
          onClick={playback.toggle}
        >
          {playback.playing ? '❚❚' : '▶︎'}
        </button>
        <button
          type="button"
          aria-label="Step forward one frame"
          disabled={frozen || frame >= range.last}
          onClick={() => playback.step(1)}
        >
          ▶︎|
        </button>
        <input
          type="range"
          className="scrubber"
          aria-label="Frame"
          min={range.first}
          max={range.last}
          step={1}
          value={frame}
          disabled={frozen}
          list={collisionInRange ? ticksId : undefined}
          onChange={(e) => playback.seek(Number(e.target.value))}
        />
        {collisionInRange && (
          <datalist id={ticksId}>
            <option value={collisionFrame} label="collision" />
          </datalist>
        )}
      </div>
      <div className="playback-row">
        <span className="readout" data-testid="frame-readout">
          {frozen
            ? `frame ${frame}, the only frame`
            : `frame ${frame} of ${range.first}–${range.last}`}{' '}
          · {observed} of {total} tracks observed
        </span>
        {collisionInRange && (
          <button type="button" className="link-button" onClick={() => playback.seek(collisionFrame)}>
            Jump to collision (frame {collisionFrame})
          </button>
        )}
        <label className="rate">
          <select
            aria-label="Playback rate"
            value={playback.rate}
            disabled={frozen}
            onChange={(e) => playback.setRate(Number(e.target.value) as Rate)}
          >
            {RATES.map((r) => (
              <option key={r} value={r}>
                {r} frames/s
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
