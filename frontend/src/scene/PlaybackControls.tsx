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

type TransportKind = 'back' | 'play' | 'pause' | 'forward';

/*
 * The transport symbols are drawn here rather than typed as Unicode glyphs ("|◀︎", "▶︎", "❚❚", "▶︎|"), which look
 * different on every platform. The shapes are the same four symbols; the buttons' accessible names are their own
 * aria-labels below, so the drawing is aria-hidden. Filled shapes carry `transport-fill`, bars carry none.
 */
function TransportIcon({ kind }: { kind: TransportKind }) {
  return (
    <svg className="transport-icon" data-icon={kind} viewBox="0 0 20 20" aria-hidden="true">
      {kind === 'back' && <><path d="M5 4v12" /><path className="transport-fill" d="m15 5-8 5 8 5Z" /></>}
      {kind === 'play' && <path className="transport-fill" d="m7 4 9 6-9 6Z" />}
      {kind === 'pause' && <><path d="M7 5v10" /><path d="M13 5v10" /></>}
      {kind === 'forward' && <><path d="M15 4v12" /><path className="transport-fill" d="m5 5 8 5-8 5Z" /></>}
    </svg>
  );
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
          <TransportIcon kind="back" />
        </button>
        <button
          type="button"
          className="play"
          aria-label={playback.playing ? 'Pause' : 'Play'}
          disabled={frozen}
          onClick={playback.toggle}
        >
          <TransportIcon kind={playback.playing ? 'pause' : 'play'} />
        </button>
        <button
          type="button"
          aria-label="Step forward one frame"
          disabled={frozen || frame >= range.last}
          onClick={() => playback.step(1)}
        >
          <TransportIcon kind="forward" />
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
