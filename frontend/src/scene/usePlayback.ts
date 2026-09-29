import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { advance, clampFrame, DEFAULT_RATE, isStatic, startFrame, type FrameRange, type Rate } from './playback';

export interface Playback {
  /** The frame on screen, or null when there is no frame to show. */
  frame: number | null;
  /** False until the viewer plays, steps, scrubs or jumps. */
  touched: boolean;
  playing: boolean;
  rate: Rate;
  toggle(): void;
  /** Steps pause playback. */
  step(delta: number): void;
  /** Seeking — scrubbing or jumping — pauses playback. */
  seek(frame: number): void;
  setRate(rate: Rate): void;
}

/**
 * Playback state for one scene.
 *
 * Until the viewer touches the controls, the playhead IS `defaultFrame` (the static
 * view's reference frame), so it follows that frame as data arrives or the focus
 * changes — the untouched view is exactly the static view. Once touched, the
 * playhead is the viewer's and nothing moves it but them.
 *
 * Seeking while playing pauses: a playhead that kept advancing under the viewer's
 * drag would be fighting them for the control.
 */
export function usePlayback(range: FrameRange | null, defaultFrame: number | null): Playback {
  const [userFrame, setUserFrame] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState<Rate>(DEFAULT_RATE);

  const frame =
    range === null ? null : clampFrame(userFrame ?? defaultFrame ?? range.first, range);

  // The rAF loop and the handlers read the playhead through a ref, so the loop is not
  // torn down and restarted (losing its carried time) on every frame it produces.
  const frameRef = useRef(frame);
  useLayoutEffect(() => {
    frameRef.current = frame;
  }, [frame]);

  useEffect(() => {
    if (!playing || range === null || isStatic(range)) return;
    let handle = 0;
    let last: number | null = null;
    let clock = { frame: frameRef.current ?? range.first, carryMs: 0, playing: true };

    const tick = (now: number) => {
      const dt = last === null ? 0 : now - last;
      last = now;
      const next = advance(clock, dt, rate, range);
      if (next.frame !== clock.frame) setUserFrame(next.frame);
      clock = next;
      if (!next.playing) {
        setPlaying(false);
        return;
      }
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing, rate, range]);

  const toggle = useCallback(() => {
    if (range === null || isStatic(range)) return;
    if (playing) {
      setPlaying(false);
      return;
    }
    setUserFrame(startFrame(frameRef.current ?? range.first, range));
    setPlaying(true);
  }, [playing, range]);

  const seek = useCallback(
    (target: number) => {
      if (range === null) return;
      setPlaying(false);
      setUserFrame(clampFrame(target, range));
    },
    [range],
  );

  const step = useCallback(
    (delta: number) => {
      if (range === null) return;
      seek((frameRef.current ?? range.first) + delta);
    },
    [range, seek],
  );

  return { frame, touched: userFrame !== null, playing, rate, toggle, step, seek, setRate };
}
