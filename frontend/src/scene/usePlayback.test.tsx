import { act, renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { FrameRange } from './playback';
import { usePlayback } from './usePlayback';

/*
 * The playhead's "playing" flag follows the playable range. A refetch can make the range empty (no track has a vertex)
 * or a single frame while a replay is running. The animation loop already refuses to run then, but the flag used to stay
 * true, so when a later response made the range playable again the replay started on its own, without any input. The
 * flag now goes false as soon as there is nothing to play, and only a press of Play sets it again.
 */

const RANGE: FrameRange = { first: 0, last: 10 };
const ONE_FRAME: FrameRange = { first: 4, last: 4 };

type Props = { range: FrameRange | null };
function start(initial: FrameRange | null = RANGE) {
  return renderHook(({ range }: Props) => usePlayback(range, 0), { initialProps: { range: initial } as Props });
}

describe('usePlayback: the range goes away while playing', () => {
  it('stops playing when the range becomes null', () => {
    const { result, rerender } = start();
    act(() => result.current.toggle());
    expect(result.current.playing).toBe(true);
    rerender({ range: null });
    expect(result.current.playing).toBe(false);
  });

  it('does not start again by itself when the range returns', () => {
    const { result, rerender } = start();
    act(() => result.current.toggle());
    rerender({ range: null });
    rerender({ range: RANGE });
    expect(result.current.playing).toBe(false);
    expect(result.current.frame).not.toBeNull();
  });

  it('plays again only when Play is pressed', () => {
    const { result, rerender } = start();
    act(() => result.current.toggle());
    rerender({ range: null });
    rerender({ range: RANGE });
    expect(result.current.playing).toBe(false);
    act(() => result.current.toggle());
    expect(result.current.playing).toBe(true);
  });

  it('stops when the range shrinks to a single frame, and stays stopped when it grows back', () => {
    const { result, rerender } = start();
    act(() => result.current.toggle());
    rerender({ range: ONE_FRAME });
    expect(result.current.playing).toBe(false);
    rerender({ range: RANGE });
    expect(result.current.playing).toBe(false);
  });

  it('keeps playing when the range changes but is still playable', () => {
    const { result, rerender } = start();
    act(() => result.current.toggle());
    rerender({ range: { first: 0, last: 20 } });
    expect(result.current.playing).toBe(true);
  });

  it('is not playing before anyone presses Play, with a range or without one', () => {
    expect(start(RANGE).result.current.playing).toBe(false);
    expect(start(null).result.current.playing).toBe(false);
  });

  it('ignores Play while there is no range', () => {
    const { result } = start(null);
    act(() => result.current.toggle());
    expect(result.current.playing).toBe(false);
  });
});
