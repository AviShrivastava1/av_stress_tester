import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { PlaybackControls } from './PlaybackControls';
import type { Playback } from './usePlayback';

/*
 * The transport buttons draw their symbols as inline SVG, not as Unicode glyphs, so they look the same on
 * every platform. The accessible names are the buttons' own aria-labels and must not change with the drawing.
 */

const RANGE = { first: 0, last: 90 };

function playback(over: Partial<Playback> = {}): Playback {
  return { frame: 40, touched: false, playing: false, rate: 10, toggle: vi.fn(), step: vi.fn(), seek: vi.fn(), setRate: vi.fn(), ...over };
}

function renderControls(over: Partial<Playback> = {}) {
  return render(<PlaybackControls range={RANGE} playback={playback(over)} observed={3} total={3} collisionFrame={null} />);
}

const BUTTONS = [
  ['Step back one frame', 'back'],
  ['Play', 'play'],
  ['Step forward one frame', 'forward'],
] as const;

describe('the transport buttons', () => {
  it.each(BUTTONS)('%s draws an inline SVG icon and no glyph text', (name, kind) => {
    renderControls();
    const button = screen.getByRole('button', { name });
    const icon = button.querySelector('svg.transport-icon');
    expect(icon).not.toBeNull();
    expect(icon).toHaveAttribute('aria-hidden', 'true');
    expect(icon).toHaveAttribute('data-icon', kind);
    expect(button.textContent).toBe('');
  });

  it('swaps the play icon for the pause icon, and the label with it, while playing', () => {
    const { rerender } = renderControls();
    expect(screen.getByRole('button', { name: 'Play' }).querySelector('svg')).toHaveAttribute('data-icon', 'play');
    rerender(<PlaybackControls range={RANGE} playback={playback({ playing: true })} observed={3} total={3} collisionFrame={null} />);
    const pause = screen.getByRole('button', { name: 'Pause' });
    expect(pause.querySelector('svg')).toHaveAttribute('data-icon', 'pause');
    expect(screen.queryByRole('button', { name: 'Play' })).not.toBeInTheDocument();
  });

  it('keeps the disabled states: no step back on the first frame, no step forward on the last', () => {
    const first = renderControls({ frame: 0 });
    expect(screen.getByRole('button', { name: 'Step back one frame' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Step forward one frame' })).toBeEnabled();
    first.unmount();
    renderControls({ frame: 90 });
    expect(screen.getByRole('button', { name: 'Step back one frame' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Step forward one frame' })).toBeDisabled();
  });

  it('freezes all three on a single-frame scene', () => {
    render(<PlaybackControls range={{ first: 5, last: 5 }} playback={playback({ frame: 5 })} observed={1} total={1} collisionFrame={null} />);
    const row = screen.getAllByRole('button').slice(0, 3);
    for (const b of row) expect(b).toBeDisabled();
  });
});
