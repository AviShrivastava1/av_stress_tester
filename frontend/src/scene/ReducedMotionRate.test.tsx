import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_RATE, RATES } from './playback';
import { usePlayback } from './usePlayback';

/*
 * The default playback rate under prefers-reduced-motion (INVENTORY.md, "Keyboard and reduced-motion replay controls": "default
 * the rate to the lowest when reduced motion is on"). The reading taken where the inventory leaves a question open ("whether
 * reduced motion should change the default rate or disable play"): only the DEFAULT changes. Play stays available, nothing
 * starts by itself (it never did), and the rate is still the viewer's to change.
 *
 *   reduced motion on        the rate starts at the lowest the control offers (RATES[0])
 *   reduced motion off, or the browser cannot say
 *                            the rate starts at DEFAULT_RATE, as before
 *   a rate the viewer picked is never replaced, whatever the setting does afterwards
 *   until the viewer picks one, the default follows the setting if it changes while the page is open
 */

type Listener = () => void;

/** A stand-in for window.matchMedia whose reduced-motion answer the test can flip. */
function stubReducedMotion(initial: boolean) {
  let reduced = initial;
  const listeners = new Set<Listener>();
  vi.stubGlobal(
    'matchMedia',
    vi.fn((query: string) => ({
      media: query,
      get matches() {
        return query === '(prefers-reduced-motion: reduce)' ? reduced : false;
      },
      addEventListener: (_type: string, listener: Listener) => listeners.add(listener),
      removeEventListener: (_type: string, listener: Listener) => listeners.delete(listener),
    })),
  );
  return {
    set(next: boolean) {
      reduced = next;
      act(() => listeners.forEach((listener) => listener()));
    },
    listeners: () => listeners.size,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

const RANGE = { first: 0, last: 90 };
const open = () => renderHook(() => usePlayback(RANGE, 0));

describe('the default rate', () => {
  it('is the lowest rate when the viewer prefers reduced motion', () => {
    stubReducedMotion(true);
    const { result } = open();
    expect(result.current.rate).toBe(RATES[0]);
    expect(RATES[0]).toBe(5);
  });

  it('is the usual default when the viewer does not', () => {
    stubReducedMotion(false);
    expect(open().result.current.rate).toBe(DEFAULT_RATE);
    expect(DEFAULT_RATE).toBe(10);
  });

  it('is the usual default when the browser has no way to say', () => {
    vi.stubGlobal('matchMedia', undefined);
    expect(open().result.current.rate).toBe(DEFAULT_RATE);
  });

  it('asks about reduced motion and nothing else', () => {
    stubReducedMotion(true);
    open();
    const asked = (window.matchMedia as unknown as ReturnType<typeof vi.fn>).mock.calls.map((c) => c[0]);
    expect(new Set(asked)).toEqual(new Set(['(prefers-reduced-motion: reduce)']));
  });
});

describe('a rate the viewer chose', () => {
  it('is kept when the setting turns on afterwards', () => {
    const media = stubReducedMotion(false);
    const { result } = open();
    act(() => result.current.setRate(30));
    media.set(true);
    expect(result.current.rate).toBe(30);
  });

  it('is kept when the setting turns off afterwards', () => {
    const media = stubReducedMotion(true);
    const { result } = open();
    act(() => result.current.setRate(20));
    media.set(false);
    expect(result.current.rate).toBe(20);
  });

  it('can be the lowest or any other rate, whatever the setting', () => {
    stubReducedMotion(true);
    const { result } = open();
    for (const rate of RATES) {
      act(() => result.current.setRate(rate));
      expect(result.current.rate).toBe(rate);
    }
  });
});

describe('before the viewer has chosen', () => {
  it('the default follows the setting while the page is open', () => {
    const media = stubReducedMotion(false);
    const { result } = open();
    expect(result.current.rate).toBe(DEFAULT_RATE);
    media.set(true);
    expect(result.current.rate).toBe(RATES[0]);
    media.set(false);
    expect(result.current.rate).toBe(DEFAULT_RATE);
  });

  it('stops listening when the scene goes away', () => {
    const media = stubReducedMotion(true);
    const { unmount } = open();
    expect(media.listeners()).toBeGreaterThan(0);
    unmount();
    expect(media.listeners()).toBe(0);
  });
});

describe('play is still there', () => {
  it('toggling starts playback at the lowest rate, and it never starts by itself', () => {
    stubReducedMotion(true);
    const { result } = open();
    expect(result.current.playing).toBe(false);
    act(() => result.current.toggle());
    expect(result.current.playing).toBe(true);
    expect(result.current.rate).toBe(5);
  });
});
