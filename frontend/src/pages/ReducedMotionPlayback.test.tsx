import { act, fireEvent, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The replay's rate control under prefers-reduced-motion, on the real page (scene/usePlayback.ts has the rule, and
 * scene/ReducedMotionRate.test.tsx the hook on its own): the rate select opens on the lowest rate, a replay still plays
 * when asked, and it moves at that rate. With the setting off nothing changes from before.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
const challenger = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -28], heading: Math.PI / 2 });
const perturbed = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 });
const data = {
  detail: makeDetail({ stress_tested: true, stress_outcome: 'collision_found', collision_timestep: 25 }),
  trajectories: makeTrajectories([sdc, challenger]),
  perturbed: makePerturbed({ target_idx: 1, collision_timestep: 25, baseline: challenger, perturbed }),
};

function stubMatchMedia(reduced: boolean) {
  vi.stubGlobal('matchMedia', (query: string) => ({
    media: query,
    matches: query === '(prefers-reduced-motion: reduce)' ? reduced : false,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
}

async function openScene() {
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return ok(data.trajectories);
    if (req.path.endsWith('/perturbed')) return ok(data.perturbed);
    return ok(data.detail);
  });
  renderApp(`/scenarios/${ID}`);
  await screen.findByTestId('frame-readout');
  vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame'] });
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const rateSelect = () => screen.getByRole('combobox', { name: 'Playback rate' }) as HTMLSelectElement;
const frameShown = () => Number(/frame (\d+)/.exec(screen.getByTestId('frame-readout').textContent ?? '')![1]);
const playButton = () => screen.getByRole('button', { name: /^(Play|Pause)$/ });
const tick = (ms: number) => act(() => vi.advanceTimersByTime(ms));

describe('with reduced motion on', () => {
  it('opens the rate control on the lowest rate, and says so in its own words', async () => {
    stubMatchMedia(true);
    await openScene();
    expect(rateSelect().value).toBe('5');
    expect(rateSelect().selectedOptions[0]!.textContent).toBe('5 frames/s');
  });

  it('does not play until asked, and then plays at the lowest rate', async () => {
    stubMatchMedia(true);
    await openScene();
    expect(frameShown()).toBe(25);
    tick(5000);
    expect(frameShown()).toBe(25); // nothing started by itself

    expect(playButton()).toBeEnabled();
    fireEvent.click(playButton());
    tick(2000); // about 10 frames at 5 frames/s (the first tick carries no time)
    expect(frameShown()).toBeGreaterThanOrEqual(25 + 9);
    expect(frameShown()).toBeLessThanOrEqual(25 + 10);
  });

  it('lets the viewer pick a faster rate, and keeps it', async () => {
    stubMatchMedia(true);
    await openScene();
    fireEvent.change(rateSelect(), { target: { value: '30' } });
    expect(rateSelect().value).toBe('30');
    fireEvent.click(playButton());
    tick(1000);
    expect(frameShown()).toBeGreaterThanOrEqual(25 + 28);
  });
});

describe('with reduced motion off', () => {
  it('opens the rate control on the usual rate and plays at it', async () => {
    stubMatchMedia(false);
    await openScene();
    expect(rateSelect().value).toBe('10');
    fireEvent.click(playButton());
    tick(2000);
    expect(frameShown()).toBeGreaterThanOrEqual(25 + 19);
    expect(frameShown()).toBeLessThanOrEqual(25 + 20);
  });

  it('is the same when the browser cannot say', async () => {
    vi.stubGlobal('matchMedia', undefined);
    await openScene();
    expect(rateSelect().value).toBe('10');
  });
});
