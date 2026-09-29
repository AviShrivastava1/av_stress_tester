import { act, fireEvent, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });

// SDC along +x; challenger heading +y; the perturbed run collides at frame 25.
// The bystander is unobserved over frames 50-54 (a validity gap).
const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
const challenger = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -28], heading: Math.PI / 2 });
const perturbedTrack = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 });
const bystander = makeTrack({
  agent_idx: 2,
  frames: [...range(0, 50), ...range(55, 91)],
  origin: [30, 30],
});

const collision = {
  detail: makeDetail({ stress_tested: true, stress_outcome: 'collision_found', collision_timestep: 25 }),
  trajectories: makeTrajectories([sdc, challenger, bystander]),
  perturbed: makePerturbed({
    target_idx: 1,
    collision_timestep: 25,
    baseline: challenger,
    perturbed: perturbedTrack,
  }),
};

function server(data: { detail: unknown; trajectories: unknown; perturbed: unknown }) {
  return (req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return ok(data.trajectories);
    if (req.path.endsWith('/perturbed')) return ok(data.perturbed);
    return ok(data.detail);
  };
}

/** Loads the page with real timers, then hands requestAnimationFrame to a fake clock. */
async function openScene(data = collision) {
  stubFetch(server(data));
  const view = renderApp(`/scenarios/${ID}`);
  await screen.findByTestId('frame-readout');
  vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame'] });
  return view;
}

afterEach(() => {
  vi.useRealTimers();
});

const readout = () => screen.getByTestId('frame-readout').textContent ?? '';
const frameShown = () => Number(/frame (\d+)/.exec(readout())![1]);
const scrubber = () => screen.getByRole('slider', { name: 'Frame' });
const playButton = () => screen.getByRole('button', { name: /^(Play|Pause)$/ });
const tick = (ms: number) => act(() => vi.advanceTimersByTime(ms));

function footprintRoles() {
  return [...document.querySelectorAll<SVGElement>('[data-footprint]')].map((el) => ({
    role: el.dataset.footprint,
    colliding: el.dataset.colliding === 'true',
  }));
}

describe('playback — before the viewer touches anything', () => {
  it('opens on the static view: the collision frame, with its note', async () => {
    await openScene();
    expect(frameShown()).toBe(25);
    expect(screen.getByText(/Positions at frame 25, the first colliding frame/)).toBeInTheDocument();
    expect(readout()).toMatch(/frame 25 of 0–90 · 4 of 4 tracks observed/);
    expect(playButton()).toHaveAccessibleName('Play');
  });

  it('never plays on its own', async () => {
    await openScene();
    tick(5000);
    expect(frameShown()).toBe(25);
  });
});

describe('playback — playing', () => {
  it('advances whole frames at the chosen rate, then stops on the last frame', async () => {
    await openScene();
    fireEvent.click(playButton());
    expect(playButton()).toHaveAccessibleName('Pause');

    tick(2000); // about 20 frames at 10 frames/s (the first tick carries no time)
    const after2s = frameShown();
    expect(after2s).toBeGreaterThanOrEqual(25 + 19);
    expect(after2s).toBeLessThanOrEqual(25 + 20);

    tick(60_000);
    expect(frameShown()).toBe(90);
    expect(playButton()).toHaveAccessibleName('Play');
  });

  it('runs faster at a higher rate', async () => {
    await openScene();
    fireEvent.change(screen.getByRole('combobox', { name: 'Playback rate' }), { target: { value: '30' } });
    fireEvent.click(playButton());
    tick(1000);
    expect(frameShown()).toBeGreaterThanOrEqual(25 + 28);
  });

  it('restarts from the first frame when played at the end', async () => {
    await openScene();
    fireEvent.change(scrubber(), { target: { value: '90' } });
    fireEvent.click(playButton());
    expect(frameShown()).toBe(0);
  });

  it('pauses on the pause button', async () => {
    await openScene();
    fireEvent.click(playButton());
    tick(500);
    fireEvent.click(playButton());
    const paused = frameShown();
    tick(2000);
    expect(frameShown()).toBe(paused);
  });
});

describe('playback — scrubbing and stepping', () => {
  it('pauses when the viewer scrubs mid-playback, and stays where they put it', async () => {
    await openScene();
    fireEvent.click(playButton());
    tick(500);
    fireEvent.change(scrubber(), { target: { value: '60' } });

    expect(playButton()).toHaveAccessibleName('Play');
    expect(frameShown()).toBe(60);
    tick(2000);
    expect(frameShown()).toBe(60);
  });

  it('steps one frame each way, pausing, and stops at the ends', async () => {
    await openScene();
    fireEvent.click(playButton());
    fireEvent.click(screen.getByRole('button', { name: 'Step forward one frame' }));
    expect(playButton()).toHaveAccessibleName('Play');
    const here = frameShown();
    fireEvent.click(screen.getByRole('button', { name: 'Step back one frame' }));
    expect(frameShown()).toBe(here - 1);

    fireEvent.change(scrubber(), { target: { value: '0' } });
    expect(screen.getByRole('button', { name: 'Step back one frame' })).toBeDisabled();
    fireEvent.change(scrubber(), { target: { value: '90' } });
    expect(screen.getByRole('button', { name: 'Step forward one frame' })).toBeDisabled();
  });

  it('answers the keyboard when the scene has focus, and not otherwise', async () => {
    await openScene();
    const scene = screen.getByLabelText(/Space plays or pauses/);
    fireEvent.keyDown(scene, { key: 'ArrowRight' });
    expect(frameShown()).toBe(26);
    fireEvent.keyDown(scene, { key: 'ArrowLeft' });
    fireEvent.keyDown(scene, { key: 'ArrowLeft' });
    expect(frameShown()).toBe(24);
    fireEvent.keyDown(scene, { key: ' ' });
    expect(playButton()).toHaveAccessibleName('Pause');
    fireEvent.keyDown(scene, { key: ' ' });
    expect(playButton()).toHaveAccessibleName('Play');

    fireEvent.keyDown(scrubber(), { key: ' ' }); // outside the scene frame: its keys are its own
    expect(playButton()).toHaveAccessibleName('Play');
  });

  it('drops the opening note once the viewer has moved the playhead', async () => {
    await openScene();
    fireEvent.click(screen.getByRole('button', { name: 'Step forward one frame' }));
    expect(screen.queryByText(/the first colliding frame/)).not.toBeInTheDocument();
  });
});

describe('playback — what each frame shows', () => {
  it('marks the colliding pair on the collision frame and on no other', async () => {
    await openScene();
    expect(footprintRoles().filter((f) => f.colliding).map((f) => f.role)).toEqual(['sdc', 'challenger_perturbed']);

    fireEvent.click(screen.getByRole('button', { name: 'Step forward one frame' }));
    expect(footprintRoles().some((f) => f.colliding)).toBe(false);

    fireEvent.click(screen.getByRole('button', { name: /Jump to collision/ }));
    expect(frameShown()).toBe(25);
    expect(footprintRoles().filter((f) => f.colliding)).toHaveLength(2);
  });

  it('draws no footprint for a track inside its validity gap', async () => {
    await openScene();
    fireEvent.change(scrubber(), { target: { value: '52' } });
    expect(footprintRoles().map((f) => f.role)).not.toContain('other');
    expect(readout()).toMatch(/3 of 4 tracks observed/);

    fireEvent.change(scrubber(), { target: { value: '55' } });
    expect(footprintRoles().map((f) => f.role)).toContain('other');
  });

  it('leaves no animation frame scheduled after unmounting mid-playback', async () => {
    const { unmount } = await openScene();
    fireEvent.click(playButton());
    tick(100);
    expect(vi.getTimerCount()).toBeGreaterThan(0);
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('playback — nothing, or almost nothing, to play', () => {
  it('disables every control for a single-frame scene', async () => {
    await openScene({
      detail: makeDetail(),
      trajectories: makeTrajectories([makeTrack({ agent_idx: 0, is_sdc: true, frames: [7] })]),
      perturbed: makePerturbed(),
    });
    expect(readout()).toMatch(/frame 7, the only frame · 1 of 1 tracks observed/);
    expect(playButton()).toBeDisabled();
    expect(scrubber()).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Step back one frame' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Step forward one frame' })).toBeDisabled();

    fireEvent.keyDown(screen.getByLabelText(/Space plays or pauses/), { key: ' ' });
    tick(1000);
    expect(frameShown()).toBe(7);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('says so, with no controls, when no track has any frame', async () => {
    stubFetch(
      server({
        detail: makeDetail(),
        trajectories: makeTrajectories([makeTrack({ agent_idx: 0, is_sdc: true, frames: [] })]),
        perturbed: makePerturbed(),
      }),
    );
    renderApp(`/scenarios/${ID}`);
    expect(await screen.findByText(/none has an observed frame/)).toBeInTheDocument();
    expect(screen.queryByRole('slider')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Play' })).not.toBeInTheDocument();
  });
});
