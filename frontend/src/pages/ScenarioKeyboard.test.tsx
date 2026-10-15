import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The keys the focused scene answers, beyond Space and the arrows (pages/detail/ScenePanel.tsx onKeyDown):
 *
 *   Home and End   the first and last frame the scene has (the ends of the scrubber, frameRange)
 *   C              the collision frame (the first colliding one), the same frame "Jump to collision" goes to (scene/playback.ts
 *                  collisionTarget); a scene with no collision frame in range has no such key
 *
 * Each key is scoped to the scene frame as Space and the arrows are. A key pressed with Ctrl, Alt or Meta is left alone:
 * Alt+Left and Cmd+Left are the browser's Back, Ctrl+Home scrolls the page and Ctrl+C or Cmd+C copies, none of them ours.
 * The scene has no ARIA role, so a screen reader in browse mode keeps its own letter keys; a key that does nothing is not
 * cancelled (preventDefault), so it still reaches the browser and assistive technology.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
const challenger = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -28], heading: Math.PI / 2 });
const perturbedTrack = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 });

function scene(collisionFrame: number | null) {
  return {
    detail: makeDetail({
      stress_tested: collisionFrame !== null,
      stress_outcome: collisionFrame !== null ? 'collision_found' : null,
      collision_timestep: collisionFrame,
    }),
    trajectories: makeTrajectories([sdc, challenger]),
    perturbed:
      collisionFrame === null
        ? makePerturbed()
        : makePerturbed({ target_idx: 1, collision_timestep: collisionFrame, baseline: challenger, perturbed: perturbedTrack }),
  };
}

async function open(data: ReturnType<typeof scene>) {
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return ok(data.trajectories);
    if (req.path.endsWith('/perturbed')) return ok(data.perturbed);
    return ok(data.detail);
  });
  renderApp(`/scenarios/${ID}`);
  await screen.findByTestId('frame-readout');
}

const frameShown = () => Number(/frame (\d+)/.exec(screen.getByTestId('frame-readout').textContent ?? '')![1]);
const sceneFrame = () => screen.getByLabelText(/Space plays or pauses/);
const playButton = () => screen.getByRole('button', { name: /^(Play|Pause)$/ });
const hint = () => document.querySelector('.keyboard-hint')!.textContent!;
/** fireEvent returns false when the handler cancelled the event. */
const press = (el: Element, init: KeyboardEventInit) => fireEvent.keyDown(el, init);

describe('Home, End and C on a scene with a collision frame', () => {
  it('Home goes to the first frame and End to the last, each cancelling the key', async () => {
    await open(scene(25));
    expect(frameShown()).toBe(25);

    expect(press(sceneFrame(), { key: 'End' })).toBe(false);
    expect(frameShown()).toBe(90);
    expect(press(sceneFrame(), { key: 'Home' })).toBe(false);
    expect(frameShown()).toBe(0);
  });

  it('C goes to the first colliding frame from either side, in either case', async () => {
    await open(scene(25));
    fireEvent.keyDown(sceneFrame(), { key: 'End' });
    expect(press(sceneFrame(), { key: 'c' })).toBe(false);
    expect(frameShown()).toBe(25);

    fireEvent.keyDown(sceneFrame(), { key: 'Home' });
    press(sceneFrame(), { key: 'C' });
    expect(frameShown()).toBe(25);
  });

  it('C goes to the frame "Jump to collision" goes to', async () => {
    await open(scene(25));
    const scrubber = () => screen.getByRole('slider', { name: 'Frame' });
    fireEvent.change(scrubber(), { target: { value: '60' } });
    expect(frameShown()).toBe(60);
    fireEvent.click(screen.getByRole('button', { name: /Jump to collision/ }));
    const viaButton = frameShown();
    expect(viaButton).toBe(25);

    fireEvent.change(scrubber(), { target: { value: '60' } });
    expect(frameShown()).toBe(60);
    fireEvent.keyDown(sceneFrame(), { key: 'c' });
    expect(frameShown()).toBe(viaButton);
  });

  it.each([
    ['the first frame the scene has', 0],
    ['the last frame the scene has', 90],
  ])('C reaches a collision at %s', async (_name, collisionFrame) => {
    await open(scene(collisionFrame));
    expect(frameShown()).toBe(collisionFrame);
    expect(screen.getByRole('button', { name: /Jump to collision/ })).toBeInTheDocument();

    const elsewhere = collisionFrame === 0 ? 'End' : 'Home';
    fireEvent.keyDown(sceneFrame(), { key: elsewhere });
    expect(frameShown()).not.toBe(collisionFrame);
    expect(press(sceneFrame(), { key: 'c' })).toBe(false);
    expect(frameShown()).toBe(collisionFrame);
    expect(hint()).toContain('and C for the collision frame');
  });

  it('each key pauses a replay that is running, as scrubbing does', async () => {
    await open(scene(25));
    for (const key of ['Home', 'End', 'c']) {
      fireEvent.click(playButton());
      expect(playButton()).toHaveAccessibleName('Pause');
      fireEvent.keyDown(sceneFrame(), { key });
      expect(playButton()).toHaveAccessibleName('Play');
    }
  });

  it('Home and End are no-ops, and still cancelled, when already at that frame', async () => {
    await open(scene(25));
    fireEvent.keyDown(sceneFrame(), { key: 'Home' });
    expect(press(sceneFrame(), { key: 'Home' })).toBe(false);
    expect(frameShown()).toBe(0);
    fireEvent.keyDown(sceneFrame(), { key: 'End' });
    expect(press(sceneFrame(), { key: 'End' })).toBe(false);
    expect(frameShown()).toBe(90);
  });
});

describe('the keys have no effect outside the scene', () => {
  it('a key on the scrubber, a button, or the page does nothing and is not cancelled', async () => {
    await open(scene(25));
    const targets = [
      screen.getByRole('slider', { name: 'Frame' }),
      screen.getByRole('button', { name: 'Whole scene' }),
      screen.getByRole('button', { name: /Jump to collision/ }),
      document.body,
    ];
    for (const target of targets) {
      for (const key of ['Home', 'End', 'c']) {
        expect(press(target, { key })).toBe(true);
        expect(frameShown()).toBe(25);
      }
    }
  });
});

describe('a key with Ctrl, Alt or Meta is left to the browser', () => {
  it.each([
    ['Home', { ctrlKey: true }],
    ['End', { ctrlKey: true }],
    ['c', { ctrlKey: true }],
    ['c', { metaKey: true }],
    ['Home', { metaKey: true }],
    ['ArrowLeft', { altKey: true }],
    ['ArrowLeft', { metaKey: true }],
    ['ArrowRight', { altKey: true }],
    [' ', { ctrlKey: true }],
  ])('%s with %o does nothing and is not cancelled', async (key, modifier) => {
    await open(scene(25));
    expect(press(sceneFrame(), { key, ...modifier })).toBe(true);
    expect(frameShown()).toBe(25);
    expect(playButton()).toHaveAccessibleName('Play');
  });
});

describe('a scene with no collision frame in range', () => {
  it.each([
    ['no collision frame at all', null],
    ['a collision frame past the last frame the scene has', 200],
  ])('%s: C does nothing and is not cancelled, Home and End still work', async (_name, collisionFrame) => {
    await open(scene(collisionFrame));
    const start = frameShown();
    expect(press(sceneFrame(), { key: 'c' })).toBe(true);
    expect(frameShown()).toBe(start);

    fireEvent.keyDown(sceneFrame(), { key: 'End' });
    expect(frameShown()).toBe(90);
    fireEvent.keyDown(sceneFrame(), { key: 'Home' });
    expect(frameShown()).toBe(0);
  });

  it('a scene with a single frame has nothing to go to and raises nothing', async () => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/trajectories')) {
        return ok(makeTrajectories([makeTrack({ agent_idx: 0, is_sdc: true, frames: [7] })]));
      }
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
      return ok(makeDetail());
    });
    renderApp(`/scenarios/${ID}`);
    await screen.findByTestId('frame-readout');
    for (const key of ['Home', 'End', 'c']) fireEvent.keyDown(sceneFrame(), { key });
    expect(frameShown()).toBe(7);
  });
});

describe('the hint and the scene label list the keys that exist', () => {
  it('with a collision frame: Home, End and C are in both', async () => {
    await open(scene(25));
    expect(hint()).toBe(
      'Focus the scene, then use Space to play, ← → to step, Home and End for the first and last frame, and C for the collision frame.',
    );
    expect(sceneFrame()).toHaveAttribute(
      'aria-label',
      'Scene. Space plays or pauses; the arrow keys step one frame; Home and End go to the first and last frame; C jumps to the collision frame.',
    );
  });

  it.each([null, 200])('with no collision frame in range (%s): Home and End only, and no mention of C', async (collisionFrame) => {
    await open(scene(collisionFrame));
    expect(hint()).toBe(
      'Focus the scene, then use Space to play, ← → to step, and Home and End for the first and last frame.',
    );
    expect(sceneFrame()).toHaveAttribute(
      'aria-label',
      'Scene. Space plays or pauses; the arrow keys step one frame; Home and End go to the first and last frame.',
    );
  });
});
