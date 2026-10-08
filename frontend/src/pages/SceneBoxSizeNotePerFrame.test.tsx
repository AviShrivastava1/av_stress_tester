import { cleanup, fireEvent, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { renderApp } from '../test/renderApp';

/*
 * The note under the scene, in the three cases the drawn colliding pair (the SDC and the perturbed
 * challenger, whichever of them are drawn) can be in:
 *
 *   every one drawn at its scalar size       -> the existing text, unchanged (SceneBoxSizeNote.test.tsx pins it)
 *   every one drawn from per-frame sizes     -> the new text
 *   some of each                             -> the third text
 *
 * Shown, as before, only when the perturbed path is drawn AND a collision frame is set.
 */

const OLD =
  "Boxes are drawn at each agent's size in its first observed frame. The collision check used " +
  "each agent's size in every frame, so at the collision frame the drawn boxes may not touch. " +
  'Contact is decided at the precision of the stored positions, up to about a millimetre at the ' +
  'coordinates in this data.';
const PER_FRAME =
  "Boxes are drawn at each agent's stored size for the frame shown, the size the collision check used. " +
  'Contact is decided at the precision of the stored positions, up to about a millimetre at the ' +
  'coordinates in this data.';
const MIXED =
  "Some boxes are drawn at the agent's size in its first observed frame, because no per-frame sizes " +
  "are stored for that agent. The collision check used each agent's size in every frame, so at the " +
  'collision frame the drawn boxes may not touch. Contact is decided at the precision of the stored ' +
  'positions, up to about a millimetre at the coordinates in this data.';
const ALL = [OLD, PER_FRAME, MIXED];
const BANNED = [
  'error bar', 'margin', 'uncertainty', 'safe', 'safety', 'cause', 'caused', 'causes',
  'wrong', 'bug', 'inaccurate', 'incorrect',
];

const ID = 'synthetic-0001';
const FRAMES = range(0, 91);
const sizes = { lengths_m: FRAMES.map((f) => 4.5 + f * 0.001), widths_m: FRAMES.map((f) => 2 + f * 0.0005) };

const sdcScalar = makeTrack({ agent_idx: 0, is_sdc: true, frames: FRAMES, origin: [-20, 0] });
const sdcPerFrame = { ...sdcScalar, ...sizes };
const loggedScalar = makeTrack({ agent_idx: 1, frames: FRAMES, origin: [0, -28], heading: Math.PI / 2 });
const loggedPerFrame = { ...loggedScalar, ...sizes };
const perturbedScalar = makeTrack({ agent_idx: 1, frames: FRAMES, origin: [1, -28], heading: Math.PI / 2 });
const perturbedPerFrame = { ...perturbedScalar, ...sizes };

const DETAIL = makeDetail({ stress_tested: true, stress_outcome: 'collision_found', collision_timestep: 25 });

interface Scene {
  sdc?: typeof sdcScalar | null;     // null: the SDC is not in the geometry
  logged?: typeof loggedScalar;
  perturbed?: typeof perturbedScalar | null; // null: no perturbed path
  collisionFrame?: number | null;
}

function open({ sdc = sdcScalar, logged = loggedScalar, perturbed = perturbedScalar, collisionFrame = 25 }: Scene) {
  const agents = [sdc, logged].filter((t): t is NonNullable<typeof t> => t !== null);
  const response = makePerturbed({
    target_idx: 1, collision_timestep: collisionFrame, baseline: logged, perturbed,
  });
  vi.stubGlobal('fetch', (input: Request) => {
    const path = new URL(input.url).pathname;
    const body = path.endsWith('/perturbed') ? response : path.endsWith('/trajectories') ? makeTrajectories(agents) : DETAIL;
    return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } }));
  });
  return renderApp(`/scenarios/${ID}`);
}

/** Which of the three texts is on the page: exactly one, or none. */
function shown(): string[] {
  return ALL.filter((text) => screen.queryAllByText(text).length > 0);
}

describe('box-size note: which text', () => {
  it.each([
    ['every drawn pair member at its scalar size', {}, OLD],
    ['the SDC and the perturbed challenger both from per-frame sizes', { sdc: sdcPerFrame, perturbed: perturbedPerFrame }, PER_FRAME],
    ['the SDC per-frame, the perturbed challenger scalar', { sdc: sdcPerFrame, perturbed: perturbedScalar }, MIXED],
    ['the SDC scalar, the perturbed challenger per-frame', { sdc: sdcScalar, perturbed: perturbedPerFrame }, MIXED],
  ] as [string, Scene, string][])('%s', async (_name, scene, expected) => {
    open(scene);
    await screen.findByTestId('frame-readout');
    expect(shown()).toEqual([expected]);
    expect(screen.getAllByText(expected)).toHaveLength(1);
  });

  it('judges the SDC and the perturbed challenger only, not the logged challenger or other agents', async () => {
    open({ sdc: sdcPerFrame, perturbed: perturbedPerFrame, logged: loggedScalar });
    await screen.findByTestId('frame-readout');
    expect(shown()).toEqual([PER_FRAME]);
    cleanup();
    open({ sdc: sdcScalar, perturbed: perturbedScalar, logged: loggedPerFrame });
    await screen.findByTestId('frame-readout');
    expect(shown()).toEqual([OLD]);
  });

  it('uses the pair members that are drawn: no SDC in the geometry', async () => {
    open({ sdc: null, perturbed: perturbedPerFrame });
    await screen.findByTestId('frame-readout');
    expect(shown()).toEqual([PER_FRAME]);
    cleanup();
    open({ sdc: null, perturbed: perturbedScalar });
    await screen.findByTestId('frame-readout');
    expect(shown()).toEqual([OLD]);
  });

  it('treats a track whose arrays do not fit its path as scalar-drawn', async () => {
    const short = { ...perturbedScalar, lengths_m: sizes.lengths_m.slice(1), widths_m: sizes.widths_m };
    open({ sdc: sdcPerFrame, perturbed: short });
    await screen.findByTestId('frame-readout');
    expect(shown()).toEqual([MIXED]);
  });
});

describe('box-size note: when it is not shown, whatever the sizes', () => {
  it('without a collision frame', async () => {
    open({ sdc: sdcPerFrame, perturbed: perturbedPerFrame, collisionFrame: null });
    await screen.findByTestId('frame-readout');
    expect(shown()).toEqual([]);
  });

  it('without a perturbed path', async () => {
    open({ sdc: sdcPerFrame, perturbed: null });
    await screen.findByTestId('frame-readout');
    expect(shown()).toEqual([]);
  });
});

describe('box-size note: wording', () => {
  it('the new text never says the boxes touch, and says they use the stored size for the frame shown', async () => {
    expect(PER_FRAME).not.toMatch(/touch/i);
    open({ sdc: sdcPerFrame, perturbed: perturbedPerFrame });
    await screen.findByTestId('frame-readout');
    const text = screen.getByText(PER_FRAME).textContent ?? '';
    expect(text).not.toMatch(/touch/i);
    expect(text).toMatch(/stored size for the frame shown/);
    expect(text).toMatch(/precision of the stored positions/);
  });

  it('the third text keeps the precision sentence and is conditional about touching', async () => {
    open({ sdc: sdcPerFrame, perturbed: perturbedScalar });
    await screen.findByTestId('frame-readout');
    const text = screen.getByText(MIXED).textContent ?? '';
    expect(text).toMatch(/Some boxes/);
    expect(text).toMatch(/may not touch/);
    expect(text).toMatch(/precision of the stored positions/);
  });

  it.each([
    ['existing', {}, OLD],
    ['per-frame', { sdc: sdcPerFrame, perturbed: perturbedPerFrame }, PER_FRAME],
    ['mixed', { sdc: sdcPerFrame, perturbed: perturbedScalar }, MIXED],
  ] as [string, Scene, string][])('the %s text uses none of the banned words, in text, tooltips or labels', async (_n, scene, expected) => {
    open(scene);
    await screen.findByTestId('frame-readout');
    const element = screen.getByText(expected);
    const texts: string[] = [];
    for (const el of [element, ...Array.from(element.querySelectorAll<HTMLElement>('*'))]) {
      texts.push(el.textContent ?? '');
      for (const attr of ['title', 'aria-label', 'aria-description', 'alt']) {
        const value = el.getAttribute(attr);
        if (value) texts.push(value);
      }
    }
    for (const word of BANNED) {
      for (const text of texts) expect(text, `"${word}"`).not.toMatch(new RegExp(`\\b${word}\\b`, 'i'));
    }
  });
});

describe('box-size note: playback does not change it', () => {
  it('stays the same text while the viewer plays, steps and jumps', async () => {
    open({ sdc: sdcPerFrame, perturbed: perturbedPerFrame });
    await screen.findByTestId('frame-readout');
    fireEvent.click(screen.getByRole('button', { name: 'Step forward one frame' }));
    expect(shown()).toEqual([PER_FRAME]);
    fireEvent.change(screen.getByRole('slider', { name: 'Frame' }), { target: { value: '3' } });
    expect(shown()).toEqual([PER_FRAME]);
    fireEvent.click(screen.getByRole('button', { name: /Jump to collision/ }));
    expect(shown()).toEqual([PER_FRAME]);
  });
});
