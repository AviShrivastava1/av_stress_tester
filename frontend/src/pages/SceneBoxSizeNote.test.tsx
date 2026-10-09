import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { HealthResponse, StatsResponse } from '../api/client';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import {
  makeDetail, makePage, makePerturbed, makeRow, makeTrack, makeTrajectories, range,
} from '../test/fixtures';
import { renderApp } from '../test/renderApp';

/*
 * A note under the scene about how the boxes are drawn.
 *
 * Every track in this file has no per-frame size arrays, which is how data exported before
 * the per-frame columns existed arrives. The page then draws ONE box size per agent, from its
 * first observed frame, and this is the note it shows. The search's exact collision check used
 * each agent's size in every frame, so at the collision frame such boxes may not touch even
 * though the check saw contact. The note says so, conditionally, and only where the page marks
 * a colliding pair: when the perturbed path is drawn AND a collision frame is set.
 *
 * Tracks WITH per-frame arrays, and the other two texts of the note, are in
 * SceneBoxSizeNotePerFrame.test.tsx.
 */

const NOTE =
  "Boxes are drawn at each agent's size in its first observed frame. The collision check used " +
  "each agent's size in every frame, so at the collision frame the drawn boxes may not touch. " +
  'Contact is decided at the precision of the stored positions, up to about a millimetre at the ' +
  'coordinates in this data.';
const BANNED = [
  'error bar', 'margin', 'uncertainty', 'safe', 'safety', 'cause', 'caused', 'causes',
  'wrong', 'bug', 'inaccurate', 'incorrect',
];
const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });
const serverError: Reply = { status: 500, body: { detail: 'boom' } };

// SDC along +x; challenger heading +y; the perturbed run collides at frame 25.
const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
const challenger = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -28], heading: Math.PI / 2 });
const perturbedTrack = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 });

const FOUND_DETAIL = makeDetail({ stress_tested: true, stress_outcome: 'collision_found', collision_timestep: 25 });
const COLLIDING = makePerturbed({
  target_idx: 1, collision_timestep: 25, baseline: challenger, perturbed: perturbedTrack,
});
const GEOMETRY = makeTrajectories([sdc, challenger]);

interface Replies {
  detail?: Reply;
  trajectories?: Reply;
  /** 'pending' = the request never answers. */
  perturbed?: Reply | 'pending';
}

function open(replies: Replies = {}) {
  const detail = replies.detail ?? ok(FOUND_DETAIL);
  const trajectories = replies.trajectories ?? ok(GEOMETRY);
  const perturbed = replies.perturbed ?? ok(COLLIDING);
  const asResponse = (r: Reply) => new Response(JSON.stringify(r.body), {
    status: r.status, headers: { 'Content-Type': 'application/json' },
  });
  vi.stubGlobal('fetch', (input: Request) => {
    const path = new URL(input.url).pathname;
    if (path.endsWith('/perturbed')) {
      return perturbed === 'pending' ? new Promise<Response>(() => {}) : Promise.resolve(asResponse(perturbed));
    }
    return Promise.resolve(asResponse(path.endsWith('/trajectories') ? trajectories : detail));
  });
  return renderApp(`/scenarios/${ID}`);
}

/** Wait until the Scene panel has settled one way or the other, so an absence means something. */
async function settled() {
  await screen.findByText('Min perturbation');
  await waitFor(() => expect(screen.queryByText('Loading geometry…')).not.toBeInTheDocument());
}
const note = () => screen.queryByText(NOTE);

afterEach(() => {
  vi.useRealTimers();
});

describe('box-size note: when it is shown', () => {
  it('appears when the perturbed path is drawn and a collision frame is set', async () => {
    open();
    await screen.findByTestId('frame-readout');
    expect(note()).toBeInTheDocument();
    expect(note()?.tagName).toBe('P');
    expect(note()).toHaveClass('note');
  });
});

describe('box-size note: when it is not shown', () => {
  it('for a scenario with no stored result', async () => {
    open({ detail: ok(makeDetail()), perturbed: ok(makePerturbed()) });
    await screen.findByTestId('frame-readout');
    expect(note()).not.toBeInTheDocument();
  });

  it('for a completed search that found no collision (nothing drawn for the perturbed path)', async () => {
    open({
      detail: ok(makeDetail({ stress_tested: true, stress_outcome: 'no_collision_found', no_collision_found: true })),
      perturbed: ok(makePerturbed()),
    });
    await screen.findByTestId('frame-readout');
    expect(note()).not.toBeInTheDocument();
  });

  it('when the perturbed path is drawn but carries no collision frame', async () => {
    open({
      detail: ok(makeDetail({ stress_tested: true, stress_outcome: 'no_collision_found', no_collision_found: true })),
      perturbed: ok(makePerturbed({
        target_idx: 1, collision_timestep: null, baseline: challenger, perturbed: perturbedTrack,
      })),
    });
    await screen.findByTestId('frame-readout');
    expect(screen.getAllByTestId('frame-readout')).toHaveLength(1);   // the scene IS drawn
    expect(note()).not.toBeInTheDocument();
  });

  // A stored result whose geometry has not been exported: the response carries the collision frame
  // and the delta, but no baseline or perturbed path, so no colliding pair is drawn or marked.
  it('when the response carries a collision frame but the perturbed path is not drawn', async () => {
    open({
      perturbed: ok(makePerturbed({
        target_idx: 1, delta: [0.02, -0.01, 0.009, -0.011], min_perturbation: 0.02, collision_timestep: 25,
      })),
    });
    await screen.findByTestId('frame-readout');
    expect(note()).not.toBeInTheDocument();
  });

  it('for a scenario whose only attempt was refused', async () => {
    open({
      detail: ok(makeDetail({ stress_tested: false, stress_attempted: true, last_attempt_outcome: 'replay_infeasible' })),
      perturbed: ok(makePerturbed()),
    });
    await screen.findByTestId('frame-readout');
    expect(note()).not.toBeInTheDocument();
  });

  it('when there is no geometry to draw', async () => {
    open({ trajectories: ok(makeTrajectories([])), perturbed: ok(makePerturbed()) });
    await settled();
    expect(screen.queryByTestId('frame-readout')).not.toBeInTheDocument();
    expect(note()).not.toBeInTheDocument();
  });

  it('when the geometry request fails', async () => {
    open({ trajectories: serverError });
    await settled();
    await screen.findByText(/Scene geometry could not be loaded/);
    expect(note()).not.toBeInTheDocument();
  });

  it('while the perturbed path is loading', async () => {
    open({ perturbed: 'pending' });
    await screen.findByTestId('frame-readout');
    expect(screen.getByText('Loading the perturbed path…')).toBeInTheDocument();
    expect(note()).not.toBeInTheDocument();
  });

  it('when the perturbed request fails', async () => {
    open({ perturbed: serverError });
    await screen.findByTestId('frame-readout');
    expect(await screen.findByText(/The perturbed path could not be loaded/)).toBeInTheDocument();
    expect(note()).not.toBeInTheDocument();
  });
});

describe('box-size note: wording', () => {
  it('is exactly the approved text', async () => {
    open();
    await screen.findByTestId('frame-readout');
    expect(note()?.textContent).toBe(NOTE);
  });

  it('makes the three claims: first observed frame, per-frame sizes in the check, positional precision', async () => {
    open();
    await screen.findByTestId('frame-readout');
    const text = note()?.textContent ?? '';
    expect(text).toMatch(/drawn at each agent's size in its first observed frame/);
    expect(text).toMatch(/collision check used each agent's size in every frame/);
    expect(text).toMatch(/precision of the stored positions/);
  });

  it('is conditional: it claims nothing about this scenario in particular', async () => {
    open();
    await screen.findByTestId('frame-readout');
    expect(note()?.textContent).toMatch(/may not touch/);
  });

  // textContent skips tooltips and accessible names, which are shown on hover and read out.
  it.each(BANNED)('uses the word "%s" in none of its text, tooltips or accessible names', async (word) => {
    open();
    await screen.findByTestId('frame-readout');
    const element = note()!;
    const texts: string[] = [];
    for (const el of [element, ...Array.from(element.querySelectorAll<HTMLElement>('*'))]) {
      texts.push(el.textContent ?? '');
      for (const attr of ['title', 'aria-label', 'aria-description', 'alt']) {
        const value = el.getAttribute(attr);
        if (value) texts.push(value);
      }
      for (const id of (el.getAttribute('aria-labelledby') ?? '').split(/\s+/).filter(Boolean)) {
        const labelled = document.getElementById(id)?.textContent;
        if (labelled) texts.push(labelled);
      }
    }
    for (const text of texts) {
      expect(text, `"${word}" in the note`).not.toMatch(new RegExp(`\\b${word}\\b`, 'i'));
    }
    // and nothing in the panel's other notes added it either
    const panel = element.closest('section')!;
    expect(within(panel).queryByText(new RegExp(`\\b${word}\\b`, 'i'))).not.toBeInTheDocument();
  });
});

describe('box-size note: other pages', () => {
  it('is not on the list page', async () => {
    stubFetch(() => ok(makePage([makeRow({ stress_tested: true, min_perturbation: 0.022 })], null)));
    renderApp('/');
    await screen.findByRole('link', { name: ID });
    expect(note()).not.toBeInTheDocument();
    expect(screen.queryByText(/first observed frame/)).not.toBeInTheDocument();
  });

  it('is not on the Corpus page', async () => {
    const stats: StatsResponse = {
      total_scenarios: 100, stress_tested: 18, collisions_found: 18, no_collision_found: 0,
      replay_infeasible: 2, heading_blend_singularity: 0, no_challenger: 0, stress_errors: 0,
      robustly_safe: 0, with_geometry: 20, fragility_min: 0.5, fragility_max: 4.5, fragility_mean: 2,
    };
    const health: HealthResponse = { status: 'ok', database: 'connected', postgis: '3.6' };
    stubFetch((req: RecordedRequest) => ok(req.path === '/health' ? health : stats));
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Stored results' });
    expect(note()).not.toBeInTheDocument();
    expect(screen.queryByText(/first observed frame/)).not.toBeInTheDocument();
  });
});

describe('box-size note: playback does not remove or change it', () => {
  const stays = () => expect(note()?.textContent).toBe(NOTE);

  it('survives playing, pausing, stepping, keys and the jump to the collision frame', async () => {
    open();
    await screen.findByTestId('frame-readout');
    stays();

    fireEvent.click(screen.getByRole('button', { name: 'Play' }));
    await act(async () => { await new Promise((r) => setTimeout(r, 60)); });
    stays();
    fireEvent.click(screen.getByRole('button', { name: 'Pause' }));
    stays();

    fireEvent.click(screen.getByRole('button', { name: 'Step forward one frame' }));
    stays();
    fireEvent.click(screen.getByRole('button', { name: 'Step back one frame' }));
    stays();

    const scene = screen.getByLabelText(/Space plays or pauses/);
    fireEvent.keyDown(scene, { key: 'ArrowRight' });
    fireEvent.keyDown(scene, { key: ' ' });
    fireEvent.keyDown(scene, { key: ' ' });
    stays();

    fireEvent.change(screen.getByRole('slider', { name: 'Frame' }), { target: { value: '3' } });
    expect(screen.getByTestId('frame-readout')).toHaveTextContent('frame 3 of');
    stays();
    fireEvent.click(screen.getByRole('button', { name: /Jump to collision/ }));
    expect(screen.getByTestId('frame-readout')).toHaveTextContent('frame 25 of');
    stays();
  });

  it('is still there once the viewer has moved the playhead and the opening-frame note is gone', async () => {
    open();
    await screen.findByTestId('frame-readout');
    expect(screen.getByText(/Positions at frame 25, the first colliding frame/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Step forward one frame' }));
    expect(screen.queryByText(/Positions at frame 25, the first colliding frame/)).not.toBeInTheDocument();
    expect(note()).toBeInTheDocument();
  });

  it('is shown once, not repeated', async () => {
    open();
    await screen.findByTestId('frame-readout');
    expect(screen.getAllByText(NOTE)).toHaveLength(1);
  });
});
