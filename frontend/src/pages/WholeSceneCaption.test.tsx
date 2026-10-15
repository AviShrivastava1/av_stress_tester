import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { OTHERS_AS_LOGGED_NOTE, othersAsLoggedNote, type DrawnTrack } from '../scene/sceneModel';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The caption under the whole-scene replay (scene/sceneModel.ts OTHERS_AS_LOGGED_NOTE): the SDC and every other agent are the
 * recorded tracks, because the search rewrites the challenger's track only. It is shown where a reader needs it, the whole-scene
 * framing with the perturbed challenger and at least one other agent drawn, and nowhere else.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });
const NOTE =
  "The SDC and every other agent are drawn as logged. The search changes the challenger's track only, so none of them reacts to the perturbed challenger.";

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
const logged = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -28], heading: Math.PI / 2 });
const perturbed = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 });
const parked = makeTrack({ agent_idx: 2, frames: range(0, 91), origin: [10, 10] });

function open(options: { others?: boolean; perturbed?: boolean } = {}) {
  const { others = true, perturbed: withPerturbed = true } = options;
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged, ...(others ? [parked] : [])]));
    if (req.path.endsWith('/perturbed')) {
      return ok(makePerturbed(withPerturbed ? { target_idx: 1, collision_timestep: 25, baseline: logged, perturbed } : { target_idx: 1 }));
    }
    return ok(makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', collision_timestep: 25, min_perturbation: 0.1 }));
  });
  renderApp(`/scenarios/${ID}`);
}

const frame = () => screen.findByTestId('frame-readout');
const framing = (name: string) => within(screen.getByRole('group', { name: 'Framing' })).getByRole('button', { name });
const scenePanel = () => screen.getByRole('heading', { name: 'Replay' }).closest('section')!;

describe('the caption under the whole-scene replay', () => {
  it('is shown, word for word and visible, in the whole-scene framing with the perturbed challenger and another agent', async () => {
    open();
    await frame();
    await userEvent.click(framing('Whole scene'));
    const note = within(scenePanel()).getByText(NOTE);
    expect(note).toBeVisible();
    expect(note).toHaveClass('note');
    expect(OTHERS_AS_LOGGED_NOTE).toBe(NOTE);
  });

  it('is not shown in the challenger-and-SDC framing, where no other agent is drawn', async () => {
    open();
    await frame();
    expect(framing('Challenger & SDC')).toHaveAttribute('aria-pressed', 'true');
    expect(within(scenePanel()).queryByText(NOTE)).not.toBeInTheDocument();
  });

  it('follows the framing back and forth', async () => {
    open();
    await frame();
    await userEvent.click(framing('Whole scene'));
    expect(within(scenePanel()).getByText(NOTE)).toBeInTheDocument();
    await userEvent.click(framing('Challenger & SDC'));
    expect(within(scenePanel()).queryByText(NOTE)).not.toBeInTheDocument();
    await userEvent.click(framing('Whole scene'));
    expect(within(scenePanel()).getByText(NOTE)).toBeInTheDocument();
  });

  it('stays while the replay plays and is scrubbed: it describes every frame, not the opening one', async () => {
    open();
    await frame();
    await userEvent.click(framing('Whole scene'));
    await userEvent.click(screen.getByRole('button', { name: 'Step forward one frame' }));
    expect(within(scenePanel()).getByText(NOTE)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /^Jump to collision/ }));
    expect(within(scenePanel()).getByText(NOTE)).toBeInTheDocument();
  });

  it('is not shown when there is no perturbed track to qualify, even in the whole-scene framing', async () => {
    open({ perturbed: false });
    await frame();
    await userEvent.click(framing('Whole scene'));
    expect(within(scenePanel()).queryByText(NOTE)).not.toBeInTheDocument();
  });

  it('is not shown when no other agent is drawn', async () => {
    open({ others: false });
    await frame();
    await userEvent.click(framing('Whole scene'));
    expect(within(scenePanel()).queryByText(NOTE)).not.toBeInTheDocument();
  });

  it('appears once, and the offset note under the stored perturbation does not repeat it', async () => {
    open();
    await frame();
    await userEvent.click(framing('Whole scene'));
    expect(screen.getAllByText(NOTE)).toHaveLength(1);
    expect(document.body.textContent!.split('none of them reacts').length - 1).toBe(1);
  });
});

describe('othersAsLoggedNote', () => {
  const d = (role: DrawnTrack['role']): DrawnTrack => ({ key: role, role, track: sdc });
  it('is the caption only for a perturbed challenger shown with an other agent', () => {
    expect(othersAsLoggedNote([d('sdc'), d('challenger_perturbed'), d('other')])).toBe(NOTE);
    expect(othersAsLoggedNote([d('sdc'), d('challenger_logged'), d('challenger_perturbed'), d('other')])).toBe(NOTE);
    expect(othersAsLoggedNote([d('sdc'), d('challenger_perturbed')])).toBeNull();
    expect(othersAsLoggedNote([d('sdc'), d('challenger_logged'), d('other')])).toBeNull();
    expect(othersAsLoggedNote([d('other')])).toBeNull();
    expect(othersAsLoggedNote([])).toBeNull();
  });
});
