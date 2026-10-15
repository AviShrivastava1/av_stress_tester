import { fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { WAYMO_ATTRIBUTION } from '../components/Footer';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';
import { LIMITS } from './MethodPage';

/*
 * The "Copy brief" button on the scenario page (pages/detail/BriefButton.tsx; the text is pages/detail/brief.ts). It puts the brief on
 * the clipboard and says so. If the browser will not (no clipboard API, a refusal), it says that and shows the same text in a
 * box to select and copy by hand: the brief is never lost and a copy that did not happen is never reported as one.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });
const FRAMES = range(0, 91);
const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: FRAMES, path: FRAMES.map(() => [0, 0]), headings: FRAMES.map(() => 0) });
const logged = makeTrack({ agent_idx: 1, frames: FRAMES, origin: [0, 40] });
const perturbed = makeTrack({ agent_idx: 1, frames: FRAMES, path: FRAMES.map((f) => [0, f < 25 ? 2.5 : 1]), headings: FRAMES.map(() => 0) });

const collision = {
  detail: makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found', last_attempt_outcome: 'collision_found', min_perturbation: 0.0226, collision_timestep: 25, challengers_searched: 3, challengers_total: 20 }),
  perturbed: makePerturbed({ target_idx: 1, collision_timestep: 25, delta: [0.1, 0.2, 0.3, 0.4], delta_labels: ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'], baseline: logged, perturbed }),
  trajectories: makeTrajectories([sdc, logged]),
};
const none = {
  detail: makeDetail({ stress_tested: true, stress_attempted: true, stress_outcome: 'no_collision_found', last_attempt_outcome: 'no_collision_found' }),
  perturbed: makePerturbed(),
  trajectories: makeTrajectories([sdc, logged]),
};

function serve(data: typeof collision) {
  return (req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return ok(data.trajectories);
    if (req.path.endsWith('/perturbed')) return ok(data.perturbed);
    return ok(data.detail);
  };
}

function setClipboard(value: unknown) {
  Object.defineProperty(navigator, 'clipboard', { value, configurable: true });
}

afterEach(() => {
  Reflect.deleteProperty(navigator, 'clipboard');
  vi.unstubAllGlobals();
});

async function open(data = collision) {
  stubFetch(serve(data));
  renderApp(`/scenarios/${ID}`);
  await screen.findByRole('group', { name: 'Search outcome' });
  await screen.findByTestId('frame-readout');
}
const button = () => screen.getByRole('button', { name: 'Copy brief' });
const box = () => screen.queryByRole('textbox', { name: 'Brief' }) as HTMLTextAreaElement | null;

describe('copying the brief', () => {
  it('puts the brief on the clipboard and says it did', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setClipboard({ writeText });
    await open();
    fireEvent.click(button());
    await screen.findByText('Copied the brief.');
    expect(writeText).toHaveBeenCalledTimes(1);
    const text = writeText.mock.calls[0]![0] as string;
    expect(text).toMatch(new RegExp(`^AV Scenario Stress-Tester: scenario ${ID}\\n${window.location.origin}/scenarios/${ID}\\n`));
    expect(screen.getByRole('status')).toHaveTextContent('Copied the brief.');
    expect(box()).toBeNull();
  });

  it('copies the values the page shows, read from the API', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setClipboard({ writeText });
    await open();
    fireEvent.click(button());
    await screen.findByText('Copied the brief.');
    const text = writeText.mock.calls[0]![0] as string;
    expect(text).toContain('Outcome: collision found\n');
    expect(text).toContain('Min perturbation: 0.023 (');
    expect(text).toContain('Collision frame: 25 (');
    expect(text).toContain('Recording: frames 0 to 90');
    expect(text).toContain('Gap before first contact: The boxes were 0.500 m apart on frame 24');
    expect(text).toContain('Challenger: agent 1, 3 of 20 challengers searched');
    expect(text).toContain('  initial speed (m/s): 0.1000');
  });

  it('includes the Waymo attribution and every Method limit', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setClipboard({ writeText });
    await open();
    fireEvent.click(button());
    await screen.findByText('Copied the brief.');
    const text = writeText.mock.calls[0]![0] as string;
    expect(text.endsWith(WAYMO_ATTRIBUTION)).toBe(true);
    for (const limit of LIMITS) expect(text).toContain(limit);
  });

  it('copies a scenario with no collision, without a collision frame or a perturbation', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setClipboard({ writeText });
    await open(none as unknown as typeof collision);
    fireEvent.click(button());
    await screen.findByText('Copied the brief.');
    const text = writeText.mock.calls[0]![0] as string;
    expect(text).toContain('Outcome: no collision found');
    expect(text).not.toMatch(/Collision frame|Min perturbation|Gap before first contact|Perturbation components/);
    expect(text.endsWith(WAYMO_ATTRIBUTION)).toBe(true);
  });
});

describe('when the copy does not happen', () => {
  it('says so and shows the same text to copy by hand, when the browser refuses', async () => {
    const writeText = vi.fn().mockRejectedValue(new DOMException('denied', 'NotAllowedError'));
    setClipboard({ writeText });
    await open();
    fireEvent.click(button());
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Could not copy to the clipboard. The brief is below: select it and copy it yourself.');
    expect(screen.queryByText('Copied the brief.')).not.toBeInTheDocument();
    const attempted = writeText.mock.calls[0]![0] as string;
    expect(box()).not.toBeNull();
    expect(box()!.value).toBe(attempted); // the same text, not a second build
    expect(box()!.readOnly).toBe(true);
    expect(box()!.value.endsWith(WAYMO_ATTRIBUTION)).toBe(true);
  });

  it('says so and shows the text when the browser has no clipboard API at all', async () => {
    setClipboard(undefined);
    await open();
    fireEvent.click(button());
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not copy to the clipboard.');
    expect(box()!.value).toContain(`scenario ${ID}`);
  });

  it('says so when writeText throws instead of rejecting', async () => {
    setClipboard({ writeText: () => { throw new Error('boom'); } });
    await open();
    fireEvent.click(button());
    expect(await screen.findByRole('alert')).toBeInTheDocument();
    expect(box()).not.toBeNull();
  });

  it('a later press that works replaces the failure with the confirmation', async () => {
    const writeText = vi.fn().mockRejectedValueOnce(new Error('denied')).mockResolvedValue(undefined);
    setClipboard({ writeText });
    await open();
    fireEvent.click(button());
    await screen.findByRole('alert');
    fireEvent.click(button());
    await screen.findByText('Copied the brief.');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(box()).toBeNull();
    expect(writeText).toHaveBeenCalledTimes(2);
  });

  it('shows the box for a scenario with no collision as well', async () => {
    setClipboard({ writeText: vi.fn().mockRejectedValue(new Error('denied')) });
    await open(none as unknown as typeof collision);
    fireEvent.click(button());
    await screen.findByRole('alert');
    expect(box()!.value).toContain('Outcome: no collision found');
  });
});

describe('the button', () => {
  it('is not offered while the scenario cannot be loaded, and nothing is copied', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setClipboard({ writeText });
    stubFetch((req: RecordedRequest): Reply =>
      req.path.endsWith('/trajectories') || req.path.endsWith('/perturbed') ? ok(collision.trajectories) : { status: 500, body: { detail: 'down' } });
    renderApp(`/scenarios/${ID}`);
    await screen.findByRole('alert');
    expect(screen.queryByRole('button', { name: 'Copy brief' })).not.toBeInTheDocument();
    expect(writeText).not.toHaveBeenCalled();
  });

  it('copies nothing and reports nothing before it is pressed', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setClipboard({ writeText });
    await open();
    expect(writeText).not.toHaveBeenCalled();
    expect(screen.queryByText('Copied the brief.')).not.toBeInTheDocument();
    await waitFor(() => expect(button()).toBeEnabled());
  });
});
