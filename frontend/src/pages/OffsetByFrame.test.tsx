import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The lab's first slice, in the Perturbation panel on the scenario page (pages/detail/OffsetByFrame.tsx, scene/offsets.ts): next
 * to the stored perturbation's components (named, with their units), the distance between the logged challenger and the perturbed
 * challenger on every frame both were observed, worked out in the browser from the two tracks /perturbed serves. Stored results
 * only: nothing is simulated.
 *
 * The logged challenger runs along +x one metre a frame. The perturbed one runs beside it: 0.1 m off per frame up to frame 60,
 * then closing back at 0.2 m per frame, so the offset is 0.1 * frame to a peak of 6 m on frame 60 and 0 on frame 90.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });
const LABELS = ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'];

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [0, -30] });
const logged = makeTrack({ agent_idx: 1, frames: range(0, 91) }); // (f, 0)
const sideways = (f: number) => (f <= 60 ? 0.1 * f : 6 - 0.2 * (f - 60));
function perturbedTrack(frames = range(0, 91)) {
  return makeTrack({ agent_idx: 1, frames, path: frames.map((f) => [f, sideways(f)]), headings: frames.map(() => 0) });
}

async function open(options: { perturbed?: ReturnType<typeof perturbedTrack> | null; collision?: number; labels?: string[] | null } = {}) {
  const { perturbed = perturbedTrack(), collision = 25, labels = LABELS } = options;
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged]));
    if (req.path.endsWith('/perturbed')) {
      return ok(
        makePerturbed({
          target_idx: 1,
          collision_timestep: collision,
          delta: [0.5, 0.01, -0.2, 0.003],
          delta_labels: labels,
          baseline: perturbed === null ? null : logged,
          perturbed,
        }),
      );
    }
    return ok(makeDetail({ stress_tested: true, stress_outcome: 'collision_found', collision_timestep: collision, min_perturbation: 0.5 }));
  });
  renderApp(`/scenarios/${ID}`);
  await screen.findByRole('heading', { name: 'Perturbation' });
  await screen.findByText('initial speed (m/s)');
}

const block = () => screen.queryByRole('group', { name: 'Offset from the recording, by frame' });

describe('the offset by frame', () => {
  it('names the offset on the collision frame, the largest, and the last shared frame', async () => {
    await open();
    const b = block()!;
    expect(b).not.toBeNull();
    expect(b).toHaveTextContent('At the collision frame (frame 25)2.500 m');
    expect(b).toHaveTextContent('Largest, on frame 606.000 m');
    expect(b).toHaveTextContent('On the last frame both were observed (frame 90)0.000 m');
  });

  it('sits in the Perturbation panel beside the stored components with their units', async () => {
    await open();
    const panel = screen.getByRole('heading', { name: 'Perturbation' }).closest('section')!;
    expect(within(panel).getByText('initial speed (m/s)')).toBeInTheDocument();
    expect(within(panel).getByText('steering bias (rad)')).toBeInTheDocument();
    expect(panel.contains(block())).toBe(true);
  });

  it('says what the distance is, and that nothing was simulated', async () => {
    await open();
    expect(block()).toHaveTextContent(
      'The distance between the logged challenger and the perturbed challenger on each frame where both were observed, ' +
        'taken from the two stored tracks (their positions only), in metres. Every other agent is as logged.',
    );
  });

  it('draws the offsets as a line chart with a text description of it', async () => {
    await open();
    const chart = within(block()!).getByRole('figure');
    expect(chart).toHaveAccessibleName(
      'Offset in metres by frame: from 0.000 m on frame 0 to 0.000 m on frame 90, largest 6.000 m on frame 60',
    );
    expect(chart.querySelector('polyline')).not.toBeNull();
    expect(chart.querySelector('svg')).toHaveAttribute('aria-hidden', 'true'); // the name and the table carry the numbers
  });

  it('plots the largest offset at the top of the chart and zero offset at its base', async () => {
    await open();
    const points = block()!.querySelector('polyline')!.getAttribute('points')!.split(' ');
    expect(points).toHaveLength(91);
    // The drawing is 320 wide and 80 high with 4 of margin left, right and below and 6 above: frame 0 is at x = 4, frame 90 at
    // x = 316, an offset of 0 sits on the base (y = 76) and the largest, 6 m on frame 60, on the top margin (y = 6).
    expect(points[0]).toBe('4.00,76.00');
    expect(points[60]).toBe('212.00,6.00');
    expect(points[90]).toBe('316.00,76.00');
    expect(points[30]).toBe('108.00,41.00'); // 3 m on frame 30: halfway up
  });

  it('labels the ends of the line and what the top of the chart stands for', async () => {
    await open();
    expect(block()!.querySelector('figcaption')).toHaveTextContent('frame 0chart top: 6.000 mframe 90');
  });

  it('lists the offset on every shared frame in a table that opens on request', async () => {
    await open();
    const details = block()!.querySelector('details')!;
    expect(details.open).toBe(false);
    expect(within(details).queryByRole('table')).toBeNull(); // not drawn until it is opened
    await userEvent.click(within(details).getByText('Offset at every frame'));
    await within(details).findByRole('table');
    const rows = within(details).getAllByRole('row').slice(1);
    expect(rows).toHaveLength(91);
    expect(rows[25]).toHaveTextContent('252.500');
    expect(rows[60]).toHaveTextContent('606.000');
    expect(within(details).getAllByRole('columnheader').map((h) => h.textContent)).toEqual(['Frame', 'Offset (m)']);
  });

  it('draws the table again only while it is open', async () => {
    await open();
    const details = block()!.querySelector('details')!;
    await userEvent.click(within(details).getByText('Offset at every frame'));
    await within(details).findByRole('table');
    await userEvent.click(within(details).getByText('Offset at every frame'));
    await waitFor(() => expect(within(details).queryByRole('table')).toBeNull());
  });

  it('draws a flat line, with no NaN in it, when the two tracks are on the same spot throughout', async () => {
    await open({ perturbed: makeTrack({ agent_idx: 1, frames: range(0, 91) }) }); // exactly the logged track
    expect(block()).toHaveTextContent('Largest, on frame 00.000 m');
    const points = block()!.querySelector('polyline')!.getAttribute('points')!;
    expect(points).not.toMatch(/NaN|Infinity/);
    expect(points.split(' ')).toHaveLength(91);
  });

  it('copes with a single frame both tracks share', async () => {
    await open({ perturbed: makeTrack({ agent_idx: 1, frames: [25], path: [[25, 3]], headings: [0] }), collision: 25 });
    expect(block()).toHaveTextContent('At the collision frame (frame 25)3.000 m');
    expect(block()).toHaveTextContent('On the last frame both were observed (frame 25)3.000 m');
    expect(block()!.querySelector('polyline')!.getAttribute('points')).not.toMatch(/NaN|Infinity/);
  });

  it('is shown even when no perturbation vector is stored, as long as both tracks are', async () => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, logged]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ target_idx: 1, collision_timestep: 25, delta: null, baseline: logged, perturbed: perturbedTrack() }));
      return ok(makeDetail({ stress_tested: true, stress_outcome: 'collision_found', collision_timestep: 25 }));
    });
    renderApp(`/scenarios/${ID}`);
    await screen.findByText('No stored perturbation.');
    expect(await screen.findByRole('group', { name: 'Offset from the recording, by frame' })).toBeInTheDocument();
  });

  it('counts only the frames both tracks were observed on', async () => {
    // Frames 0 to 38 and 50 to 89: 79 shared frames, and nothing in the gap from 39 to 49.
    await open({ perturbed: perturbedTrack([...range(0, 39), ...range(50, 90)]) });
    await userEvent.click(within(block()!).getByText('Offset at every frame'));
    await within(block()!).findByRole('table');
    const rows = within(block()!.querySelector('details')!).getAllByRole('row').slice(1);
    expect(rows).toHaveLength(79);
    const shown = rows.map((r) => Number(within(r).getAllByRole('cell')[0]!.textContent));
    expect(shown.some((frame) => frame >= 39 && frame <= 49)).toBe(false); // the gap: no frame is borrowed from beside it
  });

  it('leaves out the collision line when the collision frame is not one both tracks share', async () => {
    await open({ perturbed: perturbedTrack(range(30, 90)), collision: 25 });
    expect(block()).not.toHaveTextContent('At the collision frame');
    expect(block()).toHaveTextContent('Largest, on frame 606.000 m');
  });

  it('is not there without a perturbed path, and the stored components still are', async () => {
    await open({ perturbed: null });
    expect(block()).toBeNull();
    expect(screen.getByText('initial speed (m/s)')).toBeInTheDocument();
  });

  it('is not there when the two tracks share no frame', async () => {
    await open({ perturbed: makeTrack({ agent_idx: 1, frames: range(200, 205), path: range(200, 205).map((f) => [f, 0]), headings: range(200, 205).map(() => 0) }) });
    expect(block()).toBeNull();
  });

  it('shows the figures from the data it is given', async () => {
    // The same rule on other numbers: 0.2 per frame all the way, so 5 at frame 25 and 18 at frame 90.
    const frames = range(0, 91);
    await open({ perturbed: makeTrack({ agent_idx: 1, frames, path: frames.map((f) => [f, 0.2 * f]), headings: frames.map(() => 0) }) });
    expect(block()).toHaveTextContent('At the collision frame (frame 25)5.000 m');
    expect(block()).toHaveTextContent('Largest, on frame 9018.000 m');
  });
});
