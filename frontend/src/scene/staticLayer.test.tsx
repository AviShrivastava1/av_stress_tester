import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';
import * as geometry from './geometry';

// `segments` is called only by the static layer (paths), once per track per render,
// so its call count is a direct measure of how often that layer renders.
vi.mock('./geometry', async (importOriginal) => {
  const real = await importOriginal<typeof import('./geometry')>();
  return { ...real, segments: vi.fn(real.segments) };
});

const ok = (body: unknown): Reply => ({ status: 200, body });
const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91) });
const challenger = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -20], heading: Math.PI / 2 });
const bystander = makeTrack({ agent_idx: 2, frames: range(0, 91), origin: [200, 200] });

describe('static layer', () => {
  it('does not re-render as the playhead moves, only when the framing changes', async () => {
    stubFetch((req: RecordedRequest) => {
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, challenger, bystander]));
      if (req.path.endsWith('/perturbed')) {
        return ok(makePerturbed({ target_idx: 1, collision_timestep: 25, baseline: challenger, perturbed: challenger }));
      }
      return ok(makeDetail());
    });
    renderApp('/scenarios/synthetic-0001');
    await screen.findByTestId('frame-readout');

    const segmentsSpy = vi.mocked(geometry.segments);
    const settled = segmentsSpy.mock.calls.length;
    const stepForward = screen.getByRole('button', { name: 'Step forward one frame' });
    for (let i = 0; i < 5; i++) fireEvent.click(stepForward);
    fireEvent.change(screen.getByRole('slider', { name: 'Frame' }), { target: { value: '70' } });
    expect(screen.getByTestId('frame-readout')).toHaveTextContent('frame 70');
    expect(segmentsSpy.mock.calls.length).toBe(settled);

    // Positive control: a framing change DOES re-render it, so the spy is live.
    fireEvent.click(screen.getByRole('button', { name: 'Whole scene' }));
    expect(segmentsSpy.mock.calls.length).toBeGreaterThan(settled);
  });
});
