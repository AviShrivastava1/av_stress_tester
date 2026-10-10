import { screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { makeDetail, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The header of the replay panel: the recording's frame spacing, with its rate. ScenePanel.tsx derives the rate
 * from the same FRAME_INTERVAL_S it prints, so the two cannot disagree.
 */

afterEach(() => {
  vi.unstubAllGlobals();
});

const ok = (body: unknown): Reply => ({ status: 200, body });
const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });

describe('the replay panel header', () => {
  it('gives the recording rate and the frame interval', async () => {
    stubFetch((req: RecordedRequest) =>
      ok(req.path.endsWith('/trajectories') ? makeTrajectories([sdc]) : req.path.endsWith('/perturbed') ? { baseline: null, perturbed: null } : makeDetail()),
    );
    renderApp('/scenarios/synthetic-0001');
    expect(await screen.findByText('10 Hz · 0.1 s / frame')).toBeVisible();
    expect(screen.queryByText('0.1 s / frame')).not.toBeInTheDocument();
  });
});
