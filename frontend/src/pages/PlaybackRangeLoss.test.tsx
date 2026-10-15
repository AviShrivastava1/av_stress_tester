import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { UseQueryResult } from '@tanstack/react-query';
import { describe, expect, it } from 'vitest';
import type { PerturbedResponse, TrajectoryResponse } from '../api/client';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { ScenePanel } from './detail/ScenePanel';

/*
 * The component half of the playback fix: on a real scene panel, press Play, let the geometry go away (a refetch that
 * returns no tracks), and let it come back. The button must read "Play" the whole time after the geometry is gone and
 * the replay must not resume by itself; pressing Play starts it again.
 */

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 31), origin: [-20, 0] });
const other = makeTrack({ agent_idx: 1, frames: range(0, 31), origin: [0, -28], heading: Math.PI / 2 });

const query = <T,>(data: T): UseQueryResult<T> =>
  ({ data, isPending: false, isError: false, isFetching: false, error: null, refetch: () => undefined }) as unknown as UseQueryResult<T>;
const withTracks = (agents = [sdc, other]) => query<TrajectoryResponse>(makeTrajectories(agents));
const perturbed = query<PerturbedResponse>(makePerturbed());

function panel(trajectories: UseQueryResult<TrajectoryResponse>) {
  return <ScenePanel trajectories={trajectories} perturbed={perturbed} detail={makeDetail()} />;
}

describe('the scene panel when the geometry goes away while the replay is playing', () => {
  it('reads Play after the geometry is gone and again after it returns, and plays only when pressed', async () => {
    const user = userEvent.setup();
    const { rerender } = render(panel(withTracks()));
    await user.click(await screen.findByRole('button', { name: 'Play' }));
    expect(screen.getByRole('button', { name: 'Pause' })).toBeVisible();

    rerender(panel(withTracks([])));
    expect(screen.queryByRole('button', { name: 'Pause' })).not.toBeInTheDocument();

    rerender(panel(withTracks()));
    expect(await screen.findByRole('button', { name: 'Play' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Pause' })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Play' }));
    expect(screen.getByRole('button', { name: 'Pause' })).toBeVisible();
  });
});
