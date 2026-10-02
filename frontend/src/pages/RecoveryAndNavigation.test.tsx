import { act, fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createMemoryRouter, RouterProvider, type NonIndexRouteObject } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { HealthResponse, StatsResponse } from '../api/client';
import { routes } from '../App';
import { renderApp } from '../test/renderApp';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import {
  makeDetail, makePage, makePerturbed, makeRow, makeTrack, makeTrajectories, range,
} from '../test/fixtures';

afterEach(() => {
  vi.restoreAllMocks();
});

const ok = (body: unknown): Reply => ({ status: 200, body });
const busy: Reply = { status: 503, body: { detail: 'busy' } };

describe('unknown addresses and render failures', () => {
  it('shows a page of its own for an unknown URL, with navigation intact', async () => {
    renderApp('/a-page-that-does-not-exist');
    expect(await screen.findByRole('heading', { name: 'Page not found' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Back to scenarios' })).toHaveAttribute('href', '/');
    const nav = screen.getByRole('navigation', { name: 'Main' });
    expect(nav).toHaveTextContent('Scenarios');
    expect(nav).toHaveTextContent('Corpus');
  });

  it('shows a recoverable page, with navigation, when a route throws while rendering', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    function Boom(): never {
      throw new Error('boom');
    }
    const root = routes[0] as NonIndexRouteObject;
    const withBoom: NonIndexRouteObject[] = [
      { ...root, children: [...(root.children ?? []), { path: 'boom', element: <Boom /> }] },
    ];
    const router = createMemoryRouter(withBoom, { initialEntries: ['/boom'] });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
    expect(await screen.findByRole('heading', { name: 'This page could not be displayed' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Back to scenarios' })).toHaveAttribute('href', '/');
    expect(screen.getByRole('navigation', { name: 'Main' })).toHaveTextContent('Corpus');
    expect(screen.queryByText(/Unexpected Application Error/)).not.toBeInTheDocument();
  });
});

// One test per place a failed request is shown. Each starts with that endpoint failing
// and the rest healthy, so exactly one "Try again" exists, and a click must send the
// request again and show what it returns.
describe('retry after a failed request, without reloading', () => {
  const ID = 'synthetic-0001';
  const labels = ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'];
  const stats: StatsResponse = {
    total_scenarios: 1234, stress_tested: 40, collisions_found: 17, no_collision_found: 11,
    replay_infeasible: 5, heading_blend_singularity: 3, no_challenger: 2, stress_errors: 1,
    robustly_safe: 0, with_geometry: 38, fragility_min: 0.25, fragility_max: 4.5, fragility_mean: 1.125,
  };
  const health: HealthResponse = { status: 'ok', database: 'connected', postgis: '3.6 USE_GEOS=1' };
  const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 11) });
  const challenger = makeTrack({ agent_idx: 1, frames: range(0, 11), origin: [0, 5] });

  type Endpoint = 'list' | 'detail' | 'trajectories' | 'perturbed' | 'stats' | 'health';

  /** Serves every endpoint healthily, except `failing`, which answers 503 until `heal()`. */
  function serverFailing(failing: Endpoint) {
    let healed = false;
    const calls: Record<string, number> = {};
    const respond = (req: RecordedRequest): Reply => {
      const endpoint: Endpoint =
        req.path === '/scenarios' ? 'list'
        : req.path === '/stats' ? 'stats'
        : req.path === '/health' ? 'health'
        : req.path.endsWith('/trajectories') ? 'trajectories'
        : req.path.endsWith('/perturbed') ? 'perturbed'
        : 'detail';
      calls[endpoint] = (calls[endpoint] ?? 0) + 1;
      if (endpoint === failing && !healed) return busy;
      switch (endpoint) {
        case 'list': return ok(makePage([makeRow()], null));
        case 'stats': return ok(stats);
        case 'health': return ok(health);
        case 'trajectories': return ok(makeTrajectories([sdc, challenger]));
        case 'perturbed': return ok(makePerturbed({ target_idx: 1, delta: [0.5, 0.25, 0.125, 0.0625], delta_labels: labels }));
        case 'detail': return ok(makeDetail());
      }
    };
    stubFetch(respond);
    return { heal: () => { healed = true; }, calls };
  }

  async function retry(server: ReturnType<typeof serverFailing>, endpoint: Endpoint) {
    await screen.findByRole('alert');
    expect(server.calls[endpoint]).toBe(1);
    server.heal();
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
  }

  it('on the scenario list', async () => {
    const server = serverFailing('list');
    renderApp('/');
    await retry(server, 'list');
    expect(await screen.findByRole('link', { name: ID })).toBeInTheDocument();
    expect(server.calls.list).toBe(2);
  });

  it('on the scenario summary', async () => {
    const server = serverFailing('detail');
    renderApp(`/scenarios/${ID}`);
    await retry(server, 'detail');
    expect(await screen.findByText('Min perturbation')).toBeInTheDocument();
    expect(server.calls.detail).toBe(2);
  });

  it('on the scene geometry', async () => {
    const server = serverFailing('trajectories');
    renderApp(`/scenarios/${ID}`);
    await retry(server, 'trajectories');
    expect(await screen.findByTestId('frame-readout')).toBeInTheDocument();
    expect(server.calls.trajectories).toBe(2);
  });

  it('on the perturbation vector', async () => {
    const server = serverFailing('perturbed');
    renderApp(`/scenarios/${ID}`);
    await retry(server, 'perturbed');
    expect(await screen.findByText('initial speed (m/s)')).toBeInTheDocument();
    expect(server.calls.perturbed).toBe(2);
  });

  it('on the corpus statistics', async () => {
    const server = serverFailing('stats');
    renderApp('/stats');
    await retry(server, 'stats');
    expect(await screen.findByRole('heading', { name: 'Stored results' })).toBeInTheDocument();
    expect(server.calls.stats).toBe(2);
  });

  it('on the service status', async () => {
    const server = serverFailing('health');
    renderApp('/stats');
    await retry(server, 'health');
    expect(await screen.findByText('connected')).toBeInTheDocument();
    expect(server.calls.health).toBe(2);
  });

  // A refetch that fails while earlier data is still held leaves the query in `error`
  // with `isFetching` true during the retry; this is the only case where the button's
  // "Retrying…" state is on screen.
  it('disables the button, saying so, while a retry is in flight', async () => {
    const json = (reply: Reply) => new Response(JSON.stringify(reply.body), {
      status: reply.status, headers: { 'Content-Type': 'application/json' },
    });
    let statsCalls = 0;
    let answerRetry: () => void = () => {};
    vi.stubGlobal('fetch', (input: Request) => new Promise<Response>((resolve) => {
      if (new URL(input.url).pathname === '/health') return resolve(json(ok(health)));
      statsCalls += 1;
      if (statsCalls === 1) resolve(json(ok(stats)));
      else if (statsCalls === 2) resolve(json(busy));
      else answerRetry = () => resolve(json(ok({ ...stats, total_scenarios: 4321 })));
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const router = createMemoryRouter(routes, { initialEntries: ['/stats'] });
    render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
    await screen.findByText('1,234');
    await act(async () => { await client.refetchQueries({ queryKey: ['stats'] }); });
    await screen.findByRole('alert');
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('button', { name: 'Retrying…' })).toBeDisabled();
    await act(async () => { answerRetry(); });
    expect(await screen.findByText('4,321')).toBeInTheDocument();
  });
});

describe('scene playback across scenarios', () => {
  it('starts a different cached scenario at its own reference frame, paused', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    for (const [id, collision] of [['scene-a', 2], ['scene-b', 6]] as const) {
      const sdc = makeTrack({ is_sdc: true });
      const target = makeTrack({ agent_idx: 1, origin: [0, 5] });
      client.setQueryData(['scenario', id], makeDetail({ scenario_id: id, collision_timestep: collision }));
      client.setQueryData(['scenario', id, 'trajectories'], makeTrajectories([sdc, target], id));
      client.setQueryData(['scenario', id, 'perturbed'], makePerturbed({
        scenario_id: id, target_idx: 1, baseline: target, perturbed: target, collision_timestep: collision,
      }));
    }
    const router = createMemoryRouter(routes, { initialEntries: ['/scenarios/scene-a'] });
    render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
    await screen.findByTestId('frame-readout');
    fireEvent.change(screen.getByRole('slider', { name: 'Frame' }), { target: { value: '8' } });
    fireEvent.click(screen.getByRole('button', { name: 'Play' }));
    await act(async () => { await router.navigate('/scenarios/scene-b'); });
    expect(screen.getByTestId('frame-readout')).toHaveTextContent('frame 6 of');
    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
  });
});
