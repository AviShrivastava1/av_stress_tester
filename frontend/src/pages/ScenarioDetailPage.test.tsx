import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import {
  makeDetail,
  makePage,
  makePerturbed,
  makeRow,
  makeTrack,
  makeTrajectories,
  range,
} from '../test/fixtures';
import { stubFetch, stubFetchDeferred, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });

interface Replies {
  detail?: Reply;
  trajectories?: Reply;
  perturbed?: Reply;
  list?: Reply;
}

/** Routes each request by endpoint; anything unconfigured gets a harmless default. */
function server(replies: Replies) {
  return (req: RecordedRequest): Reply => {
    if (req.path === '/scenarios') return replies.list ?? ok(makePage([], null));
    if (req.path.endsWith('/trajectories')) return replies.trajectories ?? ok(makeTrajectories([]));
    if (req.path.endsWith('/perturbed')) return replies.perturbed ?? ok(makePerturbed());
    return replies.detail ?? ok(makeDetail());
  };
}

// A collision scenario: SDC along +x, challenger heading +y, colliding at frame 25.
const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
const challengerLogged = makeTrack({
  agent_idx: 1,
  frames: range(0, 91),
  origin: [0, -28],
  heading: Math.PI / 2,
});
const challengerPerturbed = makeTrack({
  agent_idx: 1,
  frames: range(0, 91),
  origin: [1, -28],
  heading: Math.PI / 2,
});
const collisionPerturbed = makePerturbed({
  target_idx: 1,
  delta: [0.02, -0.01, 0.009, -0.011],
  delta_labels: ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'],
  min_perturbation: 0.123,
  collision_timestep: 25,
  baseline: challengerLogged,
  perturbed: challengerPerturbed,
});
const collisionDetail = makeDetail({
  stress_tested: true,
  stress_attempted: true,
  stress_outcome: 'collision_found',
  last_attempt_outcome: 'collision_found',
  min_perturbation: 0.123,
  collision_timestep: 25,
});

function footprints() {
  return [...document.querySelectorAll<SVGElement>('[data-footprint]')].map((el) => ({
    role: el.dataset.footprint,
    shape: el.dataset.shape,
    colliding: el.dataset.colliding === 'true',
    ghost: el.classList.contains('ghost'),
  }));
}

describe('ScenarioDetailPage — loading', () => {
  it('starts all three requests before any of them has answered', async () => {
    const { requests, releaseAll } = stubFetchDeferred(
      server({
        detail: ok(collisionDetail),
        trajectories: ok(makeTrajectories([sdc, challengerLogged])),
        perturbed: ok(collisionPerturbed),
      }),
    );
    renderApp(`/scenarios/${ID}`);

    // Nothing has been released yet, so a chained request could not have started.
    await waitFor(() => expect(requests).toHaveLength(3));
    expect(requests.map((r) => r.path).sort()).toEqual([
      `/scenarios/${ID}`,
      `/scenarios/${ID}/perturbed`,
      `/scenarios/${ID}/trajectories`,
    ]);

    releaseAll();
    expect(await screen.findByText('collision found')).toBeInTheDocument();
  });

  it('encodes the scenario ID in every request and shows it decoded', async () => {
    const requests = stubFetch(server({}));
    renderApp(`/scenarios/${encodeURIComponent('odd/id 50%')}`);

    await waitFor(() => expect(requests).toHaveLength(3));
    for (const r of requests) expect(r.path.startsWith('/scenarios/odd%2Fid%2050%25')).toBe(true);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('odd/id 50%');
  });
});

describe('ScenarioDetailPage — collision scenario', () => {
  it('draws footprints at the collision frame, the colliding pair marked', async () => {
    stubFetch(
      server({
        detail: ok(collisionDetail),
        trajectories: ok(makeTrajectories([sdc, challengerLogged])),
        perturbed: ok(collisionPerturbed),
      }),
    );
    renderApp(`/scenarios/${ID}`);

    expect(await screen.findByText(/Positions at frame 25, the first colliding frame/)).toBeInTheDocument();
    expect(footprints()).toEqual([
      { role: 'sdc', shape: 'box', colliding: true, ghost: false },
      { role: 'challenger_logged', shape: 'box', colliding: false, ghost: true },
      { role: 'challenger_perturbed', shape: 'box', colliding: true, ghost: false },
    ]);
    expect(screen.getByRole('button', { name: 'Challenger & SDC' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('labels the delta with the names the API sent', async () => {
    stubFetch(server({ detail: ok(collisionDetail), perturbed: ok(collisionPerturbed) }));
    renderApp(`/scenarios/${ID}`);

    const table = await screen.findByRole('table');
    expect(within(table).getByText('initial speed (m/s)')).toBeInTheDocument();
    expect(within(table).getByText('0.0200')).toBeInTheDocument();
    expect(screen.queryByTestId('parameterization-unknown')).not.toBeInTheDocument();
  });

  it('draws a dot, not a box, for a perturbed track without dimensions', async () => {
    stubFetch(
      server({
        detail: ok(collisionDetail),
        trajectories: ok(makeTrajectories([sdc, challengerLogged])),
        perturbed: ok({
          ...collisionPerturbed,
          baseline: null,
          perturbed: { ...challengerPerturbed, length_m: null, width_m: null },
        }),
      }),
    );
    renderApp(`/scenarios/${ID}`);

    await screen.findByText(/the first colliding frame/);
    expect(footprints().find((f) => f.role === 'challenger_perturbed')).toMatchObject({ shape: 'dot' });
  });

  it('frames the whole scene on request', async () => {
    const bystander = makeTrack({ agent_idx: 2, agent_type: 3, frames: range(0, 91), origin: [300, 300] });
    stubFetch(
      server({
        detail: ok(collisionDetail),
        trajectories: ok(makeTrajectories([sdc, challengerLogged, bystander])),
        perturbed: ok(collisionPerturbed),
      }),
    );
    renderApp(`/scenarios/${ID}`);

    const svg = await screen.findByRole('img');
    const focused = svg.getAttribute('viewBox')!.split(' ').map(Number);
    await userEvent.click(screen.getByRole('button', { name: 'Whole scene' }));
    const whole = screen.getByRole('img').getAttribute('viewBox')!.split(' ').map(Number);
    expect(whole[2]!).toBeGreaterThan(focused[2]!); // wider: the bystander 300 m away is now in frame
    expect(screen.getByText('Other agent: cyclist')).toBeInTheDocument();
  });
});

describe('ScenarioDetailPage — delta labels', () => {
  it('shows "parameterization unknown" and no units when labels are null', async () => {
    stubFetch(server({ perturbed: ok({ ...collisionPerturbed, delta_labels: null }) }));
    renderApp(`/scenarios/${ID}`);

    expect(await screen.findByTestId('parameterization-unknown')).toBeInTheDocument();
    const table = screen.getByRole('table');
    expect(within(table).getByText('component 1')).toBeInTheDocument();
    expect(table.textContent).not.toMatch(/m\/s|rad/);
  });

  it('treats labels that do not match the delta one-to-one as unknown', async () => {
    stubFetch(server({ perturbed: ok({ ...collisionPerturbed, delta_labels: ['initial speed (m/s)'] }) }));
    renderApp(`/scenarios/${ID}`);

    expect(await screen.findByTestId('parameterization-unknown')).toBeInTheDocument();
    expect(screen.queryByText('initial speed (m/s)')).not.toBeInTheDocument();
  });

  it('says so when there is no stored perturbation', async () => {
    stubFetch(server({}));
    renderApp(`/scenarios/${ID}`);
    expect(await screen.findByText('No stored perturbation.')).toBeInTheDocument();
  });
});

describe('ScenarioDetailPage — reference frame without a collision', () => {
  it('starts where every focused track is observed, for a late-starting agent', async () => {
    const lateAgent = makeTrack({ agent_idx: 1, frames: range(20, 41), origin: [0, 10] });
    stubFetch(server({ trajectories: ok(makeTrajectories([sdc, lateAgent])) }));
    renderApp(`/scenarios/${ID}`);

    expect(
      await screen.findByText(/Positions at frame 20, the earliest frame where all 2 focused tracks are observed/),
    ).toBeInTheDocument();
    expect(footprints().map((f) => f.role)).toEqual(['other', 'sdc']);
  });

  it('says so when no frame has every focused track observed', async () => {
    const early = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 10) });
    const late = makeTrack({ agent_idx: 1, frames: range(20, 30), origin: [0, 10] });
    stubFetch(server({ trajectories: ok(makeTrajectories([early, late])) }));
    renderApp(`/scenarios/${ID}`);

    expect(await screen.findByText(/No frame has all 2 focused tracks observed together/)).toBeInTheDocument();
  });

  it('does not use the collision frame when the perturbed path is not drawn', async () => {
    stubFetch(
      server({
        detail: ok(collisionDetail),
        trajectories: ok(makeTrajectories([sdc, challengerLogged])),
        perturbed: ok({ ...collisionPerturbed, baseline: null, perturbed: null }),
      }),
    );
    renderApp(`/scenarios/${ID}`);

    expect(
      await screen.findByText(/A stored result exists, but no perturbed path matching it has been exported/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/first colliding frame/)).not.toBeInTheDocument();
    expect(footprints().some((f) => f.colliding)).toBe(false);
  });
});

describe('ScenarioDetailPage — states', () => {
  it('shows an unknown scenario for a 404', async () => {
    stubFetch(server({ detail: { status: 404, body: { detail: 'Unknown scenario: nope' } } }));
    renderApp('/scenarios/nope');
    expect(await screen.findByRole('heading', { name: 'Unknown scenario' })).toBeInTheDocument();
  });

  it('treats no geometry and no stress test as normal states, not errors', async () => {
    stubFetch(server({}));
    renderApp(`/scenarios/${ID}`);

    expect(await screen.findByText(/No geometry to draw/)).toBeInTheDocument();
    expect(await screen.findByText('No stored perturbation.')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows a geometry 500 as an error in the scene panel only', async () => {
    stubFetch(
      server({
        detail: ok(collisionDetail),
        trajectories: {
          status: 500,
          body: { detail: 'Geometry inconsistent for synthetic-0001 agent 3: 91 coordinates, 90 measures' },
        },
        perturbed: ok(collisionPerturbed),
      }),
    );
    renderApp(`/scenarios/${ID}`);

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/Scene geometry could not be loaded/);
    expect(alert).toHaveTextContent(/Geometry inconsistent/);
    expect(screen.queryByText(/No geometry to draw/)).not.toBeInTheDocument();
    expect(await screen.findByText('collision found')).toBeInTheDocument(); // summary unaffected
    expect(screen.getByRole('table')).toBeInTheDocument(); // perturbation unaffected
  });

  it('names unknown agent types with their code', async () => {
    stubFetch(server({ trajectories: ok(makeTrajectories([sdc, makeTrack({ agent_idx: 4, agent_type: 7 })])) }));
    renderApp(`/scenarios/${ID}`);

    await screen.findByRole('img');
    const titles = [...document.querySelectorAll('[data-agent="4"] > title')].map((t) => t.textContent);
    expect(titles[0]).toMatch(/unknown type \(code 7\)/);
  });
});

describe('navigation', () => {
  it('links from the list and returns to the same filtered list', async () => {
    stubFetch(server({ list: ok(makePage([makeRow({ scenario_id: 'odd/id' })], null)) }));
    renderApp('/?tested=1');

    const link = await screen.findByRole('link', { name: 'odd/id' });
    expect(link).toHaveAttribute('href', '/scenarios/odd%2Fid');
    await userEvent.click(link);

    const back = await screen.findByRole('link', { name: /All scenarios/ });
    expect(back).toHaveAttribute('href', '/?tested=1');
  });

  it('goes back to the unfiltered list when opened directly', async () => {
    stubFetch(server({}));
    renderApp(`/scenarios/${ID}`);
    expect(await screen.findByRole('link', { name: /All scenarios/ })).toHaveAttribute('href', '/');
  });
});
