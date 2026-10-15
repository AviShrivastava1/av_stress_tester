import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { StrictMode } from 'react';
import { RouterProvider, createMemoryRouter } from 'react-router';
import { describe, expect, it, vi } from 'vitest';
import { routes } from '../App';
import type { ScenarioSummary } from '../api/client';
import { formatMagnitude } from '../domain/format';
import { describeOutcome } from '../domain/outcomes';
import { makePage, makeRow, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { LAST_FRAME_NOTE } from '../scene/sceneModel';
import { readFrontendFile } from '../test/readFrontendFile';
import { renderApp } from '../test/renderApp';
import type { RecordedRequest, Reply } from '../test/fetchStub';

/*
 * The walkthrough (pages/WalkthroughPage.tsx, domain/walkthrough.ts) shows three stored results chosen by rule from what the
 * API returns, never by a scenario id or a number written into the page:
 *
 *   smallest   the smallest min_perturbation among collisions that do NOT happen on the last recorded frame
 *   largest    the largest min_perturbation among all collisions
 *   refused    the first scenario by id whose outcome is replay infeasible
 *
 * The recording's last frame is not in the list or in /perturbed, so it is read from /trajectories (recordingLastFrame, the
 * largest timestep of every track, as the detail page's last-frame line reads it), for one candidate at a time, smallest
 * perturbation first, stopping at the first that qualifies. These fixtures are built so that each rule picks a scene other
 * than the first row that could have been taken, and the ids and numbers are synthetic.
 */

const collision = (id: string, perturbation: number, frame: number): ScenarioSummary =>
  makeRow({
    scenario_id: id,
    stress_tested: true,
    stress_attempted: true,
    stress_outcome: 'collision_found',
    last_attempt_outcome: 'collision_found',
    min_perturbation: perturbation,
    collision_timestep: frame,
    n_agents: 12,
  });
// As the API serves a refused scenario: no stored result, and the last attempt was refused. (The Corpus count of refused
// scenarios, GET /stats replay_infeasible, counts last_attempt_outcome.)
const refused = (id: string, agents = 9): ScenarioSummary =>
  makeRow({
    scenario_id: id,
    stress_attempted: true,
    stress_outcome: null,
    last_attempt_outcome: 'replay_infeasible',
    n_agents: agents,
  });

// In the API's order (most fragile first). The first refused row is NOT the first by id; the first collisions are all on the
// last frame; the largest perturbation is also on the last frame, which only the "smallest" rule excludes.
const ROWS: ScenarioSummary[] = [
  refused('wz-refused'),
  collision('wa-edge-1', 0.011, 90),
  collision('wb-edge-2', 0.012, 90),
  collision('wc-small', 0.0126, 40),
  collision('wd-mid', 0.5, 30),
  collision('we-huge-edge', 0.9, 90),
  refused('wb-refused', 7),
  makeRow({ scenario_id: 'wf-untested' }),
  // Stored as collisions but unusable: no perturbation to rank, and no collision frame to compare with the recording's end.
  { ...collision('wg-nopert', 0, 5), min_perturbation: null },
  { ...collision('wh-noframe', 0.001, 5), collision_timestep: null },
  // Carries a perturbation and a frame (an earlier result, say) but its stored outcome is not a collision.
  { ...collision('wj-notcollision', 0.0005, 20), stress_outcome: 'no_collision_found', last_attempt_outcome: 'no_collision_found' },
];

const LAST_FRAME: Record<string, number> = { default: 90 };

function tracksFor(id: string) {
  const last = LAST_FRAME[id] ?? LAST_FRAME.default!;
  // Two tracks that end at different frames: the recording's end is the later of them.
  return makeTrajectories([
    makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, last - 5) }),
    makeTrack({ agent_idx: 1, frames: range(0, last + 1), origin: [0, 10] }),
  ], id);
}

interface Options {
  rows?: ScenarioSummary[];
  pageSize?: number;
  /** Ids whose trajectories are held until `release(id)`. */
  held?: string[];
  trajectories?: (id: string) => Reply;
}

function install({ rows = ROWS, pageSize = 5, held = [], trajectories }: Options = {}) {
  const requests: RecordedRequest[] = [];
  const waiting = new Map<string, () => void>();
  const fixed = new Set<string>();
  const trajectoryIds = () => requests.filter((r) => r.path.endsWith('/trajectories')).map((r) => r.path.split('/')[2]!);

  function reply(req: RecordedRequest): Reply {
    if (req.path === '/scenarios') {
      const start = Number(req.query.get('cursor') ?? 0);
      const items = rows.slice(start, start + pageSize);
      const next = start + pageSize < rows.length ? String(start + pageSize) : null;
      return { status: 200, body: makePage(items, next) };
    }
    if (req.path.endsWith('/trajectories')) {
      const id = req.path.split('/')[2]!;
      return trajectories ? trajectories(id) : { status: 200, body: tracksFor(id) };
    }
    return { status: 404, body: { detail: 'not under test' } };
  }

  vi.stubGlobal(
    'fetch',
    (input: Request) =>
      new Promise<Response>((resolve) => {
        const url = new URL(input.url);
        const req: RecordedRequest = { path: url.pathname, query: url.searchParams };
        requests.push(req);
        const respond = () => {
          const { status, body } = reply(req);
          resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }));
        };
        const id = req.path.split('/')[2];
        if (req.path.endsWith('/trajectories') && id && held.includes(id) && !fixed.has(id)) waiting.set(id, respond);
        else respond();
      }),
  );
  return {
    requests,
    trajectoryIds,
    release: (id: string) => {
      fixed.add(id);
      waiting.get(id)?.();
      waiting.delete(id);
    },
    fix: (id: string) => fixed.add(id),
  };
}

/** The walkthrough in the real route table, with the production query client's stale time: a response the page already has is reused, not asked for again. */
function renderWalk() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 60_000 } } });
  const router = createMemoryRouter(routes, { initialEntries: ['/walkthrough'] });
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
}

/** The recordings read to choose the smallest, in the order asked: every request but the one for the largest scene. */
const probed = (server: { trajectoryIds(): string[] }) => server.trajectoryIds().filter((id) => id !== 'we-huge-edge');

const section = (name: string) => screen.getByRole('region', { name });
const SMALLEST = 'Smallest change, before the recording ends';
const LARGEST = 'Largest change that still collides';
const REFUSED = 'A refused replay';

describe('the three scenes are chosen by rule from the API', () => {
  it('picks the smallest perturbation that is not on the last recorded frame, not the first or the smallest of all', async () => {
    const server = install();
    renderWalk();
    await screen.findByRole('heading', { level: 1, name: 'Three stored results' });
    await waitFor(() => expect(within(section(SMALLEST)).getByText('wc-small')).toBeInTheDocument());

    const card = within(section(SMALLEST));
    expect(section(SMALLEST)).toHaveTextContent(
      `has the smallest stored perturbation, ${formatMagnitude(0.0126)}, among the collisions that happen before the last recorded frame. ` +
        'The collision is on frame 40; the recording runs to frame 90.',
    );
    expect(card.getByRole('link', { name: 'Open scenario wc-small' })).toHaveAttribute('href', '/scenarios/wc-small');
    // Cheap: the two smallest were looked at because they come first, the largest scene's recording once (for its last-frame
    // sentence), and nothing else.
    expect(probed(server)).toEqual(['wa-edge-1', 'wb-edge-2', 'wc-small']);
    expect(server.trajectoryIds().filter((id) => id === 'we-huge-edge')).toHaveLength(1);
    expect(server.trajectoryIds()).toHaveLength(4);
  });

  it('picks the largest perturbation among all collisions, last frame or not', async () => {
    install();
    renderWalk();
    await waitFor(() => expect(within(section(LARGEST)).getByText('we-huge-edge')).toBeInTheDocument());
    await waitFor(() => expect(within(section(SMALLEST)).getByText('wc-small')).toBeInTheDocument()); // both cards settled
    expect(section(LARGEST)).toHaveTextContent(
      `Scenario we-huge-edge has the largest stored perturbation, ${formatMagnitude(0.9)}, among all the collisions. The collision is on frame 90.`,
    );
    expect(within(section(LARGEST)).getByRole('link', { name: 'Open scenario we-huge-edge' })).toBeInTheDocument();
    expect(section(LARGEST)).not.toHaveTextContent('This is the scenario above.');
  });

  it('picks the first refused scenario by id, not the first in the list', async () => {
    install();
    renderWalk();
    await waitFor(() => expect(within(section(REFUSED)).getByText('wb-refused')).toBeInTheDocument());
    const meaning = describeOutcome('replay_infeasible').meaning;
    expect(section(REFUSED)).toHaveTextContent(
      `Scenario wb-refused is the first, by scenario ID, that the API reports as replay infeasible. ${meaning} The scene has 7 agents.`,
    );
    expect(within(section(REFUSED)).getByRole('link', { name: 'Open scenario wb-refused' })).toBeInTheDocument();
  });

  it('chooses again from different data: the ids are not in the page', async () => {
    install({
      rows: [
        collision('xa', 0.2, 10),
        collision('xb', 0.04, 20),
        collision('xc', 0.3, 30),
        refused('xd'),
        refused('xc-first'),
      ],
    });
    renderWalk();
    await waitFor(() => expect(within(section(SMALLEST)).getByText('xb')).toBeInTheDocument());
    expect(within(section(LARGEST)).getByText('xc')).toBeInTheDocument();
    expect(within(section(REFUSED)).getByText('xc-first')).toBeInTheDocument();
  });

  it('writes no scenario id and no number of its own into the page', () => {
    const source = readFrontendFile('src/pages/WalkthroughPage.tsx') + readFrontendFile('src/domain/walkthrough.ts');
    expect(source).not.toMatch(/\b[0-9a-f]{12,}\b/); // no id of the stored corpus
    const prose = [...source.matchAll(/(['"`])((?:(?!\1)[^\\\n]|\\.)*?)\1/g)].map((m) => m[2]!).filter((s) => /[A-Za-z]{4}/.test(s));
    expect(prose.filter((s) => /\d/.test(s.replace(/\$\{[^}]*\}/g, '')))).toEqual([]);
  });
});

describe('where the recording ends is read one candidate at a time, and cheaply', () => {
  it('asks for the next recording only after the one before has answered', async () => {
    const server = install({ held: ['wa-edge-1'] });
    renderWalk();
    expect(await screen.findByText('Checking where each recording ends…')).toBeInTheDocument();
    await waitFor(() => expect(probed(server)).toEqual(['wa-edge-1']));
    await new Promise((r) => setTimeout(r, 50));
    expect(probed(server)).toEqual(['wa-edge-1']); // one at a time

    await act(async () => server.release('wa-edge-1'));
    await waitFor(() => expect(within(section(SMALLEST)).getByText('wc-small')).toBeInTheDocument());
    expect(probed(server)).toEqual(['wa-edge-1', 'wb-edge-2', 'wc-small']);
  });

  it('asks only for the recordings it needs: the candidates one at a time, the largest scene once, nothing for the rest', async () => {
    const server = install();
    renderWalk();
    await waitFor(() => expect(within(section(SMALLEST)).getByText('wc-small')).toBeInTheDocument());
    expect(server.trajectoryIds().filter((id) => id === 'we-huge-edge')).toHaveLength(1); // for its last-frame sentence
    expect(server.trajectoryIds()).not.toContain('wb-refused');
    expect(server.trajectoryIds()).not.toContain('wf-untested');
    expect(server.trajectoryIds()).not.toContain('wg-nopert'); // no perturbation to rank
    expect(server.trajectoryIds()).not.toContain('wh-noframe'); // no collision frame to compare
    expect(server.trajectoryIds()).not.toContain('wj-notcollision'); // not a collision
  });

  it('does not take a scene whose recording it cannot read: no geometry means no last frame to compare with', async () => {
    const noGeometry = (id: string): Reply =>
      id === 'wc-small' ? { status: 200, body: makeTrajectories([], id) } : { status: 200, body: tracksFor(id) };
    install({ trajectories: noGeometry });
    renderWalk();
    await waitFor(() => expect(within(section(SMALLEST)).getByText('wd-mid')).toBeInTheDocument());
  });

  it('says so, and chooses nothing, when every collision is on the last recorded frame', async () => {
    const server = install({ rows: [collision('wa', 0.1, 90), collision('wb', 0.2, 90), refused('wr')] });
    renderWalk();
    await waitFor(() =>
      expect(section(SMALLEST)).toHaveTextContent(
        'Every stored collision is on the last recorded frame, or has no geometry to check that against, so no scene is shown here.',
      ),
    );
    expect([...server.trajectoryIds()].sort()).toEqual(['wa', 'wb']); // each once: wb is both a candidate and the largest
    // The largest does not depend on the last frame.
    expect(within(section(LARGEST)).getByText('wb')).toBeInTheDocument();
  });

  it('shows the error and chooses nothing when a recording cannot be read, and retries', async () => {
    let broken = true;
    const server = install({
      trajectories: (id) => (broken && id === 'wa-edge-1' ? { status: 500, body: { detail: 'boom' } } : { status: 200, body: tracksFor(id) }),
    });
    renderWalk();
    await screen.findByRole('region', { name: SMALLEST });
    const alert = await within(section(SMALLEST)).findByRole('alert');
    expect(alert).toHaveTextContent('500: boom');
    expect(within(section(SMALLEST)).queryByText(/Scenario wc-small/)).not.toBeInTheDocument();
    expect(probed(server)).toEqual(['wa-edge-1']);

    broken = false;
    await userEvent.click(within(section(SMALLEST)).getByRole('button', { name: 'Try again' }));
    await waitFor(() => expect(within(section(SMALLEST)).getByText('wc-small')).toBeInTheDocument());
  });
});

describe('when there is nothing to show', () => {
  it('says so for a corpus with no collision and no refused scenario', async () => {
    install({ rows: [makeRow({ scenario_id: 'only' })] });
    renderWalk();
    await waitFor(() =>
      expect(section(SMALLEST)).toHaveTextContent('No stored result has a collision, so there is no scene to show here.'),
    );
    expect(section(LARGEST)).toHaveTextContent('No stored result has a collision, so there is no scene to show here.');
    expect(section(REFUSED)).toHaveTextContent('No scenario is reported as replay infeasible.');
  });

  it('says the smallest and the largest are one scenario when only one collision qualifies', async () => {
    install({ rows: [collision('solo', 0.25, 12)] });
    renderWalk();
    await waitFor(() => expect(section(LARGEST)).toHaveTextContent('Scenario solo has the largest'));
    await waitFor(() => expect(section(LARGEST)).toHaveTextContent('This is the scenario above.'));
    expect(within(section(SMALLEST)).getByText('solo')).toBeInTheDocument();
    expect(section(SMALLEST)).not.toHaveTextContent('This is the scenario above.');
  });

  it('loads every page of the list before it chooses, and says so while it does', async () => {
    const server = install({ pageSize: 2 });
    renderWalk();
    await waitFor(() => expect(within(section(SMALLEST)).getByText('wc-small')).toBeInTheDocument());
    const pages = server.requests.filter((r) => r.path === '/scenarios');
    expect(pages).toHaveLength(Math.ceil(ROWS.length / 2));
    expect(pages.every((p) => p.query.get('stress_tested_only') === 'false')).toBe(true);
  });

  it('shows the list error with a retry when the list cannot be loaded', async () => {
    vi.stubGlobal('fetch', async () => new Response(JSON.stringify({ detail: 'down' }), { status: 500, headers: { 'Content-Type': 'application/json' } }));
    renderWalk();
    expect(await screen.findByRole('alert')).toHaveTextContent('500: down');
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
  });
});

describe('the page in the site', () => {
  it('is in the navigation and has its own title', async () => {
    install();
    renderApp('/');
    const nav = screen.getByRole('navigation', { name: 'Main' });
    await userEvent.click(within(nav).getByRole('link', { name: 'Walkthrough' }));
    await screen.findByRole('heading', { level: 1, name: 'Three stored results' });
    expect(document.title).toBe('Walkthrough · AV Scenario Stress-Tester');
  });

  it('says every figure is the smallest found by a finite search, not a proven minimum', async () => {
    install();
    renderWalk();
    await screen.findByRole('heading', { level: 1, name: 'Three stored results' });
    expect(document.body).toHaveTextContent(
      'These three scenarios are chosen by rule from the stored results each time the page loads, not by hand. ' +
        'Every perturbation is the smallest found by a finite search, not a proven minimum.',
    );
  });
});

describe('ties, missing values, and a list that is still arriving', () => {
  it('breaks a tie in perturbation by scenario ID, for the smallest and for the largest', async () => {
    install({ rows: [collision('tb', 0.1, 10), collision('ta', 0.1, 10)] });
    renderWalk();
    await waitFor(() => expect(within(section(SMALLEST)).getByText('ta')).toBeInTheDocument());
    expect(within(section(LARGEST)).getByText('ta')).toBeInTheDocument();
  });

  it('counts a scenario refused on its last attempt even when an earlier result is stored, and says so', async () => {
    install({ rows: [{ ...collision('wk', 0.2, 10), last_attempt_outcome: 'replay_infeasible' }, refused('wz')] });
    renderWalk();
    await waitFor(() => expect(within(section(REFUSED)).getByText('wk')).toBeInTheDocument());
    expect(section(REFUSED)).toHaveTextContent('It also has a stored result from an earlier pass: collision found.');
    // ...and it is still a collision for the other two rules.
    expect(within(section(LARGEST)).getByText('wk')).toBeInTheDocument();
  });

  it('does not take the stored outcome for the last attempt', async () => {
    install({ rows: [{ ...refused('wa-stored'), stress_outcome: 'replay_infeasible', last_attempt_outcome: 'no_collision_found' }, refused('wb')] });
    renderWalk();
    await waitFor(() => expect(within(section(REFUSED)).getByText('wb')).toBeInTheDocument());
  });

  it('leaves out of the refused card the agent count the API did not give', async () => {
    install({ rows: [{ ...refused('wr'), n_agents: null }] });
    renderWalk();
    await waitFor(() => expect(within(section(REFUSED)).getByText('wr')).toBeInTheDocument());
    expect(section(REFUSED)).not.toHaveTextContent('The scene has');
    expect(section(REFUSED)).not.toHaveTextContent('earlier pass');
    expect(section(REFUSED)).not.toHaveTextContent('null');
  });

  it('asks for no recording when there is no collision to ask about', async () => {
    const server = install({ rows: [makeRow({ scenario_id: 'only' }), refused('wr')] });
    renderWalk();
    await waitFor(() => expect(section(SMALLEST)).toHaveTextContent('No stored result has a collision'));
    expect(server.trajectoryIds()).toEqual([]);
  });

  it('does not start choosing until the whole list has arrived', async () => {
    // The smallest of all is on the last page. Choosing from the first pages alone would read another scenario's recording.
    const server = install({
      pageSize: 2,
      rows: [collision('p1', 0.5, 10), collision('p2', 0.1, 10), collision('p3', 0.05, 10), collision('p4', 0.9, 10)],
    });
    renderWalk();
    await waitFor(() => expect(within(section(SMALLEST)).getByText('p3')).toBeInTheDocument());
    expect([...server.trajectoryIds()].sort()).toEqual(['p3', 'p4']); // p3 to probe, p4 (the largest) for its last-frame sentence; never p1 or p2
  });

  it('does not skip a candidate when React runs the effect twice on a recording it already has', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } }); // cached means not asked for again
    const rows = [collision('s1', 0.1, 90), collision('s2', 0.2, 10), collision('s3', 0.3, 10)];
    // The list and the first candidate's recording are cached, so the page mounts with the first candidate already answered
    // and React's development double run of the effect happens on a recording that does not qualify.
    client.setQueryData(['scenarios', false, null], { pages: [makePage(rows, null)], pageParams: [null] });
    client.setQueryData(['scenario', 's1', 'trajectories'], tracksFor('s1'));
    const server = install({ rows });
    const router = createMemoryRouter(routes, { initialEntries: ['/walkthrough'] });
    render(
      <StrictMode>
        <QueryClientProvider client={client}>
          <RouterProvider router={router} />
        </QueryClientProvider>
      </StrictMode>,
    );
    await waitFor(() => expect(within(section(SMALLEST)).getByText('s2')).toBeInTheDocument());
    // s1 was cached and must not be asked for; s2 is the smallest and s3 the largest. (React's development double mount can start
    // a request twice, so this is the set asked for, not a count.)
    expect([...new Set(server.trajectoryIds())].sort()).toEqual(['s2', 's3']);
  });
});

describe('a card whose collision is on the last recorded frame says so', () => {
  const note = (card: HTMLElement) => [...card.querySelectorAll('.note')].map((n) => n.textContent);

  it('shows the last-frame sentence on the largest-perturbation card when its collision is on the recording\u2019s last frame', async () => {
    install();
    renderWalk();
    await waitFor(() => expect(within(section(LARGEST)).getByText('we-huge-edge')).toBeInTheDocument());
    await waitFor(() => expect(note(section(LARGEST))).toEqual([LAST_FRAME_NOTE]));
    expect(LAST_FRAME_NOTE).toBe('This is the last recorded frame; the search does not look past it.');
  });

  it('does not show it when the largest-perturbation collision is not on the last recorded frame', async () => {
    const server = install({ rows: [collision('wa', 0.1, 90), collision('wb-large', 0.8, 30), refused('wr')] });
    renderWalk();
    await waitFor(() => expect(within(section(LARGEST)).getByText('wb-large')).toBeInTheDocument());
    await waitFor(() => expect(server.trajectoryIds()).toContain('wb-large')); // its recording was read ...
    await new Promise((r) => setTimeout(r, 50));
    expect(note(section(LARGEST))).toEqual([]); // ... and it ends on frame 90, not 30
  });

  it('compares with where THIS recording ends, not with a frame number', async () => {
    const endsAt120 = (id: string): Reply => ({ status: 200, body: makeTrajectories([makeTrack({ agent_idx: 0, frames: range(0, 121) })], id) });
    install({ rows: [collision('wa', 0.9, 90)], trajectories: endsAt120 });
    renderWalk();
    await waitFor(() => expect(within(section(LARGEST)).getByText('wa')).toBeInTheDocument());
    await new Promise((r) => setTimeout(r, 80));
    expect(note(section(LARGEST))).toEqual([]);
  });

  it('says nothing about the last frame when the recording has no geometry to read it from', async () => {
    const none = (id: string): Reply => ({ status: 200, body: makeTrajectories([], id) });
    install({ rows: [collision('wa', 0.9, 90)], trajectories: none });
    renderWalk();
    await waitFor(() => expect(within(section(LARGEST)).getByText('wa')).toBeInTheDocument());
    await new Promise((r) => setTimeout(r, 80));
    expect(note(section(LARGEST))).toEqual([]);
  });

  it('says it could not check, and does not assume, when the recording cannot be read', async () => {
    install({ rows: [collision('wa', 0.9, 90)], trajectories: () => ({ status: 500, body: { detail: 'boom' } }) });
    renderWalk();
    await waitFor(() => expect(within(section(LARGEST)).getByText('wa')).toBeInTheDocument());
    await waitFor(() =>
      expect(note(section(LARGEST))).toEqual(['Whether this is the last recorded frame could not be checked.']),
    );
    expect(section(LARGEST)).not.toHaveTextContent(LAST_FRAME_NOTE);
  });

  it('reads where the LARGEST scene\u2019s recording ends, not another scene\u2019s', async () => {
    // The smallest scene's recording runs on to frame 120; the largest scene's ends on 90, where its collision is.
    LAST_FRAME['wc-small'] = 120;
    try {
      install();
      renderWalk();
      await waitFor(() => expect(within(section(SMALLEST)).getByText('wc-small')).toBeInTheDocument());
      expect(section(SMALLEST)).toHaveTextContent('the recording runs to frame 120');
      await waitFor(() => expect(note(section(LARGEST))).toEqual([LAST_FRAME_NOTE]));
    } finally {
      delete LAST_FRAME['wc-small'];
    }
  });

  it('never puts it on the smallest card or the refused card', async () => {
    install();
    renderWalk();
    await waitFor(() => expect(within(section(SMALLEST)).getByText('wc-small')).toBeInTheDocument());
    await waitFor(() => expect(note(section(LARGEST))).toEqual([LAST_FRAME_NOTE]));
    expect(note(section(SMALLEST))).toEqual([]);
    expect(note(section(REFUSED))).toEqual([]);
  });

  it('asks for the recording of the largest scene once, even when it is also the scene the smallest rule reads', async () => {
    const server = install({ rows: [collision('solo', 0.3, 12)] });
    renderWalk();
    await waitFor(() => expect(section(LARGEST)).toHaveTextContent('This is the scenario above.'));
    expect(server.trajectoryIds()).toEqual(['solo']);
  });

  it('uses the constant the replay uses and has no second copy of the sentence', () => {
    const source = readFrontendFile('src/pages/WalkthroughPage.tsx') + readFrontendFile('src/domain/walkthrough.ts');
    const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, ''); // the backing comments quote the sentence; the code must not
    expect(code).toMatch(/searchWindowNote/);
    expect(code).not.toMatch(/last recorded frame; the search does not look past it/);
  });
});
