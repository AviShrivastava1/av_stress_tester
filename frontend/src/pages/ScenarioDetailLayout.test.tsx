import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * The detail page's header, its result overview and its panel headings.
 *
 * The replay comes first. The header carries the scenario ID and the stored outcome, so the outcome is
 * on the first screen at every width; the overview under the replay carries the rest of what the stored
 * search found (minimum perturbation, collision frame, challengers searched); the Summary panel keeps the
 * scenario's own numbers. Nothing here claims a collision, a perturbed path or a comparison unless the
 * response carries one.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 91), origin: [-20, 0] });
const challengerLogged = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [0, -28], heading: Math.PI / 2 });
const challengerPerturbed = makeTrack({ agent_idx: 1, frames: range(0, 91), origin: [1, -28], heading: Math.PI / 2 });

const COLLISION_DETAIL = makeDetail({
  stress_tested: true,
  stress_attempted: true,
  stress_outcome: 'collision_found',
  last_attempt_outcome: 'collision_found',
  min_perturbation: 0.123456,
  collision_timestep: 25,
  challengers_searched: 3,
  challengers_total: 20,
});
const COLLISION_PERTURBED = makePerturbed({
  target_idx: 1,
  delta: [0.02, -0.01, 0.009, -0.011],
  delta_labels: ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'],
  min_perturbation: 0.123456,
  collision_timestep: 25,
  baseline: challengerLogged,
  perturbed: challengerPerturbed,
});

const TOOLTIP = 'The search perturbs one heuristically chosen challenger, not every agent.';

interface Replies {
  detail?: Reply;
  trajectories?: Reply;
  perturbed?: Reply;
}
function open(replies: Replies = {}) {
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return replies.trajectories ?? ok(makeTrajectories([]));
    if (req.path.endsWith('/perturbed')) return replies.perturbed ?? ok(makePerturbed());
    return replies.detail ?? ok(makeDetail());
  });
  return renderApp(`/scenarios/${ID}`);
}
const collisionScenario = () =>
  open({
    detail: ok(COLLISION_DETAIL),
    trajectories: ok(makeTrajectories([sdc, challengerLogged])),
    perturbed: ok(COLLISION_PERTURBED),
  });
/** A stored result whose perturbed path was never exported: the response has no perturbed track. */
const resultWithoutPath = () =>
  open({
    detail: ok(COLLISION_DETAIL),
    trajectories: ok(makeTrajectories([sdc, challengerLogged])),
    perturbed: ok({ ...COLLISION_PERTURBED, baseline: null, perturbed: null }),
  });

const overview = () => screen.findByRole('region', { name: 'Search result' });
const outcomeGroup = () => screen.findByRole('group', { name: 'Search outcome' });
const before = (a: Node, b: Node) => Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
const dtLabels = () => [...document.querySelectorAll('.facts dt')].map((el) => el.textContent);

describe('the order of the page', () => {
  it('puts the replay first, then the result overview, then the run details', async () => {
    collisionScenario();
    const replay = (await screen.findByRole('heading', { name: 'Replay', level: 2 })).closest('section')!;
    const result = await overview();
    const analysis = (await screen.findByRole('heading', { name: 'Run details', level: 2 })).closest('section')!;
    expect(before(replay, result)).toBe(true);
    expect(before(result, analysis)).toBe(true);
  });

  it('no longer lays the replay out beside the side panels', async () => {
    collisionScenario();
    await overview();
    expect(document.querySelector('.detail-grid')).toBeNull();
    expect(document.querySelector('.detail-intro')).toBeNull();
  });
});

describe('the outcome in the header', () => {
  it('shows the stored outcome once, in the header, above the replay', async () => {
    collisionScenario();
    const group = await outcomeGroup();
    expect(within(group).getByText('collision found')).toBeVisible();
    expect(screen.getAllByText('collision found')).toHaveLength(1);
    expect(group.closest('.detail-heading')).not.toBeNull();
    const replay = (await screen.findByRole('heading', { name: 'Replay', level: 2 })).closest('section')!;
    expect(before(group, replay)).toBe(true);
  });

  it('says a refused attempt left no stored result, and what the last attempt was, in the header', async () => {
    open({
      detail: ok(makeDetail({ stress_attempted: true, last_attempt_outcome: 'replay_infeasible' })),
      trajectories: ok(makeTrajectories([sdc])),
    });
    const group = await outcomeGroup();
    expect(within(group).getByText('no stored result')).toBeVisible();
    expect(within(group).getByText('replay infeasible')).toBeVisible();
  });

  it('says "not tested" in the header for a scenario nothing has reached', async () => {
    open({ detail: ok(makeDetail()), trajectories: ok(makeTrajectories([sdc])) });
    expect(within(await outcomeGroup()).getByText('not tested')).toBeVisible();
  });

  it('is not shown until the scenario has loaded', async () => {
    open({ detail: { status: 500, body: { detail: 'boom' } } });
    expect((await screen.findAllByRole('alert')).length).toBeGreaterThan(0);
    expect(screen.queryByRole('group', { name: 'Search outcome' })).not.toBeInTheDocument();
  });
});

describe('the result overview', () => {
  it('carries the rest of the stored result: minimum perturbation, collision frame, challengers searched', async () => {
    collisionScenario();
    const region = await overview();
    expect(region).toBeVisible();
    expect(within(region).queryByText('collision found')).not.toBeInTheDocument();
    expect(within(region).getByText('Min perturbation')).toBeInTheDocument();
    expect(within(region).getByText('0.123')).toHaveAttribute('title', '0.123456');
    expect(within(region).getByText('Collision frame').nextElementSibling).toHaveTextContent('frame 25');
    expect(within(region).getByText('Challengers searched').nextElementSibling).toHaveTextContent('3 of 20');
  });

  it('keeps the tooltip that says only one challenger is searched', async () => {
    collisionScenario();
    const region = await overview();
    expect(within(region).getByText('Challengers searched')).toHaveAttribute('title', TOOLTIP);
  });

  it('shows a dash for each figure the response does not have', async () => {
    open({ detail: ok(makeDetail()) });
    const region = await overview();
    expect(within(region).getAllByText('—')).toHaveLength(3);
    expect(within(region).queryByText(/frame \d/)).not.toBeInTheDocument();
  });

  it('says "Min perturbation" once on the page, and the Summary panel no longer repeats the result rows', async () => {
    collisionScenario();
    await overview();
    expect(screen.getAllByText('Min perturbation')).toHaveLength(1);
    const labels = dtLabels();
    for (const moved of ['Outcome', 'Min perturbation', 'Collision frame', 'Challengers searched']) {
      expect(labels).not.toContain(moved);
    }
    expect(labels).toEqual(expect.arrayContaining(['Fragility', 'Min TTC', 'Min PET', 'Challenger', 'Method', 'Agents', 'Shard']));
  });

  it('appears only once the scenario has loaded', async () => {
    open({ detail: { status: 500, body: { detail: 'boom' } } });
    expect((await screen.findAllByRole('alert')).length).toBeGreaterThan(0);
    expect(screen.queryByRole('region', { name: 'Search result' })).not.toBeInTheDocument();
  });
});

describe('the header', () => {
  it('keeps the eyebrow "SCENARIO REPLAY" (it is shown for scenarios that have no collision too)', async () => {
    open();
    expect(await screen.findByText('SCENARIO REPLAY')).toBeInTheDocument();
    expect(screen.queryByText(/FAILURE/)).not.toBeInTheDocument();
  });

  it('labels the comparison "logged / counterfactual" only when a perturbed track is drawn', async () => {
    collisionScenario();
    expect(await screen.findByText('logged / counterfactual')).toBeVisible();
    expect(screen.queryByText(/Compare the recorded motion/)).not.toBeInTheDocument();
    expect(screen.queryByText('logged scene')).not.toBeInTheDocument();
  });

  it.each([
    ['a stored result whose perturbed path was not exported', resultWithoutPath],
    ['a scenario with nothing stored', () => open({ trajectories: ok(makeTrajectories([sdc])) })],
  ])('makes no claim of a comparison for %s', async (_name, openIt) => {
    openIt();
    await screen.findByRole('region', { name: 'Search result' });
    await screen.findByTestId('frame-readout');
    expect(screen.queryByText(/counterfactual/)).not.toBeInTheDocument();
    expect(screen.queryByText(/simulated challenger/)).not.toBeInTheDocument();
    expect(screen.getByText('logged scene')).toBeInTheDocument();
  });
});

/*
 * With the intro sentence gone, the label "logged / counterfactual" is the one statement that the perturbed track is a
 * simulated comparison, so it must be visible at every width. jsdom does not apply the stylesheet, so the CSS half of
 * this is a check on styles.css as text: no rule, in any media block, may hide a label that is in the detail header.
 */
interface CssRule { selector: string; declarations: string }
function cssRules(text: string): CssRule[] {
  const stripped = text.replace(/\/\*[\s\S]*?\*\//g, '');
  // A rule is `selector { declarations }` with no braces inside, so a rule inside @media matches on its own.
  return [...stripped.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({
    selector: m[1]!.trim().replace(/\s+/g, ' '),
    declarations: m[2]!.replace(/\s+/g, ''),
  }));
}
const HIDES = /(?:^|;)(?:display:none|visibility:hidden|opacity:0|font-size:0)(?:!important)?(?:;|$)/;
/** Selectors that can reach the header label: anything under .detail-heading, or the bare .dataset-label. */
const reachesHeaderLabel = (selector: string) =>
  selector.split(',').some((part) => /\.detail-heading|\.detail-outcome/.test(part) || part.trim() === '.dataset-label');

// The stylesheet's text, read from disk: vitest hands a `?raw` import of a .css file an empty string, and the project
// has no @types/node, so Node's fs and cwd come from the process global (Node 22, as package.json "engines" requires).
// vitest's root, and so its cwd, is frontend/.
const nodeProcess = (globalThis as unknown as {
  process: { cwd(): string; getBuiltinModule(id: 'node:fs'): { readFileSync(path: string, encoding: 'utf8'): string } };
}).process;
const css = nodeProcess.getBuiltinModule('node:fs').readFileSync(`${nodeProcess.cwd()}/src/styles.css`, 'utf8');

describe('the comparison label stays visible', () => {
  it('sits in the header row with the outcome, above the replay', async () => {
    collisionScenario();
    const label = await screen.findByText('logged / counterfactual');
    const group = await outcomeGroup();
    expect(label.parentElement).toBe(group.parentElement);
    expect(label.parentElement).toHaveClass('detail-heading-meta');
    const replay = (await screen.findByRole('heading', { name: 'Replay', level: 2 })).closest('section')!;
    expect(before(label, replay)).toBe(true);
  });

  it('is not hidden by any rule in styles.css, in any media block', () => {
    const rules = cssRules(css);
    expect(rules.length).toBeGreaterThan(100); // the parser really read the stylesheet
    expect(rules.some((r) => r.selector.includes('.detail-heading-meta'))).toBe(true);
    const hiding = rules.filter((r) => reachesHeaderLabel(r.selector) && HIDES.test(r.declarations));
    expect(hiding.map((r) => r.selector)).toEqual([]);
  });

  it('the matcher itself finds a rule that hides it, at any width', () => {
    const rules = cssRules('@media(max-width:820px) { .a { x:1 } .detail-heading .dataset-label { display:none; } }');
    expect(rules.filter((r) => reachesHeaderLabel(r.selector) && HIDES.test(r.declarations))).toHaveLength(1);
    const other = cssRules('.list-header .dataset-label { display:none; } .detail-heading-meta { display:flex; }');
    expect(other.filter((r) => reachesHeaderLabel(r.selector) && HIDES.test(r.declarations))).toHaveLength(0);
  });
});

describe('the panel headings', () => {
  const heading = (name: string) => screen.findByRole('heading', { name, level: 2 });

  it('names the replay panel "Replay", with a kicker that says what it compares', async () => {
    collisionScenario();
    await screen.findByTestId('frame-readout'); // the scene has loaded, so what the page claims is settled
    await heading('Replay');
    expect(screen.getByText('RECORDED VS PERTURBED, FRAME BY FRAME')).toBeVisible();
    expect(screen.queryByRole('heading', { name: /^(Scene|Collision replay)$/ })).not.toBeInTheDocument();
  });

  it.each([
    ['a stored result without a perturbed path', resultWithoutPath],
    ['a scenario with nothing stored', () => open({ trajectories: ok(makeTrajectories([sdc])) })],
  ])('says "RECORDED SCENE", not a comparison or a collision, for %s', async (_name, openIt) => {
    openIt();
    await screen.findByTestId('frame-readout'); // loaded: not the first render, where this would hold trivially
    await heading('Replay');
    expect(screen.getByText('RECORDED SCENE')).toBeInTheDocument();
    expect(screen.queryByText(/RECORDED VS PERTURBED/)).not.toBeInTheDocument();
    expect(screen.queryByText(/collision replay/i)).not.toBeInTheDocument();
  });

  it('names the summary panel "Run details" and the perturbation panel "Perturbation" under "SEARCH RESULT"', async () => {
    collisionScenario();
    await heading('Run details');
    expect(screen.queryByRole('heading', { name: 'Summary' })).not.toBeInTheDocument();
    expect(screen.getByText('RESULT CONTEXT')).toBeInTheDocument();
    await heading('Perturbation');
    expect(screen.getByText('SEARCH RESULT')).toBeInTheDocument();
    expect(screen.queryByText('SEARCH PARAMETERS')).not.toBeInTheDocument();
  });
});
