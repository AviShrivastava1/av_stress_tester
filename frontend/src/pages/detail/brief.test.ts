import { describe, expect, it } from 'vitest';
import { WAYMO_ATTRIBUTION } from '../../components/Footer';
import { LAST_FRAME_NOTE } from '../../scene/sceneModel';
import { makeDetail, makePerturbed, makeTrack, range } from '../../test/fixtures';
import { LIMITS } from '../MethodPage';
import { buildBrief } from './brief';
import { FRAME_INTERVAL_S } from './ScenePanel';

/*
 * The text brief (pages/detail/brief.ts): plain text built from the values the scenario page shows, every number read from the API
 * response in hand, ending with the Waymo attribution. The tracks here are stationary boxes (4 long, 2 wide), so the gap before
 * first contact is a distance along y worked out by hand: the SDC's box spans y from -1 to 1, and the perturbed challenger is
 * centred at y = 2.5 (half a metre clear) until the collision frame.
 */

const ORIGIN = 'https://example.test';
const FRAMES = range(0, 91);
const LABELS = ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'];

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: FRAMES, path: FRAMES.map(() => [0, 0]), headings: FRAMES.map(() => 0) });
const logged = makeTrack({ agent_idx: 1, frames: FRAMES, origin: [0, 40] });
const perturbed = makeTrack({
  agent_idx: 1,
  frames: FRAMES,
  path: FRAMES.map((f) => [0, f < 25 ? 2.5 : 1]),
  headings: FRAMES.map(() => 0),
});

const detail = makeDetail({
  scenario_id: 'abc-1',
  stress_tested: true,
  stress_attempted: true,
  stress_outcome: 'collision_found',
  last_attempt_outcome: 'collision_found',
  min_perturbation: 0.0226,
  collision_timestep: 25,
  challengers_searched: 3,
  challengers_total: 20,
  fragility_score: 1.5,
  min_ttc: 0.5,
  min_pet: -0.1,
  n_agents: 12,
  baseline_offset_at_collision: 0.4,
  baseline_replay_error: 0.6,
});
const perturbedResponse = makePerturbed({
  scenario_id: 'abc-1',
  target_idx: 7,
  collision_timestep: 25,
  delta: [0.1234, -0.05, 0.3, 0.02],
  delta_labels: LABELS,
  baseline: logged,
  perturbed,
});

const full = () => buildBrief({ origin: ORIGIN, scenarioId: 'abc-1', detail, perturbed: perturbedResponse, agents: [sdc, logged] });

const EXPECTED_FULL = [
  'AV Scenario Stress-Tester: scenario abc-1',
  'https://example.test/scenarios/abc-1',
  '',
  'SEARCH RESULT',
  'Outcome: collision found',
  'Min perturbation: 0.023 (a weighted norm over components of mixed units; the smallest found by a finite search, not a proven minimum)',
  `Collision frame: 25 (frames are ${FRAME_INTERVAL_S} s apart)`,
  'Recording: frames 0 to 90',
  'Gap before first contact: The boxes were 0.500 m apart on frame 24, the frame before first contact on frame 25.',
  'Worked out in this browser from the stored positions and box sizes of the SDC and the perturbed challenger; the search does not store it. Contact is decided at the precision of the stored positions, up to about a millimetre at the coordinates in this data, so no finer figure is claimed.',
  'Challenger: agent 7, 3 of 20 challengers searched',
  'Perturbation components, as stored:',
  '  initial speed (m/s): 0.1234',
  '  initial heading (rad): -0.0500',
  '  acceleration bias (m/s^2): 0.3000',
  '  steering bias (rad): 0.0200',
  '',
  'RUN DETAILS',
  'Fragility: 1.500',
  'Min TTC: 0.50 s',
  'Min PET: -0.10 s (overlap)',
  'Baseline offset at collision: 0.400 m',
  'Replay drift, whole track (max): 0.600 m',
  'Agents: 12',
  '',
  'HOW TO READ THIS',
  ...LIMITS.map((limit) => `- ${limit}`),
  '',
  'ATTRIBUTION',
  WAYMO_ATTRIBUTION,
].join('\n');

describe('a brief for a scenario with a collision', () => {
  it('is exactly this text, every figure read from the data', () => {
    expect(full()).toBe(EXPECTED_FULL);
  });

  it('ends with the Waymo attribution, verbatim, as its own paragraph', () => {
    const text = full();
    expect(text.endsWith(`\n\nATTRIBUTION\n${WAYMO_ATTRIBUTION}`)).toBe(true);
    expect(text.split(WAYMO_ATTRIBUTION)).toHaveLength(2); // once
  });

  it('carries every limit the Method page states, each in full', () => {
    const text = full();
    expect(LIMITS.length).toBeGreaterThanOrEqual(6);
    for (const limit of LIMITS) expect(text).toContain(`- ${limit}\n`);
  });

  it('is plain text: no markup, no tabs, no trailing spaces, no carriage returns', () => {
    const text = full();
    expect(text).not.toMatch(/[<>\t\r]/);
    expect(text.split('\n').filter((line) => /\s$/.test(line))).toEqual([]);
  });

  it('reads its figures from the data it is given', () => {
    const other = makeDetail({
      ...detail,
      scenario_id: 'zz-9',
      min_perturbation: 0.5,
      collision_timestep: 31,
      fragility_score: 2.25,
      min_ttc: 999,
      min_pet: null,
      n_agents: 77,
      challengers_searched: 1,
      challengers_total: 5,
    });
    const text = buildBrief({
      origin: 'http://localhost:5173',
      scenarioId: 'zz-9',
      detail: other,
      perturbed: makePerturbed({ ...perturbedResponse, scenario_id: 'zz-9', target_idx: 2, collision_timestep: 31, delta: [1, 2, 3, 4], delta_labels: LABELS }),
      agents: [sdc, logged],
    });
    expect(text).toContain('scenario zz-9\nhttp://localhost:5173/scenarios/zz-9');
    expect(text).toContain('Min perturbation: 0.500 (');
    expect(text).toContain('Collision frame: 31 (');
    expect(text).toContain('Fragility: 2.250');
    expect(text).toContain('Min TTC: none'); // the 999 sentinel is not a measurement
    expect(text).toContain('Min PET: not recorded'); // null is not zero
    expect(text).toContain('Agents: 77');
    expect(text).toContain('Challenger: agent 2, 1 of 5 challengers searched');
    expect(text).toContain('  initial speed (m/s): 1.0000');
    expect(text).not.toContain('abc-1');
    expect(text).not.toContain('0.023');
  });

  it.each([
    [0.8, 'Min PET: 0.80 s\n'],
    [0, 'Min PET: 0.00 s (overlap)\n'],
    [-0, 'Min PET: 0.00 s (overlap)\n'], // negative zero prints as 0.00, and is still an overlap
    [-1.25, 'Min PET: -1.25 s (overlap)\n'],
    [0.01, 'Min PET: 0.01 s\n'],
  ])('tags a PET of %s as overlap only when it is zero or less, and never a TTC', (pet, line) => {
    const text = buildBrief({ origin: ORIGIN, scenarioId: 'abc-1', detail: makeDetail({ ...detail, min_pet: pet, min_ttc: -0.5 }), perturbed: perturbedResponse, agents: [sdc, logged] });
    expect(text).toContain(line);
    expect(text).toContain('Min TTC: -0.50 s\n'); // a TTC is never tagged, whatever its sign
  });

  it('adds the last-frame sentence only when the collision is on the last recorded frame', () => {
    expect(full()).not.toContain(LAST_FRAME_NOTE);
    const onLast = buildBrief({
      origin: ORIGIN,
      scenarioId: 'abc-1',
      detail: makeDetail({ ...detail, collision_timestep: 90 }),
      perturbed: makePerturbed({ ...perturbedResponse, collision_timestep: 90 }),
      agents: [sdc, logged],
    });
    expect(onLast).toContain(`Recording: frames 0 to 90\n${LAST_FRAME_NOTE}\n`);
  });

  it('leaves out the recording, the last-frame sentence and the gap when the tracks were not loaded', () => {
    const text = buildBrief({ origin: ORIGIN, scenarioId: 'abc-1', detail, perturbed: perturbedResponse, agents: undefined });
    expect(text).not.toContain('Recording:');
    expect(text).not.toContain('Gap before first contact');
    expect(text).toContain('Collision frame: 25');
  });

  it('names components only when there is exactly one label for each', () => {
    const unlabelled = buildBrief({
      origin: ORIGIN,
      scenarioId: 'abc-1',
      detail,
      perturbed: makePerturbed({ ...perturbedResponse, delta_labels: null }),
      agents: [sdc, logged],
    });
    expect(unlabelled).toContain('Perturbation components: parameterization unknown, so they are not named and carry no units');
    expect(unlabelled).not.toContain('initial speed');
    expect(unlabelled).not.toContain('0.1234');
    const short = buildBrief({
      origin: ORIGIN, scenarioId: 'abc-1', detail, agents: [sdc, logged],
      perturbed: makePerturbed({ ...perturbedResponse, delta_labels: LABELS.slice(0, 3) }),
    });
    expect(short).toContain('parameterization unknown');
  });
});

describe('a brief for a scenario with no collision', () => {
  const base = { origin: ORIGIN, scenarioId: 'abc-1', agents: [sdc, logged] };

  it('has no collision frame, perturbation, gap or components, and says what the search found', () => {
    const text = buildBrief({
      ...base,
      detail: makeDetail({
        scenario_id: 'abc-1', stress_tested: true, stress_attempted: true, stress_outcome: 'no_collision_found',
        last_attempt_outcome: 'no_collision_found', min_perturbation: null, collision_timestep: null, fragility_score: 0.75,
      }),
      perturbed: makePerturbed({ scenario_id: 'abc-1' }),
    });
    expect(text).toContain('Outcome: no collision found');
    for (const absent of ['Min perturbation', 'Collision frame', 'Gap before first contact', 'Perturbation components', 'Challenger:']) {
      expect(text).not.toContain(absent);
    }
    expect(text).toContain('Fragility: 0.750');
    expect(text.endsWith(WAYMO_ATTRIBUTION)).toBe(true);
    for (const limit of LIMITS) expect(text).toContain(limit);
  });

  it.each([
    ['a scenario nothing has reached', makeDetail({ scenario_id: 'abc-1' }), 'Outcome: not tested'],
    [
      'a scenario whose only attempt was refused',
      makeDetail({ scenario_id: 'abc-1', stress_attempted: true, last_attempt_outcome: 'replay_infeasible' }),
      'Outcome: no stored result; last attempt: replay infeasible',
    ],
    [
      'a stored result written before outcomes were recorded',
      makeDetail({ scenario_id: 'abc-1', stress_tested: true, min_perturbation: 0.1, collision_timestep: 4 }),
      'Outcome: outcome not recorded',
    ],
    [
      'an outcome this interface does not know',
      makeDetail({ scenario_id: 'abc-1', stress_tested: true, stress_outcome: 'surprise' }),
      'Outcome: unrecognized: surprise',
    ],
    [
      'a later refused attempt over a stored collision',
      makeDetail({ scenario_id: 'abc-1', stress_tested: true, stress_outcome: 'collision_found', last_attempt_outcome: 'replay_infeasible' }),
      'Outcome: collision found; last attempt: replay infeasible',
    ],
  ])('says the outcome for %s', (_name, d, line) => {
    const text = buildBrief({ ...base, detail: d, perturbed: makePerturbed({ scenario_id: 'abc-1' }) });
    expect(text).toContain(`${line}\n`);
  });

  it('leaves out the replay offsets when nothing is stored, and says "none" for a search that found no collision', () => {
    const untested = buildBrief({ ...base, detail: makeDetail({ scenario_id: 'abc-1' }), perturbed: undefined });
    expect(untested).not.toContain('Baseline offset');
    expect(untested).not.toContain('Replay drift');
    const noCollision = buildBrief({
      ...base,
      detail: makeDetail({ scenario_id: 'abc-1', stress_tested: true, stress_outcome: 'no_collision_found', baseline_offset_at_collision: null, baseline_replay_error: 0.3 }),
      perturbed: undefined,
    });
    expect(noCollision).toContain('Baseline offset at collision: none, no collision found');
    expect(noCollision).toContain('Replay drift, whole track (max): 0.300 m');
    const missing = buildBrief({
      ...base,
      detail: makeDetail({ scenario_id: 'abc-1', stress_tested: true, stress_outcome: 'collision_found', baseline_offset_at_collision: null, baseline_replay_error: null }),
      perturbed: undefined,
    });
    expect(missing).toContain('Baseline offset at collision: not recorded for this result');
    expect(missing).toContain('Replay drift, whole track (max): not recorded for this result');
  });
});

describe('the words in a brief', () => {
  // The same list the site's copy lint applies to every page (pages/copyLint.test.tsx): none of these words, as whole words.
  const BANNED = [/\blive\b/i, /\bsafety\b/i, /\bend-to-end\b/i, /\badversarial\b/i, /\bsafe\b/i];

  it.each([
    ['a collision', () => full()],
    [
      'no collision',
      () =>
        buildBrief({
          origin: ORIGIN, scenarioId: 'abc-1', agents: [sdc, logged], perturbed: makePerturbed({ scenario_id: 'abc-1' }),
          detail: makeDetail({ scenario_id: 'abc-1', stress_tested: true, stress_outcome: 'no_collision_found', robustly_safe: true }),
        }),
    ],
    ['nothing tested', () => buildBrief({ origin: ORIGIN, scenarioId: 'abc-1', agents: undefined, perturbed: undefined, detail: makeDetail({ scenario_id: 'abc-1' }) })],
  ])('has none of the words the site does not use (%s)', (_name, text) => {
    for (const word of BANNED) expect(text()).not.toMatch(word);
  });

  it('does not repeat the server’s "robustly safe" flag, which the site’s copy rules keep to the Corpus page', () => {
    const text = buildBrief({
      origin: ORIGIN, scenarioId: 'abc-1', agents: undefined, perturbed: undefined,
      detail: makeDetail({ scenario_id: 'abc-1', stress_tested: true, stress_outcome: 'no_collision_found', robustly_safe: true }),
    });
    expect(text).not.toMatch(/robustly/i);
  });
});
