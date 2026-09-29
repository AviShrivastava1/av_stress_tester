import { describe, expect, it } from 'vitest';
import { makeDetail, makePerturbed, makeTrack } from '../test/fixtures';
import {
  buildScene,
  canFocusInteraction,
  focusTracks,
  perturbedPathNote,
  referenceFrameNote,
} from './sceneModel';

const sdc = makeTrack({ agent_idx: 0, is_sdc: true });
const challenger = makeTrack({ agent_idx: 1 });
const bystander = makeTrack({ agent_idx: 2, agent_type: 2 });

describe('buildScene', () => {
  it('draws the logged challenger from /perturbed, not a second copy from /trajectories', () => {
    const baseline = makeTrack({ agent_idx: 1, origin: [0, 99] });
    const perturbedTrack = makeTrack({ agent_idx: 1, origin: [0, 50] });
    const drawn = buildScene(
      [sdc, challenger, bystander],
      makePerturbed({ target_idx: 1, baseline, perturbed: perturbedTrack }),
    );
    expect(drawn.map((d) => d.role)).toEqual(['other', 'sdc', 'challenger_logged', 'challenger_perturbed']);
    expect(drawn.find((d) => d.role === 'challenger_logged')!.track).toBe(baseline);
    expect(drawn.filter((d) => d.track.agent_idx === 1)).toHaveLength(2);
  });

  it('falls back to the /trajectories challenger when the baseline is absent', () => {
    const drawn = buildScene([sdc, challenger], makePerturbed({ target_idx: 1 }));
    expect(drawn.find((d) => d.role === 'challenger_logged')!.track).toBe(challenger);
  });

  it('draws plain agents when nothing was stress-tested', () => {
    const drawn = buildScene([sdc, challenger], makePerturbed());
    expect(drawn.map((d) => d.role)).toEqual(['other', 'sdc']);
  });

  it('still draws the challenger when no other agent geometry exists', () => {
    const drawn = buildScene(
      [],
      makePerturbed({ target_idx: 1, baseline: challenger, perturbed: makeTrack({ agent_idx: 1 }) }),
    );
    expect(drawn.map((d) => d.role)).toEqual(['challenger_logged', 'challenger_perturbed']);
  });
});

describe('focus', () => {
  it('offers the interaction focus only with both the SDC and a challenger drawn', () => {
    expect(canFocusInteraction(buildScene([sdc, challenger], makePerturbed({ target_idx: 1 })))).toBe(true);
    expect(canFocusInteraction(buildScene([sdc, challenger], makePerturbed()))).toBe(false);
    expect(canFocusInteraction(buildScene([challenger], makePerturbed({ target_idx: 1 })))).toBe(false);
  });

  it('frames the interaction on the SDC and the challenger only', () => {
    const drawn = buildScene([sdc, challenger, bystander], makePerturbed({ target_idx: 1 }));
    expect(focusTracks(drawn, 'interaction').map((d) => d.role)).toEqual(['sdc', 'challenger_logged']);
    expect(focusTracks(drawn, 'scene')).toHaveLength(3);
  });
});

describe('perturbedPathNote', () => {
  const none = makePerturbed();

  it('says nothing when a perturbed path is drawn', () => {
    expect(perturbedPathNote(makeDetail(), makePerturbed({ perturbed: challenger }))).toBeNull();
  });

  it('does not guess before the score row has loaded', () => {
    expect(perturbedPathNote(undefined, none)).toBeNull();
  });

  it('distinguishes the reasons a path is missing', () => {
    expect(perturbedPathNote(makeDetail(), none)).toMatch(/Not stress-tested/);
    expect(perturbedPathNote(makeDetail({ stress_attempted: true }), none)).toMatch(/stored no result/);
    expect(
      perturbedPathNote(makeDetail({ stress_tested: true, stress_outcome: 'collision_found' }), none),
    ).toMatch(/no perturbed path matching it has been exported/);
    expect(
      perturbedPathNote(makeDetail({ stress_tested: true, stress_outcome: 'no_collision_found' }), none),
    ).toMatch(/found no collision/);
  });
});

describe('referenceFrameNote', () => {
  it('names the rule that picked the frame', () => {
    expect(referenceFrameNote({ kind: 'collision', frame: 25 }, 3)).toMatch(/frame 25, the first colliding frame/);
    expect(referenceFrameNote({ kind: 'all_observed', frame: 20 }, 2)).toMatch(
      /frame 20, the earliest frame where all 2 focused tracks are observed/,
    );
  });

  it('does not present the fallback as the all-observed rule', () => {
    const note = referenceFrameNote({ kind: 'earliest_observed', frame: 5 }, 2);
    expect(note).toMatch(/No frame has all 2 focused tracks observed together/);
    expect(note).not.toMatch(/the earliest frame where all/);
  });
});
