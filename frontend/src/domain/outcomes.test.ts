import { describe, expect, it } from 'vitest';
import { OUTCOMES, describeOutcome, parseOutcome } from './outcomes';

describe('outcome vocabulary', () => {
  it('has exactly the six values the backend writes', () => {
    expect([...OUTCOMES].sort()).toEqual(
      [
        'collision_found',
        'error',
        'heading_blend_singularity',
        'no_challenger',
        'no_collision_found',
        'replay_infeasible',
      ].sort(),
    );
  });

  it.each(OUTCOMES)('describes %s', (outcome) => {
    const { label, meaning } = describeOutcome(outcome);
    expect(label).not.toBe('');
    expect(meaning).not.toBe('');
  });

  it('gives every outcome its own label', () => {
    const labels = OUTCOMES.map((o) => describeOutcome(o).label);
    expect(new Set(labels).size).toBe(OUTCOMES.length);
  });

  it('does not coerce an unknown string into a known outcome', () => {
    expect(parseOutcome('collision')).toBeNull();
    expect(parseOutcome('COLLISION_FOUND')).toBeNull();
    expect(parseOutcome('')).toBeNull();
    expect(parseOutcome('replay_infeasible')).toBe('replay_infeasible');
  });
});
