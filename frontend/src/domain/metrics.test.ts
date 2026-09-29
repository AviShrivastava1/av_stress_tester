import { describe, expect, it } from 'vitest';
import { NO_INTERACTION_SENTINEL, readTimeMetric } from './metrics';

describe('readTimeMetric', () => {
  it('treats 999.0 as no interaction, not a measurement', () => {
    expect(readTimeMetric(NO_INTERACTION_SENTINEL)).toEqual({ kind: 'no_interaction' });
    expect(readTimeMetric(999)).toEqual({ kind: 'no_interaction' });
  });

  it('keeps negative values as real measurements', () => {
    expect(readTimeMetric(-0.4)).toEqual({ kind: 'measured', seconds: -0.4 });
  });

  it('keeps zero as a real measurement', () => {
    expect(readTimeMetric(0)).toEqual({ kind: 'measured', seconds: 0 });
  });

  it('distinguishes missing from the sentinel', () => {
    expect(readTimeMetric(null)).toEqual({ kind: 'missing' });
    expect(readTimeMetric(undefined)).toEqual({ kind: 'missing' });
  });
});
