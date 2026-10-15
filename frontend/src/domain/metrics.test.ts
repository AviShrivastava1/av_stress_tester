import { describe, expect, it } from 'vitest';
import { NO_INTERACTION_SENTINEL, isPetOverlap, readTimeMetric } from './metrics';

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

describe('isPetOverlap', () => {
  it('is true for zero (and negative zero) and for any negative value', () => {
    expect(isPetOverlap(0)).toBe(true);
    expect(isPetOverlap(-0)).toBe(true);
    expect(isPetOverlap(-0.01)).toBe(true);
    expect(isPetOverlap(-6.7)).toBe(true);
  });

  it('is false for any positive value, however small, and for a number that is not one', () => {
    expect(isPetOverlap(0.01)).toBe(false);
    expect(isPetOverlap(Number.MIN_VALUE)).toBe(false);
    expect(isPetOverlap(5)).toBe(false);
    expect(isPetOverlap(Number.NaN)).toBe(false);
  });
});
