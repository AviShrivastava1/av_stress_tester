import { describe, expect, it } from 'vitest';
import { formatMagnitude } from './format';

describe('formatMagnitude', () => {
  it('uses fixed decimals for ordinary values', () => {
    expect(formatMagnitude(0.123456)).toBe('0.123');
    expect(formatMagnitude(12.5)).toBe('12.500');
    expect(formatMagnitude(-0.5)).toBe('-0.500');
  });

  it('switches to scientific notation below the last displayed decimal, not at it', () => {
    expect(formatMagnitude(0.001)).toBe('0.001');
    expect(formatMagnitude(0.00099)).toBe('9.90e-4');
    expect(formatMagnitude(0.0001, 4)).toBe('0.0001');
    expect(formatMagnitude(0.00009, 4)).toBe('9.00e-5');
  });

  it('keeps small values of either sign nonzero', () => {
    expect(formatMagnitude(5.96e-8)).toBe('5.96e-8');
    expect(formatMagnitude(-5.96e-8)).toBe('-5.96e-8');
    expect(formatMagnitude(1.3709068e-6, 4)).toBe('1.37e-6');
  });

  it('shows an exact zero as zero', () => {
    expect(formatMagnitude(0)).toBe('0.000');
    expect(formatMagnitude(0, 4)).toBe('0.0000');
  });
});
