import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { TimeMetricCell } from './TimeMetricCell';
import { PET_OVERLAP_TITLE } from '../domain/metrics';

/*
 * The "overlap" tag on a PET reading. What the number means is in the engine (src/danger/pet_engine.py): PET is the gap
 * between two agents' visits to one conflict zone, and a value of zero or less means the two visits share at least one
 * sampled frame, so both agents' boxes touched the same zone on a common frame. Zero is therefore marked exactly like a
 * negative value. A positive value, however small, is a real gap and is not marked. The 999 sentinel is "no shared
 * conflict zone", not a measurement, and null is missing.
 */

const tag = () => screen.queryByText('overlap');

describe('TimeMetricCell on a PET reading', () => {
  it.each([
    ['zero', 0, '0.00 s'],
    ['negative zero', -0, '0.00 s'],
    ['a small negative', -0.01, '-0.01 s'],
    ['a negative', -6.7, '-6.70 s'],
  ])('tags %s as overlap and shows the number', (_name, value, text) => {
    const { container } = render(<TimeMetricCell value={value} metric="pet" />);
    expect(container).toHaveTextContent(text);
    expect(tag()).toBeVisible();
    expect(tag()).toHaveAttribute('title', PET_OVERLAP_TITLE);
  });

  it.each([
    ['a small positive', 0.01, '0.01 s'],
    ['a positive', 2.5, '2.50 s'],
  ])('does not tag %s', (_name, value, text) => {
    const { container } = render(<TimeMetricCell value={value} metric="pet" />);
    expect(container).toHaveTextContent(text);
    expect(tag()).not.toBeInTheDocument();
  });

  it('shows the 999 sentinel as "none", with no number and no tag', () => {
    const { container } = render(<TimeMetricCell value={999} metric="pet" />);
    expect(container).toHaveTextContent('none');
    expect(container).not.toHaveTextContent('999');
    expect(tag()).not.toBeInTheDocument();
  });

  it('shows null as a dash, with no tag', () => {
    const { container } = render(<TimeMetricCell value={null} metric="pet" />);
    expect(container).toHaveTextContent('—');
    expect(tag()).not.toBeInTheDocument();
  });

  it('never tags a time-to-collision reading, whatever its value', () => {
    const { container } = render(<TimeMetricCell value={0} metric="ttc" />);
    expect(container).toHaveTextContent('0.00 s');
    expect(tag()).not.toBeInTheDocument();
  });

  it('says what the tag means, in the words the engine supports, with no banned word', () => {
    expect(PET_OVERLAP_TITLE).toBe(
      'PET of zero or less: the SDC and another agent were both in a conflict zone at the same time, on at least one shared sampled frame.',
    );
    expect(PET_OVERLAP_TITLE).not.toMatch(/\b(live|safe|safety|end-to-end|adversarial)\b/i);
  });
});
