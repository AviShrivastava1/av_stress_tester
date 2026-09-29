import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { Footer, WAYMO_ATTRIBUTION } from './Footer';

describe('Footer', () => {
  it('carries the Waymo license attribution verbatim', () => {
    // Asserted against the literal sentence, not the constant alone, so an edit to
    // the constant cannot pass by also being what the test compares against.
    expect(WAYMO_ATTRIBUTION).toBe(
      'This software was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available at waymo.com/open/terms, and your access and use of such work are governed by the terms and conditions therein.',
    );
    render(<Footer />);
    expect(screen.getByText(WAYMO_ATTRIBUTION)).toBeInTheDocument();
  });
});
