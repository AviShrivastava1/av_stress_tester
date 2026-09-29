import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { OUTCOMES, describeOutcome } from '../domain/outcomes';
import { makeRow } from '../test/fixtures';
import { OutcomeBadge, OutcomeCell } from './OutcomeBadge';

describe('OutcomeBadge', () => {
  it.each(OUTCOMES)('renders %s with its own label', (outcome) => {
    render(<OutcomeBadge outcome={outcome} />);
    expect(screen.getByText(describeOutcome(outcome).label)).toBeInTheDocument();
  });

  it('shows an unrecognized outcome verbatim instead of guessing', () => {
    render(<OutcomeBadge outcome="partial_collision" />);
    expect(screen.getByText('unrecognized: partial_collision')).toBeInTheDocument();
  });
});

describe('OutcomeCell', () => {
  it('says "not tested" when no pass has reached the scenario', () => {
    render(<OutcomeCell row={makeRow()} />);
    expect(screen.getByText('not tested')).toBeInTheDocument();
  });

  it('does not back-infer an outcome for a stored result that predates the column', () => {
    render(
      <OutcomeCell
        row={makeRow({ stress_tested: true, stress_outcome: null, min_perturbation: 0.1 })}
      />,
    );
    expect(screen.getByText('outcome not recorded')).toBeInTheDocument();
    expect(screen.queryByText('collision found')).not.toBeInTheDocument();
  });

  it('shows the last attempt when it concluded differently from the stored result', () => {
    render(
      <OutcomeCell
        row={makeRow({
          stress_tested: true,
          stress_attempted: true,
          stress_outcome: 'collision_found',
          last_attempt_outcome: 'replay_infeasible',
        })}
      />,
    );
    expect(screen.getByText('collision found')).toBeInTheDocument();
    expect(screen.getByText(/last attempt:/)).toBeInTheDocument();
    expect(screen.getByText('replay infeasible')).toBeInTheDocument();
  });

  it('does not repeat the last attempt when it agrees with the stored result', () => {
    render(
      <OutcomeCell
        row={makeRow({
          stress_tested: true,
          stress_attempted: true,
          stress_outcome: 'no_collision_found',
          last_attempt_outcome: 'no_collision_found',
        })}
      />,
    );
    expect(screen.getAllByText('no collision found')).toHaveLength(1);
    expect(screen.queryByText(/last attempt:/)).not.toBeInTheDocument();
  });

  it('shows a refused attempt on a row with no stored result', () => {
    render(
      <OutcomeCell
        row={makeRow({ stress_attempted: true, last_attempt_outcome: 'heading_blend_singularity' })}
      />,
    );
    expect(screen.getByText('no stored result')).toBeInTheDocument();
    expect(screen.getByText('heading-blend singularity')).toBeInTheDocument();
  });

  it('shows "robustly safe" only from the server boolean', () => {
    const { rerender } = render(
      <OutcomeCell
        row={makeRow({ stress_tested: true, stress_outcome: 'no_collision_found', robustly_safe: false })}
      />,
    );
    expect(screen.queryByText('robustly safe')).not.toBeInTheDocument();

    rerender(
      <OutcomeCell
        row={makeRow({ stress_tested: true, stress_outcome: 'no_collision_found', robustly_safe: true })}
      />,
    );
    expect(screen.getByText('robustly safe')).toBeInTheDocument();
  });
});
