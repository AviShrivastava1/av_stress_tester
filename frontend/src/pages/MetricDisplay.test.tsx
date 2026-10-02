import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { makeDetail, makePage, makePerturbed, makeRow, makeTrajectories } from '../test/fixtures';
import { renderApp } from '../test/renderApp';

const ok = (body: unknown): Reply => ({ status: 200, body });

describe('small nonzero values on the list', () => {
  it('shows a value below the displayed precision in scientific notation, with the exact value as a tooltip', async () => {
    const tiny = 1.2345678e-7;
    stubFetch(() => ok(makePage([makeRow({ min_perturbation: tiny })], null)));
    renderApp();
    const row = (await screen.findByRole('link', { name: 'synthetic-0001' })).closest('tr')!;
    const cell = within(row).getByText('1.23e-7');
    expect(cell).toHaveAttribute('title', String(tiny));
  });

  it('still shows an ordinary value with three decimals', async () => {
    stubFetch(() => ok(makePage([makeRow({ min_perturbation: 0.123456 })], null)));
    renderApp();
    const row = (await screen.findByRole('link', { name: 'synthetic-0001' })).closest('tr')!;
    expect(within(row).getByText('0.123')).toHaveAttribute('title', '0.123456');
  });
});

describe('small nonzero values on the detail page', () => {
  const labels = ['initial speed (m/s)', 'initial heading (rad)', 'acceleration bias (m/s^2)', 'steering bias (rad)'];
  const delta = [1.3709068e-6, 0, -0.001, 0.25];

  function open() {
    stubFetch((req: RecordedRequest) => {
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
      if (req.path.endsWith('/perturbed')) {
        return ok(makePerturbed({ target_idx: 1, delta, delta_labels: labels, min_perturbation: 2.5e-6 }));
      }
      return ok(makeDetail({ min_perturbation: 2.5e-6 }));
    });
    renderApp('/scenarios/synthetic-0001');
  }

  it('shows the minimum perturbation in the summary without rounding it to zero', async () => {
    open();
    const term = await screen.findByText('Min perturbation');
    const value = term.nextElementSibling as HTMLElement;
    expect(value).toHaveTextContent('2.50e-6');
    expect(within(value).getByText('2.50e-6')).toHaveAttribute('title', String(2.5e-6));
  });

  it('shows each delta component to four places, or in scientific notation when smaller', async () => {
    open();
    const row = (label: string) => screen.findByText(label).then((th) => th.closest('tr')!);
    const speed = within(await row(labels[0]!)).getByRole('cell');
    expect(speed).toHaveTextContent('1.37e-6');
    expect(speed).toHaveAttribute('title', String(delta[0]));
    expect(within(await row(labels[1]!)).getByRole('cell')).toHaveTextContent('0.0000');
    expect(within(await row(labels[2]!)).getByRole('cell')).toHaveTextContent('-0.0010');
    expect(within(await row(labels[3]!)).getByRole('cell')).toHaveTextContent('0.2500');
  });
});

describe('the infinite-TTC sentinel', () => {
  it('does not define it as the absence of closing motion', async () => {
    stubFetch(() => ok(makePage([makeRow({ min_ttc: 999 })], null)));
    renderApp();
    const none = await screen.findByText('none');
    expect(none).toHaveAttribute('title', expect.stringContaining('constant-velocity'));
    expect(none.getAttribute('title')).not.toContain('never closes');
  });
});
