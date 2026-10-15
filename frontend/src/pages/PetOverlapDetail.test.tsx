import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { PET_OVERLAP_TITLE } from '../domain/metrics';
import { makeDetail, makePerturbed, makeTrajectories } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/* The same tag on the detail page's "Min PET" row: zero and negative are marked, a positive value is not. */

const ok = (body: unknown): Reply => ({ status: 200, body });

function open(minPet: number | null) {
  stubFetch((req: RecordedRequest): Reply => {
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
    if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
    return ok(makeDetail({ min_pet: minPet }));
  });
  return renderApp('/scenarios/synthetic-0001');
}

async function petCell() {
  const label = await screen.findByText('Min PET');
  return label.nextElementSibling as HTMLElement;
}

describe('Min PET on the detail page', () => {
  it.each([
    ['zero', 0, '0.00 s'],
    ['a small negative', -0.01, '-0.01 s'],
  ])('marks %s as overlap', async (_name, value, text) => {
    open(value);
    const cell = await petCell();
    expect(cell).toHaveTextContent(text);
    expect(within(cell).getByText('overlap')).toHaveAttribute('title', PET_OVERLAP_TITLE);
  });

  it('does not mark a small positive value', async () => {
    open(0.01);
    const cell = await petCell();
    expect(cell).toHaveTextContent('0.01 s');
    expect(within(cell).queryByText('overlap')).not.toBeInTheDocument();
  });
});
