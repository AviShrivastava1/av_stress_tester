import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import type { HealthResponse, StatsResponse } from '../api/client';
import { describeOutcome, ROBUSTLY_SAFE } from '../domain/outcomes';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { makePage } from '../test/fixtures';
import { renderApp } from '../test/renderApp';

const ok = (body: unknown): Reply => ({ status: 200, body });

// Synthetic counts, each distinct so a value found in the wrong place is detectable.
function makeStats(overrides: Partial<StatsResponse> = {}): StatsResponse {
  return {
    total_scenarios: 1234,
    stress_tested: 40,
    collisions_found: 17,
    no_collision_found: 11,
    replay_infeasible: 5,
    heading_blend_singularity: 3,
    no_challenger: 2,
    stress_errors: 1,
    robustly_safe: 0,
    with_geometry: 38,
    fragility_min: 0.25,
    fragility_max: 4.5,
    fragility_mean: 1.125,
    ...overrides,
  };
}

function makeHealth(overrides: Partial<HealthResponse> = {}): HealthResponse {
  return { status: 'ok', database: 'connected', postgis: '3.6 USE_GEOS=1', ...overrides };
}

function server(replies: { stats?: Reply; health?: Reply }) {
  return (req: RecordedRequest): Reply => {
    if (req.path === '/stats') return replies.stats ?? ok(makeStats());
    if (req.path === '/health') return replies.health ?? ok(makeHealth());
    return ok(makePage([], null));
  };
}

function stat(id: string) {
  const el = document.querySelector<HTMLElement>(`[data-stat="${id}"]`);
  if (!el) throw new Error(`no stat ${id}`);
  return { value: el.querySelector('dd')!.textContent, text: el.textContent ?? '', el };
}

function panel(name: string) {
  return screen.getByRole('heading', { name, level: 2 }).closest('section')!;
}

describe('StatsPage — where each count appears', () => {
  it('shows every count as reported, in the group its column belongs to', async () => {
    stubFetch(server({}));
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Stored results' });

    const stored = panel('Stored results');
    const latest = panel('Latest attempts');
    const corpus = panel('Corpus');

    for (const id of ['stress_tested', 'collisions_found', 'no_collision_found', 'robustly_safe']) {
      expect(stored.querySelector(`[data-stat="${id}"]`)).not.toBeNull();
    }
    for (const id of ['replay_infeasible', 'heading_blend_singularity', 'no_challenger', 'stress_errors']) {
      expect(latest.querySelector(`[data-stat="${id}"]`)).not.toBeNull();
    }
    expect(corpus.querySelector('[data-stat="with_geometry"]')).not.toBeNull();

    expect(stat('total_scenarios').value).toBe('1,234');
    expect(stat('stress_tested').value).toBe('40');
    expect(stat('collisions_found').value).toBe('17');
    expect(stat('no_collision_found').value).toBe('11');
    expect(stat('replay_infeasible').value).toBe('5');
    expect(stat('heading_blend_singularity').value).toBe('3');
    expect(stat('no_challenger').value).toBe('2');
    expect(stat('stress_errors').value).toBe('1');
    expect(stat('with_geometry').value).toBe('38');
    expect(stat('fragility_mean').value).toBe('1.125');
  });

  it('labels the collision count by what the SQL counts, not as an outcome', async () => {
    stubFetch(server({}));
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Stored results' });
    const collisions = stat('collisions_found');
    expect(collisions.text).toContain('Stored results with a perturbation (collision found)');
    expect(collisions.text).toMatch(/Counted from the stored perturbation/);
  });

  it('uses the outcome vocabulary for the outcome-based counts', async () => {
    stubFetch(server({}));
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Latest attempts' });
    expect(stat('no_collision_found').text).toContain(describeOutcome('no_collision_found').label);
    expect(stat('stress_errors').text).toContain(describeOutcome('error').label);
    expect(stat('replay_infeasible').text).toContain(describeOutcome('replay_infeasible').meaning);
  });

  it('shows robustly safe as the server count, zero included, with the shared wording', async () => {
    stubFetch(server({}));
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Stored results' });
    const rs = stat('robustly_safe');
    expect(rs.value).toBe('0');
    expect(rs.text).toContain(ROBUSTLY_SAFE.label);
    expect(rs.text).toContain(ROBUSTLY_SAFE.meaning);
  });

  it('never renders a percentage or a total across the groups', async () => {
    stubFetch(server({}));
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Latest attempts' });
    const text = document.body.textContent ?? '';
    expect(text).not.toMatch(/%/);
    // 17 + 11 = 28, the two stored-result counts added up, is the tempting total; it must
    // not appear as a stat of its own. (The latest-attempt sum, 11, cannot be checked this
    // way: it coincides with the real no_collision_found count.)
    const values = [...document.querySelectorAll('[data-stat] dd')].map((d) => d.textContent);
    expect(values).not.toContain('28');
    expect(document.querySelectorAll('[data-stat]')).toHaveLength(16);
  });
});

describe('StatsPage — normal states that are not errors', () => {
  it('shows zero exported geometry as a plain count', async () => {
    stubFetch(server({ stats: ok(makeStats({ with_geometry: 0 })) }));
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Corpus', level: 2 });
    expect(stat('with_geometry').value).toBe('0');
    expect(stat('with_geometry').el.querySelector('.badge')).toBeNull();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows an empty corpus with dashes for the fragility range', async () => {
    stubFetch(
      server({
        stats: ok(
          makeStats({
            total_scenarios: 0,
            stress_tested: 0,
            collisions_found: 0,
            no_collision_found: 0,
            fragility_min: null,
            fragility_max: null,
            fragility_mean: null,
          }),
        ),
      }),
    );
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Corpus', level: 2 });
    expect(stat('total_scenarios').value).toBe('0');
    expect(stat('fragility_min').value).toBe('—');
    expect(stat('fragility_mean').value).toBe('—');
    expect(stat('fragility_max').value).toBe('—');
  });
});

describe('StatsPage — service health', () => {
  it('shows what /health reports, not what a 200 implies', async () => {
    stubFetch(server({ health: ok(makeHealth({ status: 'ok', database: 'connected-replica' })) }));
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Service health' });
    expect(await screen.findByText('connected-replica')).toBeInTheDocument();
    expect(stat('status').value).toBe('ok');
    expect(stat('postgis').value).toBe('3.6 USE_GEOS=1');
  });

  it('shows missing PostGIS as degraded, not as an error', async () => {
    stubFetch(server({ health: ok(makeHealth({ postgis: null })) }));
    renderApp('/stats');
    await screen.findByRole('heading', { name: 'Service health' });
    const postgis = await screen.findByText('not available');
    expect(postgis).toHaveClass('badge-warning');
    expect(stat('postgis').text).toMatch(/geometry endpoints return empty/);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('keeps the health panel when /stats fails, and the counts when /health fails', async () => {
    stubFetch(server({ stats: { status: 500, body: { detail: 'boom' } } }));
    const first = renderApp('/stats');
    expect(await screen.findByRole('alert')).toHaveTextContent('500: boom');
    expect(await screen.findByText('connected')).toBeInTheDocument();
    first.unmount();

    stubFetch(server({ health: { status: 503, body: { detail: 'busy' }, headers: { 'Retry-After': '1' } } }));
    renderApp('/stats');
    const alert = await screen.findByRole('alert');
    expect(within(panel('Service health')).getByRole('alert')).toBe(alert);
    expect(alert).toHaveTextContent(/The API is busy/);
    expect(stat('stress_tested').value).toBe('40');
  });
});

describe('navigation', () => {
  it('reaches the corpus page from the header, and back', async () => {
    stubFetch(server({}));
    renderApp('/');
    const nav = screen.getByRole('navigation', { name: 'Main' });
    expect(within(nav).getByRole('link', { name: 'Scenarios' })).toHaveClass('active');

    await userEvent.click(within(nav).getByRole('link', { name: 'Corpus' }));
    expect(await screen.findByRole('heading', { name: 'Corpus', level: 1 })).toBeInTheDocument();
    expect(within(nav).getByRole('link', { name: 'Corpus' })).toHaveClass('active');
    expect(within(nav).getByRole('link', { name: 'Scenarios' })).not.toHaveClass('active');

    await userEvent.click(within(nav).getByRole('link', { name: 'Scenarios' }));
    expect(
      await screen.findByRole('heading', { name: 'How small a change turns a recorded driving scene into a collision?', level: 1 }),
    ).toBeInTheDocument();
  });
});
