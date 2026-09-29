import type { ScenarioPage, ScenarioSummary } from '../api/client';

/**
 * Synthetic rows only. No real scenario IDs or fragility numbers belong in tests:
 * the stored corpus is due to be re-scored, and today's worst scenarios need not be
 * tomorrow's.
 */
export function makeRow(overrides: Partial<ScenarioSummary> = {}): ScenarioSummary {
  return {
    scenario_id: 'synthetic-0001',
    shard: 'synthetic',
    n_agents: 4,
    min_ttc: 1.5,
    min_pet: 0.8,
    fragility_score: 1.0,
    min_perturbation: null,
    collision_timestep: null,
    stress_method: null,
    stress_tested: false,
    stress_attempted: false,
    stress_outcome: null,
    last_attempt_outcome: null,
    challengers_total: null,
    challengers_searched: null,
    no_collision_found: false,
    robustly_safe: false,
    ...overrides,
  };
}

export function makePage(items: ScenarioSummary[], nextCursor: string | null): ScenarioPage {
  return { items, next_cursor: nextCursor, limit: 25 };
}
