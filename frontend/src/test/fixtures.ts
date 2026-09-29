import type {
  AgentTrack,
  PerturbedResponse,
  ScenarioDetail,
  ScenarioPage,
  ScenarioSummary,
  TrajectoryResponse,
} from '../api/client';

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

/**
 * A straight, fully observed track along +x unless overridden. `frames` sets the
 * observed frames; path, timesteps and headings are kept index-aligned from it.
 */
export function makeTrack(
  overrides: Partial<AgentTrack> & { frames?: number[]; origin?: [number, number]; heading?: number } = {},
): AgentTrack {
  const { frames = range(0, 11), origin = [0, 0], heading = 0, ...rest } = overrides;
  return {
    agent_idx: 0,
    agent_type: 1,
    is_sdc: false,
    length_m: 4.5,
    width_m: 2.0,
    path: frames.map((f) => [origin[0] + f * Math.cos(heading), origin[1] + f * Math.sin(heading)]),
    timesteps: frames,
    headings: frames.map(() => heading),
    ...rest,
  };
}

export function range(start: number, endExclusive: number): number[] {
  return Array.from({ length: endExclusive - start }, (_, i) => start + i);
}

export function makeDetail(overrides: Partial<ScenarioDetail> = {}): ScenarioDetail {
  return { ...makeRow(), delta: null, ...overrides };
}

export function makePerturbed(overrides: Partial<PerturbedResponse> = {}): PerturbedResponse {
  return {
    scenario_id: 'synthetic-0001',
    target_idx: null,
    delta: null,
    delta_labels: null,
    min_perturbation: null,
    collision_timestep: null,
    baseline: null,
    perturbed: null,
    ...overrides,
  };
}

export function makeTrajectories(agents: AgentTrack[], scenarioId = 'synthetic-0001'): TrajectoryResponse {
  return { scenario_id: scenarioId, agents };
}
