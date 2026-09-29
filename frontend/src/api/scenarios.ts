import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { api, unwrap, type ScenarioPage } from './client';

export interface ScenarioListParams {
  stressTestedOnly: boolean;
  /** Omitted means the server's default page size. */
  limit?: number;
}

export async function fetchScenarioPage(
  params: ScenarioListParams,
  cursor: string | null,
  signal?: AbortSignal,
): Promise<ScenarioPage> {
  const result = await api.GET('/scenarios', {
    params: {
      query: {
        stress_tested_only: params.stressTestedOnly,
        limit: params.limit,
        cursor: cursor ?? undefined,
      },
    },
    signal,
  });
  return unwrap(result);
}

/**
 * The ranked list, one keyset page at a time.
 *
 * The cursor is opaque: whatever `next_cursor` the server returned goes back
 * verbatim, and nothing here ever builds, decodes, or offsets one. The filter and
 * page size are part of the query key, so changing either starts a new sequence
 * from the first page — a cursor minted under one filter is never sent with
 * another.
 *
 * Pages are kept in the order received and rows are never re-sorted: the server's
 * fragility-descending order is the ranking.
 */
export function useScenarioPages(params: ScenarioListParams) {
  return useInfiniteQuery({
    queryKey: ['scenarios', params.stressTestedOnly, params.limit ?? null],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) => fetchScenarioPage(params, pageParam, signal),
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? null,
  });
}

// ── one scenario ─────────────────────────────────────────────────────────────────
//
// Three separate queries, deliberately not chained: none needs another's result to
// build its request, so the detail page starts all three at once and each panel
// resolves (or fails) on its own. A trajectories 500 does not hide the score.

export function useScenarioDetail(scenarioId: string) {
  return useQuery({
    queryKey: ['scenario', scenarioId],
    queryFn: async ({ signal }) =>
      unwrap(
        await api.GET('/scenarios/{scenario_id}', {
          params: { path: { scenario_id: scenarioId } },
          signal,
        }),
      ),
  });
}

export function useTrajectories(scenarioId: string) {
  return useQuery({
    queryKey: ['scenario', scenarioId, 'trajectories'],
    queryFn: async ({ signal }) =>
      unwrap(
        await api.GET('/scenarios/{scenario_id}/trajectories', {
          params: { path: { scenario_id: scenarioId } },
          signal,
        }),
      ),
  });
}

export function usePerturbed(scenarioId: string) {
  return useQuery({
    queryKey: ['scenario', scenarioId, 'perturbed'],
    queryFn: async ({ signal }) =>
      unwrap(
        await api.GET('/scenarios/{scenario_id}/perturbed', {
          params: { path: { scenario_id: scenarioId } },
          signal,
        }),
      ),
  });
}
