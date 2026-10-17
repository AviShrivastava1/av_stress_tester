import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { useEffect } from 'react';
import { api, unwrap, type ScenarioPage, type ScenarioSummary } from './client';

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

/**
 * Every row of the list, with no filter: the pages are followed to the end, and `rows` is null until the last one has arrived
 * (a partial list is never handed out as the whole). The query key is the list page's own for the unfiltered list, so the
 * pages are shared with it. After a failed page the cursor is not followed again by itself; `retry` asks for it.
 */
export function useAllScenarios(): {
  rows: ScenarioSummary[] | null;
  error: unknown;
  retrying: boolean;
  retry(): void;
} {
  const query = useScenarioPages({ stressTestedOnly: false });
  const { hasNextPage, isFetchingNextPage, isFetchNextPageError, fetchNextPage } = query;
  const pageCount = query.data?.pages.length ?? 0;
  useEffect(() => {
    if (hasNextPage && !isFetchingNextPage && !isFetchNextPageError) void fetchNextPage();
  }, [hasNextPage, isFetchingNextPage, isFetchNextPageError, fetchNextPage, pageCount]);

  const failed = query.isError;
  return {
    rows: query.isSuccess && !hasNextPage ? query.data.pages.flatMap((p) => p.items) : null,
    error: failed ? query.error : null,
    retrying: query.isFetching,
    retry: () => void (isFetchNextPageError ? fetchNextPage() : query.refetch()),
  };
}

// ── one scenario ─────────────────────────────────────────────────────────────────
//
// Three separate queries, deliberately not chained: none needs another's result to
// build its request, so the detail page starts all three at once and each panel
// resolves (or fails) on its own. A trajectories 500 does not hide the score.

export function useScenarioDetail(scenarioId: string, options: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: ['scenario', scenarioId],
    enabled: options.enabled ?? true,
    queryFn: async ({ signal }) =>
      unwrap(
        await api.GET('/scenarios/{scenario_id}', {
          params: { path: { scenario_id: scenarioId } },
          signal,
        }),
      ),
  });
}

export function useTrajectories(scenarioId: string, options: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: ['scenario', scenarioId, 'trajectories'],
    enabled: options.enabled ?? true,
    queryFn: async ({ signal }) =>
      unwrap(
        await api.GET('/scenarios/{scenario_id}/trajectories', {
          params: { path: { scenario_id: scenarioId } },
          signal,
        }),
      ),
  });
}

export function usePerturbed(scenarioId: string, options: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: ['scenario', scenarioId, 'perturbed'],
    enabled: options.enabled ?? true,
    queryFn: async ({ signal }) =>
      unwrap(
        await api.GET('/scenarios/{scenario_id}/perturbed', {
          params: { path: { scenario_id: scenarioId } },
          signal,
        }),
      ),
  });
}

/**
 * Where a stored result came from (GET /scenarios/{id}/provenance). Not started with the page: it is asked for when the reader
 * opens the provenance panel (`enabled`), so a page that is only looked at costs no extra request.
 */
export function useProvenance(scenarioId: string, options: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: ['scenario', scenarioId, 'provenance'],
    enabled: options.enabled ?? true,
    queryFn: async ({ signal }) =>
      unwrap(
        await api.GET('/scenarios/{scenario_id}/provenance', {
          params: { path: { scenario_id: scenarioId } },
          signal,
        }),
      ),
  });
}

/**
 * The convergence trace of a scenario's stored search: GET /scenarios/{id}/trace. Asked for when the provenance panel is open and the
 * provenance has loaded and holds a search, so a page that is only looked at costs no extra request.
 */
export function useTrace(scenarioId: string) {
  return useQuery({
    queryKey: ['scenario', scenarioId, 'trace'],
    queryFn: async ({ signal }) =>
      unwrap(
        await api.GET('/scenarios/{scenario_id}/trace', {
          params: { path: { scenario_id: scenarioId } },
          signal,
        }),
      ),
  });
}

/**
 * The other searches recorded for a scenario: GET /scenarios/{id}/other-searches. Asked for when the "Other searches of this scenario"
 * section is opened (`enabled`), so a page that is only looked at costs no extra request.
 */
export function useOtherSearches(scenarioId: string, options: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: ['scenario', scenarioId, 'other-searches'],
    enabled: options.enabled ?? true,
    queryFn: async ({ signal }) =>
      unwrap(
        await api.GET('/scenarios/{scenario_id}/other-searches', {
          params: { path: { scenario_id: scenarioId } },
          signal,
        }),
      ),
  });
}

/**
 * The sensitivity slices recorded around a scenario's stored result: GET /scenarios/{id}/slice. Asked for when the "Sensitivity slice"
 * section is opened (`enabled`), so a page that is only looked at costs no extra request.
 */
export function useSensitivitySlices(scenarioId: string, options: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: ['scenario', scenarioId, 'slice'],
    enabled: options.enabled ?? true,
    queryFn: async ({ signal }) =>
      unwrap(
        await api.GET('/scenarios/{scenario_id}/slice', {
          params: { path: { scenario_id: scenarioId } },
          signal,
        }),
      ),
  });
}
