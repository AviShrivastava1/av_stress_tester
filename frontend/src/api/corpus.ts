import { useQuery } from '@tanstack/react-query';
import { api, unwrap } from './client';

// Two independent queries, started together: the corpus counts and the service
// health each load, or fail, on their own.

export function useStats(options: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: ['stats'],
    queryFn: async ({ signal }) => unwrap(await api.GET('/stats', { signal })),
    enabled: options.enabled ?? true,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: async ({ signal }) => unwrap(await api.GET('/health', { signal })),
  });
}
