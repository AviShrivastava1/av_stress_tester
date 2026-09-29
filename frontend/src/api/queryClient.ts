import { QueryClient } from '@tanstack/react-query';
import { ApiError } from './client';

const MAX_BACKOFF_MS = 8000;

/**
 * Retry only what can succeed on retry: a 503 (the API's "all connections busy",
 * which is transient by design) and network failures. A 404 or 500 will say the same
 * thing the second time.
 */
export function shouldRetry(failureCount: number, error: unknown): boolean {
  if (error instanceof ApiError) return error.status === 503 && failureCount < 3;
  return failureCount < 2;
}

/** Honour the server's Retry-After when it sent one; otherwise back off exponentially. */
export function retryDelay(attempt: number, error: unknown): number {
  if (error instanceof ApiError && error.retryAfterSeconds !== null) {
    return error.retryAfterSeconds * 1000;
  }
  return Math.min(1000 * 2 ** attempt, MAX_BACKOFF_MS);
}

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: shouldRetry,
        retryDelay,
        // The data changes only when an offline pass writes to Postgres; refetching
        // every page of an infinite list whenever the tab regains focus buys nothing.
        staleTime: 60_000,
        refetchOnWindowFocus: false,
      },
    },
  });
}
