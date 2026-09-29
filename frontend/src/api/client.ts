import createClient from 'openapi-fetch';
import type { components, paths } from './schema';

// Aliases only — every shape here is generated from openapi.json.
export type ScenarioSummary = components['schemas']['ScenarioSummary'];
export type ScenarioPage = components['schemas']['ScenarioPage'];
export type ScenarioDetail = components['schemas']['ScenarioDetail'];
export type AgentTrack = components['schemas']['AgentTrack'];
export type TrajectoryResponse = components['schemas']['TrajectoryResponse'];
export type PerturbedResponse = components['schemas']['PerturbedResponse'];

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

export const api = createClient<paths>({
  baseUrl: API_BASE_URL,
  // Looked up per call rather than captured when the client is created, so tests can
  // stub globalThis.fetch after this module has loaded.
  fetch: (request) => globalThis.fetch(request),
});

/**
 * The body of a successful openapi-fetch call, or the ApiError for a failed one.
 * Network failures are not caught here: they stay as the TypeError fetch throws.
 */
export function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  if (result.data === undefined) throw ApiError.from(result.response, result.error);
  return result.data;
}

/** A non-2xx response. Network failures stay as the TypeError fetch throws. */
export class ApiError extends Error {
  readonly status: number;
  /** From `Retry-After` (seconds), which the API sends with its pool-exhausted 503. */
  readonly retryAfterSeconds: number | null;

  constructor(status: number, retryAfterSeconds: number | null, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.retryAfterSeconds = retryAfterSeconds;
  }

  static from(response: Response, body: unknown): ApiError {
    const header = response.headers.get('Retry-After');
    const parsed = header === null ? NaN : Number.parseInt(header, 10);
    const detail =
      typeof body === 'object' && body !== null && 'detail' in body &&
      typeof (body as { detail: unknown }).detail === 'string'
        ? (body as { detail: string }).detail
        : response.statusText;
    return new ApiError(
      response.status,
      Number.isFinite(parsed) && parsed >= 0 ? parsed : null,
      `${response.status}: ${detail || 'request failed'}`,
    );
  }
}
