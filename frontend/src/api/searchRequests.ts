import { API_BASE_URL, ApiError } from './client';
import { fetchWithDeadline } from './request';

/*
 * Queued search requests: a scenario id and a seed, filed through POST /search-requests and run LATER, OFFLINE, by the owner.
 * Nothing runs when one is filed. OFF unless the site is built with VITE_SEARCH_REQUESTS=true; the API route exists only when
 * the API is switched on too (src/api/search_requests.py), and while it is off it is not in openapi.json, so these types are
 * written here by hand, from that module's SearchRequestBody and SearchRequestResponse, and not generated.
 */

/** The seed's bounds: the API's (src/api/search_requests.py MAX_SEED), the database's integer column. */
export const MIN_SEED = 0;
export const MAX_SEED = 2_147_483_647;

/**
 * Whether this build offers the request form. Read from the build-time variable and true only for exactly "true": unset
 * (the default, and what every committed env file and the deploy configuration leave it), empty, "1", "yes" or "TRUE" are off.
 */
export function searchRequestsEnabled(): boolean {
  return import.meta.env.VITE_SEARCH_REQUESTS === 'true';
}

export interface FiledSearchRequest {
  request_id: string;
  scenario_id: string;
  seed: number;
  /** "filed" when this call filed it; "existing" when the same scenario and seed had been filed before. */
  state: 'filed' | 'existing';
}

function isFiled(body: unknown): body is FiledSearchRequest {
  if (typeof body !== 'object' || body === null) return false;
  const b = body as Record<string, unknown>;
  return typeof b.request_id === 'string' && typeof b.scenario_id === 'string' && Number.isInteger(b.seed) &&
    (b.state === 'filed' || b.state === 'existing');
}

/** The API answered, but not with a filed request: whether it filed anything is not known from here. */
export class UnexpectedAnswerError extends Error {
  constructor() {
    super('The API answered with something this page does not understand.');
    this.name = 'UnexpectedAnswerError';
  }
}

/** The seed a text field holds, or null when it is not a whole number from MIN_SEED to MAX_SEED. */
export function parseSeed(text: string): number | null {
  if (!/^\d{1,10}$/.test(text)) return null;
  const seed = Number(text);
  return seed >= MIN_SEED && seed <= MAX_SEED ? seed : null;
}

/** File one request. A refusal is an ApiError (404, 422, 429 or 503, with Retry-After in seconds where the API sent it). */
export async function fileSearchRequest(scenarioId: string, seed: number): Promise<FiledSearchRequest> {
  const response = await fetchWithDeadline(
    new Request(`${API_BASE_URL}/search-requests`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario_id: scenarioId, seed }),
    }),
  );
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) throw ApiError.from(response, body);
  if (!isFiled(body)) throw new UnexpectedAnswerError();
  return body;
}

/** "about a minute", "about 3 minutes", "about 2 hours"; "later" when the API gave no time. */
export function formatWait(seconds: number | null): string {
  if (seconds === null) return 'later';
  if (seconds <= 90) return 'about a minute';
  if (seconds < 3240) return `about ${Math.round(seconds / 60)} minutes`; // under 54 minutes; from there it is "an hour"
  const hours = Math.round(seconds / 3600);
  return hours === 1 ? 'about an hour' : `about ${hours} hours`;
}
