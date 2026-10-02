import { API_BASE_URL, ApiError } from './client';
import { RequestTimeoutError } from './request';

export function describeError(error: unknown): string {
  if (error instanceof RequestTimeoutError) return 'The API is taking longer than expected. Please try again.';
  if (error instanceof ApiError) {
    if (error.status === 503) {
      return 'The API is busy (all database connections in use). Try again in a moment.';
    }
    return `The API returned an error — ${error.message}`;
  }
  return `Could not reach the API at ${API_BASE_URL}.`;
}

export function isNotFound(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404;
}
