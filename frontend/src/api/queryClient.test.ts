import { describe, expect, it } from 'vitest';
import { ApiError } from './client';
import { retryDelay, shouldRetry } from './queryClient';
import { RequestTimeoutError } from './request';

describe('retry policy', () => {
  it('retries a 503, up to three times', () => {
    const busy = new ApiError(503, 1, '503: busy');
    expect(shouldRetry(0, busy)).toBe(true);
    expect(shouldRetry(2, busy)).toBe(true);
    expect(shouldRetry(3, busy)).toBe(false);
  });

  it('does not retry other HTTP errors', () => {
    expect(shouldRetry(0, new ApiError(404, null, '404'))).toBe(false);
    expect(shouldRetry(0, new ApiError(500, null, '500'))).toBe(false);
  });

  it('does not retry a request that already waited out its deadline', () => {
    expect(shouldRetry(0, new RequestTimeoutError())).toBe(false);
  });

  it('retries network failures twice', () => {
    const offline = new TypeError('Failed to fetch');
    expect(shouldRetry(1, offline)).toBe(true);
    expect(shouldRetry(2, offline)).toBe(false);
  });

  it("waits for the server's Retry-After when given", () => {
    expect(retryDelay(0, new ApiError(503, 2, '503'))).toBe(2000);
  });

  it('backs off exponentially, capped, otherwise', () => {
    expect(retryDelay(0, new TypeError('x'))).toBe(1000);
    expect(retryDelay(1, new TypeError('x'))).toBe(2000);
    expect(retryDelay(10, new TypeError('x'))).toBe(8000);
  });
});
