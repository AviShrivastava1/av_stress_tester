import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError, unwrap } from './client';

afterEach(() => {
  vi.useRealTimers();
});

/** Records how a promise settled without awaiting it, so an assertion can fail fast
 *  instead of waiting on a promise that never settles. */
function track(promise: Promise<unknown>) {
  const outcome: { state: 'pending' | 'fulfilled' | 'rejected'; value?: unknown } = { state: 'pending' };
  promise.then(
    (value) => { outcome.state = 'fulfilled'; outcome.value = value; },
    (error) => { outcome.state = 'rejected'; outcome.value = error; },
  );
  return outcome;
}

/** A fetch that never answers, and honours its signal the way a browser's does. */
function stubStalledFetch() {
  vi.stubGlobal('fetch', (request: Request) => new Promise((_resolve, reject) => {
    if (request.signal.aborted) return reject(request.signal.reason);
    request.signal.addEventListener('abort', () => reject(request.signal.reason), { once: true });
  }));
}

/** A fetch whose headers arrive at once and whose body never finishes. Aborting the
 *  signal errors the body stream, as a browser's fetch does. */
function stubStalledBodyFetch() {
  vi.stubGlobal('fetch', (request: Request) => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('{"total_scenarios":'));
        request.signal.addEventListener('abort', () => controller.error(request.signal.reason), { once: true });
      },
    });
    return Promise.resolve(new Response(body, { status: 200, headers: { 'Content-Type': 'application/json' } }));
  });
}

describe('request deadline', () => {
  it('is still waiting just before 30 s and times out just after', async () => {
    vi.useFakeTimers();
    stubStalledFetch();
    const outcome = track(api.GET('/stats'));
    await vi.advanceTimersByTimeAsync(29_999);
    expect(outcome.state).toBe('pending');
    await vi.advanceTimersByTimeAsync(2);
    expect(outcome.state).toBe('rejected');
    // The message names the deadline the two timings above just showed is enforced.
    expect(outcome.value).toMatchObject({
      name: 'RequestTimeoutError',
      message: expect.stringContaining('30 seconds'),
    });
  });

  it('reports a timeout even when the platform rejects with a bare AbortError', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', (request: Request) => new Promise((_resolve, reject) => {
      request.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true });
    }));
    const outcome = track(api.GET('/stats'));
    await vi.advanceTimersByTimeAsync(30_001);
    expect(outcome.state).toBe('rejected');
    expect(outcome.value).toMatchObject({ name: 'RequestTimeoutError' });
  });

  it('covers reading the response body, not only the headers', async () => {
    vi.useFakeTimers();
    stubStalledBodyFetch();
    const outcome = track(api.GET('/stats'));
    await vi.advanceTimersByTimeAsync(29_999);
    expect(outcome.state).toBe('pending');
    await vi.advanceTimersByTimeAsync(2);
    expect(outcome.state).toBe('rejected');
    expect(outcome.value).toMatchObject({ name: 'RequestTimeoutError' });
  });

  it('leaves no timer running after a request completes', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', async () => new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } }));
    await api.GET('/stats');
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('caller cancellation', () => {
  it('is reported as the caller\'s abort and clears the deadline timer', async () => {
    vi.useFakeTimers();
    stubStalledFetch();
    const controller = new AbortController();
    const outcome = track(api.GET('/stats', { signal: controller.signal }));
    // openapi-fetch has asynchronous middleware; let it start the request first.
    await vi.advanceTimersByTimeAsync(0);
    controller.abort();
    await vi.advanceTimersByTimeAsync(0);
    expect(outcome.state).toBe('rejected');
    expect(outcome.value).toMatchObject({ name: 'AbortError' });
    expect(vi.getTimerCount()).toBe(0);
  });

  it('does not start waiting on a signal that is already aborted', async () => {
    vi.useFakeTimers();
    stubStalledFetch();
    const controller = new AbortController();
    controller.abort();
    const outcome = track(api.GET('/stats', { signal: controller.signal }));
    await vi.advanceTimersByTimeAsync(0);
    expect(outcome.state).toBe('rejected');
    expect(outcome.value).toMatchObject({ name: 'AbortError' });
    expect(vi.getTimerCount()).toBe(0);
  });

  it('is not relabelled as a timeout when the deadline would have passed later', async () => {
    vi.useFakeTimers();
    stubStalledFetch();
    const controller = new AbortController();
    const outcome = track(api.GET('/stats', { signal: controller.signal }));
    await vi.advanceTimersByTimeAsync(10_000);
    controller.abort();
    await vi.advanceTimersByTimeAsync(40_000);
    expect(outcome.state).toBe('rejected');
    expect(outcome.value).toMatchObject({ name: 'AbortError' });
  });

  it('cancels a body read that is already under way', async () => {
    vi.useFakeTimers();
    stubStalledBodyFetch();
    const controller = new AbortController();
    const outcome = track(api.GET('/stats', { signal: controller.signal }));
    await vi.advanceTimersByTimeAsync(1_000);
    controller.abort();
    await vi.advanceTimersByTimeAsync(0);
    expect(outcome.state).toBe('rejected');
    expect(outcome.value).toMatchObject({ name: 'AbortError' });
    expect(vi.getTimerCount()).toBe(0);
  });
});

// The wrapper rebuilds the Response from its bytes. openapi-fetch reads only the status,
// Content-Length and Transfer-Encoding to decide whether there is a body, and
// ApiError reads statusText and Retry-After. These pin that none of it changes.
describe('the rebuilt response behaves like the original', () => {
  it('passes a 204 through as an empty success', async () => {
    vi.stubGlobal('fetch', async () => new Response(null, { status: 204 }));
    const result = await api.GET('/stats');
    expect(result.response.status).toBe(204);
    expect(result.data).toBeUndefined();
    expect(result.error).toBeUndefined();
  });

  it('parses a JSON body that carries a Content-Length', async () => {
    const text = JSON.stringify({ total_scenarios: 7 });
    vi.stubGlobal('fetch', async () => new Response(text, {
      status: 200,
      headers: { 'Content-Type': 'application/json', 'Content-Length': String(text.length) },
    }));
    const result = await api.GET('/stats');
    expect(result.data).toEqual({ total_scenarios: 7 });
  });

  it('parses a JSON body whose Content-Encoding and Content-Length describe the wire form, not the decoded bytes', async () => {
    const text = JSON.stringify({ total_scenarios: 7, padding: 'x'.repeat(200) });
    vi.stubGlobal('fetch', async () => new Response(text, {
      status: 200,
      headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip', 'Content-Length': '40' },
    }));
    const result = await api.GET('/stats');
    expect(result.data).toMatchObject({ total_scenarios: 7 });
  });

  it('parses a JSON body that has no Content-Length', async () => {
    vi.stubGlobal('fetch', async () => new Response('{"total_scenarios":7}', {
      status: 200, headers: { 'Content-Type': 'application/json' },
    }));
    const result = await api.GET('/stats');
    expect(result.data).toEqual({ total_scenarios: 7 });
  });

  it('treats a 200 with no body as no data', async () => {
    vi.stubGlobal('fetch', async () => new Response('', { status: 200 }));
    const result = await api.GET('/stats');
    expect(result.data).toBeUndefined();
  });

  it('keeps the status, the JSON error body and Retry-After that ApiError reads', async () => {
    vi.stubGlobal('fetch', async () => new Response(JSON.stringify({ detail: 'busy' }), {
      status: 503,
      statusText: 'Service Unavailable',
      headers: { 'Content-Type': 'application/json', 'Retry-After': '2' },
    }));
    const result = await api.GET('/stats');
    expect(result.response.status).toBe(503);
    expect(result.error).toEqual({ detail: 'busy' });
    let thrown: unknown;
    try { unwrap(result); } catch (error) { thrown = error; }
    expect(thrown).toBeInstanceOf(ApiError);
    expect(thrown).toMatchObject({ status: 503, retryAfterSeconds: 2, message: '503: busy' });
  });

  it('keeps statusText when the error body is not JSON', async () => {
    vi.stubGlobal('fetch', async () => new Response('upstream down', { status: 502, statusText: 'Bad Gateway' }));
    const result = await api.GET('/stats');
    let thrown: unknown;
    try { unwrap(result); } catch (error) { thrown = error; }
    expect(thrown).toMatchObject({ status: 502, message: '502: Bad Gateway' });
  });
});
