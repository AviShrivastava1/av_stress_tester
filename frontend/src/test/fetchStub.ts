import { vi } from 'vitest';

export interface RecordedRequest {
  /** As sent: percent-encoding is preserved, so tests can check what went on the wire. */
  path: string;
  query: URLSearchParams;
}

export type Reply = { status: number; body: unknown; headers?: Record<string, string> };

function record(input: Request): RecordedRequest {
  const url = new URL(input.url);
  return { path: url.pathname, query: url.searchParams };
}

function toResponse({ status, body, headers }: Reply): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}

/**
 * Stubs globalThis.fetch. `respond` sees each request's path and decoded query and
 * returns what the server would; every request is recorded so tests can assert on
 * exactly what the client sent.
 */
export function stubFetch(respond: (req: RecordedRequest) => Reply) {
  const requests: RecordedRequest[] = [];
  vi.stubGlobal('fetch', async (input: Request) => {
    const req = record(input);
    requests.push(req);
    return toResponse(respond(req));
  });
  return requests;
}

/**
 * Like stubFetch, but no response is delivered until `releaseAll()`. Lets a test see
 * every request a page issues BEFORE any of them completes — i.e. whether any request
 * waited on another.
 */
export function stubFetchDeferred(respond: (req: RecordedRequest) => Reply) {
  const requests: RecordedRequest[] = [];
  const pending: (() => void)[] = [];
  vi.stubGlobal(
    'fetch',
    (input: Request) =>
      new Promise<Response>((resolve) => {
        const req = record(input);
        requests.push(req);
        pending.push(() => resolve(toResponse(respond(req))));
      }),
  );
  return {
    requests,
    releaseAll: () => pending.splice(0).forEach((release) => release()),
  };
}
