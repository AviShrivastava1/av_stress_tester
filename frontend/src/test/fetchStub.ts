import { vi } from 'vitest';

export interface RecordedRequest {
  path: string;
  query: URLSearchParams;
}

type Reply = { status: number; body: unknown; headers?: Record<string, string> };

/**
 * Stubs globalThis.fetch. `respond` sees each request's path and decoded query and
 * returns what the server would; every request is recorded so tests can assert on
 * exactly what the client sent.
 */
export function stubFetch(respond: (req: RecordedRequest) => Reply) {
  const requests: RecordedRequest[] = [];
  vi.stubGlobal('fetch', async (input: Request) => {
    const url = new URL(input.url);
    const req = { path: url.pathname, query: url.searchParams };
    requests.push(req);
    const { status, body, headers } = respond(req);
    return new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json', ...headers },
    });
  });
  return requests;
}
