export const REQUEST_TIMEOUT_MS = 30_000;

export class RequestTimeoutError extends Error {
  constructor() {
    super(`The request took longer than ${REQUEST_TIMEOUT_MS / 1000} seconds.`);
    this.name = 'RequestTimeoutError';
  }
}

/** Bound the complete response, including its body, and preserve navigation cancellation. */
export async function fetchWithDeadline(request: Request): Promise<Response> {
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort(request.signal.reason);
  if (request.signal.aborted) cancel();
  else request.signal.addEventListener('abort', cancel, { once: true });
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort(new RequestTimeoutError());
  }, REQUEST_TIMEOUT_MS);
  try {
    const response = await globalThis.fetch(new Request(request, { signal: controller.signal }));
    // openapi-fetch parses JSON after fetch returns. Read the body while the
    // deadline is still active, so stalled downloads are bounded too.
    const bytes = await response.arrayBuffer();
    return new Response(bytes.byteLength ? bytes : null, {
      status: response.status, statusText: response.statusText, headers: response.headers,
    });
  } catch (error) {
    if (timedOut) throw new RequestTimeoutError();
    throw error;
  } finally {
    clearTimeout(timeout);
    request.signal.removeEventListener('abort', cancel);
  }
}
