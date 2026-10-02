import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

async function loadClient(baseUrl: string | undefined) {
  vi.resetModules();
  vi.stubEnv('VITE_API_BASE_URL', baseUrl);
  return import('./client');
}

describe('API base URL', () => {
  it('has trailing slashes and surrounding whitespace removed', async () => {
    const { API_BASE_URL } = await loadClient('  https://api.example.test//  ');
    expect(API_BASE_URL).toBe('https://api.example.test');
  });

  it('is what requests are sent to, with a single slash before the path', async () => {
    const seen: string[] = [];
    vi.stubGlobal('fetch', async (request: Request) => {
      seen.push(request.url);
      return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
    });
    const { api } = await loadClient(' https://api.example.test/ ');
    await api.GET('/stats');
    expect(seen).toEqual(['https://api.example.test/stats']);
  });

  it('falls back to the local API when unset', async () => {
    const { API_BASE_URL } = await loadClient(undefined);
    expect(API_BASE_URL).toBe('http://localhost:8000');
  });
});
