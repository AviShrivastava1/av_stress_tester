import { describe, expect, it } from 'vitest';
import { requireApiBaseUrl } from './requireApiBaseUrl';

describe('requireApiBaseUrl', () => {
  it('refuses a production build with no API URL, which would ship a site pointed at localhost', () => {
    // The message must say what would ship, not merely that something is wrong.
    expect(() => requireApiBaseUrl('production', {})).toThrow(/not set.*localhost:8000/);
    expect(() => requireApiBaseUrl('production', { VITE_API_BASE_URL: '  ' })).toThrow(
      /not set.*localhost:8000/,
    );
  });

  it('refuses plain http to a non-local host, which an https site cannot call', () => {
    expect(() =>
      requireApiBaseUrl('production', { VITE_API_BASE_URL: 'http://api.example.onrender.com' }),
    ).toThrow(/https/);
  });

  it('refuses a value that is not a URL', () => {
    expect(() => requireApiBaseUrl('production', { VITE_API_BASE_URL: 'api.example.com' })).toThrow(
      /not a URL/,
    );
  });

  it('accepts https, and http on localhost for a local production preview', () => {
    expect(() =>
      requireApiBaseUrl('production', { VITE_API_BASE_URL: 'https://api.example.onrender.com' }),
    ).not.toThrow();
    expect(() =>
      requireApiBaseUrl('production', { VITE_API_BASE_URL: 'http://localhost:8000' }),
    ).not.toThrow();
  });

  it('leaves development and test modes alone: they use the localhost default', () => {
    expect(() => requireApiBaseUrl('development', {})).not.toThrow();
    expect(() => requireApiBaseUrl('test', {})).not.toThrow();
  });
});
