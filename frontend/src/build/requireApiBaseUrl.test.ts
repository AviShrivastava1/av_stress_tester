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

/*
 * What a production API base URL may not carry. It is baked into the public bundle, so credentials in it would be
 * published; a query string or fragment makes endpoint resolution ambiguous (the client appends paths to it). The
 * error names the setting and the problem and never prints the value, which can hold the credentials.
 */
const SECRET = 'hunter2-secret';
const refuse = (value: string) => {
  try {
    requireApiBaseUrl('production', { VITE_API_BASE_URL: value });
  } catch (e) {
    return (e as Error).message;
  }
  throw new Error(`expected ${'a refusal'}`);
};

describe('requireApiBaseUrl, refused forms', () => {
  it.each([
    ['credentials (user and password)', `https://user:${SECRET}@api.example.com`, /credentials/],
    ['credentials (user only)', `https://${SECRET}@api.example.com`, /credentials/],
    ['credentials (empty user, password only)', `https://:${SECRET}@api.example.com`, /credentials/],
    ['credentials on a non-https URL', `http://user:${SECRET}@api.example.com`, /credentials/],
    ['empty user information (just an "@")', 'https://@api.example.com', /credentials/],
    // WHATWG parsing reads a backslash as a slash for https, so these are credentials the raw "://" pattern does not see.
    ['credentials written with backslashes', `https:\\\\user:${SECRET}@api.example.com`, /credentials/],
    ['a password only, written with backslashes', `https:\\\\:${SECRET}@api.example.com`, /credentials/],
    ['a query string', `https://api.example.com?token=${SECRET}`, /query string/],
    ['a query string after a path', `https://api.example.com/v1?token=${SECRET}`, /query string/],
    ['an empty query string', 'https://api.example.com/?', /query string/],
    ['a fragment', `https://api.example.com#${SECRET}`, /fragment/],
    ['a fragment after a path', `https://api.example.com/v1#${SECRET}`, /fragment/],
    ['an empty fragment', 'https://api.example.com/#', /fragment/],
  ])('refuses %s', (_name, value, problem) => {
    const message = refuse(value);
    expect(message).toMatch(/^VITE_API_BASE_URL /);
    expect(message).toMatch(problem);
    expect(message).not.toContain(SECRET);
    expect(message).not.toContain('api.example.com');
  });

  it('does not print the value in the older refusals either', () => {
    const notUrl = refuse(`${SECRET}.example.com`);
    const http = refuse(`http://${SECRET}.example.com`);
    expect(notUrl).toMatch(/not a URL/);
    expect(http).toMatch(/https/);
    expect(notUrl).not.toContain(SECRET);
    expect(http).not.toContain(SECRET);
  });
});

describe('requireApiBaseUrl, accepted forms', () => {
  it.each([
    ['plain https', 'https://api.example.com'],
    ['a trailing slash', 'https://api.example.com/'],
    ['several trailing slashes', 'https://api.example.com///'],
    ['a path prefix', 'https://api.example.com/v1'],
    ['a path prefix with a trailing slash', 'https://api.example.com/v1/'],
    ['a port', 'https://api.example.com:8443/v1'],
    ['surrounding whitespace (trimmed, as before)', '  https://api.example.com/v1  '],
    ['http on localhost', 'http://localhost:8000'],
    ['http on 127.0.0.1', 'http://127.0.0.1:8000/v1'],
  ])('accepts %s', (_name, value) => {
    expect(() => requireApiBaseUrl('production', { VITE_API_BASE_URL: value })).not.toThrow();
  });

  it('still refuses plain http to a non-local host', () => {
    expect(refuse('http://api.example.com/v1')).toMatch(/https/);
  });
});
