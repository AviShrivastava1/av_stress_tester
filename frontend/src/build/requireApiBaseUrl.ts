/**
 * A production build must be told where the API is.
 *
 * client.ts falls back to http://localhost:8000 when VITE_API_BASE_URL is unset. That is
 * right for `npm run dev` and silently wrong for a deployed site: the build succeeds, and
 * every visitor sees "Could not reach the API at http://localhost:8000". The variable is
 * baked in at build time, so the build is the place to refuse.
 *
 * Plain http is refused for anything but localhost: an https page cannot call an http
 * API (mixed content), so such a URL would fail in exactly the same way. http on
 * localhost stays allowed for a local `vite build && vite preview`.
 *
 * The value is baked into the public bundle, so three more forms are refused. Credentials in it (a user name or
 * password before the host) would be published with the site. A query string or a fragment makes the endpoint
 * ambiguous: the client appends each path to this base, so "https://api.example.com?x=1" followed by "/stats" is not
 * a URL anyone meant. A path prefix ("https://api.example.com/v1") is a real deployment shape and stays allowed, and
 * so do trailing slashes, which client.ts strips. None of the messages prints the value: it can contain the
 * credentials this check exists to keep out of the build, and a build log is not a private place.
 *
 * Called from vite.config.ts with the env Vite itself would expose to the bundle.
 */
export function requireApiBaseUrl(mode: string, env: Record<string, string | undefined>): void {
  if (mode !== 'production') return;

  const raw = env.VITE_API_BASE_URL?.trim();
  if (!raw) {
    throw new Error(
      'VITE_API_BASE_URL is not set, so this production build would point the site at ' +
        "http://localhost:8000. Set it to the API's public URL (on Vercel: Project " +
        'Settings > Environment Variables, then redeploy).',
    );
  }

  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error('VITE_API_BASE_URL is not a URL. Give it as https://host with an optional path prefix.');
  }
  // The authority is everything between "://" and the first "/", "?" or "#". An "@" in it means user information, even
  // an empty one that URL parsing drops (https://@host), so the raw text is checked as well as the parsed fields.
  if (url.username !== '' || url.password !== '' || /^[a-z][a-z0-9+.-]*:\/\/[^/?#]*@/i.test(raw)) {
    throw new Error(
      'VITE_API_BASE_URL must not contain credentials (a user name or password before the host): ' +
        'they would be published in the build.',
    );
  }
  if (raw.includes('?')) {
    throw new Error(
      'VITE_API_BASE_URL must not contain a query string (anything after "?"): the client appends paths to it.',
    );
  }
  if (raw.includes('#')) {
    throw new Error(
      'VITE_API_BASE_URL must not contain a fragment (anything after "#"): the client appends paths to it.',
    );
  }
  const local = url.hostname === 'localhost' || url.hostname === '127.0.0.1';
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && local)) {
    throw new Error(
      `VITE_API_BASE_URL must use https (an https site cannot call ${url.protocol}// APIs).`,
    );
  }
}
