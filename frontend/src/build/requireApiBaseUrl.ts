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
    throw new Error(`VITE_API_BASE_URL is not a URL: ${JSON.stringify(raw)}`);
  }
  const local = url.hostname === 'localhost' || url.hostname === '127.0.0.1';
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && local)) {
    throw new Error(
      `VITE_API_BASE_URL must use https (an https site cannot call ${url.protocol}// APIs): ${raw}`,
    );
  }
}
