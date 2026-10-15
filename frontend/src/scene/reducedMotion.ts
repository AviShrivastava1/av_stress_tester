import { useSyncExternalStore } from 'react';

/*
 * Whether the viewer has asked their system for reduced motion, as the browser reports it through the standard media query
 * `prefers-reduced-motion: reduce`. A browser that cannot answer (no matchMedia, as in a test environment or a very old
 * browser) is read as "no preference": nothing is assumed about a viewer who has not said.
 *
 * It is read through useSyncExternalStore so that the answer follows the setting while the page is open: the query's
 * "change" event is the source of truth, and the hook stops listening when its component goes away. Only this one query is
 * asked about.
 */
const QUERY = '(prefers-reduced-motion: reduce)';

function mediaQuery(): MediaQueryList | null {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function' ? window.matchMedia(QUERY) : null;
}

function subscribe(onChange: () => void): () => void {
  const query = mediaQuery();
  if (query === null) return () => undefined;
  query.addEventListener('change', onChange);
  return () => query.removeEventListener('change', onChange);
}

const getSnapshot = () => mediaQuery()?.matches ?? false;
const getServerSnapshot = () => false;

export function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
