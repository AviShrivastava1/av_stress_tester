import { useEffect } from 'react';
import { PROJECT_NAME } from './components/Footer';

/*
 * The browser tab, the history list and a screen reader's page announcement read document.title, and index.html has one
 * title for the whole app. Each route now sets its own, in one form: "<page> · AV Scenario Stress-Tester".
 *
 *   "<page> · "              a middle dot with a space either side, as the app's other separators are written
 *                            (the eyebrows, "ILLUSTRATION · not measured data"); it separates the part that changes
 *                            from the part that does not
 *   "AV Scenario Stress-Tester"   PROJECT_NAME (components/Footer.tsx), the same name as the <title> in index.html and the
 *                            home link's accessible name
 *
 * Every route calls this on arrival and whenever its page string changes, so moving between routes (or between two scenarios,
 * which are one route with a different id) replaces the title and none is left over. The only data a title can carry is the
 * scenario id, which is read from the address; no count or other number is written into one.
 */
export function useDocumentTitle(page: string): void {
  useEffect(() => {
    document.title = `${page} · ${PROJECT_NAME}`;
  }, [page]);
}
