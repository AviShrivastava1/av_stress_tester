import { useState, type ReactNode } from 'react';

/*
 * The "Copy brief" control. What each state says, and what backs it:
 *
 *   "Copied the brief."
 *       navigator.clipboard.writeText resolved: the browser accepted the text. Nothing is said before the button is pressed
 *   "Could not copy to the clipboard. The brief is below: select it and copy it yourself."
 *       there is no clipboard API in this browser, or writeText threw or was refused (a permission, an insecure page). The
 *       same text that was handed to the clipboard is shown in a read-only box, so the brief is never lost and a copy that did
 *       not happen is never reported as one
 */
type State = { kind: 'idle' } | { kind: 'copied' } | { kind: 'failed'; text: string };

export function BriefButton({ build, children }: { build: () => string; children?: ReactNode }) {
  const [state, setState] = useState<State>({ kind: 'idle' });

  async function copy() {
    const text = build();
    try {
      // A browser with no clipboard API makes this line throw a TypeError, which is the same failure as a refusal.
      await navigator.clipboard.writeText(text);
      setState({ kind: 'copied' });
    } catch {
      setState({ kind: 'failed', text });
    }
  }

  return (
    <div className="brief">
      <button type="button" onClick={() => void copy()}>
        Copy brief
      </button>
      <span role="status">{state.kind === 'copied' ? 'Copied the brief.' : ''}</span>
      {children}
      {state.kind === 'failed' && (
        <>
          <p className="status-error" role="alert">
            Could not copy to the clipboard. The brief is below: select it and copy it yourself.
          </p>
          <textarea
            className="brief-text"
            aria-label="Brief"
            readOnly
            rows={14}
            value={state.text}
            onFocus={(e) => e.currentTarget.select()}
          />
        </>
      )}
    </div>
  );
}
