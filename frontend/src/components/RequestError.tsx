import { describeError } from '../api/errors';

export function RequestError({ error, retry, retrying = false, context }: {
  error: unknown;
  retry(): unknown;
  retrying?: boolean;
  context?: string;
}) {
  return (
    <div className="request-error">
      <p className="status status-error" role="alert">
        {context && <>{context} </>}{describeError(error)}
      </p>
      <button type="button" onClick={() => void retry()} disabled={retrying}>
        {retrying ? 'Retrying…' : 'Try again'}
      </button>
    </div>
  );
}
