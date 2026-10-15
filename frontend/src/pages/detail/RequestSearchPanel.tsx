import { useMutation } from '@tanstack/react-query';
import { useState, type FormEvent } from 'react';
import { ApiError } from '../../api/client';
import { describeError } from '../../api/errors';
import {
  fileSearchRequest,
  formatWait,
  MAX_SEED,
  MIN_SEED,
  parseSeed,
  searchRequestsEnabled,
  UnexpectedAnswerError,
  type FiledSearchRequest,
} from '../../api/searchRequests';

/*
 * The words on the request form, and what backs each. It renders only when the site is built with VITE_SEARCH_REQUESTS=true
 * (api/searchRequests.ts searchRequestsEnabled); otherwise nothing at all is rendered and nothing is requested.
 *
 *   h2 "Request a search" and the paragraph under it
 *       POST /search-requests (src/api/search_requests.py) files a scenario id and a seed and answers; nothing else happens.
 *       "The owner may run it offline": the owner reads what is waiting (deploy/pending_search_requests.sql) and decides; there
 *       is no worker, no schedule and no queue consumer in the repository, so "no time is promised" is the whole truth and the
 *       page must not say or imply "soon", "later today" or "will be run". "Nothing runs when you press the button": the
 *       endpoint computes nothing. "The results on this page do not change when it is run": this page reads the stored
 *       results (scenario_scores and its two geometry tables); the endpoint writes only search_requests, and a requested run
 *       is kept apart from the stored results (DESIGN-21.md section 3: its own tables, never the same rows). That separation
 *       is how the owner runs a request, not something this page checks, and no endpoint tells this page when one is run
 *   "Filed: request {id} for seed {n}. It is queued: the owner may run it offline, at no stated time, and this page does not
 *    update. Nothing has run."
 *       201 with state "filed", and "Already filed" for 200 with state "existing" (the same scenario and seed is one request,
 *       so the first is returned). "Queued" is the status the row is stored with (search_requests.status, default queued).
 *       "This page does not update": nothing polls for a request, and there is no endpoint that could answer
 *   "The seed must be a whole number from 0 to 2147483647."
 *       SearchRequestBody.seed: a strict integer, ge 0, le MAX_SEED (api/searchRequests.ts has the same bounds)
 *   "Whether the request was filed is not known. Pressing the button again is harmless: ..."
 *       a 500, a timeout, no network, or a 2xx that is not a filed request: each may have come after the request was stored
 *       (the endpoint commits, then answers), so the page does not say "nothing was filed". The same scenario and seed is one
 *       request (the function returns the first), so filing again cannot make a second
 *   "This scenario is not in the stored results, so nothing was filed."
 *       the 404 (the scenario must exist in scenario_scores)
 *   "You have reached a limit on requests ..." and "Requests cannot be filed right now ..."
 *       the 429 (this visitor's own limits) and the 503 (the shared limits, or the pool is busy), each with Retry-After in
 *       seconds, which formatWait puts in words. The limits are the constants in src/api/search_requests.py; no number from
 *       them is written here
 */

const SEED_ERROR = `The seed must be a whole number from ${MIN_SEED} to ${MAX_SEED}.`;

function refusal(error: unknown): string {
  if (error instanceof ApiError) {
    const wait = formatWait(error.retryAfterSeconds);
    switch (error.status) {
      case 404:
        return 'This scenario is not in the stored results, so nothing was filed.';
      case 422:
        return `The request was not accepted, so nothing was filed. ${SEED_ERROR}`;
      case 429:
        return `You have reached a limit on requests, so nothing was filed. Try again ${wait === 'later' ? 'later' : `in ${wait}`}.`;
      case 503:
        return `Requests cannot be filed right now, so nothing was filed. Try again ${wait === 'later' ? 'later' : `in ${wait}`}.`;
    }
  }
  // Anything else (a 500, a timeout, no network, an answer that is not a filed request) may have happened after the request
  // was stored, so "nothing was filed" would be a claim this page cannot make. Filing again is harmless: the same scenario
  // and seed is one request.
  const what = error instanceof UnexpectedAnswerError ? `${error.message}` : describeError(error);
  return `${what} Whether the request was filed is not known. Pressing the button again is harmless: the same scenario and seed is one request.`;
}

function Outcome({ filed }: { filed: FiledSearchRequest }) {
  return (
    <p className="status" role="status">
      {filed.state === 'filed' ? 'Filed' : 'Already filed'}: request <span className="mono">{filed.request_id}</span> for seed{' '}
      {filed.seed}. It is queued: the owner may run it offline, at no stated time, and this page does not update. Nothing has
      run.
    </p>
  );
}

function RequestForm({ scenarioId }: { scenarioId: string }) {
  const [text, setText] = useState('');
  const [invalid, setInvalid] = useState(false);
  const request = useMutation({ mutationFn: (seed: number) => fileSearchRequest(scenarioId, seed) });

  function submit(event: FormEvent) {
    event.preventDefault();
    if (request.isPending) return;
    const seed = parseSeed(text);
    if (seed === null) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    request.mutate(seed);
  }

  return (
    <section className="search-request" aria-labelledby="search-request-title">
      <h2 id="search-request-title">Request a search</h2>
      <p className="note">
        Ask for this scenario to be searched with a seed of your choice. The request is filed, and the owner may run it
        offline; no time is promised. Nothing runs when you press the button. The results on this page do not change when it
        is run.
      </p>
      <form className="search-request-form" onSubmit={submit} noValidate>
        <label htmlFor="search-request-seed">Seed</label>
        <input
          id="search-request-seed"
          type="text"
          inputMode="numeric"
          autoComplete="off"
          value={text}
          aria-invalid={invalid}
          aria-describedby={invalid ? 'search-request-seed-error' : undefined}
          onChange={(event) => {
            setText(event.target.value.trim());
            setInvalid(false);
            if (!request.isPending) request.reset();
          }}
        />
        <button type="submit" disabled={request.isPending || text === ''}>
          {request.isPending ? 'Filing…' : 'File request'}
        </button>
      </form>
      {invalid && (
        <p id="search-request-seed-error" className="status status-error" role="alert">
          {SEED_ERROR}
        </p>
      )}
      {request.isError && (
        <p className="status status-error" role="alert">
          {refusal(request.error)}
        </p>
      )}
      {request.isSuccess && <Outcome filed={request.data} />}
    </section>
  );
}

/** The request form, or nothing at all when this build does not offer it. */
export function RequestSearchPanel({ scenarioId }: { scenarioId: string }) {
  if (!searchRequestsEnabled()) return null;
  return <RequestForm scenarioId={scenarioId} />;
}
