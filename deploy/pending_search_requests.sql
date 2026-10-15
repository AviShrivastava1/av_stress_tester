-- The requests waiting to be run offline, oldest first. Read-only: the transaction cannot write.
--
--   psql "$AV_STRESS_DATABASE_URL" -f deploy/pending_search_requests.sql
--
-- Connect as the database's default user (the owner): no other role can read search_requests.
-- Marking one done, failed or cancelled is a separate, deliberate UPDATE by hand; nothing here does it.
\set ON_ERROR_STOP on
BEGIN READ ONLY;
SELECT request_id,
       scenario_id,
       seed,
       status,
       requested_at,
       now() - requested_at AS waiting
  FROM search_requests
 WHERE status IN ('queued', 'running')
 ORDER BY requested_at, request_id;
COMMIT;
