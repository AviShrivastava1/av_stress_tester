-- Queued search requests: the table, the one function that writes to it, and the one role that may call it.
--
-- RUN BY HAND, as the database's default user (the owner of the three result tables), after the schema exists.
-- Nothing in src/ runs it, and no schema initialiser in src/scoring mentions it. Re-running it is safe.
--
--   psql "$AV_STRESS_DATABASE_URL" -f deploy/search_requests.sql
--   psql "$AV_STRESS_DATABASE_URL" -c '\password av_req_insert'      -- silent prompt; the password is never typed into SQL
--
-- What a request is: a scenario id and a seed, filed by the API and run LATER, OFFLINE, by the owner. Nothing here
-- computes anything, and nothing here touches scenario_scores, scenario_agents or perturbed_paths.
--
-- The role av_req_insert has NO privilege on any table. It can call file_search_request() and nothing else. A bare
-- INSERT grant would let a holder of the credential skip every limit below; the function counts and inserts in one
-- transaction, under one lock, with the limits passed in by the API.

CREATE TABLE IF NOT EXISTS search_requests (
    request_id     uuid        PRIMARY KEY,
    scenario_id    text        NOT NULL,
    seed           integer     NOT NULL CHECK (seed BETWEEN 0 AND 2147483647),
    -- HMAC-SHA256 of the requester's address under a secret salt, hex. The address itself is never stored.
    requester_hash text        NOT NULL CHECK (requester_hash ~ '^[0-9a-f]{64}$'),
    requested_at   timestamptz NOT NULL DEFAULT now(),
    status         text        NOT NULL DEFAULT 'queued'
                               CHECK (status IN ('queued', 'running', 'done', 'failed', 'expired', 'cancelled')),
    -- The same scenario and seed is one request: filing it again returns the first.
    UNIQUE (scenario_id, seed)
);
CREATE INDEX IF NOT EXISTS search_requests_status_requested_at ON search_requests (status, requested_at);
CREATE INDEX IF NOT EXISTS search_requests_requester_requested_at ON search_requests (requester_hash, requested_at);

-- outcome is one of: filed, existing, requester_active, requester_daily, queue_full, global_daily.
-- retry_after_seconds is set for the four refusals.
CREATE OR REPLACE FUNCTION file_search_request(
    p_request_id uuid,
    p_scenario_id text,
    p_seed integer,
    p_requester_hash text,
    p_max_active_per_requester integer,
    p_max_per_requester_per_day integer,
    p_max_per_day integer,
    p_max_queue integer
) RETURNS TABLE (outcome text, filed_request_id uuid, retry_after_seconds integer)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_catalog
AS $fn$
DECLARE
    v_existing uuid;
    v_count    integer;
    v_oldest   timestamptz;
BEGIN
    -- One filing at a time, so two requests cannot both pass the same count.
    PERFORM pg_advisory_xact_lock(hashtext('av_stress.search_requests'));

    SELECT r.request_id INTO v_existing
      FROM search_requests r
     WHERE r.scenario_id = p_scenario_id AND r.seed = p_seed;
    IF FOUND THEN
        RETURN QUERY SELECT 'existing'::text, v_existing, NULL::integer;
        RETURN;
    END IF;

    SELECT count(*) INTO v_count FROM search_requests r
     WHERE r.requester_hash = p_requester_hash AND r.status IN ('queued', 'running');
    IF v_count >= p_max_active_per_requester THEN
        RETURN QUERY SELECT 'requester_active'::text, NULL::uuid, 3600;
        RETURN;
    END IF;

    SELECT count(*), min(r.requested_at) INTO v_count, v_oldest FROM search_requests r
     WHERE r.requester_hash = p_requester_hash AND r.requested_at > now() - interval '1 day';
    IF v_count >= p_max_per_requester_per_day THEN
        RETURN QUERY SELECT 'requester_daily'::text, NULL::uuid,
            GREATEST(1, ceil(extract(epoch FROM v_oldest + interval '1 day' - now()))::integer);
        RETURN;
    END IF;

    SELECT count(*) INTO v_count FROM search_requests r WHERE r.status = 'queued';
    IF v_count >= p_max_queue THEN
        RETURN QUERY SELECT 'queue_full'::text, NULL::uuid, 3600;
        RETURN;
    END IF;

    SELECT count(*), min(r.requested_at) INTO v_count, v_oldest FROM search_requests r
     WHERE r.requested_at > now() - interval '1 day';
    IF v_count >= p_max_per_day THEN
        RETURN QUERY SELECT 'global_daily'::text, NULL::uuid,
            GREATEST(1, ceil(extract(epoch FROM v_oldest + interval '1 day' - now()))::integer);
        RETURN;
    END IF;

    INSERT INTO search_requests (request_id, scenario_id, seed, requester_hash)
    VALUES (p_request_id, p_scenario_id, p_seed, p_requester_hash);
    RETURN QUERY SELECT 'filed'::text, p_request_id, NULL::integer;
END
$fn$;

-- The role. LOGIN, no other attribute; its password is set by hand (see the top of this file).
DO $role$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'av_req_insert') THEN
        CREATE ROLE av_req_insert LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
END
$role$;

DO $grant$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO av_req_insert', current_database());
END
$grant$;
GRANT USAGE ON SCHEMA public TO av_req_insert;

-- Nobody but the owner reads or writes the table directly, and nobody but av_req_insert calls the function.
REVOKE ALL ON TABLE search_requests FROM PUBLIC;
REVOKE ALL ON FUNCTION file_search_request(uuid, text, integer, text, integer, integer, integer, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION file_search_request(uuid, text, integer, text, integer, integer, integer, integer) TO av_req_insert;

-- The read-only API role (DEPLOY.md step 2) is granted SELECT on tables created later through a default privilege.
-- It must not read this one: the stored-result read path never touches requests.
DO $revoke$
BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'av_api_ro') THEN
        REVOKE ALL ON TABLE search_requests FROM av_api_ro;
    END IF;
END
$revoke$;
