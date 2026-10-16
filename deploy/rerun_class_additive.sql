-- Run manifests and the re-run class: the tables the offline pipeline records its runs and its extra searches in.
--
-- RUN BY HAND, as the database's default user (the owner of the three result tables), on a COPY of the database first:
--   psql "$COPY_DATABASE_URL" -v ON_ERROR_STOP=1 -f deploy/rerun_class_additive.sql
-- and then, only if the copy behaves, on the live one (DEPLOY.md, "Run manifests"). Nothing in src/ runs it, and no schema initialiser
-- mentions it. It is idempotent: run it twice and the second run changes nothing. It has been run only on throw-away databases.
--
-- What it does NOT do, on purpose: it changes no existing table, column, constraint, index, default or privilege on
-- scenario_scores, scenario_agents or perturbed_paths; it writes no row to them; it adds no key to search_provenance (that JSONB is
-- hashed into stress_run_id, so a new key would change the id of every result that is searched again); it adds no foreign key
-- to or from them (the data-replacement procedure in DEPLOY.md drops and recreates those three tables).
--
-- The offline writer is src/scoring/manifest.py. Today the API reads only the view pipeline_runs_public and pipeline_run_results
-- (src/api/routes.py), and DEPLOY.md grants the read-only role those two. The role also gets SELECT on the other new tables
-- (the default privilege of DEPLOY.md step 2 would give it anyway; they hold results and no secrets), and nothing on pipeline_runs
-- itself, whose command line, host label and notes are not public.
--
-- Requires: PostgreSQL 13 or later (gen_random_uuid() is built in). Nothing here needs PostGIS.

BEGIN;

-- ── #20: one row per run of the offline pipeline ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    pipeline_run_id  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    kind             text        NOT NULL CHECK (kind IN ('score', 'stress', 'export', 'rerun')),
    status           text        NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'finished', 'failed', 'abandoned')),
    started_at       timestamptz NOT NULL DEFAULT now(),
    finished_at      timestamptz,
    -- What ran. NULL always means "not recorded", never "none".
    git_revision     text        CHECK (git_revision IS NULL OR git_revision ~ '^[0-9a-f]{40}$'),
    git_dirty        boolean,
    python_version   text,
    package_versions jsonb       CHECK (package_versions IS NULL OR jsonb_typeof(package_versions) = 'object'),
    shard_name       text,
    shard_sha256     text        CHECK (shard_sha256 IS NULL OR shard_sha256 ~ '^[0-9a-f]{64}$'),
    parameters       jsonb       CHECK (parameters IS NULL OR jsonb_typeof(parameters) = 'object'),   -- seeds, budgets, k, ...
    counts           jsonb       CHECK (counts IS NULL OR jsonb_typeof(counts) = 'object'),           -- scored, searched, refused, ...
    -- Never served by the API (see the view below): they can hold paths, hostnames and anything typed on a command line.
    command_line     text,
    host_label       text,
    notes            text,
    CHECK (finished_at IS NULL OR finished_at >= started_at),
    CHECK ((status = 'running') = (finished_at IS NULL))
);
CREATE INDEX IF NOT EXISTS pipeline_runs_started_at ON pipeline_runs (started_at DESC);

-- Which stored results a run wrote or touched. No foreign key to scenario_scores: see the header. A reader joins on the identity gate.
CREATE TABLE IF NOT EXISTS pipeline_run_results (
    pipeline_run_id uuid NOT NULL REFERENCES pipeline_runs (pipeline_run_id) ON DELETE CASCADE,
    scenario_id     text NOT NULL,
    stress_run_id   text,
    PRIMARY KEY (pipeline_run_id, scenario_id)
);
CREATE INDEX IF NOT EXISTS pipeline_run_results_result ON pipeline_run_results (scenario_id, stress_run_id);

-- ── #1, #16, #17 (and the run behind #4): every search that is not the stored result, and the stored result's own twin ──────────
CREATE TABLE IF NOT EXISTS search_runs (
    -- compute_stress_run_id() of this search: a content hash of scenario, target, delta, method and the provenance (which holds the
    -- seed and the budget). A search that IS the stored result has the stored result's id and is found by joining on it.
    search_run_id     text        PRIMARY KEY,
    scenario_id       text        NOT NULL,
    -- The scene it was searched against, copied at search time (NULL = not recorded, the same carve-out as scenario_scores).
    scene_fingerprint text,
    sdc_idx           integer,
    target_idx        integer     NOT NULL,
    -- Which model's units the four components of `delta` carry (the same two words search_provenance records). NULL = not recorded.
    delta_parameterization text   CHECK (delta_parameterization IS NULL OR delta_parameterization IN ('bicycle', 'linear')),
    kind              text        NOT NULL CHECK (kind IN ('challenger', 'seed', 'budget', 'trace_rerun')),
    challenger_rank   integer     CHECK (challenger_rank IS NULL OR challenger_rank >= 1),   -- 1 = nearest, for kind = 'challenger'
    seed              integer,
    popsize           integer,
    maxiter           integer,
    tol               double precision,
    outcome           text        NOT NULL,                                                  -- the same outcome words as scenario_scores
    min_perturbation  double precision,
    collision_timestep integer,
    delta             double precision[],
    n_iter            integer,
    n_eval            integer,
    search_provenance jsonb,
    -- For kind = 'trace_rerun': the stored result this search re-executes, and how far its delta is from the stored one.
    reruns_stress_run_id       text,
    max_abs_delta_difference   double precision,
    pipeline_run_id   uuid        REFERENCES pipeline_runs (pipeline_run_id) ON DELETE SET NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    CHECK ((kind = 'trace_rerun') = (reruns_stress_run_id IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS search_runs_scenario ON search_runs (scenario_id, kind);
CREATE INDEX IF NOT EXISTS search_runs_pipeline_run ON search_runs (pipeline_run_id);

-- ── #4: the best weighted norm after each generation of one search ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS search_traces (
    search_run_id        text             NOT NULL REFERENCES search_runs (search_run_id) ON DELETE CASCADE,
    generation           integer          NOT NULL CHECK (generation >= 0),
    n_eval_cumulative    integer          NOT NULL CHECK (n_eval_cumulative >= 0),
    best_weighted_norm   double precision,            -- NULL until a candidate exists that collides
    PRIMARY KEY (search_run_id, generation)
);

-- ── #15: a two-dimensional slice of the search space around a stored result ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sensitivity_slices (
    slice_id          bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scenario_id       text        NOT NULL,
    -- The stored result the slice is drawn around. A reader shows it only while this still equals scenario_scores.stress_run_id.
    stress_run_id     text        NOT NULL,
    scene_fingerprint text,
    sdc_idx           integer,
    axis_x            smallint    NOT NULL CHECK (axis_x BETWEEN 0 AND 3),
    axis_y            smallint    NOT NULL CHECK (axis_y BETWEEN 0 AND 3),
    -- Required: the axes are numbered components of a delta, and a number means nothing until it is known which model's four
    -- components they are. The API names them from this column and from nothing else.
    delta_parameterization text   NOT NULL CHECK (delta_parameterization IN ('bicycle', 'linear')),
    x_values          double precision[] NOT NULL,
    y_values          double precision[] NOT NULL,
    weighted_norm     double precision[] NOT NULL,       -- row-major, y outer, x inner
    collides          boolean[]          NOT NULL,       -- same layout
    held_delta        double precision[] NOT NULL,       -- all four components of the stored delta; the two not on an axis are held at these
    pipeline_run_id   uuid        REFERENCES pipeline_runs (pipeline_run_id) ON DELETE SET NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    CHECK (axis_x <> axis_y),
    CHECK (cardinality(x_values) BETWEEN 2 AND 101 AND cardinality(y_values) BETWEEN 2 AND 101),
    CHECK (cardinality(weighted_norm) = cardinality(x_values) * cardinality(y_values)),
    CHECK (cardinality(collides) = cardinality(weighted_norm)),
    CHECK (cardinality(held_delta) = 4),
    UNIQUE (scenario_id, stress_run_id, axis_x, axis_y)
);

-- ── #18: one row per shard that has been loaded ──────────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS shards (
    shard_name      text        PRIMARY KEY,                 -- the value scenario_scores.shard already holds
    sha256          text        CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'),
    scenarios_in_shard integer  CHECK (scenarios_in_shard IS NULL OR scenarios_in_shard >= 0),
    pipeline_run_id uuid        REFERENCES pipeline_runs (pipeline_run_id) ON DELETE SET NULL,
    recorded_at     timestamptz NOT NULL DEFAULT now()
);

-- ── what the read-only API may read ──────────────────────────────────────────────────────────────────────────────────────────────
-- DEPLOY.md step 2 gives the read-only role SELECT on every table the owner creates later (ALTER DEFAULT PRIVILEGES). That is right
-- for the tables above, which hold results and no secrets, and wrong for pipeline_runs, whose command_line, host_label and notes
-- must not be public. So: the raw table is closed to the role, and a view with the public columns is opened.
CREATE OR REPLACE VIEW pipeline_runs_public AS
SELECT pipeline_run_id, kind, status, started_at, finished_at, git_revision, git_dirty, python_version,
       package_versions, shard_name, shard_sha256, parameters, counts
  FROM pipeline_runs;

DO $grants$
BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'av_api_ro') THEN
        REVOKE ALL ON TABLE pipeline_runs FROM av_api_ro;
        GRANT SELECT ON pipeline_runs_public TO av_api_ro;
        -- The default privilege covers the other new tables only when this file runs as the role that set it; say it outright.
        GRANT SELECT ON pipeline_run_results, search_runs, search_traces, sensitivity_slices, shards TO av_api_ro;
    END IF;
END
$grants$;

COMMIT;
