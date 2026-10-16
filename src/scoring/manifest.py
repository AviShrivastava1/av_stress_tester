"""
manifest.py — the offline writer of run manifests (the tables of deploy/rerun_class_additive.sql).

One row of `pipeline_runs` per run of the offline pipeline: what ran (the git revision and whether the working tree was dirty, the
versions of Python and of the libraries that matter), on what (the shard's name and sha256), with what (the parameters that vary a
run: seeds, budgets), how it went (status, times, counts), and which stored results it wrote (`pipeline_run_results`).

Rules, each pinned by tests/test_manifest.py:

  * NULL means "not recorded", never "none". A repository that is not there (Colab often is not a checkout), a shard that is not
    hashed, a package that is not installed: the field is left NULL or the key left out. Nothing is guessed.
  * It writes only the manifest tables. It never touches scenario_scores, scenario_agents or perturbed_paths, never calls
    update_stress_results, and never changes what a result's identity is.
  * It does not create the tables. They come from deploy/rerun_class_additive.sql, run by hand; a missing table is a clear error here.
  * The command line is stored (so a run can be repeated) but with credentials removed first, and it is never served by the API.
  * A run is finished once: `finish_run` only moves a run that is still running.

Usage, from a notebook or a script, as the database owner (the same connection the pipeline already writes with):

    run = begin_run(conn, 'stress', parameters={'de_seed': 1, 'de_popsize': 15}, shard_path=path_to_the_tfrecord)
    ...                                   # the pass
    link_result(conn, run, scenario_id, stress_run_id)    # for each result the pass stored
    finish_run(conn, run, counts={'scored': 100, 'searched': 18})
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import uuid

KINDS = ('score', 'stress', 'export', 'rerun')

#: The libraries whose versions a manifest records; the same list the API serves (src/api/manifest.py PACKAGES, less Python itself).
PACKAGES = ('numpy', 'scipy', 'shapely', 'torch', 'psycopg2')

_MISSING = ('pipeline_runs', 'pipeline_run_results')


class ManifestTablesMissing(RuntimeError):
    """The manifest tables are not in this database: run deploy/rerun_class_additive.sql (by hand, on a copy first)."""


def _git(args, cwd):
    try:
        done = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def git_state(path=None) -> dict:
    """
    {'git_revision': the 40-hex commit or None, 'git_dirty': True/False or None}, for the repository containing `path` (default:
    this repository). Both None when there is no repository or git cannot be run. Dirty means `git status --porcelain` lists
    anything, untracked files included: such a tree is not reproduced by the revision alone.
    """
    cwd = path or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    revision = (_git(['rev-parse', 'HEAD'], cwd) or '').strip()
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        return {'git_revision': None, 'git_dirty': None}
    status = _git(['status', '--porcelain'], cwd)
    return {'git_revision': revision, 'git_dirty': None if status is None else status.strip() != ''}


def package_versions() -> dict:
    """{'python': ..., 'numpy': ...} for the installed packages of PACKAGES; one that is not installed is left out."""
    versions = {'python': platform.python_version()}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def sha256_file(path, chunk=1 << 20) -> str:
    """The sha256 of a file, read in chunks (a shard is hundreds of megabytes)."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


_URL_PASSWORD = re.compile(r'(://[^/\s:@]*:)[^@\s/]+(@)')
_SECRET_FLAG = re.compile(r'(?i)(--?(?:password|passwd|pwd|token|secret|api[-_]?key)(?:=|\s+))(\S+)')
_SECRET_ENV = re.compile(r'(?i)\b((?:PG)?PASSWORD|[A-Z_]*TOKEN|[A-Z_]*SECRET|[A-Z_]*API_KEY)=(\S+)')


def redact_command_line(text: str) -> str:
    """A command line with the credentials it carries removed: a password in a URL, a value after a secret-looking flag or assignment."""
    text = _URL_PASSWORD.sub(r'\1***\2', text)
    text = _SECRET_FLAG.sub(r'\1***', text)
    return _SECRET_ENV.sub(r'\1=***', text)


def _require_tables(cur):
    for table in _MISSING:
        cur.execute('SELECT to_regclass(%s) IS NOT NULL', (table,))
        if not cur.fetchone()[0]:
            raise ManifestTablesMissing(f'{table} is not in this database; run deploy/rerun_class_additive.sql by hand (on a copy first).')


def begin_run(conn, kind, *, parameters=None, shard_path=None, shard_name=None, shard_sha256=None, command_line=None,
              host_label=None, notes=None, repo_path=None, hash_shard=True) -> str:
    """
    Record that a run has started; returns its id. Captures the git revision and dirty flag, the package versions and, from
    `shard_path`, the shard's name and sha256 (`hash_shard=False` skips the hash: it is left NULL, not guessed). `command_line`
    defaults to this process's argv; credentials are removed before it is stored. Commits.
    """
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}, not {kind!r}')
    if shard_path is not None:
        shard_name = shard_name or os.path.basename(shard_path)
        if shard_sha256 is None and hash_shard:
            shard_sha256 = sha256_file(shard_path)
    if command_line is None:
        command_line = ' '.join(sys.argv)
    state = git_state(repo_path)
    run_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        _require_tables(cur)
        cur.execute(
            """INSERT INTO pipeline_runs (pipeline_run_id, kind, git_revision, git_dirty, python_version, package_versions,
                                          shard_name, shard_sha256, parameters, command_line, host_label, notes)
               VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb, %s, %s, %s)""",
            (run_id, kind, state['git_revision'], state['git_dirty'], platform.python_version(), json.dumps(package_versions()),
             shard_name, shard_sha256, json.dumps(parameters) if parameters is not None else None,
             redact_command_line(command_line), host_label, notes))
    conn.commit()
    return run_id


def link_result(conn, run_id, scenario_id, stress_run_id) -> None:
    """Record that this run stored the result (scenario_id, stress_run_id). Idempotent. Commits."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO pipeline_run_results (pipeline_run_id, scenario_id, stress_run_id) VALUES (%s, %s, %s)
               ON CONFLICT (pipeline_run_id, scenario_id) DO UPDATE SET stress_run_id = EXCLUDED.stress_run_id""",
            (run_id, scenario_id, stress_run_id))
    conn.commit()


def finish_run(conn, run_id, *, status='finished', counts=None) -> bool:
    """
    Move a run that is still running to `finished` (or `failed`, or `abandoned`) with its counts and the finish time. Returns False,
    and changes nothing, for a run that is not running or does not exist: a finished run is not finished twice. Commits.
    """
    if status not in ('finished', 'failed', 'abandoned'):
        raise ValueError(f'status must be finished, failed or abandoned, not {status!r}')
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE pipeline_runs SET status = %s, finished_at = now(), counts = %s::jsonb
                WHERE pipeline_run_id = %s AND status = 'running'""",
            (status, json.dumps(counts) if counts is not None else None, run_id))
        changed = cur.rowcount == 1
    conn.commit()
    return changed
