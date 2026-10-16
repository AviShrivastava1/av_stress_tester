"""
manifest.py — which parts of a run manifest the API serves, and how.

`pipeline_runs` (deploy/rerun_class_additive.sql) holds one row per run of the offline pipeline. The API reads it only through the
view `pipeline_runs_public`, which leaves out the command line, the host label and the notes. Three of the view's columns are JSON
(`package_versions`, `parameters`, `counts`) and are served here through an ALLOW-LIST of keys and types, never whole: a key added to
one of them tomorrow, by a later pipeline or by hand, is not public by being there. A value of the wrong type is served as null (not
repaired, not an error): "not recorded" is an honest answer and a guess is not.

Stdlib only, so the deployed API (which installs neither numpy nor scipy) can import it.
"""

import math
import re

#: Python and the five packages the manifest records. A version is a short string of digits, letters and . + -
PACKAGES = ['python', 'numpy', 'scipy', 'shapely', 'torch', 'psycopg2']
_VERSION = re.compile(r'[0-9A-Za-z.+\-]{1,64}')

#: The only words `kind` and `status` may carry (the migration's own CHECKs; the API does not rely on them being there).
KINDS = ['score', 'stress', 'export', 'rerun']
STATUSES = ['running', 'finished', 'failed', 'abandoned']

_SHARD_NAME = re.compile(r'[A-Za-z0-9._\-]{1,200}')
_REVISION = re.compile(r'[0-9a-f]{40}')
_SHA256 = re.compile(r'[0-9a-f]{64}')

#: The integer counts a run may report.
COUNTS = ['scenarios', 'scored', 'searched', 'refused', 'errors', 'stored_results', 'collisions_found', 'no_collision_found']

#: The parameters a run may report: integers, one number, and a list of integers (at most MAX_SEEDS entries).
PARAMETER_INTS = ['de_seed', 'de_popsize', 'de_maxiter', 'challengers_k']
PARAMETER_NUMBERS = ['de_tol']
PARAMETER_INT_LISTS = ['seeds']
MAX_SEEDS = 100


def valid_shard_name(v):
    """A file name: letters, digits, dot, underscore and hyphen, 1 to 200 characters. A path, a space or anything longer is not one."""
    return isinstance(v, str) and _SHARD_NAME.fullmatch(v) is not None


def valid_revision(v):
    return isinstance(v, str) and _REVISION.fullmatch(v) is not None


def valid_sha256(v):
    return isinstance(v, str) and _SHA256.fullmatch(v) is not None


def valid_kind(v):
    return isinstance(v, str) and v in KINDS


def valid_status(v):
    return isinstance(v, str) and v in STATUSES


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_version(v):
    return isinstance(v, str) and _VERSION.fullmatch(v) is not None


def _is_int_list(v):
    return isinstance(v, list) and len(v) <= MAX_SEEDS and all(_is_int(x) for x in v)


def select_package_versions(blob) -> dict | None:
    """The six served versions, each as recorded or null. None when there is no object at all."""
    if not isinstance(blob, dict):
        return None
    return {name: (blob[name] if name in blob and _is_version(blob[name]) else None) for name in PACKAGES}


def select_counts(blob) -> dict | None:
    if not isinstance(blob, dict):
        return None
    return {name: (blob[name] if name in blob and _is_int(blob[name]) and blob[name] >= 0 else None) for name in COUNTS}


def select_parameters(blob) -> dict | None:
    if not isinstance(blob, dict):
        return None
    served = {}
    for name in PARAMETER_INTS:
        served[name] = blob[name] if name in blob and _is_int(blob[name]) else None
    for name in PARAMETER_NUMBERS:
        served[name] = blob[name] if name in blob and _is_number(blob[name]) else None
    for name in PARAMETER_INT_LISTS:
        served[name] = blob[name] if name in blob and _is_int_list(blob[name]) else None
    return served
