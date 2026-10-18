"""
build_geometry_fixtures.py — builds small scene-geometry fixtures from a LOCAL shard into a directory that git ignores.

The fixtures hold Waymo-derived geometry (agent states over time), so this is behind the one switch for Waymo-derived outputs
(waymo_derived_outputs.json, read by src/waymo_derived.py): it refuses to run unless the switch is on, and `build_fixtures` takes the switch's value as
an argument so a test can exercise it with the switch on without editing the file. The licence reply is outstanding (email of 2026-10-09 to
open-dataset@waymo.com); until it is settled the switch is off, no fixture exists, and none may be committed.

Where it writes: tests/geometry_fixtures_local/ (FIXTURE_DIR), which .gitignore lists. It refuses a directory that git does not ignore, so a fixture
cannot land somewhere it could be added by accident. Nothing here is ever to be committed; tests/test_waymo_derived_files_not_tracked.py fails if anything is.

What a fixture is: for one scenario, the SDC and the nearest few challengers (the ranking perturbation_space.rank_challengers), every frame:
    {"scenario_id", "sdc_idx", "agents": [{"agent_idx", "agent_type", "states": [[...7 numbers per frame...]], "valid": [bool per frame]}]}
plus MANIFEST.json (the sha256 of each file). Invalid frames are written as they are in the shard (the shard zero-fills them); `valid` says which are real.

Usage (switch on, a shard on this machine):
    python tools/build_geometry_fixtures.py --shard /path/to/local.tfrecord --max-scenarios 5
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

FIXTURE_DIR = os.path.join(PROJECT, 'tests', 'geometry_fixtures_local')
MANIFEST = 'MANIFEST.json'
DEFAULT_CHALLENGERS = 3
DEFAULT_MAX_SCENARIOS = 5


class FixturesRefused(RuntimeError):
    """The fixtures were not built, and nothing was written. The message says why."""


def is_git_ignored(path: str) -> bool:
    """Whether git ignores `path` (which need not exist). False for a path outside the repository."""
    rel = os.path.relpath(os.path.abspath(path), PROJECT)
    if rel.startswith('..'):
        return False
    probe = os.path.join(rel, '.fixture-probe')
    return subprocess.run(['git', 'check-ignore', '-q', '--no-index', probe], cwd=PROJECT).returncode == 0


def iter_shard_scenes(shard_path: str, limit: int):
    """(scenario_id, states, validity, types, sdc_idx) of the first `limit` scenarios of a local shard. Needs the shard parser (the Waymo protobufs)."""
    from src.data.loader import ShardLoader
    from src.data.parser import ScenarioParser
    for i, raw in enumerate(ShardLoader(shard_path)):
        if i >= limit:
            return
        p = ScenarioParser(raw)
        yield p.get_scenario_id(), p.get_agent_states(), p.get_agent_validity(), p.get_agent_types(), p.get_sdc_index()


def fixture_of(scenario_id, states, validity, types, sdc_idx, challengers: int) -> dict:
    from src.optimization.perturbation_space import rank_challengers
    keep = [int(sdc_idx)] + [int(j) for j in rank_challengers(states, validity, sdc_idx)[:challengers]]
    return {
        'scenario_id': str(scenario_id),
        'sdc_idx': int(sdc_idx),
        'agents': [{'agent_idx': j, 'agent_type': int(types[j]),
                    'states': [[float(v) for v in row] for row in states[j]],
                    'valid': [bool(v) for v in validity[j]]} for j in keep],
    }


def build_fixtures(shard_path, out_dir, *, enabled, max_scenarios=DEFAULT_MAX_SCENARIOS, challengers=DEFAULT_CHALLENGERS, scenes=None) -> list:
    """
    Writes one JSON file per scenario and the manifest into `out_dir`; returns the paths. Refuses, having written nothing, unless `enabled` is exactly
    True, unless `out_dir` is a directory git ignores, and unless the counts are sensible. `scenes` replaces the shard reader (a test passes it).
    """
    if enabled is not True:
        raise FixturesRefused('Waymo-derived outputs are switched off (waymo_derived_outputs.json); no fixture was built.')
    if not is_git_ignored(out_dir):
        raise FixturesRefused(f'{out_dir} is not a directory git ignores; fixtures are only written where they cannot be committed by accident.')
    if isinstance(max_scenarios, bool) or not isinstance(max_scenarios, int) or not 1 <= max_scenarios <= 50:
        raise FixturesRefused(f'max_scenarios is from 1 to 50, not {max_scenarios!r}')
    if isinstance(challengers, bool) or not isinstance(challengers, int) or not 1 <= challengers <= 10:
        raise FixturesRefused(f'challengers is from 1 to 10, not {challengers!r}')
    source = scenes if scenes is not None else iter_shard_scenes(shard_path, max_scenarios)
    built = []
    for n, (scenario_id, states, validity, types, sdc_idx) in enumerate(source):
        if n >= max_scenarios:
            break
        built.append((f'{scenario_id}.json', fixture_of(scenario_id, states, validity, types, sdc_idx, challengers)))
    if not built:
        raise FixturesRefused('the shard gave no scenario; nothing was written')
    os.makedirs(out_dir, exist_ok=True)
    paths, digests = [], {}
    for name, fixture in built:
        data = (json.dumps(fixture, separators=(',', ':')) + '\n').encode('utf-8')
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
        digests[name] = hashlib.sha256(data).hexdigest()
    manifest = os.path.join(out_dir, MANIFEST)
    with open(manifest, 'w', encoding='utf-8') as fh:
        json.dump({'note': 'Waymo-derived geometry. Never commit. Built by tools/build_geometry_fixtures.py.', 'files': digests}, fh, indent=1, sort_keys=True)
        fh.write('\n')
    return [*paths, manifest]


def main(argv=None) -> int:
    from src.waymo_derived import outputs_enabled
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--shard', required=True, help='a .tfrecord shard on this machine')
    ap.add_argument('--out', default=FIXTURE_DIR, help='defaults to the ignored directory tests/geometry_fixtures_local')
    ap.add_argument('--max-scenarios', type=int, default=DEFAULT_MAX_SCENARIOS)
    ap.add_argument('--challengers', type=int, default=DEFAULT_CHALLENGERS)
    args = ap.parse_args(argv)
    try:
        paths = build_fixtures(args.shard, args.out, enabled=outputs_enabled(), max_scenarios=args.max_scenarios, challengers=args.challengers)
    except FixturesRefused as e:
        print(f'refused: {e}', file=sys.stderr)
        return 2
    print(f'wrote {len(paths)} files to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
