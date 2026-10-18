"""
refresh_collision_results.py — rebuilds the results-only golden record of the collision results the public API serves.

The record (tests/data/public_collision_results_<date>.json) holds, for every scenario whose stored result is a collision, only what the
public API serves as a RESULT: scenario id, the challenger agent's index (target_idx), the four components of the perturbation and which
model's units they carry, the weighted norm (min_perturbation), the collision frame and the stored result's run id. It holds no scene
geometry, no trajectory and no coordinate of any agent: GET /perturbed also serves the tracks, and this tool reads only its target_idx and
drops the rest the moment it is parsed.

HOW IT IS REFRESHED (by hand, never by a test):

    # the real API app on a local scratch database restored from the dump, every connection read-only, on 127.0.0.1
    python tools/refresh_collision_results.py --base-url http://127.0.0.1:8123 --out tests/data/public_collision_results_2026-10-18.json

A base URL that is not on this machine (localhost, 127.0.0.1, ::1) is refused unless --allow-remote is given; nothing in the test suite calls
this tool with a network, and the suite never calls the hosted API. The tests (tests/test_collision_results_golden.py) read the committed file only.
"""

import argparse
import ipaddress
import json
import sys
import urllib.parse
import urllib.request

FIELDS = ('scenario_id', 'target_idx', 'delta', 'delta_parameterization', 'min_perturbation', 'collision_timestep', 'stress_run_id')
NOTE = ('Results only, as served by the public API: no scene geometry, no trajectory, no coordinate of any agent. '
        'Refreshed by tools/refresh_collision_results.py (see its docstring); never by a test.')
PAGE = 100


def is_local(base_url: str) -> bool:
    host = urllib.parse.urlparse(base_url).hostname or ''
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def http_get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


def collision_ids(get_json, base_url: str) -> list:
    """Every scenario whose stored result is a collision, in the order the API lists them, then sorted by id."""
    ids, cursor = [], None
    while True:
        query = f'limit={PAGE}&stress_tested_only=true' + (f'&cursor={urllib.parse.quote(cursor)}' if cursor else '')
        page = get_json(f'{base_url}/scenarios?{query}')
        ids += [r['scenario_id'] for r in page['items'] if r.get('stress_outcome') == 'collision_found']
        cursor = page.get('next_cursor')
        if not cursor:
            return sorted(ids)


def row_for(get_json, base_url: str, scenario_id: str) -> dict:
    detail = get_json(f'{base_url}/scenarios/{scenario_id}')
    provenance = get_json(f'{base_url}/scenarios/{scenario_id}/provenance')
    target_idx = get_json(f'{base_url}/scenarios/{scenario_id}/perturbed')['target_idx']       # the tracks in that response are dropped here
    return {
        'scenario_id': scenario_id,
        'target_idx': target_idx,
        'delta': detail['delta'],
        'delta_parameterization': (provenance.get('search') or {}).get('delta_parameterization'),
        'min_perturbation': detail['min_perturbation'],
        'collision_timestep': detail['collision_timestep'],
        'stress_run_id': provenance['stress_run_id'],
    }


def build_record(get_json, base_url: str, source: str) -> dict:
    base_url = base_url.rstrip('/')
    return {'note': NOTE, 'source': source, 'rows': [row_for(get_json, base_url, i) for i in collision_ids(get_json, base_url)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--source', default='the real API app on a local scratch database restored from the dump, read-only connections, GET requests only')
    ap.add_argument('--allow-remote', action='store_true', help='allow a base URL that is not on this machine')
    args = ap.parse_args(argv)
    if not is_local(args.base_url) and not args.allow_remote:
        print(f'refused: {args.base_url} is not on this machine; pass --allow-remote to read a remote API', file=sys.stderr)
        return 2
    record = build_record(http_get_json, args.base_url, args.source)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1)
        fh.write('\n')
    print(f'wrote {len(record["rows"])} rows to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
