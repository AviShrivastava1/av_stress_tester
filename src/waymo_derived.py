"""
waymo_derived.py — the one switch for everything that exports or commits Waymo-derived imagery or geometry.

The Waymo licence reply is outstanding (email of 2026-10-09 to open-dataset@waymo.com). Until it is settled, every exporter (an image, an SVG, a GIF
or a video of a scene; a fixture of scene geometry) is built, tested and left OFF. The switch is the file waymo_derived_outputs.json at the repository
root, {"enabled": false}; this module and frontend/src/waymoDerivedOutputs.ts are the only two readers of it, and there is no second constant.

The tooling takes the value as an argument (tools/build_geometry_fixtures.py `build_fixtures(..., enabled=...)`), so tests exercise it with the switch
on without editing the file. A file that is missing, is not exactly {"enabled": <true or false>}, or says anything else is an error, never "on".
"""

import json
import os

SWITCH_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'waymo_derived_outputs.json')


class SwitchError(ValueError):
    """The switch file is not exactly {"enabled": true} or {"enabled": false}."""


def read_switch(path: str = SWITCH_FILE) -> bool:
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise SwitchError(f'{path}: cannot be read as JSON ({e})') from None
    if not isinstance(data, dict) or set(data) != {'enabled'} or not isinstance(data['enabled'], bool):
        raise SwitchError(f'{path}: must be exactly {{"enabled": true}} or {{"enabled": false}}')
    return data['enabled']


def outputs_enabled(path: str = SWITCH_FILE) -> bool:
    """Whether Waymo-derived outputs are switched on, from the file."""
    return read_switch(path)
