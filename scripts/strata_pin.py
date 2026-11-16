"""Pinned Strata component identity, read from `.state/strata-pin.json` (defaults = the first install).

The wrapper (`strata_dev.py`) takes the release tag, engine archive and SHA256 values from here instead of constants, so
`strata_update.py` can move the pin one signed step and move it back. Pure functions; no network, no subprocess.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root


ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = dev_state_root(ROOT) / 'strata-pin.json'

# The original install (Terminal 0.2.7, Control 0.1.53, engine 0.1.52): the pin when no file exists.
DEFAULT_PIN = {
    'schema': 1,
    'release_tag': 'v0.2.7',
    'terminal': '0.2.7',
    'control': '0.1.53',
    'engine': '0.1.52',
    'engine_archive': 'strata-engine-0.1.52-windows-x86_64-cuda-cuda13.tar.gz',
    'engine_sha256': 'bfa68c6b2e92f9c273caaf937d9ae164918e6de4cca2dc1f1d79d32470f0915a',
    'install_sha256': '6e0c8b2ffa033a9a40850bf16d017365de07a30be5a21f540d23c0c75d94bce9',
    'feed_sequence': 116,
    'bench_baseline': None,
    'previous': None,
}
REQUIRED = ('release_tag', 'control', 'engine', 'engine_archive', 'engine_sha256', 'install_sha256')


def validate(pin: dict) -> dict:
    for key in REQUIRED:
        if not isinstance(pin.get(key), str) or not pin[key]:
            raise ValueError(f'Strata pin needs a non-empty {key}')
    if len(pin['engine_sha256']) != 64 or len(pin['install_sha256']) != 64:
        raise ValueError('Strata pin SHA256 values must be 64 hex digits')
    int(pin['engine_sha256'], 16)
    int(pin['install_sha256'], 16)
    return pin


def load(path: Path | None = None) -> dict:
    path = Path(os.environ.get('LSS_STRATA_PIN', path or PIN_PATH))
    if not path.exists():
        return copy.deepcopy(DEFAULT_PIN)
    return validate(json.loads(path.read_text(encoding='utf-8-sig')))


def save(pin: dict, path: Path | None = None) -> None:
    path = Path(os.environ.get('LSS_STRATA_PIN', path or PIN_PATH))
    validate(pin)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(pin, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def advance(pin: dict, new: dict) -> dict:
    """The pin after one accepted step: `new` becomes current and the old pin (without its own history) is `previous`."""
    old = {key: value for key, value in pin.items() if key != 'previous'}
    result = {**copy.deepcopy(DEFAULT_PIN), **copy.deepcopy(new), 'previous': old}
    return validate(result)


def revert(pin: dict) -> dict:
    """The pin before the last step; refuses when there is none. The reverted-from version is kept as `rejected`."""
    previous = pin.get('previous')
    if not previous:
        raise ValueError('Strata pin has no previous version to return to')
    rejected = sorted(set(previous.get('rejected', [])) | {pin['engine']})
    return validate({**copy.deepcopy(DEFAULT_PIN), **copy.deepcopy(previous), 'previous': None, 'rejected': rejected})


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split('.'))
