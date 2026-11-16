"""Policy-owned output paths. Creates empty build endpoints only; existing data
is left for the OS layout mover. No deletion, relocation or global TEMP setting.
"""
import argparse
import hashlib
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root


import storage_tiers as tiers


def build_directory(repo, policy):
    identity = hashlib.sha256(str(Path(repo).absolute()).casefold().encode()).hexdigest()[:16]
    base = Path(policy['build_root']) if 'build_root' in policy else dev_state_root(Path(repo)) / 'build-data'
    return base / identity


def prepare_build(repo, policy):
    repo = Path(repo).absolute()
    target = build_directory(repo, policy)
    rows = []
    for name in ('build', 'dist'):
        old, new = repo / name, target / name
        if old.exists() or tiers.is_link(old):
            rows.append({'path': str(old), 'resolved': str(old.resolve()), 'created': False})
            continue
        new.mkdir(parents=True, exist_ok=True)
        tiers.make_junction(old, new)
        rows.append({'path': str(old), 'resolved': str(new), 'created': True})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-build'])
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(prepare_build(args.repo, tiers.load_policy())))


if __name__ == '__main__':
    main()
