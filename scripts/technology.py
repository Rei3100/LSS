"""Read-only LSS technology lookup: curated roles + pinned original experiment sections.

    python scripts/technology.py search "SAM Audio"
    python scripts/technology.py show sam-audio
    python scripts/technology.py assets sam-audio [--hash]
    python scripts/technology.py check [--freshness]
    python scripts/technology.py coverage

No model imports, downloads, credential access, inference or recursive drive scans.
The JSON ledger owns current scoped decisions; original Git records retain their
headings, tables, surrounding qualifications and exact identities. A search miss
means no indexed record, never not installed or not tried. No mutation commands.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

if __package__:
    from .git_objects import history_checkout, read_blobs
else:
    from git_objects import history_checkout, read_blobs

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/research/technology_ledger.json"


def git(*args: str, root: Path = ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise ValueError(f"Git source unavailable: {' '.join(args)}")
    return result.stdout


def normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold())


def read_ledger(path: Path = LEDGER) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported technology ledger version")
    for collection in ("sources", "technologies"):
        ids = [row["id"] for row in data[collection]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {collection} IDs")
    source_ids = {s["id"] for s in data["sources"]}
    for source in data["sources"]:
        if source.get("commit"):
            if not re.fullmatch(r"[a-f0-9]{40}", source["commit"]):
                raise ValueError(f"Source must pin a full commit: {source['id']}")
        elif not re.fullmatch(r"[a-f0-9]{64}", source.get("sha256", "")):
            raise ValueError(f"Local snapshot must pin a SHA-256: {source['id']}")
        path_value = source["path"]
        if (Path(path_value).is_absolute() or ".." in Path(path_value).parts
                or not path_value.endswith((".md", ".tsv"))):
            raise ValueError(f"Unsafe/non-document source: {source['id']}")
    for tech in data["technologies"]:
        for field in ("id", "name", "category", "implementation", "adoption", "configuration"):
            if not tech.get(field):
                raise ValueError(f"Missing {field}: {tech.get('id')}")
        if not tech.get("roles"):
            raise ValueError(f"Missing role records: {tech['id']}")
        for role in tech["roles"]:
            for field in ("use", "input", "assembly", "population", "observation", "decision", "reopen"):
                if not role.get(field):
                    raise ValueError(f"Missing role {field}: {tech['id']}")
            if not role.get("sources") or not set(role["sources"]) <= source_ids:
                raise ValueError(f"Unknown/missing role source: {tech['id']}")
    return data


def sections(text: str) -> list[dict]:
    """Retain complete section text and exact source lines, including table context."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if re.match(r"^#{1,6} ", line)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    records = []
    for start, end in zip(starts, starts[1:] + [len(lines)]):
        body = "\n".join(lines[start:end]).strip()
        if body:
            records.append({"heading": lines[start].lstrip("# ") if lines[start] else "Preamble",
                            "line": start + 1, "end_line": end, "text": body})
    return records


def source_records(source: dict, *, root: Path = ROOT, blobs: dict | None = None) -> list[dict]:
    if source.get("commit"):
        request = f"{source['commit']}:{source['path']}"
        raw = (read_blobs([request], root=root) if blobs is None else blobs).get(request)
        if raw is None:
            raise ValueError(f"Git source unavailable: {request}")
        text = raw.decode('utf-8-sig')
    else:
        path = root / source["path"]
        history = history_checkout()
        if not path.is_file() and history is not None and (history / source["path"]).is_file():
            path = history / source["path"]  # the pinned sha256 below still decides identity
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"Snapshot identity mismatch: {source['id']}")
        text = raw.decode("utf-8-sig")
    if source["path"].endswith(".tsv"):
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        # Preserve historical columns without interpreting an Instrumental state
        # as a family-wide availability or adoption decision.
        return [{"heading": row.get("model_id", f"row-{i}"), "line": i,
                 "end_line": i, "text": json.dumps(row, ensure_ascii=False)}
                for i, row in enumerate(reader, 2)]
    return sections(text)


def search(data: dict, query: str, *, root: Path = ROOT) -> list[dict]:
    needle = normalize(query)
    if not needle:
        raise ValueError("Give a nonempty technology, role, condition or assembly")
    results = []
    for tech in data["technologies"]:
        if needle in normalize(json.dumps(tech, ensure_ascii=False)):
            results.append({"kind": "curated", "id": tech["id"], "record": tech})
    blobs = read_blobs([f"{s['commit']}:{s['path']}" for s in data['sources'] if s.get('commit')], root=root)
    for source in data["sources"]:
        try:
            for record in source_records(source, root=root, blobs=blobs):
                if needle in normalize(record["text"]):
                    results.append({"kind": "original", "source": source, **record})
        except (ValueError, OSError) as exc:
            results.append({"kind": "unavailable_source", "source": source["id"], "error": str(exc)})
    # Put the named experiment section before incidental mentions in long
    # contracts; all originals remain available and their authority is unchanged.
    return sorted(results, key=lambda r: 0 if r['kind'] == 'curated' else
                  1 if r['kind'] == 'unavailable_source' else
                  2 if needle in normalize(r.get('heading', '')) else 3)


def static_spec(path: Path, symbol: str) -> dict:
    """Read only literal asset identity fields; never import provider code."""
    wanted = {"cache_name", "file_name", "size_bytes", "sha256"}
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets):
            if not isinstance(node.value, ast.Dict):
                break
            result = {}
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value in wanted:
                    result[key.value] = ast.literal_eval(value)
            if set(result) == wanted:
                return result
    raise ValueError(f"No complete literal asset identity: {path}:{symbol}")


def observe_asset(path: Path, spec: dict, *, hash_file: bool = False) -> dict:
    row = {"path": str(path), "expected_bytes": spec.get("size_bytes"),
           "expected_sha256": spec.get("sha256"), "status": "MISSING_AT_THIS_PATH"}
    try:
        row["resolved_path"] = str(path.resolve())
        row["bytes"] = path.stat().st_size
        row["status"] = ("PRESENT_IDENTITY_UNCHECKED" if spec.get("size_bytes") is None else
                         "PRESENT_SIZE_MATCH_HASH_UNCHECKED" if row["bytes"] == spec["size_bytes"] else "SIZE_MISMATCH")
        if hash_file and row["status"] != "SIZE_MISMATCH":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 << 20), b""):
                    digest.update(chunk)
            row["sha256"] = digest.hexdigest()
            row["status"] = ("HASH_OBSERVED_NO_EXPECTATION" if not spec.get("sha256") else
                             "HASH_MATCH" if row["sha256"] == spec["sha256"] else "HASH_MISMATCH")
    except FileNotFoundError:
        pass
    except OSError as exc:
        row["status"] = "OBSERVATION_UNAVAILABLE"
        row["error_type"] = type(exc).__name__
    return row


def assets(data: dict, tech_id: str, *, hash_file: bool = False, root: Path = ROOT) -> list[dict]:
    tech = next((t for t in data["technologies"] if t["id"] == tech_id), None)
    if tech is None:
        raise ValueError(f"Unknown technology ID: {tech_id}; use search first")
    roots = [("development", root / ".models"), ("installed", Path(data["installed_model_root"]))]
    if os.environ.get("LSS_MODEL_DIR"):
        roots.insert(0, ("environment_override", Path(os.environ["LSS_MODEL_DIR"])))
    result = []
    for asset in tech.get("assets", []):
        if asset.get("path"):
            path = Path(asset["path"])
            result.append({"store": "research", "model_root": str(path.parent), "asset": asset.get("kind", "research_asset"),
                           "reuse_instruction": tech.get("reuse", {}).get("development_use", "Use the recorded execution loader and exact paths; verify identity before inference."),
                           **observe_asset(path, asset, hash_file=hash_file)})
            continue
        spec = static_spec(root / asset["module"], asset["symbol"])
        for label, model_root in roots:
            result.append({"store": label, "model_root": str(model_root), "asset": f"{asset['module']}:{asset['symbol']}",
                           **observe_asset(model_root / spec["cache_name"] / spec["file_name"],
                                           spec, hash_file=hash_file)})
    return result


def reusable_locations(observations: list[dict]) -> list[dict]:
    """Suggest existing complete declared file sets, without claiming runtime readiness."""
    stores = {}
    for row in observations:
        stores.setdefault((row['store'], row['model_root']), []).append(row)
    return [{'store': store, 'model_root': model_root,
             'identity': 'HASH_MATCH' if all(r['status'] == 'HASH_MATCH' for r in rows)
                         else 'PRESENT_FILES_IDENTITY_UNCHECKED' if any(r['status'] in {'PRESENT_IDENTITY_UNCHECKED', 'HASH_OBSERVED_NO_EXPECTATION'} for r in rows)
                         else 'SIZE_MATCH_HASH_UNCHECKED',
             'next_action': rows[0].get('reuse_instruction', 'Reuse this existing store via LSS_MODEL_DIR; no re-download or login is indicated by a missing file in another store.')}
            for (store, model_root), rows in stores.items()
            if rows and all(r['status'] in {'HASH_MATCH', 'PRESENT_SIZE_MATCH_HASH_UNCHECKED', 'PRESENT_IDENTITY_UNCHECKED', 'HASH_OBSERVED_NO_EXPECTATION'} for r in rows)]


def check(data: dict, *, freshness: bool = False, root: Path = ROOT) -> dict:
    errors, stale, coverage = [], [], []
    requests = [f"{s['commit']}:{s['path']}" for s in data['sources'] if s.get('commit')]
    if freshness:
        requests += [f"{s['follow_ref']}:{s['path']}" for s in data['sources'] if s.get('follow_ref')]
    requests += [f"{entry['commit']}:{entry['path']}" for t in data['technologies'] for entry in t.get('execution', [])]
    blobs = read_blobs(requests, root=root)
    for source in data["sources"]:
        try:
            records = source_records(source, root=root, blobs=blobs)
            coverage.append({"source": source["id"], "records": len(records),
                             "commit": source.get("commit"), "sha256": source.get("sha256"), "path": source["path"]})
            if freshness and source.get("follow_ref"):
                old = blobs[f"{source['commit']}:{source['path']}"]
                current = blobs[f"{source['follow_ref']}:{source['path']}"]
                if current is None:
                    stale.append({"source": source["id"], "status": "FOLLOW_REF_UNAVAILABLE_PIN_RETAINED"})
                elif old != current:
                    stale.append({"source": source["id"], "status": "NEW_SOURCE_CONTENT", "follow_ref": source["follow_ref"]})
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    for technology in data['technologies']:
        for locator in technology.get('execution', []):
            request = f"{locator['commit']}:{locator['path']}"
            if blobs[request] is None:
                errors.append(f'Git execution source unavailable: {request}')
    map_path = root / 'docs/research/TECHNOLOGY.md'
    if map_path.is_file() and map_path.read_text(encoding='utf-8') != render_map(data):
        errors.append('Generated TECHNOLOGY.md differs from the ledger; regenerate with technology.py map')
    return {"status": "INVALID" if errors else "STALE" if stale else "VALID",
            "technologies": len(data["technologies"]), "sources": coverage,
            "freshness_checked": freshness, "stale": stale, "errors": errors,
            "boundary": data["coverage_boundary"]}


def render_map(data: dict) -> str:
    lines = [
        '# LSS Technology Map', '',
        'Generated from `technology_ledger.json` by `python scripts/technology.py map`. '
        'The ledger owns current scoped records; original Evidence and live artifacts remain authoritative. '
        'Do not maintain another hand-written technology scoreboard here.', '',
        '## One axis: useful wanted audio', '',
        'The goal is a useful standalone wanted asset from a finished 2ch mix, autonomously and locally. '
        'Singing/choir remains wanted music. Single-occurrence/no-reference processing remains first-class. '
        'Existing released material is an identity/redundancy question. Unknown material survives. '
        'Recorded extraction and generated/restored material stay distinct. '
        'The user chooses wanted sound, not checkpoints or A/B winners. '
        'Intent owner: [PRODUCT_INTENT](../product/PRODUCT_INTENT.md); '
        'processing/order contract: [SOURCE_TO_ASSET](../engine/SOURCE_TO_ASSET.md); '
        'donor/identity rules: [RECOVERY_AND_REFERENCES](../engine/RECOVERY_AND_REFERENCES.md).', '',
        '```mermaid', 'flowchart TD',
        '  S[Finished stereo source + wanted sound] --> O[Observe regions, instrumentation and known occurrences]',
        '  O --> P[Choose role, input, order and representation]',
        '  P --> N[Original or no-op]', '  P --> D[Specialist separation / query / local cleanup]',
        '  P --> C[Components + discarded-side recovery]',
        '  P --> R[Aligned occurrence / reference evidence]',
        '  N --> A[Assemble compatible recorded regions and components]',
        '  D --> A', '  C --> A', '  R --> A',
        '  A --> Q[Wanted preservation, residue, continuity, stereo and identity]',
        '  Q --> B[CurrentBest revision / HELD / UNKNOWN]',
        '  B --> T[One musical identity, title, tags and rollback]',
        '  P --> G[Explicitly labelled Restoration]', '  G --> Q', '```', '',
        'This is the decision space, not a claim that every branch is implemented. '
        'The records below distinguish shipped/source/research and exact tested roles.', '',
        '## Find before choosing or reinstalling', '',
        'Run with the current checkout Python (`.venv/Scripts/python.exe`). '
        'These commands do not download, load a model, run inference or touch credentials.', '',
        '```text', 'python scripts/technology.py search "SAM Audio"',
        'python scripts/technology.py show sam-audio',
        'python scripts/technology.py assets sam-audio',
        'python scripts/technology.py search "dialogue-first"',
        'python scripts/technology.py search "sample rate"',
        'python scripts/technology.py check --freshness',
        'python scripts/technology.py coverage', '```', '',
        '`assets` stats only the declared files in the development/installed model roots '
        'and optional `LSS_MODEL_DIR`. `--hash` explicitly reads those weights to compare identity; '
        'even HASH_MATCH is not an inference/readiness test. A missing file at one root '
        'does not establish authentication failure or global absence. '
        'Complete existing asset sets are returned as reuse locations: point the development job\'s '
        '`LSS_MODEL_DIR` at that store instead of downloading again. When physical placement must change, '
        'use a verified relocation and update all owning references together '
        '([DATA_LIFECYCLE](../dev/DATA_LIFECYCLE.md)); preserve the installed runtime\'s access. '
        '`search` normalizes spaces/hyphens/underscores; `--json` before the subcommand '
        'returns every matched original section with its full context. Ordinary output '
        'shows a bounded excerpt plus the exact source location. No indexed match means UNKNOWN.', '',
        '## Current role map', '',
        f"Reviewed {data['reviewed_at']}; {len(data['technologies'])} technology/method records. "
        'Adoption is recorded scope, not a live installed-runtime check. '
        'Use `show <id>` for input, ordered assembly, population, observed result, reopening condition and sources.', '',
        '| Family | Lookup ID / technology | Implementation and adoption boundary |', '|---|---|---|',
    ]
    for tech in data['technologies']:
        lines.append(f"| {tech['category']} | `{tech['id']}` — {tech['name']} | {tech['adoption']} |")
    lines += ['', '## A pattern is more than a model name', '',
              'A role record contains: **use × checkpoint/config × input/parent × channel/representation '
              '× stage order × conditioning × assembly × population**. '
              'A load/dependency failure is not a quality result; a whole-cue failure does not close a dirty-stem role. '
              'A mean, maximum, subtraction, mask projection and generated repair are different assemblies. '
              'Direct music selection, candidate retrieval and whole-span EXACT are different permissions.', '',
              '**Known contradictions resolved by scope:** SAM base-tv is implemented and its cached Product '
              'weights are a different question from official gated repository access or earlier small-model '
              'temporal trials. HyperACE standard-loader failure is superseded by successful own-module execution. '
              'The legacy Instrumental TSV is not the status of every possible use. '
              'Guide speed/pitch, residual boost, ensembles, USS and splicing already have executed cells; '
              'they are technique families with remaining patterns, not blanket untried proposals. '
              'Uniform instrument chains do not constitute the practitioner\'s per-cue method.', '',
              '## Coverage and recoverable originals', '', data['coverage_boundary'], '',
              'Every heading and table in each declared Markdown source, and every row/column of the legacy '
              'model TSV, participates in lookup. Numeric rows are not automatically converted to adoption '
              'decisions. Full-section context retains baselines, denominators, limitations and corrections. '
              'Follow-ref freshness compares the pinned file blob with the current declared branch, '
              'not commit dates or branch-head equality. Archived/missing follow refs retain the pinned source '
              'but are reported for review. It does not discover new unregistered files.', '',
              '| Source ID | Immutable locator | Scope |', '|---|---|---|']
    for source in data['sources']:
        locator = f"{source['commit'][:12]}:{source['path']}" if source.get('commit') else f"{source['path']} (SHA-256 {source['sha256'][:12]}…)"
        lines.append(f"| `{source['id']}` | `{locator}` | {source['scope']} |")
    lines += ['', '## Keep this one record current', '',
              'Before declaring a technology unavailable/untried, selecting a route or repeating a failed cell, '
              'search this entry, inspect its exact prior Evidence and check only the local assets the decision needs. '
              'Do not reinstall because a historical official URL was gated. '
              'If sources disagree, inspect implementation/artifact/measurement rather than voting or preferring the newest prose.', '',
              'After a completed trial or changed installation/adoption, update the same technology role in '
              '`technology_ledger.json`: exact checkpoint/config, input, ordered assembly, population/controls, '
              'benefit/harm, decision, reopening trigger and durable source. '
              'Add a new role when a new pattern differs; retain prior outcomes through source provenance and Git. '
              'Extend `sources` for a new Evidence family; advance an existing pin only after reviewing its delta. '
              'Keep original arrays/audio/plans in their owning Evidence root; do not copy them into the ledger.', '',
              'Regenerate this view with `python scripts/technology.py map > docs/research/TECHNOLOGY.md`, '
              'then run `python scripts/technology.py check --freshness`. '
              'For a failed/stopped experiment record that status; a saved running marker is never proof of a live job. '
              'No automatic experiment runner, global task registry or runtime Product dependency is introduced.', '']
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    commands = parser.add_subparsers(dest="command", required=True)
    lookup = commands.add_parser("search")
    lookup.add_argument("query")
    lookup.add_argument("--limit", type=int, default=12)
    show = commands.add_parser("show")
    show.add_argument("id")
    local = commands.add_parser("assets")
    local.add_argument("id")
    local.add_argument("--hash", action="store_true", help="explicitly read/hash declared weights, never load a model")
    validate = commands.add_parser("check")
    validate.add_argument("--freshness", action="store_true")
    commands.add_parser("coverage")
    commands.add_parser("map", help="print the generated human map; no file mutation")
    args = parser.parse_args(argv)
    try:
        data = read_ledger()
        if args.command == "map":
            print(render_map(data), end="")
            return 0
        if args.command == "search":
            results = search(data, args.query)
            output = {"query": args.query, "matches": len(results), "results": results if args.json else results[:max(1, args.limit)],
                      "boundary": "Search miss is UNKNOWN, never not-installed/not-tried. Original records are not current decisions."}
            if args.json:
                print(json.dumps(output, ensure_ascii=False, indent=2))
            else:
                print(f"{len(results)} matches (showing at most {max(1, args.limit)}); use --json for complete text")
                for row in output["results"]:
                    if row["kind"] == "curated":
                        tech = row["record"]
                        print(f"\n[CURRENT SCOPED RECORD] {tech['id']} | {tech['name']} | {tech['adoption']}")
                        print(f"  implementation: {tech['implementation']}; config: {tech['configuration']}")
                        for role in tech["roles"]:
                            print(f"  {role['use']} | {role['decision']} | {role['assembly']}")
                            print(f"  {role['population']}: {role['observation']}")
                        print(f"  full record: python scripts/technology.py show {tech['id']}")
                    elif row["kind"] == "original":
                        source = row["source"]
                        label = 'SUPPLIED REFERENCE' if source.get('scope', '').startswith('SUPPLIED_REFERENCE') else 'ORIGINAL RECORD'
                        print(f"\n[{label}] {source['id']} | {row['heading']}")
                        location = f"git show {source['commit']}:{source['path']}" if source.get("commit") else source['path']
                        print(f"  {location} (lines {row['line']}-{row['end_line']})")
                        # Keep the matching line plus nearby table/narrative context.
                        lines = row["text"].splitlines()
                        hit = next((i for i, line in enumerate(lines) if normalize(args.query) in normalize(line)), 0)
                        print("\n".join(lines[max(0, hit - 1):hit + 4]))
                    else:
                        print(json.dumps(row, ensure_ascii=False))
                print(output["boundary"])
            return 0 if results else 3
        if args.command == "show":
            output = next((t for t in data["technologies"] if t["id"] == args.id), None)
            if output is None:
                raise ValueError(f"Unknown ID: {args.id}; use search")
            refs = {ref for role in output["roles"] for ref in role["sources"]}
            output = {**output, "source_locations": [s for s in data["sources"] if s["id"] in refs]}
        elif args.command == "assets":
            observations = assets(data, args.id, hash_file=args.hash)
            current = next(t for t in data['technologies'] if t['id'] == args.id)
            output = {"technology": args.id, "observations": observations,
                      "inventory_status": "DECLARED_FILES_OBSERVED" if observations else "UNKNOWN_NOT_CATALOGUED",
                      "execution_locators": current.get('execution', []),
                      "reuse_locations": reusable_locations(observations),
                      "boundary": "File identity only; not an inference/readiness test. Missing at one path is not an authentication failure."}
        else:
            output = check(data, freshness=getattr(args, "freshness", False))
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1 if output.get("status") == "INVALID" else 2 if output.get("status") == "STALE" else 0
    except (ValueError, KeyError, OSError, SyntaxError) as exc:
        print(f"technology: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
