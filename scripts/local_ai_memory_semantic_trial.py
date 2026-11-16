"""Read-only paired evaluation of existing devmem hybrid on 22 frozen support cases.

No index refresh, source scan, Product change, or implicit embedding fallback.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time

import devmem
import local_llm

ROOT = Path(__file__).resolve().parents[1]
INDEX = dev_evidence_root(ROOT) / 'release-evidence/local-ai-practical-20261026/index/devmem.sqlite'
BENCH = dev_evidence_root(ROOT) / 'release-evidence/local-ai-search-r2/devmem-source-diversification.json'
OUTPUT_ROOT = dev_evidence_root(ROOT) / 'release-evidence/local-ai-memory-semantic-r1'
INDEX_SHA = '7aaf019bff969c27699bd87bcc3d34306af80d1e93064eb510326c3146d715ec'
BENCH_SHA = '3f6a2b2bc1bd0df04b548df3c5481a30c938a85930b16237179cf51c52e2f5a4'
MODEL = 'qwen3-embedding-0.6b'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def exact_control(query):
    return bool(re.fullmatch(r'[0-9a-fA-F]{7,40}', query) or
                re.fullmatch(r'[^\s/\\]+(?:[/\\][^\s/\\]+)+', query))


def support_hit(hits, source, lines):
    start, end = lines
    return any(hit['path'] == source and hit['lines'][0] <= end and hit['lines'][1] >= start for hit in hits)


def hit_identity(hits):
    return [(hit['path'], hit['lines'], hit['commit']) for hit in hits]


def lexical(con, query):
    ids, note = devmem.search(con, query, k=8, mode='auto', embedder=None)
    return ids[:8], devmem.hits(con, ids, query, 8, None)


def preflight(con, document):
    cases = document.get('cases', [])
    if len(cases) != 22 or len({row['id'] for row in cases}) != 22:
        raise ValueError('frozen 22-case benchmark missing or duplicated')
    counts = con.execute('SELECT count(*), count(e.hash) FROM chunk c LEFT JOIN emb e ON c.hash=e.hash').fetchone()
    invalid = con.execute('SELECT count(*) FROM emb WHERE length(vec) != 2048').fetchone()[0]
    if counts != (4379, 4379) or invalid:
        raise ValueError('incomplete or invalid existing embeddings')
    baseline = []
    for case in cases:
        ids, hits = lexical(con, case['query'])
        present = support_hit(hits, case['support_source'], case['support_lines'])
        if present != case['baseline_support_hit'] or hit_identity(hits) != hit_identity(case['baseline']['top8']):
            raise ValueError('frozen baseline did not reproduce: ' + case['id'])
        baseline.append((ids, hits))
    return baseline


def candidate(con, query, embedder):
    if exact_control(query):
        ids, hits = lexical(con, query)
        return ids, hits, {'exact_control_bypass': True}
    ids, note = devmem.search(con, query, k=8, mode='hybrid', embedder=embedder)
    if note.get('dense') is not True: raise RuntimeError('query embedding unavailable; no silent fallback')
    ids = ids[:8]
    return ids, devmem.hits(con, ids, query, 8, None), note


def evaluate(con, document, baseline, embedder):
    cases = []
    for case, (base_ids, base_hits) in zip(document['cases'], baseline):
        query = case['query']
        query_start = time.perf_counter()
        ids, hits, note = candidate(con, query, embedder)
        query_ms = (time.perf_counter() - query_start) * 1000
        cases.append({'id': case['id'], 'set': case['set'], 'query': query,
            'support_source': case['support_source'], 'support_lines': case['support_lines'],
            'baseline_ids': base_ids, 'candidate_ids': ids,
            'baseline_support': support_hit(base_hits, case['support_source'], case['support_lines']),
            'candidate_support': support_hit(hits, case['support_source'], case['support_lines']),
            'baseline_target_source': any(h['path'] == case['historical_expected_source'] for h in base_hits),
            'candidate_target_source': any(h['path'] == case['historical_expected_source'] for h in hits),
            'candidate_query_ms': round(query_ms, 3), 'candidate_hits': hits, 'note': note})
    controls = []
    for control in document['exact_identifier_controls']:
        q = control['query']
        if not exact_control(q): raise ValueError('invalid exact control')
        ids, _ = lexical(con, q)
        candidate_ids, candidate_hits, note = candidate(con, q, embedder)
        controls.append({'query': q, 'baseline_ids': ids, 'candidate_ids': candidate_ids,
                         'same_ordered_chunk_ids': ids == candidate_ids, 'hits': candidate_hits, 'note': note})
    return {'cases': cases, 'exact_controls': controls,
            'counts': {'n': len(cases), 'baseline_support_top8': sum(c['baseline_support'] for c in cases),
                'hybrid_support_top8': sum(c['candidate_support'] for c in cases),
                'hybrid_target_source_top8': sum(c['candidate_target_source'] for c in cases),
                'lost_support': [c['id'] for c in cases if c['baseline_support'] and not c['candidate_support']],
                'gained_support': [c['id'] for c in cases if c['candidate_support'] and not c['baseline_support']]}}


def run(output, *, serving=local_llm.serving):
    from dev_hot_paths import require_hot
    require_evidence_writable(output)
    output = output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output.exists():
        raise ValueError('output must be new and under semantic trial evidence')
    model = local_llm.get_model(MODEL)
    for label, path in [('index', INDEX), ('benchmark', BENCH), ('output', output), ('model', model.path),
                        ('runtime', local_llm.llama_dir()), ('environment', sys.prefix)]: require_hot(path, label)
    if sha(INDEX) != INDEX_SHA: raise ValueError('frozen index SHA mismatch')
    if sha(BENCH) != BENCH_SHA: raise ValueError('frozen benchmark SHA mismatch')
    document = json.loads(BENCH.read_text(encoding='utf-8-sig'))
    conn = sqlite3.connect(INDEX.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    started = time.monotonic()
    try:
        baseline = preflight(conn, document)  # Before model server or GPU lease.
        model_sha = sha(model.path)
        if not model.sha256 or model_sha.lower() != model.sha256.lower(): raise ValueError('embedding model SHA mismatch')
        identity = {'index_sha256': INDEX_SHA, 'benchmark_sha256': sha(BENCH),
            'devmem_sha256': sha(Path(devmem.__file__)), 'trial_sha256': sha(Path(__file__)),
            'model_sha256': model_sha, 'model': model.name, 'model_ctx': model.ctx,
            'model_args': list(model.args), 'query_instruction': devmem.QUERY_INSTRUCT, 'runtime_build': local_llm.BUILD,
            'runtime_sha256': sha(local_llm.llama_dir() / 'llama-server.exe'),
            'runtime_dlls_sha256': {p.name: sha(p) for p in sorted(local_llm.llama_dir().glob('*.dll'))},
            'embedding_provenance': 'prior devmem-hybrid-probe identifies qwen3-embedding-0.6b; original run does not independently pin model SHA',
            'historical_embedding_receipt_sha256': sha(dev_evidence_root(ROOT) / 'release-evidence/local-ai-practical-20261026/devmem-hybrid-probe.json')}
        with serving(MODEL, purpose='frozen devmem paired semantic evaluation') as base:
            result = evaluate(conn, document, baseline, devmem.Embedder(base))
        if sha(INDEX) != INDEX_SHA: raise RuntimeError('frozen index changed during evaluation')
        result.update(status='complete', identity=identity, wall_s=round(time.monotonic()-started, 3),
            scope='paired frozen support retrieval experiment; no default retrieval or Product change')
    finally:
        conn.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = run(args.output)
    print(json.dumps({'status': result['status'], 'counts': result['counts']}, ensure_ascii=False))


if __name__ == '__main__': main()
