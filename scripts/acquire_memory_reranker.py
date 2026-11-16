"""Acquire one pinned public reranker on SSD; never run a model or scan archives.

Launch only via quiet_dev_checks --reranker-acquire, which supplies the CPU cap,
single processor and no-nudge storage reservation. Existing partial bytes resume;
an existing complete file must match its full SHA before reuse. No registry or
Product mutation occurs. The public Hub metadata was checked on 2026-10-27.
"""
try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

from pathlib import Path
import time

from acquire_audio_probe import acquire, save
from dev_hot_paths import require_hot

ROOT = Path(__file__).resolve().parents[1]
DEST = Path('C:/LSS-Work/local-ai-models/qwen3-reranker-0.6b')
OUT = dev_evidence_root(ROOT) / 'release-evidence/local-ai-memory-rerank-r1'
REPO = 'ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF'
REV = 'a02f48bb4f057028298c21fa033da2b30d7742d5'
FILE = 'qwen3-reranker-0.6b-q8_0.gguf'
SIZE = 639153184
SHA = '22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48'


def main():
    for path in (DEST, OUT):
        require_hot(path, 'quiet reranker acquisition')
    DEST.mkdir(parents=True, exist_ok=True)
    require_evidence_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'running', 'repository': REPO, 'revision': REV,
               'file': FILE, 'size_bytes': SIZE, 'sha256': SHA,
               'destination': str(DEST), 'license': 'apache-2.0',
               'rate_mib_s': 0.25, 'gpu_used': False, 'inference': False,
               'archive_read': False, 'registry_changed': False,
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    save(OUT / 'acquisition.json', receipt)
    try:
        receipt['transfer'] = acquire(FILE, SIZE, SHA, rate=0.25 * 1024**2,
            destination=DEST, repository=REPO, revision=REV,
            progress=lambda row: save(OUT / 'acquisition-progress.json', row))
        receipt['status'] = 'verified'
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__)
        # Exception text may include temporary signed download URLs.
    finally:
        save(OUT / 'acquisition.json', receipt)
    return 0 if receipt['status'] == 'verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
