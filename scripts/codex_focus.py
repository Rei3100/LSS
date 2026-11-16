"""Codex batch admission against the user's brief, not a self-selected target.

Every Codex queue has a bounded focus and a separate review bound to the queue
and original brief bytes. The reviewer reads the commands and requested outcome,
states why this work is needed now and what decision its result changes. Support
returns to the same primary outcome; support does not become a new primary goal.
Review meaning and reviewer independence require judgement; this code validates
their bindings, not their truth. Native Codex hooks are not required by this gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

FIELDS = ('objective', 'user_result', 'kind', 'done_when', 'return_to', 'brief_path')


def canonical_sha(queue: dict) -> str:
    value = dict(queue)
    value.pop('review_path', None)
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain an object')
    return value


def _path(base: Path, value: str) -> Path:
    return (base / value).resolve()


def _validated(path: Path, queue: dict) -> tuple[dict, dict, str]:
    if not isinstance(queue.get('steps'), list) or not queue['steps']:
        raise ValueError('reviewed queue requires non-empty steps')
    focus = queue.get('focus')
    if not isinstance(focus, dict) or any(not isinstance(focus.get(k), str) or not focus[k].strip() for k in FIELDS):
        raise ValueError('focus requires non-empty ' + ', '.join(FIELDS))
    if focus['kind'] not in ('primary', 'support'):
        raise ValueError('focus.kind must be primary or support')
    brief_path = _path(path.parent, focus['brief_path'])
    brief_bytes = brief_path.read_bytes()
    brief = json.loads(brief_bytes.decode('utf-8-sig'))
    if (not isinstance(brief, dict) or not isinstance(brief.get('user_request'), str)
            or not brief['user_request'].strip() or not isinstance(brief.get('requested_outcomes'), list)
            or not brief['requested_outcomes'] or any(not isinstance(x, str) or not x.strip()
                                                     for x in brief['requested_outcomes'])):
        raise ValueError('brief requires user_request and non-empty requested_outcomes strings')
    review_path = queue.get('review_path')
    if not isinstance(review_path, str) or not review_path.strip():
        raise ValueError('review_path is required')
    review = _read(_path(path.parent, review_path))
    reviewer = review.get('reviewer')
    if (review.get('queue_sha256') != canonical_sha(queue)
            or review.get('brief_sha256') != hashlib.sha256(brief_bytes).hexdigest()
            or review.get('decision') != 'ALLOW' or not isinstance(reviewer, str)
            or not reviewer.strip() or reviewer.strip().lower() in ('root', '/root', 'self', 'main', 'parent')
            or review.get('requested_outcome') not in brief['requested_outcomes']
            or any(not isinstance(review.get(k), str) or not review[k].strip()
                   for k in ('why_required_now', 'what_changes_next'))):
        raise ValueError('independent ALLOW review must bind queue, brief, and requested outcome')
    return focus, review, hashlib.sha256(brief_bytes).hexdigest()


def admit(path: Path, queue: dict) -> None:
    if not (os.environ.get('CODEX_SESSION_ID') or os.environ.get('CODEX_THREAD_ID')):
        return
    try:
        focus, review, brief_sha = _validated(path, queue)
        if focus['kind'] == 'support':
            primary_ref = queue.get('primary_queue')
            if not isinstance(primary_ref, str) or not primary_ref.strip():
                raise ValueError('support requires primary_queue')
            primary_path = _path(path.parent, primary_ref)
            primary_focus, primary_review, primary_sha = _validated(primary_path, _read(primary_path))
            if primary_focus['kind'] != 'primary':
                raise ValueError('support cannot reference support')
            if (focus['objective'] != primary_focus['objective'] or brief_sha != primary_sha
                    or review['requested_outcome'] != primary_review['requested_outcome']):
                raise ValueError('support must match primary objective, brief and requested outcome')
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f'Codex focus admission: {exc}') from exc
