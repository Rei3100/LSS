"""Guarded Aider 0.86.2 prompt/parser proposal trial through a local-only client.

These controls are trusted program logic, not an OS sandbox. Returned proposals
are unverified data; the parent owns subsequent closed-AST behavioral evaluation.
The runner never applies or executes model-produced code.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable


import ast
from pathlib import Path
from urllib.parse import urlsplit


class GuardError(ValueError):
    pass


def local_base_url(value: str) -> str:
    try:
        url = urlsplit(value)
        if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username is not None
                or url.password is not None or url.path != '/v1' or url.query or url.fragment
                or not url.port or not 1 <= url.port <= 65535):
            raise GuardError('only http://127.0.0.1:PORT/v1 is permitted')
    except ValueError as exc:
        raise GuardError('only http://127.0.0.1:PORT/v1 is permitted') from exc
    return value


class CasePaths:
    """Only two exact existing regular files in one non-symlink case directory."""
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        if root.is_symlink() or not self.root.is_dir():
            raise GuardError('invalid case root')
        self.allowed = {self.root / 'candidate.py', self.root / 'contract.txt'}
        for path in self.allowed:
            if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
                raise GuardError('missing or linked case file')

    def resolve(self, path) -> Path:
        raw = Path(path)
        if '..' in raw.parts:
            raise GuardError('parent path refused')
        path = raw if raw.is_absolute() else self.root / raw
        # Require both the lexical path and resolved target to be allowlisted.
        if path not in self.allowed or path.is_symlink() or path.resolve(strict=True) != path:
            raise GuardError('path outside case allowlist')
        if not path.is_file():
            raise GuardError('case file is not regular')
        return path


def guarded_types(editblock_coder, input_output, parse_edits):
    """Wrap the pinned classes; no stock apply_updates path is used."""
    class GuardedIO(input_output):
        def read_text(self, filename, *args, **kwargs):
            return super().read_text(str(self.case_paths.resolve(filename)), *args, **kwargs)

        def read_image(self, filename):
            raise GuardError('images unavailable')

        def write_text(self, *args, **kwargs):
            raise GuardError('writes unavailable')

        def confirm_ask(self, *args, **kwargs):
            return False

        def offer_url(self, *args, **kwargs):
            return False

    class GuardedCoder(editblock_coder):
        def abs_root_path(self, path):
            return str(self.case_paths.resolve(path))

        def check_for_file_mentions(self, content):
            # No model- or user-named file may be added automatically.
            return None

        def run_shell_commands(self):
            if self.shell_commands:
                raise GuardError('shell output refused')
            return ''

        def get_edits(self):
            # Empty valid_fnames disables upstream fuzzy remapping of unknown paths.
            edits = list(parse_edits(self.partial_response_content, self.fence, []))
            if any(edit[0] is None for edit in edits):
                raise GuardError('shell output refused')
            return edits

        def apply_updates(self):
            # Do not call the upstream preview: it reads names before permission.
            edits = self.get_edits()
            if self.shell_commands or len(edits) != 1:
                raise GuardError('exactly one edit and no shell commands required')
            for edit in edits:
                if len(edit) != 3 or self.case_paths.resolve(edit[0]) != self.case_paths.root / 'candidate.py':
                    raise GuardError('only candidate edits permitted')
            current = self.io.read_text(self.case_paths.root / 'candidate.py')
            for edit in edits:
                _, before, after = edit
                if not before or current.count(before) != 1:
                    raise GuardError('edit must uniquely match candidate')
                # Exact in-memory replacement avoids Aider do_replace touching paths.
                current = current.replace(before, after, 1)
            tree = ast.parse(current)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != '_normalise':
                raise GuardError('expected one _normalise function')
            self.proposal = current
            return set()

        def apply_edits(self, edits, *args, **kwargs):
            raise GuardError('direct apply unavailable')

        def apply_edits_dry_run(self, edits):
            raise GuardError('upstream preview unavailable')

    return GuardedCoder, GuardedIO

# Audited files in the installed 0.86.2 wheel. A mismatch refuses the run.
AUDITED_SHA256 = {
    'coders/base_coder.py': 'a621097e9d258629f2d28f2f083fb772ba956017b9f7ea7535a40544b751d081',
    'coders/editblock_coder.py': '244df08f5ccd19928bcdac1901ae591a7a786d12a6f70d31400baaefb5a136a3',
    'io.py': '7f62a79c8202f2a3793410bac4c983e417fb886513b90faf0e0579d5bb7d047b',
    'models.py': '593cff096ba8e19ea4147f6e1e79e215f1d6c9c3978477aea7040b9dc2b5188b',
}


def single_edit_response(text):
    """Accept one explicit candidate block, with no prose or other path mentions."""
    import re
    if not isinstance(text, str) or len(text.encode('utf-8')) > 100_000:
        raise GuardError('invalid response size')
    pattern = r'candidate\.py\s*\n```(?:python)?\s*\n<<<<<<< SEARCH\s*\n[\s\S]*?\n=======\s*\n[\s\S]*?\n>>>>>>> REPLACE\s*\n```\s*'
    if not re.fullmatch(pattern, text):
        raise GuardError('expected exactly one candidate edit block')
    return text


def package_identity():
    import hashlib
    import importlib.metadata
    import aider
    if importlib.metadata.version('aider-chat') != '0.86.2':
        raise GuardError('unexpected aider version')
    folder = Path(aider.__file__).resolve().parent
    expected_folder = (Path(__file__).resolve().parents[1] / '.cache/local-ai/aider-env/Lib/site-packages/aider').resolve()
    if folder != expected_folder:
        raise GuardError('aider imported from unexpected environment')
    digests = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in AUDITED_SHA256}
    if digests != AUDITED_SHA256:
        raise GuardError('audited aider source mismatch')
    dependencies = {name: importlib.metadata.version(name) for name in ('litellm', 'openai', 'httpx')}
    return {'version': '0.86.2', 'source_sha256': digests, 'dependencies': dependencies}


def snapshot_case(paths):
    import hashlib
    entries = list(paths.root.iterdir())
    if set(entries) != paths.allowed:
        raise GuardError('unexpected case files')
    return {p.name: hashlib.sha256(paths.resolve(p).read_bytes()).hexdigest() for p in sorted(entries)}


def build_prompt(paths):
    """Construct only pinned Aider prompts and the two explicit case files."""
    from aider import models
    from aider.coders.editblock_coder import EditBlockCoder, find_original_update_blocks
    from aider.io import InputOutput
    class LocalModel(models.Model):
        def get_model_info(self, name):
            return {'max_input_tokens': 4096, 'max_output_tokens': 1024, 'litellm_provider': 'openai'}
        def validate_environment(self):
            return {'missing_keys': [], 'keys_in_environment': True}
    model = LocalModel('openai/local', weak_model=False, editor_model=False)
    model.weak_model = model
    model.editor_model = model
    coder_cls, io_cls = guarded_types(EditBlockCoder, InputOutput, find_original_update_blocks)
    coder_cls.case_paths = paths
    io_cls.case_paths = paths
    io = io_cls(pretty=False, fancy_input=False, input_history_file=None, chat_history_file=None,
                llm_history_file=None, dry_run=True)
    coder = coder_cls(model, io, fnames=[str(paths.root / 'candidate.py')],
        read_only_fnames=[str(paths.root / 'contract.txt')], use_git=False, dry_run=True,
        auto_commits=False, dirty_commits=False, auto_lint=False, auto_test=False,
        map_tokens=0, suggest_shell_commands=False, detect_urls=False, stream=False,
        analytics=type('NoAnalytics', (), {'event': lambda *a, **kw: None})())
    coder.cur_messages = [{'role': 'user', 'content': 'Propose the requested change as one SEARCH/REPLACE block for candidate.py only. Return no other text.'}]
    return coder, coder.format_messages().all_messages()


def request_local(base_url, messages, timeout=900, capture=None):
    import http.client
    import json
    url = urlsplit(local_base_url(base_url))
    body = json.dumps({'model': 'local', 'messages': messages, 'max_tokens': 1024,
                       'temperature': 0, 'stream': False,
                       'chat_template_kwargs': {'enable_thinking': False}}).encode('utf-8')
    if capture is not None:
        import hashlib
        capture['request_sha256'] = hashlib.sha256(body).hexdigest()
    if len(body) > 250_000 or not 1 <= timeout <= 1200:
        raise GuardError('request size or timeout invalid')
    conn = http.client.HTTPConnection('127.0.0.1', url.port, timeout=timeout)
    try:
        conn.request('POST', '/v1/chat/completions', body=body,
                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer local-dummy'})
        response = conn.getresponse()
        data = response.read(250_001)
        if capture is not None:
            import hashlib
            capture.update(http_status=response.status, raw_response_utf8=data.decode('utf-8', errors='replace'),
                           raw_response_sha256=hashlib.sha256(data).hexdigest(), response_bytes=len(data),
                           response_truncated=len(data) > 250_000)
        if response.status != 200 or len(data) > 250_000:
            raise GuardError('local response rejected')
        result = json.loads(data)
        choice = result['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise GuardError('local response incomplete')
        return single_edit_response(choice['message']['content'])
    finally:
        conn.close()


def run_proposal(case_root, base_url, capture=None):
    """Build a proposal without applying it; the caller owns later validation."""
    from dev_hot_paths import require_hot
    require_hot(case_root, 'Aider owned case')
    require_hot(Path(__file__).resolve().parents[1] / '.cache/local-ai/aider-env', 'Aider runtime')
    identity = package_identity()
    root = Path(case_root)
    parent = (Path(__file__).resolve().parents[1] / '.cache/local-ai').resolve()
    if root.is_symlink() or root.parent.resolve() != parent or not root.name.startswith('aider-case-'):
        raise GuardError('case must be an owned direct aider-case-* directory')
    paths = CasePaths(root)
    before = snapshot_case(paths)
    import contextlib
    import sys
    with contextlib.redirect_stdout(sys.stderr):
        coder, messages = build_prompt(paths)
    coder.partial_response_content = request_local(base_url, messages, capture=capture) if capture is not None else request_local(base_url, messages)
    coder.shell_commands = []
    coder.apply_updates()
    if snapshot_case(paths) != before:
        raise GuardError('case changed during proposal')
    return {'status': 'proposal_only', 'unverified': True, 'identity': identity, 'case_sha256': before,
            'proposal': coder.proposal, 'model': 'local', 'base_url': local_base_url(base_url),
            'scope': 'Aider prompt and edit parser; exact in-memory replacement; not applied or evaluated'}


def main():
    import argparse
    import json
    from dev_hot_paths import require_hot
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, type=Path)
    p.add_argument('--base-url', required=True)
    p.add_argument('--transcript', type=Path)
    args = p.parse_args()
    capture = {} if args.transcript else None
    if args.transcript:
        root = Path(__file__).resolve().parents[1]
        require_evidence_writable(args.transcript)
        target = args.transcript.resolve()
        allowed = ((dev_evidence_root(root) / 'release-evidence').resolve(),
                   (root / '.cache/local-ai').resolve())
        if not any(target.is_relative_to(base) for base in allowed):
            p.error('transcript must be beneath release-evidence or .cache/local-ai')
        require_hot(target, 'Aider transcript')
        if target.exists() or not target.parent.is_dir():
            p.error('transcript must be a new file in an existing directory')
    try:
        result = run_proposal(args.case, args.base_url, capture)
        if capture is not None: capture['status'] = 'proposal_only'
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        if capture is not None:
            capture.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if capture is not None:
            with target.open('x', encoding='utf-8') as stream:
                json.dump(capture, stream, ensure_ascii=False, indent=2)
                stream.write('\n')


if __name__ == '__main__':
    main()
