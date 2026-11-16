"""Run a queue of development steps serially, in the background, with receipts (replaces hand-written chain scripts).

    python scripts/batch.py start queue.json     # launch detached (hidden_run mechanism), prints the pid
    python scripts/batch.py run queue.json       # execute the steps in this process (what `start` launches)
    python scripts/batch.py status queue.json --json  # process identity, queue hash and saved steps

One queue path is one attempt: an OS lock prevents concurrent execution and a
prior receipt/DONE is retained, not overwritten. Explicit retries use a new queue
path/marker after inspecting the old receipt; arbitrary commands are not assumed
idempotent. `after_queues: ["upstream.json"]` observes bound upstream receipts and
owner identity rather than waiting forever for a crashed producer's marker.
`wait_timeout_seconds` optionally bounds queue/legacy-marker waits. Old `after`
markers remain supported but cannot prove process liveness.

queue.json:
    {"done": "optional/marker/path", "steps": [
      {"name": "infer", "cmd": ["python", "x.py"], "cwd": "optional dir", "gpu": true,
       "after": ["marker files that must exist first"], "stop_first": ["command-line regexes to stop first"],
       "required": false, "duty": 0.5}]}

Relative paths resolve against the queue file's directory. Each step's output goes to `<queue stem>-<name>.log`,
receipts to `<queue stem>.receipt.json`, the runner's own output to `<queue stem>.log`. A failing step does not stop
the queue unless it is `"required": true`; a failed required step makes the runner exit non-zero. The `done` marker is
written last (also after a failure, so watchers wake): one line per step `name rc duration`, then `status ok` or
`status failed <step> rc=<rc>`; the receipt carries the same `status`. A marker without a status line counts as ok. A queue
whose `after` marker holds `status failed` runs none of its steps: it records the reason, writes its own marker with
`status failed after <marker>` and exits non-zero.

GPU steps are wrapped in `scripts/gpu_run.py --purpose <name>` (lease, idle priority, its own load cap). CPU steps get
the same suspend/resume load cap (`duty`, default 0.3 of a 4 s period, at very low I/O priority) so a queue never saturates the desktop;
`"duty": 1` opts out. Every step starts only while the PC has memory headroom (`gpu_run.wait_for_memory`). `stop_first` pauses a lower-priority chain: processes whose command line matches are stopped
(with their children) before the step starts; this runner is never stopped.
"""
from __future__ import annotations

try:
    from scripts.dev_state import require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import require_evidence_writable


import argparse
import hashlib
import math
import json
import os
import re
import shutil
import subprocess
import sys
import time
from uuid import uuid4
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PERIOD = 4.0
DEFAULT_DUTY = 0.3
POLL = float(os.environ.get("LSS_BATCH_POLL", "5"))  # seconds between checks for `after` markers
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _slug(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "step"


def load_queue(path: Path, *, raw: bytes | None = None) -> dict:
    queue = json.loads((path.read_bytes() if raw is None else raw).decode('utf-8-sig'))
    if not isinstance(queue, dict):
        raise ValueError('queue must be an object')
    steps = queue.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("queue needs a non-empty 'steps' list")
    names = set()
    if queue.get('done') is not None and (not isinstance(queue['done'], str) or not queue['done'].strip()):
        raise ValueError('done must be a nonempty path')
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError('step must be an object')
        if (not isinstance(step.get('name'), str) or not step['name'].strip()
                or not isinstance(step.get("cmd"), list) or not step["cmd"]
                or any(not isinstance(arg, str) or '\0' in arg for arg in step['cmd']) or not step['cmd'][0].strip()):
            raise ValueError(f"step needs 'name' and a non-empty 'cmd' list: {step!r}")
        if step["name"] in names:
            raise ValueError(f"duplicate step name {step['name']!r}")
        names.add(step["name"])
        for key in ('gpu', 'required'):
            if key in step and not isinstance(step[key], bool):
                raise ValueError(f'{key} must be a boolean')
        if 'cwd' in step and (not isinstance(step['cwd'], str) or not step['cwd'].strip()):
            raise ValueError('cwd must be a nonempty path')
        duty = step.get('duty', DEFAULT_DUTY)
        if isinstance(duty, bool) or not isinstance(duty, (int, float)) or not math.isfinite(duty) or not 0 < duty <= 1:
            raise ValueError(f"invalid duty for {step['name']}")
        for key in ('after', 'after_queues', 'storage_paths', 'archive_paths', 'stop_first'):
            if key in step and (not isinstance(step[key], list) or any(not isinstance(p, str) or not p.strip() for p in step[key])):
                raise ValueError(f'{key} must be a list of nonempty strings')
        if 'storage_gb' in step and (isinstance(step['storage_gb'], bool) or not isinstance(step['storage_gb'], (int, float)) or not math.isfinite(step['storage_gb']) or step['storage_gb'] < 0):
            raise ValueError('storage_gb must be finite and nonnegative')
        if 'wait_timeout_seconds' in step and (isinstance(step['wait_timeout_seconds'], bool) or not isinstance(step['wait_timeout_seconds'], (int,float)) or not math.isfinite(step['wait_timeout_seconds']) or step['wait_timeout_seconds'] <= 0):
            raise ValueError('wait_timeout_seconds must be finite and positive')
        if any(_resolve(path.parent, dependency).resolve() == path.resolve() for dependency in step.get('after_queues', [])):
            raise ValueError('queue cannot depend on itself')
    slugs = [_slug(name) for name in names]
    if len(slugs) != len(set(slugs)):
        raise ValueError('step names collide on the log path')
    protected = {path.resolve(), *(path.with_suffix(suffix).resolve() for suffix in ('.receipt.json','.launch.json','.queue.lock','.log')),
                 *((path.parent/f'{path.stem}-{_slug(name)}.log').resolve() for name in names)}
    if queue.get('done') and _resolve(path.parent, queue['done']).resolve() in protected:
        raise ValueError('done marker collides with queue metadata')
    return queue


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def stop_matching(patterns: list[str]) -> list[int]:
    """Stop (with children) every process whose command line matches a pattern, except this runner and its parents."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"],
        capture_output=True, text=True, creationflags=NOWIN).stdout
    rows = json.loads(out) if out.strip() else []
    rows = [rows] if isinstance(rows, dict) else rows
    parent = {r["ProcessId"]: r["ParentProcessId"] for r in rows}
    protected, pid = set(), os.getpid()
    while pid and pid not in protected:  # this runner and its launcher chain
        protected.add(pid)
        pid = parent.get(pid, 0)
    stopped = []
    for row in rows:
        cmdline = row.get("CommandLine") or ""
        if row["ProcessId"] in protected or not any(re.search(p, cmdline) for p in patterns):
            continue
        subprocess.run(["taskkill", "/PID", str(row["ProcessId"]), "/T", "/F"], capture_output=True, creationflags=NOWIN)
        stopped.append(row["ProcessId"])
    return stopped


def marker_failure(marker: Path) -> str | None:
    """The `status failed ...` text of a done marker, or None (ok, or an old marker without a status line)."""
    try:
        lines = marker.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if line.startswith("status "):
            return line[len("status "):] if line.startswith("status failed") else None
    return None


def dependency_state(path: Path) -> str:
    """Recorded completion or current owner proof for an explicit upstream queue."""
    from process_state import observe
    if not path.is_file():
        return 'missing upstream queue definition'
    receipt = path.with_suffix('.receipt.json')
    if not receipt.exists():
        launch = path.with_suffix('.launch.json')
        if not launch.exists():
            return 'waiting'
        data = json.loads(launch.read_text(encoding='utf-8'))
        if data.get('stage') == 'failed':
            return 'failed launch'
        live = observe(data.get('owner'))
        return 'waiting' if live == 'RUNNING' else f'unverified or stopped launch: {live}'
    data = json.loads(receipt.read_text(encoding='utf-8'))
    if not data.get('queue_sha256'):
        return 'unverified upstream input identity'
    if data.get('queue_sha256') and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != data['queue_sha256']:
        return 'queue changed after recorded attempt'
    if data.get('ended'):
        return 'ok' if data.get('status') == 'ok' else f"failed upstream: {data.get('status')}"
    live = observe(data.get('owner'))
    return 'waiting' if live == 'RUNNING' else f'unverified or stopped upstream: {live}'


def publish_marker(marker: Path, text: str) -> None:
    """Expose a complete marker once, never partial bytes or another attempt's file."""
    require_evidence_writable(marker)
    marker.parent.mkdir(parents=True,exist_ok=True)
    temporary = marker.with_name(marker.name+'.'+uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary,marker)  # atomic create-if-absent on the same volume
    finally:
        temporary.unlink(missing_ok=True)


def run_command(command: list[str], cwd: Path, log: Path, duty: float, *, storage_gb: float | None = None,
                storage_paths=(), archive_paths=()) -> int:
    require_evidence_writable(log)
    import storage_budget
    command = storage_budget.command(command, required_gb=storage_gb, purpose=log.stem,
                                     paths=[str(cwd), str(log.parent), *storage_paths, *archive_paths],
                                     archive_paths=archive_paths)
    import gpu_run  # the one load-cap implementation
    with open(log, "a", encoding="utf-8") as out:
        gpu_run.wait_for_memory(log.stem, log=lambda line: out.write(line + "\n"))
    flags = 0 if os.name != "nt" else NOWIN | subprocess.IDLE_PRIORITY_CLASS
    exe = str(Path(shutil.which(command[0]) or command[0]).resolve())
    with open(log, "ab") as out:
        with storage_budget.owned_child([exe, *command[1:]], cwd=cwd, stdin=subprocess.DEVNULL, stdout=out,
                                       stderr=subprocess.STDOUT, creationflags=flags) as child:
            return gpu_run._run_capped(child, duty, PERIOD)


def _admission_failure(path: Path, queue: dict, error: Exception) -> int:
    status = f"failed {error}"
    receipt = {"queue": str(path), "pid": os.getpid(), "started": _now(), "ended": _now(),
               "status": status, "steps": []}
    receipt_path = path.with_suffix(".receipt.json")
    require_evidence_writable(receipt_path)
    # Refusing a new attempt never rewrites evidence from an earlier execution.
    if receipt_path.exists():
        print(status, file=sys.stderr)
        return 1
    tmp = receipt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, receipt_path)
    if queue.get("done"):
        marker = _resolve(path.parent, queue["done"])
        require_evidence_writable(marker)
        marker.parent.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(f"status {status}\n", encoding="utf-8")
    print(status, file=sys.stderr)
    return 1


def run_queue(path: Path) -> int:
    from storage_tiers import Lock
    require_evidence_writable(path.with_suffix('.queue.lock'))
    path = path.resolve()
    lock = Lock(path.with_suffix('.queue.lock'))
    try:
        lock.__enter__()
    except RuntimeError as exc:
        print(f'queue not started: {exc}', file=sys.stderr)
        return 1
    try:
        if path.with_suffix('.receipt.json').exists():
            print('Existing attempt retained: inspect status/receipt; use a new queue path for an explicitly reviewed retry.', file=sys.stderr)
            return 1
        return _run_queue_owned(path)
    finally:
        lock.__exit__(None, None, None)


def _run_queue_owned(path: Path) -> int:
    require_evidence_writable(path.with_suffix(".receipt.json"))
    path = path.resolve()
    base, stem = path.parent, path.stem
    raw = path.read_bytes()
    queue = load_queue(path, raw=raw)
    if queue.get('done') and _resolve(base, queue['done']).exists():
        raise ValueError('Existing done marker retained; use a distinct attempt marker')
    import codex_focus
    try:
        codex_focus.admit(path, queue)
    except ValueError as exc:
        return _admission_failure(path, queue, exc)
    receipt_path = base / f"{stem}.receipt.json"
    from process_state import identity
    receipt = {"queue": str(path), "pid": os.getpid(), 'owner': identity(),
               'queue_sha256': hashlib.sha256(raw).hexdigest(), "started": _now(), "ended": None, "status": None,
               "steps": []}

    def save() -> None:
        tmp = receipt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, receipt_path)

    save()
    try:
        failure = None  # "<step> rc=<rc>" of a failed required step, or "after <marker>" when an upstream queue failed
        for step in queue["steps"]:
            name = step["name"]
            cwd = _resolve(base, step["cwd"]) if step.get("cwd") else base
            log = base / f"{stem}-{_slug(name)}.log"
            row = {"name": name, "cmd": step["cmd"], "cwd": str(cwd), "log": str(log), "gpu": bool(step.get("gpu")),
                   "start": None, "end": None, "duration": None, "rc": None}
            receipt["steps"].append(row)
            save()
            wait_start = time.monotonic()
            for dependency in step.get('after_queues', []):
                while True:
                    state = dependency_state(_resolve(base,dependency))
                    if state != 'waiting':
                        break
                    if step.get('wait_timeout_seconds') and time.monotonic()-wait_start >= step['wait_timeout_seconds']:
                        state = 'dependency wait timeout'
                        break
                    time.sleep(POLL)
                if state != 'ok':
                    failure = f'after queue {dependency}: {state}'
                    row['skipped'] = failure
                    break
            if failure:
                break
            for marker in step.get("after", []):
                while not _resolve(base, marker).exists():
                    if step.get('wait_timeout_seconds') and time.monotonic()-wait_start >= step['wait_timeout_seconds']:
                        raise TimeoutError('upstream marker wait timed out')
                    time.sleep(POLL)
                upstream = marker_failure(_resolve(base, marker))
                if upstream:
                    failure = f"after {marker}"
                    row["skipped"] = f"upstream marker {marker} reports: {upstream}"
                    break
            if failure:
                break
            if step.get("stop_first"):
                row["stopped"] = stop_matching(step["stop_first"])
            command = list(step["cmd"])
            duty = float(step.get("duty", DEFAULT_DUTY))
            if step.get("gpu"):
                command = [sys.executable, str(HERE / "gpu_run.py"), "--purpose", name, "--", *command]
                duty = 1.0  # gpu_run caps its own tree
            start = time.time()
            row["start"] = _now()
            save()
            try:
                rc = run_command(command, cwd, log, duty, storage_gb=step.get("storage_gb"),
                                 storage_paths=[str(_resolve(base, p)) for p in step.get("storage_paths", [])],
                                 archive_paths=[str(_resolve(base, p)) for p in step.get("archive_paths", [])])
            except BaseException as exc:
                with open(log, "ab") as out:
                    out.write(f"[batch] {type(exc).__name__}: {exc}\n".encode('utf-8', errors='replace'))
                rc = 130 if isinstance(exc, KeyboardInterrupt) else -1
                row['error_type'] = type(exc).__name__
                failure = f'{name} exception={type(exc).__name__}'
            row["rc"], row["end"], row["duration"] = rc, _now(), round(time.time() - start, 1)
            save()
            if failure or (rc != 0 and step.get("required")):
                failure = failure or f"{name} rc={rc}"
                break
    except BaseException as exc:
        failure = f"runner exception={type(exc).__name__}"
        receipt['error_type'] = type(exc).__name__
        for row in receipt['steps']:
            if row.get('rc') is None:
                row['error_type'] = type(exc).__name__
                row['skipped'] = 'Runner interrupted before this step completed'
    completed_status = f"failed {failure}" if failure else "ok"
    receipt['status'] = 'publishing completion'
    save()
    if queue.get("done"):
        try:
            marker = _resolve(base, queue["done"])
            publish_marker(marker,"".join(f"{r['name']} {r['rc']} {r['duration']}\n" for r in receipt["steps"]
                                      if r["rc"] is not None) + f"status {completed_status}\n")
        except OSError as exc:
            receipt['status'] = f'failed completion publication={type(exc).__name__}'
            receipt['publication_error_type'] = type(exc).__name__
            receipt['ended'] = _now()
            save()
            return 1
    receipt['status'] = completed_status
    receipt['ended'] = _now()
    save()
    return 1 if failure else 0


def start_queue(path: Path) -> int:
    import hidden_run  # same launch mechanism and dev-jobs record as any hidden job
    require_evidence_writable(path.with_suffix(".launch.json"))
    require_evidence_writable(path.with_suffix(".log"))
    path = path.resolve()
    if path.with_suffix('.receipt.json').exists():
        print('Existing attempt retained; inspect its receipt before a new attempt.', file=sys.stderr)
        return 1
    queue = load_queue(path)  # fail here, not in the background
    import codex_focus
    try:
        codex_focus.admit(path, queue)
    except ValueError as exc:
        return _admission_failure(path, queue, exc)
    return hidden_run.main(["--log", str(path.with_suffix(".log")), '--receipt', str(path.with_suffix('.launch.json')), "--", sys.executable, str(Path(__file__).resolve()),
                            "run", str(path)], storage_supervisor=True)


def status(path: Path, *, as_json: bool = False) -> int:
    receipt_path = path.resolve().with_suffix(".receipt.json")
    if not receipt_path.exists():
        launch_path = path.resolve().with_suffix('.launch.json')
        if launch_path.exists():
            from process_state import observe
            launch = json.loads(launch_path.read_text(encoding='utf-8'))
            live = observe(launch.get('owner'))
            print(json.dumps({'state':'LAUNCHING' if live == 'RUNNING' else live,
                              'boundary':'Launcher observed; queue receipt is not published yet', 'log':launch.get('log')}))
            return 0 if live == 'RUNNING' else 2 if live == 'UNKNOWN' else 3
        print("no receipt yet")
        return 1
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    from process_state import observe
    live = 'TERMINAL' if receipt.get('ended') else observe(receipt.get('owner'))
    label = receipt.get('status') if live == 'TERMINAL' else live
    changed = (hashlib.sha256(path.read_bytes()).hexdigest() != receipt['queue_sha256']) if path.exists() and receipt.get('queue_sha256') else None
    if as_json:
        print(json.dumps({'queue':receipt['queue'], 'state':label, 'process_state':live,
                          'queue_changed':changed, 'steps':receipt['steps'], 'owner':receipt.get('owner')},ensure_ascii=False))
        return 0 if live in {'RUNNING','TERMINAL'} else 2 if live == 'UNKNOWN' else 3
    print(f"queue {receipt['queue']}  started {receipt['started']}  state {label}")
    if changed:
        print('queue content changed since this attempt; receipt describes the original bytes')
    print(f"{'step':<28} {'rc':>5} {'sec':>8}  state")
    for r in receipt["steps"]:
        state = "waiting" if r["start"] is None else live if r["rc"] is None else "ok" if r["rc"] == 0 else "FAILED"
        rc = "" if r["rc"] is None else r["rc"]
        sec = "" if r["duration"] is None else r["duration"]
        print(f"{r['name']:<28} {rc:>5} {sec:>8}  {state}")
    return 0 if live in {'RUNNING', 'TERMINAL'} else 2 if live == 'UNKNOWN' else 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["start", "run", "status"])
    parser.add_argument("queue", type=Path)
    parser.add_argument('--json', action='store_true', help='structured status (does not execute or resume)')
    args = parser.parse_args(argv)
    if args.action == 'status':
        return status(args.queue, as_json=args.json)
    if args.json:
        parser.error('--json applies to status only')
    return {"start": start_queue, "run": run_queue}[args.action](args.queue)


if __name__ == "__main__":
    sys.exit(main())
