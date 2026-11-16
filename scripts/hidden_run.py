"""Start a development command in the background with no window, at idle CPU priority, output to a log file.

    python scripts/hidden_run.py --log path/to/job.log -- python explore.py judge ...
    python scripts/hidden_run.py --log path/to/job.log --receipt path/to/launch.json -- python probe.py
    python scripts/hidden_run.py --status path/to/launch.json

An explicit launch receipt is exclusive and records OS creation identity, not
command arguments. It distinguishes launch failure/early exit from a running
supervisor; the command's own receipt establishes completion, not a stale PID.

Returns at once and prints the child's pid. The child gets a hidden console (CREATE_NO_WINDOW), so a venv
`python.exe` launcher and anything it starts stay invisible; never use DETACHED_PROCESS for console programs,
which makes their children open a new visible console; nothing a development job starts may pop up over the user's
desktop or require the desktop to be vacated.
Use `scripts/gpu_run.py` as the command when the job needs the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv=None, *, storage_supervisor: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", help="stdout and stderr go here")
    parser.add_argument('--receipt', type=Path, help='exclusive launch receipt with OS process identity; does not contain command arguments')
    parser.add_argument('--status', type=Path, help='observe a prior launch receipt without starting anything')
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--storage-gb", type=float, help="estimated peak additional SSD space; waits until reserved")
    parser.add_argument("--protect", action="append", default=[], help="additional directory used by this job")
    parser.add_argument("--archive", action="append", default=[], help="owned output directory eligible after job exit")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.status:
        from process_state import observe
        receipt = json.loads(args.status.read_text(encoding='utf-8'))
        if receipt.get('stage') in {'failed', 'exited'}:
            print(json.dumps({'state':receipt['stage'].upper(), 'returncode':receipt.get('returncode'), 'log':receipt.get('log')}))
            return 1 if receipt['stage']=='failed' else 3
        live = observe(receipt.get('owner'))
        print(json.dumps({'state': live, 'log': receipt.get('log'),
                          'boundary': 'Supervisor process observation; exit/result of the command is established by its own receipt/log.'}))
        return 0 if live == 'RUNNING' else 2 if live == 'UNKNOWN' else 3
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")
    if not args.log:
        parser.error('--log is required for launch')
    original_exe = Path(shutil.which(command[0]) or command[0]).resolve()
    if not original_exe.is_file():
        parser.error(f'command executable is missing: {original_exe}')
    if args.cwd and not Path(args.cwd).is_dir():
        parser.error('working directory is missing')
    log = Path(args.log).resolve()
    if args.receipt and args.receipt.resolve() == log:
        parser.error('launch receipt must differ from the log')
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open('x', encoding='utf-8') as receipt:
            json.dump({'stage': 'starting', 'log': str(log), 'owner': None}, receipt)
    child = None
    phase = 'prepare'
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        if not storage_supervisor:  # batch's supervisor reserves each step, not the entire queue
            import storage_budget
            command = storage_budget.command(command, required_gb=args.storage_gb, purpose=log.stem,
                                             paths=[str(Path(args.cwd or Path.cwd()).resolve()), str(log.parent), *args.protect,
                                                    *args.archive], archive_paths=args.archive)
        exe = str(Path(shutil.which(command[0]) or command[0]).resolve())
        flags = 0 if os.name != "nt" else subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS
        phase = 'spawn'
        with open(log, "ab") as out:
            # The job outlives its launcher, so it is admitted afresh instead of borrowing the launcher's reservation.
            launch = dict(cwd=args.cwd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                          env={k: v for k, v in os.environ.items() if k != "LSS_STORAGE_RESERVATION"})
            try:  # leave a storage_budget kill-on-close Job this was itself started under, so the job outlives it
                child = subprocess.Popen([exe, *command[1:]], creationflags=flags | (0x01000000 if os.name == "nt" else 0), **launch)
            except OSError:  # a Job that forbids breakaway gets the plain start
                child = subprocess.Popen([exe, *command[1:]], creationflags=flags, **launch)
        phase = 'publish-launch'
        if args.receipt:
            from process_state import probe
            data = {'stage':'launched', 'log':str(log), 'cwd':str(Path(args.cwd or Path.cwd()).resolve()),
                    'executable':str(original_exe), 'owner':{'pid':child.pid, 'created':probe(child.pid)['created']}}
            ended = child.poll()
            if ended is not None:
                data.update(stage='exited', returncode=ended)
            temporary = args.receipt.with_suffix(args.receipt.suffix+'.tmp')
            temporary.write_text(json.dumps(data,indent=2)+'\n', encoding='utf-8')
            os.replace(temporary,args.receipt)
        print(child.pid)
        return 0
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        if args.receipt:
            failed = {'stage':'failed', 'phase':phase, 'error_type':type(exc).__name__, 'log':str(log), 'owner':None}
            try:
                args.receipt.write_text(json.dumps(failed,indent=2)+'\n',encoding='utf-8')
            except OSError:
                pass  # No success is reported when even failure metadata cannot be published.
        raise



if __name__ == "__main__":
    sys.exit(main())
