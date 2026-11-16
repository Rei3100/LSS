"""Paired Windows CPU-cap experiment; never changes an existing process/job.

Compare the existing suspend-based 0.005 duty with a kernel Job Object hard cap,
on identical short CPU/wait workloads. Payloads start suspended, are pinned to
one processor and assigned a kill-on-close Job before first execution. No model,
GPU, network, HDD or Product access occurs. Adoption requires measured receipts.

API reference: https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_cpu_rate_control_information
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from quiet_cpu import Rate, Accounting, rate_units

ROOT = Path(__file__).resolve().parents[1]
DUTY = 0.005
PAYLOAD = '''import hashlib,json,sys,time
start=time.process_time()
data=b'x'*8192
iterations=32768 if sys.argv[1]=='cpu' else 24
for i in range(iterations):
 digest=hashlib.sha256(data).hexdigest()
 if sys.argv[1]=='wait':time.sleep(.05)
print(json.dumps({'kind':sys.argv[1],'iterations':iterations,'digest':digest,'process_cpu_s':time.process_time()-start}))
'''


def one_case(kind, method, output, cores, units):
    require_evidence_writable(output)
    import gpu_run
    from local_llm import Job
    from quiet_dev_checks import environment, pin_one_processor
    if method == 'helper':
        from quiet_cpu import run_capped
        with (output / f'{kind}-helper.json').open('xb') as log:
            metrics = run_capped([sys.executable, '-c', PAYLOAD, kind], cwd=ROOT,
                env=environment(output), stdout=log, duty=DUTY, timeout=1800)
        data = json.loads((output / f'{kind}-helper.json').read_text(encoding='utf-8'))
        if (metrics['returncode'] or metrics['active_descendants'] or data['kind'] != kind
                or data['iterations'] != (32768 if kind == 'cpu' else 24)
                or data['digest'] != hashlib.sha256(b'x'*8192).hexdigest()):
            raise ValueError('Helper did not preserve work or clean up descendants')
        return dict(metrics, kind=kind, payload=data, status='complete')
    supervisor_cpu_started = time.process_time()
    job = Job()
    child = None
    record = {'kind': kind, 'method': method, 'logical_processors': cores,
              'cpu_processors': 1, 'requested_one_processor_duty': DUTY}
    try:
        if method == 'kernel':
            limits = Rate(0x1 | 0x4, units)
            if not job._k.SetInformationJobObject(job._handle, 15, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise OSError('Cannot set job CPU hard cap')
            record.update(cpu_rate_units=units, configured_one_processor_ceiling=units * cores / 10000)
        log_path = output / f'{kind}-{method}.json'
        with log_path.open('xb') as log:
            child = subprocess.Popen([sys.executable, '-c', PAYLOAD, kind],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=environment(output), cwd=ROOT,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS | 0x4)
            pin_one_processor(child)
            job.assign(child.pid)
            nt = ctypes.WinDLL('ntdll')
            nt.NtResumeProcess.argtypes = [ctypes.c_void_p]
            nt.NtResumeProcess.restype = ctypes.c_long
            started = time.monotonic()
            if nt.NtResumeProcess(ctypes.c_void_p(int(child._handle))) != 0:
                raise OSError('Cannot resume confined payload')
            if method == 'kernel':
                code = child.wait(timeout=1800)
            else:
                code = gpu_run._run_capped(child, DUTY, 4.)
        counters = Accounting()
        k = job._k
        k.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                ctypes.c_ulong, ctypes.c_void_p]
        k.QueryInformationJobObject.restype = ctypes.c_int
        # Launcher exit precedes final descendant teardown. Wait for the actual
        # Job population to reach zero; never remove this cleanup requirement.
        cleanup_deadline = time.monotonic() + 30
        while True:
            if not k.QueryInformationJobObject(job._handle, 1, ctypes.byref(counters), ctypes.sizeof(counters), None):
                raise OSError('Cannot query job accounting')
            if not counters.active:
                break
            if time.monotonic() >= cleanup_deadline:
                raise RuntimeError(f'Payload descendants still active: {counters.active}')
            time.sleep(.05)
        wall = time.monotonic()-started
        data = json.loads(log_path.read_text(encoding='utf-8'))
        if code or data['kind'] != kind or data['iterations'] != (32768 if kind == 'cpu' else 24):
            raise ValueError('Payload did not complete the prescribed work')
        if data['digest'] != hashlib.sha256(b'x'*8192).hexdigest():
            raise ValueError('Payload result differs')
        cpu = (counters.user + counters.kernel) / 10_000_000
        record.update(status='complete', payload=data, wall_s=wall, job_cpu_s=cpu,
                      observed_mean_one_core_fraction=cpu/wall,
                      supervisor_cpu_s=time.process_time()-supervisor_cpu_started)
        return record
    finally:
        job.close()
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=30)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('Windows-only probe')
    from acquire_audio_probe import save
    from dev_hot_paths import require_hot
    import storage_budget
    output = dev_evidence_root(ROOT) / ('release-evidence/quiet-kernel-cpu-helper-20261027' if args.helper
                     else 'release-evidence/quiet-kernel-cpu-probe-20261027-r3')
    for path in (ROOT, output, Path(sys.executable)):
        require_hot(path, 'quiet CPU-rate experiment')
    require_evidence_writable(output)
    output.mkdir(exist_ok=False)
    launch_source = Path(__file__).read_bytes()
    (output / 'source-at-launch.py').write_bytes(launch_source)
    helper_source = Path(__file__).with_name('quiet_cpu.py').read_bytes()
    (output / 'helper-at-launch.py').write_bytes(helper_source)
    k = ctypes.WinDLL('kernel32')
    k.GetActiveProcessorCount.argtypes = [ctypes.c_ushort]
    k.GetActiveProcessorCount.restype = ctypes.c_ulong
    cores = int(k.GetActiveProcessorCount(0xffff))
    record = {'status': 'running', 'cases': [], 'gpu': False, 'hdd': False,
              'source_sha256': hashlib.sha256(launch_source).hexdigest(),
              'helper_sha256': hashlib.sha256(helper_source).hexdigest(),
              'rate_struct_bytes': ctypes.sizeof(Rate), 'accounting_struct_bytes': ctypes.sizeof(Accounting)}
    try:
        units = rate_units(DUTY, cores)
        with storage_budget.reserve(1, [str(ROOT), str(output)], 'quiet CPU-rate probe',
                                    wait=False, _allow_nudge=False):
            for kind in ('wait', 'cpu'):
                for method in (('helper',) if args.helper else ('kernel', 'suspend')):
                    record['stage'] = kind + ':' + method
                    record['cases'].append(one_case(kind, method, output, cores, units))
                    save(output / 'progress.json', record)
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:300])
    finally:
        save(output / 'result.json', record)
    return 0 if record['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
