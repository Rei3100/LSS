"""Windows-owned CPU hard cap for new quiet development process trees.

run_capped never attaches existing processes. Every payload starts suspended, receives one
processor affinity and a kill-on-close Job with a conservative CPU hard cap,
then resumes. Job accounting includes descendants and teardown. Quotas round
down; unsupported/unrepresentable settings refuse rather than run unrestricted.
The affinity helper can pin the current development worker before its children start.

CPU rate uses cycles per 10,000 across the system (or a stricter parent Job):
https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_cpu_rate_control_information
"""
from __future__ import annotations
import ctypes
import math
import os
import subprocess
import time


def capped_child_env(env, *, units, cores):
    child_env = dict(env)
    child_env['LSS_QUIET_CPU_ONE_CORE_DUTY'] = str(units * cores / 10000)
    return child_env


def quiet_wall_timeout(base_seconds):
    """Scale an operational wait to the confined payload's one-core ceiling."""
    try:
        duty = float(os.environ.get('LSS_QUIET_CPU_ONE_CORE_DUTY', ''))
    except (TypeError, ValueError):
        return base_seconds
    if not math.isfinite(duty) or not 0 < duty <= 1:
        return base_seconds
    return base_seconds / duty


def rate_units(duty, cores):
    if type(duty) not in (int, float) or not math.isfinite(duty) or not 0 < duty <= 1:
        raise ValueError('Invalid one-processor duty')
    if type(cores) is not int or cores < 1:
        raise ValueError('Invalid processor count')
    units = math.floor(10000 * duty / cores)
    if units < 1:
        raise ValueError('Requested ceiling is below Windows CPU-rate resolution')
    return units


class Rate(ctypes.Structure):
    _fields_ = [('ControlFlags', ctypes.c_ulong), ('CpuRate', ctypes.c_ulong)]


class Accounting(ctypes.Structure):
    _fields_ = [('user', ctypes.c_longlong), ('kernel', ctypes.c_longlong),
                ('period_user', ctypes.c_longlong), ('period_kernel', ctypes.c_longlong),
                ('faults', ctypes.c_ulong), ('total', ctypes.c_ulong),
                ('active', ctypes.c_ulong), ('terminated', ctypes.c_ulong)]


def pin_one_processor(child=None):
    """Pin a new payload, or the current development worker, to one processor."""
    if os.name != 'nt':
        raise RuntimeError('Quiet CPU enforcement requires Windows')
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    allowed, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system_mask)):
        raise OSError('Cannot inspect CPU affinity')
    one = allowed.value & -allowed.value
    handle = kernel.GetCurrentProcess() if child is None else int(child._handle)
    if not one or not kernel.SetProcessAffinityMask(ctypes.c_void_p(handle), one):
        raise OSError('Cannot constrain CPU affinity')
    return one


def run_capped(command, *, cwd, env, stdout, duty=.005, timeout=None, cleanup_timeout=300):
    if os.name != 'nt':
        raise RuntimeError('Quiet CPU enforcement requires Windows')
    from local_llm import Job
    from gpu_run import _quiet_io
    k = ctypes.WinDLL('kernel32')
    k.GetActiveProcessorCount.argtypes = [ctypes.c_ushort]
    k.GetActiveProcessorCount.restype = ctypes.c_ulong
    cores = int(k.GetActiveProcessorCount(0xffff))
    units = rate_units(duty, cores)
    child_env = capped_child_env(env, units=units, cores=cores)
    supervisor_started = time.process_time()
    job, child = Job(), None
    try:
        rate = Rate(0x1 | 0x4, units)
        if not job._k.SetInformationJobObject(job._handle, 15, ctypes.byref(rate), ctypes.sizeof(rate)):
            raise OSError('Cannot set quiet job CPU hard cap')
        child = subprocess.Popen(command, cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS | 0x4)
        pin_one_processor(child)
        job.assign(child.pid)
        _quiet_io([child.pid], set())
        nt = ctypes.WinDLL('ntdll')
        nt.NtResumeProcess.argtypes = [ctypes.c_void_p]
        nt.NtResumeProcess.restype = ctypes.c_long
        started = time.monotonic()
        if nt.NtResumeProcess(ctypes.c_void_p(int(child._handle))) != 0:
            raise OSError('Cannot resume confined quiet payload')
        code = child.wait(timeout=timeout)
        counters = Accounting()
        job._k.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                    ctypes.c_ulong, ctypes.c_void_p]
        job._k.QueryInformationJobObject.restype = ctypes.c_int
        cleanup_deadline = time.monotonic() + cleanup_timeout
        while True:
            if not job._k.QueryInformationJobObject(job._handle, 1, ctypes.byref(counters), ctypes.sizeof(counters), None):
                raise OSError('Cannot query quiet job accounting')
            if not counters.active:
                break
            if time.monotonic() >= cleanup_deadline:
                raise TimeoutError('Quiet payload descendants have not exited')
            time.sleep(.05)
        wall = time.monotonic()-started
        cpu = (counters.user + counters.kernel)/10_000_000
        return {'returncode': code, 'method': 'windows_job_cpu_hard_cap',
                'logical_processors': cores, 'cpu_processors': 1,
                'requested_one_processor_duty': duty, 'cpu_rate_units': units,
                'configured_one_processor_ceiling': units*cores/10000,
                'wall_s': wall, 'job_cpu_s': cpu, 'active_descendants': counters.active,
                'observed_mean_one_core_fraction': cpu/wall,
                'supervisor_cpu_s': time.process_time()-supervisor_started}
    finally:
        job.close()
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=30)
