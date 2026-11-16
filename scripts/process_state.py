"""Observe a declared process without signals, command lines, model imports or PID guesses.

A PID plus OS creation identity distinguishes the original process from PID reuse.
Access failures and old receipts without identity are UNKNOWN, never running proof.
"""
from __future__ import annotations

import os
from pathlib import Path


def probe(pid: int) -> dict:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return {'state': 'UNKNOWN', 'created': None}
    if os.name != 'nt':
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            return {'state': 'EXITED' if fields[0] == 'Z' else 'RUNNING', 'created': fields[19]}
        except FileNotFoundError:
            return {'state': 'EXITED', 'created': None}
        except (OSError, IndexError):
            return {'state': 'UNKNOWN', 'created': None}
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        return {'state': 'EXITED' if ctypes.get_last_error() == 87 else 'UNKNOWN', 'created': None}
    try:
        wait = kernel.WaitForSingleObject(handle, 0)
        times = [wintypes.FILETIME() for _ in range(4)]
        created = None
        if kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            created = str(times[0].dwHighDateTime << 32 | times[0].dwLowDateTime)
        return {'state': 'RUNNING' if wait == 258 else 'EXITED' if wait == 0 else 'UNKNOWN', 'created': created}
    finally:
        kernel.CloseHandle(handle)


def identity() -> dict:
    return {'pid': os.getpid(), 'created': probe(os.getpid())['created']}


def observe(owner: dict | None) -> str:
    if not owner or not owner.get('created'):
        return 'UNKNOWN'
    current = probe(owner.get('pid'))
    if current['state'] == 'EXITED':
        return 'EXITED'
    if current['created'] is None:
        return 'UNKNOWN'
    return current['state'] if current['created'] == owner['created'] else 'REPLACED'
