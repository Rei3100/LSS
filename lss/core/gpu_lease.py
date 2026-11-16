"""同じWindowsセッションのLSSワーカー間でCUDA deviceの使用を直列化する。"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import logging
import os
import threading
from collections.abc import Callable


class GpuLeaseUnavailableError(RuntimeError):
    """GPUの排他所有を保証できないため、推論を開始できない。"""


def _kernel32():
    if os.name != "nt":
        raise GpuLeaseUnavailableError("GPUの排他所有にはWindows実行環境が必要です")
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    api.CreateMutexW.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.ReleaseMutex.argtypes = [wintypes.HANDLE]
    api.ReleaseMutex.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    return api


def _failure(operation: str) -> GpuLeaseUnavailableError:
    return GpuLeaseUnavailableError(
        f"GPUの排他所有を確認できません: operation={operation}, "
        f"winerror={ctypes.get_last_error()}"
    )


class GpuLease:
    def __init__(self, api, handle, *, abandoned: bool):
        self._api = api
        self._handle = handle
        self._owner_thread = threading.get_ident()
        self.abandoned = abandoned

    def close(self) -> None:
        if self._handle is None:
            return
        if threading.get_ident() != self._owner_thread:
            raise GpuLeaseUnavailableError("GPU所有権は取得した同じスレッドで解放する必要があります")
        handle, self._handle = self._handle, None
        # 所有権を得た同じスレッドで解放する。OSは所有スレッドの異常終了も回収する。
        try:
            if not self._api.ReleaseMutex(handle):
                logging.getLogger(__name__).error(json.dumps({
                    "event": "gpu_lease_release_failed", "winerror": ctypes.get_last_error(),
                }))
        finally:
            if not self._api.CloseHandle(handle):
                logging.getLogger(__name__).error(json.dumps({
                    "event": "gpu_lease_handle_close_failed", "winerror": ctypes.get_last_error(),
                }))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def acquire(device_index: int = 0, *, on_wait: Callable[[], None] | None = None) -> GpuLease:
    if not isinstance(device_index, int) or isinstance(device_index, bool) or device_index < 0:
        raise ValueError("CUDA device indexは0以上の整数である必要があります")
    api = _kernel32()
    handle = api.CreateMutexW(None, False, f"Local\\LSS-CUDA-Device-{device_index}-v1")
    if not handle:
        raise _failure("CreateMutexW")
    try:
        state = api.WaitForSingleObject(handle, 0)
        if state == 0x102:  # WAIT_TIMEOUT
            if on_wait is not None:
                try:
                    on_wait()
                except Exception:
                    pass  # 表示の失敗で排他所有を放棄しない。
            state = api.WaitForSingleObject(handle, 0xFFFFFFFF)
        if state not in {0, 0x80}:  # WAIT_OBJECT_0 / WAIT_ABANDONED
            raise _failure("WaitForSingleObject")
        return GpuLease(api, handle, abandoned=state == 0x80)
    except BaseException:
        api.CloseHandle(handle)
        raise
