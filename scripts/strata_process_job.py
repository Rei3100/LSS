"""Windows Job ownership for a hidden, bounded subprocess tree."""
import ctypes
from ctypes import wintypes
import os
import subprocess
import uuid

JOB_ENV = "LSS_OWNED_PROCESS_JOB"


class _Basic(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD), ("min_working", ctypes.c_size_t),
                ("max_working", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD)]


class _Extended(ctypes.Structure):
    _fields_ = [("basic", _Basic), ("io", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]


class OwnedProcessJob:
    """Own one hidden process tree; descendants cannot request job breakaway.

    Methods raise on containment or signalling errors. A closed job cannot be reused.
    """
    def __init__(self):
        if os.name != "nt":
            raise OSError("Windows Job Objects are required")
        self._k = ctypes.WinDLL("kernel32", use_last_error=True)
        self._nt = ctypes.WinDLL("ntdll")
        k = self._k
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k.SetInformationJobObject.restype = wintypes.BOOL
        k.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        k.QueryInformationJobObject.restype = wintypes.BOOL
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.AssignProcessToJobObject.restype = wintypes.BOOL
        k.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        k.IsProcessInJob.restype = wintypes.BOOL
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.TerminateJobObject.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        for name in ("NtSuspendProcess", "NtResumeProcess"):
            fn = getattr(self._nt, name)
            fn.argtypes = [wintypes.HANDLE]
            fn.restype = ctypes.c_long
        self.name = "Local\\LSS-Owned-" + uuid.uuid4().hex
        ctypes.set_last_error(0)
        self._handle = k.CreateJobObjectW(None, self.name)
        creation_error = ctypes.get_last_error()
        if self._handle and creation_error == 183:  # ERROR_ALREADY_EXISTS
            k.CloseHandle(self._handle)
            self._handle = None
            raise RuntimeError("Owned Job name collision")
        self._started = False
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway
        if not k.SetInformationJobObject(self._handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("Job is closed")

    def start(self, command, env=None, cwd=None, stdout=None, stderr=None, normal_priority=False):
        self._require_open()
        if self._started:
            raise RuntimeError("Job already started")
        self._started = True
        child_env = dict(os.environ if env is None else env)
        child_env[JOB_ENV] = self.name
        child = subprocess.Popen(command, env=child_env, cwd=cwd, stdout=stdout, stderr=stderr,
                                 creationflags=0x00000004 | 0x08000000 | (0x00000020 if normal_priority else 0x00000040))
        try:
            if not self._k.AssignProcessToJobObject(self._handle, int(child._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            contained = wintypes.BOOL()
            if not self._k.IsProcessInJob(int(child._handle), self._handle, ctypes.byref(contained)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not contained.value:
                raise RuntimeError("Started process is not contained in Job")
            status = self._nt.NtResumeProcess(int(child._handle))
            if status < 0:
                raise OSError(f"NtResumeProcess failed: 0x{status & 0xffffffff:08x}")
            return child
        except BaseException:
            child.kill()
            child.wait()
            self.close()
            raise

    def members(self):
        self._require_open()
        capacity = 16
        while True:
            class ProcessList(ctypes.Structure):
                _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD),
                            ("pids", ctypes.c_size_t * capacity)]
            info = ProcessList()
            if self._k.QueryInformationJobObject(self._handle, 3, ctypes.byref(info), ctypes.sizeof(info), None):
                return [int(info.pids[i]) for i in range(info.count)]
            error = ctypes.get_last_error()
            if error != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(error)
            capacity = max(capacity * 2, info.assigned + 1)
            if capacity > 1_000_000:
                raise RuntimeError("Job process list exceeded safety limit")

    def signal(self, pids, resume):
        self._require_open()
        operation = self._nt.NtResumeProcess if resume else self._nt.NtSuspendProcess
        for pid in pids:
            # A handle pins identity across the membership check and signal.
            handle = self._k.OpenProcess(0x0800 | 0x1000, False, pid)
            if not handle:
                error = ctypes.get_last_error()
                if error == 87:  # process exited since enumeration
                    continue
                raise ctypes.WinError(error)
            try:
                contained = wintypes.BOOL()
                if not self._k.IsProcessInJob(handle, self._handle, ctypes.byref(contained)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not contained.value:
                    raise PermissionError(f"PID {pid} is not in the owned Job")
                status = operation(handle)
                # Membership can race normal CLI exit. This handle still pins
                # the correct identity; an exiting process needs no signal.
                if status < 0 and status & 0xffffffff != 0xc000010a:  # STATUS_PROCESS_IS_TERMINATING
                    raise OSError(f"Process signal failed: 0x{status & 0xffffffff:08x}")
            finally:
                self._k.CloseHandle(handle)

    def close(self):
        if self._handle:
            handle, self._handle = self._handle, None
            # Closing the last handle kills the job; explicit termination handles inherited handles too.
            terminated = self._k.TerminateJobObject(handle, 1)
            error = ctypes.get_last_error() if not terminated else 0
            self._k.CloseHandle(handle)
            if error:
                raise ctypes.WinError(error)
