import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from strata_process_job import OwnedProcessJob
from gpu_run import _run_capped


@unittest.skipUnless(os.name == "nt", "Windows Job Objects required")
class OwnedProcessJobTest(unittest.TestCase):
    def test_full_speed_still_observes_owned_members_without_suspending(self):
        child = mock.Mock()
        child.wait.return_value = 0
        observe, signal = mock.Mock(), mock.Mock()
        with mock.patch('gpu_run._quiet_io'):
            self.assertEqual(_run_capped(child, 1, 1, members=lambda: [123], signal=signal, observe=observe), 0)
        observe.assert_called_once_with([123])
        signal.assert_not_called()

    def test_full_performance_does_not_apply_background_io_priority(self):
        child = mock.Mock()
        child.wait.return_value = 0
        with mock.patch('gpu_run._quiet_io') as lower, mock.patch('gpu_run._normal_io') as normal:
            self.assertEqual(_run_capped(child, 1, 1, members=lambda: [123], background_io=False), 0)
        lower.assert_not_called()
        normal.assert_called_once()

    def test_normal_performance_restores_real_owned_process_priorities(self):
        from ctypes import wintypes
        from gpu_run import _normal_io, _quiet_io
        job = OwnedProcessJob()
        child = job.start([sys.executable, '-c', 'import time;time.sleep(10)'], normal_priority=True)
        query = job._nt.NtQueryInformationProcess
        query.argtypes = [wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG, ctypes.c_void_p]
        query.restype = wintypes.LONG
        try:
            _quiet_io([child.pid], set())
            low = wintypes.ULONG()
            self.assertEqual(query(int(child._handle), 39, ctypes.byref(low), 4, None), 0)
            self.assertEqual(low.value, 1)
            _normal_io([child.pid], {child.pid})
            for kind, expected in ((33, 2), (39, 5)):
                value = wintypes.ULONG()
                self.assertEqual(query(int(child._handle), kind, ctypes.byref(value), 4, None), 0)
                self.assertEqual(value.value, expected)
            get_priority = job._k.GetPriorityClass
            get_priority.argtypes = [wintypes.HANDLE]
            self.assertEqual(get_priority(int(child._handle)), 0x20)
        finally:
            job.close()
            child.wait(timeout=10)

    def test_signal_tolerates_only_the_terminating_status(self):
        job = OwnedProcessJob()
        child = job.start([sys.executable, '-c', 'import time;time.sleep(60)'])
        try:
            with mock.patch.object(job._nt, 'NtSuspendProcess', return_value=-1073741558):
                job.signal([child.pid], resume=False)
            with mock.patch.object(job._nt, 'NtSuspendProcess', return_value=-1073741790):
                with self.assertRaisesRegex(OSError, 'c0000022'):
                    job.signal([child.pid], resume=False)
        finally:
            job.close()
            child.wait(timeout=10)

    def test_cap_reaches_busy_orphan_after_intermediate_parent_exits(self):
        job = OwnedProcessJob()
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = None
        root = None
        try:
            burn = 'import time\nend=time.monotonic()+30\nwhile time.monotonic()<end: pass'
            intermediate = ('import subprocess,sys; p=subprocess.Popen([sys.executable,"-c",' + repr(burn) + '],'
                            'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);print(p.pid,flush=True)')
            launcher = ('import subprocess,sys,time;subprocess.run([sys.executable,"-c",' + repr(intermediate)
                        + '],check=True);time.sleep(3)')
            root = job.start([sys.executable, '-c', launcher], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pid = int(root.stdout.readline())
            handle = kernel.OpenProcess(0x1000, False, pid)
            self.assertTrue(handle)

            def cpu_seconds():
                fields = [wintypes.FILETIME() for _ in range(4)]
                self.assertTrue(kernel.GetProcessTimes(handle, *[ctypes.byref(f) for f in fields]))
                return sum((f.dwHighDateTime << 32) | f.dwLowDateTime for f in fields[2:]) / 10_000_000

            before = cpu_seconds()
            started = time.monotonic()
            self.assertEqual(_run_capped(root, .1, .1, members=job.members, signal=job.signal), 0)
            elapsed = time.monotonic() - started
            used = cpu_seconds() - before
            self.assertGreater(elapsed, 2)
            self.assertLess(used, elapsed * .35, f'orphan used {used:.3f} CPU s in {elapsed:.3f} wall s')
            self.assertIn(pid, job.members())
        finally:
            job.close()
            if handle:
                kernel.CloseHandle(handle)
            if root is not None:
                root.communicate(timeout=10)

    def test_orphan_remains_owned_and_unrelated_process_survives(self):
        flags = subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS
        control = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], creationflags=flags)
        job = OwnedProcessJob()
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        grandchild_handle = None
        try:
            code = ("import subprocess,sys; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
                    "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                    "print(p.pid,flush=True)")
            launcher = job.start([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, errors = launcher.communicate(timeout=10)
            self.assertEqual(launcher.returncode, 0, errors)
            pid = int(output)
            grandchild_handle = kernel.OpenProcess(0x00100000, False, pid)
            self.assertTrue(grandchild_handle)
            self.assertIn(pid, job.members())
            with self.assertRaises(PermissionError):
                job.signal([control.pid], resume=False)
            self.assertIsNone(control.poll())
            job.signal([pid], resume=False)
            job.signal([pid], resume=True)
            job.close()
            self.assertEqual(kernel.WaitForSingleObject(grandchild_handle, 10000), 0)
            self.assertIsNone(control.poll())
        finally:
            job.close()
            if grandchild_handle:
                kernel.CloseHandle(grandchild_handle)
            control.kill()
            control.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()

