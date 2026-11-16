"""Source-code constants and large-tree reads: verdicts stay, process count stays flat."""
import importlib.util
import os
import subprocess
import sys
import unittest
from unittest import mock

from gate_support import GUARD, RepoTest


def load_guard():
    spec = importlib.util.spec_from_file_location("public_guard_under_test", GUARD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CodeConstantTest(RepoTest):
    def test_code_constant_is_not_an_env_secret(self):
        self.admit_code("src/constants.py", ('MODEL_REVISION = "' + "a1" * 20 + '"\n').encode())
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result["counts"])

    def test_keyword_secret_in_code_is_still_blocked(self):
        self.admit_code("src/settings.py", ('api' + '_key = "' + "Z9" * 12 + '"\n').encode())
        self.blocked("ENV_SECRET")

    def test_constant_outside_code_is_still_blocked(self):
        self.write("VERSION", ('MODEL_REVISION="' + "a1" * 20 + '"\n').encode())
        self.blocked("ENV_SECRET")


class CodePathTest(RepoTest):
    """Escape sequences in source code are not paths; real drive and UNC paths in code still are."""

    def test_escaped_relative_path_and_newline_in_code_pass(self):
        body = ('SUFFIX = "' + chr(92) * 2 + "dist" + chr(92) * 2 + 'pkg"\n'
                + 'TEXT = "last 48 h:' + chr(92) + 'nrelease"\n')
        self.admit_code("src/texts.py", body.encode())
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result["counts"])

    def test_drive_and_unc_paths_in_code_are_blocked(self):
        for number, value in enumerate(('ROOT = r"E:' + chr(92) + "SyntheticMedia" + chr(92) + 'file"',
                      'ROOT = "E:' + chr(92) * 2 + "SyntheticMedia" + chr(92) * 2 + 'file"',
                      'ROOT = "e:' + chr(92) * 2 + 'syntheticmedia"',
                      'ROOT = "e' + ':/syntheticmedia"',
                      'ROOT = r"C:' + chr(92) + "Users" + chr(92) + 'SyntheticUser"',
                      'ROOT = r"' + chr(92) * 2 + "server" + chr(92) + 'share"',
                      'ROOT = "' + chr(92) * 4 + "server" + chr(92) * 2 + 'share"')):
            with self.subTest(case=number):
                self.admit_code(f"src/paths{number}.py", (value + "\n").encode())
                self.blocked("PRIVATE_PATH")
                self.write(f"src/paths{number}.py", b"ROOT = None\n")  # the next case must be blocked by itself
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result["counts"])

    def test_script_in_html_is_code(self):
        # an HTML page's script holds the same escapes: a doubled backslash before a folder is not a UNC share
        page = ("<script>const dir = 'C:" + chr(92) * 2 + "Music" + chr(92) * 2 + "Inbox';</script>\n")
        self.admit_code("src/page.html", page.encode())
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result["counts"])
        for number, value in enumerate(("'E:" + chr(92) * 2 + "SyntheticMedia'",
                                        "'C:" + chr(92) * 2 + "Users" + chr(92) * 2 + "SyntheticUser'")):
            with self.subTest(case=number):
                self.admit_code(f"src/page{number}.html", ("<script>const d = " + value + ";</script>\n").encode())
                self.blocked("PRIVATE_PATH")
                self.write(f"src/page{number}.html", b"<p></p>\n")

    def test_escaped_relative_path_outside_code_is_still_blocked(self):
        self.write("VERSION", (chr(92) * 2 + "dist" + chr(92) * 2 + "pkg\n").encode())
        self.blocked("PRIVATE_PATH")


class WordTermTest(RepoTest):
    def setUp(self):
        super().setUp()
        self.deny.write_text("word:" + "zq" + "x\n", encoding="utf-8")

    def test_word_term_blocks_the_standalone_word(self):
        self.write("VERSION", ("see " + "zq" + "x_run.wav\n").encode())
        self.blocked("DENYLIST")

    def test_word_term_ignores_a_longer_word(self):
        self.write("VERSION", ("a" + "zq" + "xb and " + "zq" + "x9\n").encode())
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result["counts"])

    def test_empty_word_term_is_a_scan_error(self):
        self.deny.write_text("word:\n", encoding="utf-8")
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 2)


class LargeTreeTest(RepoTest):
    def test_tree_scan_uses_a_constant_number_of_processes(self):
        count = 300
        for index in range(count):
            path = "src/m%03d.py" % index
            self.policy["reviewed_code_paths"].append(path)
            self.write(path, ("VALUE = %d\n" % index).encode())
        self.write_policy()
        commit = self.commit(paths=["."])
        guard = load_guard()
        spawned = []
        real_run, real_popen = subprocess.run, subprocess.Popen

        def counting_run(*args, **kwargs):
            spawned.append("run")
            return real_run(*args, **kwargs)

        class CountingPopen(real_popen):
            def __init__(self, *args, **kwargs):
                spawned.append("popen")
                super().__init__(*args, **kwargs)

        with mock.patch.dict(os.environ, self.env, clear=True), \
                mock.patch.object(guard.subprocess, "Popen", CountingPopen), \
                mock.patch.object(guard.subprocess, "run", counting_run):
            result = guard.scan(self.repo, commit)
        self.assertEqual(result["result"], "PASS", result["counts"])
        self.assertEqual(result["files_checked"], count + 2)
        self.assertLess(len(spawned), 12)


if __name__ == "__main__":
    unittest.main()
