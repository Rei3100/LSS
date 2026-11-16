"""Negative publication fixtures and real Git hook integration tests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from gate_support import GUARD, ROOT, RepoTest


class PublicGuardTests(RepoTest):
    def test_clean_working_tree(self):
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result)
        self.assertEqual(result["findings"], [])

    def test_delivered_sources_need_no_content_exceptions(self):
        paths = ["gates/public_guard.py", "gates/migrate_to_public.py", "gates/README.md",
                 ".githooks/pre-commit", ".githooks/pre-push", "tests/gates/gate_support.py",
                 "tests/gates/test_public_guard.py", "tests/gates/test_migrate_to_public.py"]
        self.policy["include"] = ["*"]
        self.policy["max_blob_bytes"] = 1024 * 1024
        for path in paths:
            self.policy["reviewed_code_paths"].append(path)
            self.write(path, (ROOT / path).read_bytes())
        self.write_policy()
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result)

    def test_clean_exact_commit_and_digest(self):
        oid = self.commit()
        self.write("VERSION", (self.literal + "\n").encode())
        proc, result = self.guard("--tree-ish", oid)
        self.assertEqual(proc.returncode, 0, result)
        self.assertEqual(result["commit_oid"], oid)
        self.assertEqual(result["object_oid"], oid)
        raw = self.git("show", oid + ":.publish-allow.json").stdout
        self.assertEqual(result["allowlist_sha256"], hashlib.sha256(raw).hexdigest())

    def test_tree_object(self):
        self.commit()
        tree = self.git("rev-parse", "HEAD^{tree}").stdout.decode().strip()
        proc, result = self.guard("--tree-ish", tree)
        self.assertEqual(proc.returncode, 0, result)
        self.assertEqual(result["object_oid"], tree)
        self.assertIsNone(result["commit_oid"])

    def test_denylist_content_case_insensitive_and_masked(self):
        self.write("VERSION", ("clean\n" + self.literal.upper() + "\n").encode())
        proc, result = self.blocked("DENYLIST")
        self.assertEqual(result["findings"][0]["line"], 2)
        self.assertNotIn(self.literal, (proc.stdout + proc.stderr).decode().casefold())

    def test_denylist_path_masked(self):
        # Do not put the denied path into the policy: exercise path scanning itself.
        self.write("src/" + self.literal.upper() + ".py", b"pass\n")
        proc, result = self.blocked("DENYLIST")
        self.assertNotIn(self.literal, (proc.stdout + proc.stderr).decode().casefold())
        self.assertTrue(any(f["path"].startswith("<masked-path:") for f in result["findings"]))

    def test_canary(self):
        self.write("VERSION", ("LSS-" + "CANARY-1234abcd\n").encode())
        self.blocked("CANARY")

    def test_fake_private_key(self):
        self.write("VERSION", ("-----BEGIN " + "RSA PRIVATE KEY-----\nFAKE\n").encode())
        proc, _ = self.blocked("PRIVATE_KEY")
        self.assertNotIn(b"FAKE", proc.stdout + proc.stderr)

    def test_fake_tokens(self):
        for prefix in ("gh" + "p_", "github_" + "pat_", "sk" + "-", "hf" + "_"):
            with self.subTest(prefix=prefix):
                self.write("VERSION", (prefix + "A" * 40 + "\n").encode())
                self.blocked("TOKEN")

    def test_cookie_jar(self):
        self.write("VERSION", ("# Netscape " + "HTTP Cookie File\n").encode())
        self.blocked("COOKIE")

    def test_env_long_assignment(self):
        self.write("VERSION", ("SYNTHETIC_VALUE=" + "X" * 32 + "\n").encode())
        self.blocked("ENV_SECRET")

    def test_private_paths(self):
        for value in ("C:" + chr(92) + "Users" + chr(92) + "SyntheticUser",
                      "E:" + chr(92) + "SyntheticMedia" + chr(92) + "file",
                      "C:/" + "FreeSoft/LSS",
                      chr(92) * 2 + "server" + chr(92) + "share",
                      chr(92) * 4 + "server" + chr(92) * 2 + "share"):
            with self.subTest(style=value[:2]):
                self.write("VERSION", (value + "\n").encode())
                self.blocked("PRIVATE_PATH")

    def test_quoted_env_long_value(self):
        self.write("VERSION", ('SYNTHETIC_VALUE="' + "long fake value " * 3 + '"\n').encode())
        self.blocked("ENV_SECRET")

    def test_extensions(self):
        for suffix in ("wav", "mp4", "safetensors"):
            with self.subTest(extension=suffix):
                path = "src/fake." + suffix
                self.admit_code(path, b"plain misleading data\n")
                self.blocked("BINARY_OR_MEDIA")

    def test_magic_misleading_extension(self):
        samples = {
            "wav": b"RIFF" + struct.pack("<I", 4) + b"WAVE",
            "mp4": struct.pack(">I", 16) + b"ftypisom" + b"\0" * 4,
            "safetensors": struct.pack("<Q", 2) + b"{}",
            "archive": b"PK\x03\x04fake", "executable": b"MZfake", "model": b"GGUFfake",
        }
        for name, data in samples.items():
            with self.subTest(kind=name):
                self.write("VERSION", data)
                self.blocked("BINARY_OR_MEDIA")

    def test_oid_approved_binary_is_allowed(self):
        self.pin("assets/synthetic.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result)

    def test_oid_approval_never_exempts_content(self):
        self.pin("assets/synthetic.bin", b"\x00" + self.literal.encode())
        self.blocked("DENYLIST")

    def test_binary_blob_is_read_as_its_printable_runs(self):
        # compressed bytes that happen to spell a drive prefix are not a path ...
        noise = b"\x00\x8fQ:" + b"/\x01\xfe" + (chr(92) * 3 + "e" + chr(92) + "i" + chr(92) + "q" + chr(92) * 3).encode() * 2
        self.pin("assets/noise.bin", noise)
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result)
        # ... a readable path, in single-byte or wide characters, still is
        readable = "Q:" + "/synthetic/library/file"
        share = chr(92) * 2 + "server" + chr(92) + "share"
        for name, path in (("narrow", readable.encode("ascii")), ("wide", readable.encode("utf-16-le")),
                           ("share", b"padding-" + share.encode("ascii"))):
            with self.subTest(kind=name):
                self.pin("assets/noise.bin", noise + path + b"\x00")
                self.blocked("PRIVATE_PATH")

    def test_binary_blob_non_ascii_literal_in_any_encoding(self):
        term = "\u79c1\u7684\u8a9e"
        self.deny.write_text(self.literal + "\n" + term + "\n", encoding="utf-8")
        for encoding in ("utf-8", "utf-16-le", "utf-16-be", "cp932"):
            with self.subTest(encoding=encoding):
                self.pin("assets/tagged.bin", b"\x00\x01" + term.encode(encoding) + b"\x02")
                self.blocked("DENYLIST")

    def test_changed_oid_approval(self):
        self.pin("docs/reviewed.md", b"reviewed\n")
        self.write("docs/reviewed.md", b"changed\n")
        self.blocked("OID_MISMATCH")

    def test_oversize(self):
        self.write("VERSION", b"X" * (self.policy["max_blob_bytes"] + 1))
        self.blocked("OVERSIZE")

    def test_unapproved_path(self):
        self.write("unexpected.txt", b"clean\n")
        self.blocked("UNAPPROVED_PATH")

    def test_glob_alone_is_not_approval(self):
        self.write("src/new.py", b"pass\n")
        self.blocked("UNAPPROVED_PATH")

    def test_fixed_self_exception_still_scans_policy(self):
        self.policy["note"] = self.literal
        self.write_policy()
        self.blocked("DENYLIST")

    def test_missing_denylist(self):
        self.deny.unlink()
        proc, result = self.blocked("SCAN_ERROR")
        self.assertEqual(proc.returncode, 2)

    def test_empty_denylist(self):
        self.deny.write_text("# comment only\n", encoding="utf-8")
        self.blocked("SCAN_ERROR")

    def test_invalid_policy(self):
        self.write(".publish-allow.json", b'{"schema_version":1,"schema_version":1}')
        self.blocked("SCAN_ERROR")

    def test_invalid_tree_ish(self):
        proc, result = self.guard("--tree-ish", "not-a-real-object")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("SCAN_ERROR", result["counts"])

    def test_bundled_ledger_needs_individual_approval(self):
        # ledger/ is no longer a hard exclusion: the product ships two ledgers. Each file is pinned by OID.
        self.pin("ledger/models.tsv", b"model_id\tarch\n")
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result)
        self.write("ledger/unreviewed.tsv", b"clean\n")
        self.blocked("UNAPPROVED_PATH")

    def test_private_ledger_stays_denied_even_when_pinned(self):
        self.policy["deny"].append("ledger/test_set.tsv")
        self.pin("ledger/test_set.tsv", b"song_id\n")
        self.blocked("DENIED_PATH")

    def test_pinned_python_source_in_a_models_package_is_admitted(self):
        # Upstream code names its package `models` (vendor/facing_the_music/core/models); weight folders keep the name blocked.
        self.pin("vendor/upstream/core/models/net.py", b"def forward(x):\n    return x\n")
        proc, result = self.guard()
        self.assertEqual(proc.returncode, 0, result)

    def test_models_folder_stays_denied_without_a_pin_or_for_non_source(self):
        self.write("vendor/upstream/core/models/net.py", b"pass\n")
        self.blocked("DENIED_PATH")

    def test_pinned_non_source_in_a_models_folder_stays_denied(self):
        self.pin("vendor/upstream/models/config.json", b"{}\n")
        self.blocked("DENIED_PATH")

    def test_priority_exclusion_beats_oid(self):
        self.pin("nested/release-evidence/entry.py", b"pass\n")
        self.blocked("DENIED_PATH")

    def test_tracked_symlink(self):
        oid = self.git("hash-object", "-w", "VERSION").stdout.decode().strip()
        self.git("add", ".publish-allow.json", "VERSION")
        self.git("update-index", "--add", "--cacheinfo", "120000," + oid + ",src/link.py")
        tree = self.git("write-tree").stdout.decode().strip()
        proc, result = self.guard("--tree-ish", tree)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("LINK_OR_SUBMODULE", result["counts"])

    def test_submodule(self):
        oid = self.commit()
        self.git("update-index", "--add", "--cacheinfo", "160000," + oid + ",src/submodule")
        proc, result = self.guard("--staged")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("LINK_OR_SUBMODULE", result["counts"])

    def test_junction(self):
        if os.name != "nt":
            self.skipTest("Windows junction integration only")
        target = self.base / "junction-target"
        target.mkdir()
        link = self.repo / "src-link"
        proc = self.run_cmd(["cmd", "/c", "mklink", "/J", link, target])
        self.assertEqual(proc.returncode, 0, "Cannot create test junction")
        try:
            self.blocked("LINK_OR_SUBMODULE")
        finally:
            link.rmdir()

    def install_hooks(self):
        (self.repo / ".githooks").mkdir()
        (self.repo / "gates").mkdir()
        shutil.copyfile(GUARD, self.repo / "gates/public_guard.py")
        for name in ("pre-push", "pre-commit"):
            path = self.repo / ".githooks" / name
            shutil.copyfile(ROOT / ".githooks" / name, path)
            path.chmod(0o755)
        self.git("config", "core.hooksPath", ".githooks")

    def test_precommit_checks_staged_tree_not_unstaged(self):
        self.install_hooks()
        self.git("add", ".publish-allow.json", "VERSION")
        self.write("VERSION", self.literal.encode())
        proc = self.git("commit", "-qm", "Synthetic staged clean", check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        self.git("add", "VERSION")
        self.write("VERSION", b"clean unstaged\n")
        proc = self.git("commit", "-qm", "Synthetic staged bad", check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"DENYLIST", proc.stdout + proc.stderr)

    def setup_push(self):
        self.install_hooks()
        remote = self.base / "remote.git"
        self.init(remote, bare=True)
        self.git("remote", "add", "origin", str(remote))
        return remote

    def run_push_hook(self, ref, oid):
        return subprocess.run(["sh", ".githooks/pre-push"], cwd=self.repo, env=self.env,
                              input=f"HEAD {oid} {ref} {'0' * 40}\n".encode(), capture_output=True)

    def test_hook_accepts_main_and_agent_refs_and_scans_bad_parents(self):
        self.install_hooks()
        clean = self.commit()
        for ref in ("refs/heads/main", "refs/heads/codex/review", "refs/heads/claude/review-2/sub"):
            proc = self.run_push_hook(ref, clean)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        self.write("VERSION", self.literal.encode())
        self.commit("Synthetic bad ancestor")
        self.write("VERSION", b"clean\n")
        tip = self.commit("Synthetic clean tip")
        for ref in ("refs/heads/codex/review", "refs/heads/claude/review"):
            proc = self.run_push_hook(ref, tip)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(b"DENYLIST", proc.stdout + proc.stderr)

    def test_hook_refuses_tags_other_branches_and_deletions(self):
        self.install_hooks()
        tip = self.commit()
        for ref, oid in (("refs/tags/release", tip), ("refs/heads/feature/test", tip),
                         ("refs/heads/codex/", tip), ("refs/heads/claude/-bad", tip),
                         ("refs/heads/main", "0" * 40), ("refs/heads/codex/test", "0" * 40)):
            proc = self.run_push_hook(ref, oid)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(b"ERROR", proc.stdout + proc.stderr)

    def test_push_allows_clean_and_blocks_bad_commit(self):
        remote = self.setup_push()
        clean_oid = self.commit()
        proc = self.git("push", "origin", "HEAD:refs/heads/main", check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        self.write("VERSION", self.literal.encode())
        self.commit("Synthetic rejected content")
        proc = self.git("push", "origin", "HEAD:refs/heads/main", check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"DENYLIST", proc.stdout + proc.stderr)
        self.assertEqual(self.git("rev-parse", "main", repo=remote).stdout.decode().strip(), clean_oid)

    def test_push_blocks_bad_parent_with_clean_tip(self):
        self.setup_push()
        self.commit()
        self.write("VERSION", self.literal.encode())
        self.commit("Synthetic bad parent")
        self.write("VERSION", b"clean\n")
        clean_tip = self.commit("Synthetic clean tip")
        proc, result = self.guard("--tree-ish", clean_tip)
        self.assertEqual(proc.returncode, 0, result)
        proc = self.git("push", "origin", "HEAD:refs/heads/main", check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"DENYLIST", proc.stdout + proc.stderr)

    def test_push_blocks_bad_commit_message(self):
        self.setup_push()
        self.commit(self.literal)
        proc = self.git("push", "origin", "HEAD:refs/heads/main", check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn(self.literal, (proc.stdout + proc.stderr).decode().casefold())

    def test_push_fails_with_missing_denylist(self):
        self.setup_push()
        self.commit()
        self.env["LSS_PUBLISH_DENYLIST"] = str(self.base / "missing-file")
        proc = self.git("push", "origin", "HEAD:refs/heads/main", check=False)
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    import unittest
    unittest.main()
