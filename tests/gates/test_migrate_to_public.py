"""Pinned-copy admission, source-state and destination-safety tests."""
import hashlib
import json
import sys

from gate_support import MIGRATE, RepoTest


class MigrationTests(RepoTest):
    def setUp(self):
        super().setUp()
        self.source = self.base / "source"
        self.init(self.source)
        self.commit()

    def source_file(self, path, data):
        target = self.source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def source_commit(self):
        self.git("add", "--all", repo=self.source)
        self.git("commit", "-qm", "Synthetic pinned source", repo=self.source)
        return self.git("rev-parse", "HEAD", repo=self.source).stdout.decode().strip()

    def allow_source(self, path):
        self.policy["include"].append(path)
        self.policy["reviewed_code_paths"].append(path)
        self.write_policy()
        self.commit("Synthetic destination policy")

    def migrate(self, oid):
        return self.run_cmd([sys.executable, "-I", "-B", MIGRATE,
                             "--source-repo", self.source, "--source-commit", oid,
                             "--repo", self.repo])

    def test_copies_only_admitted_pinned_blobs_and_manifest(self):
        self.allow_source("src/clean.py")
        content = b"print('synthetic')\n"
        self.source_file("src/clean.py", content)
        self.source_file("src/unreviewed.py", b"pass\n")
        self.source_file(".state/state.json", b"{}")
        self.source_file("release-evidence/private.py", b"pass\n")
        self.source_file("media/synthetic.wav", b"RIFF\0\0\0\0WAVE")
        self.source_file("models/synthetic.safetensors", b"fake")
        self.source_file(".venv/ignore.py", b"pass\n")
        oid = self.source_commit()
        # Unrelated source dirt is allowed, but it can never become a candidate.
        self.source_file("src/unreviewed.py", b"changed unreviewed\n")
        proc = self.migrate(oid)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        self.assertEqual((self.repo / "src/clean.py").read_bytes(), content)
        for path in ("src/unreviewed.py", ".state", "release-evidence", "media", "models", ".venv"):
            self.assertFalse((self.repo / path).exists())
        manifest = json.loads((self.repo / "gates/migration-manifest.json").read_text())
        self.assertEqual(manifest["source_commit"], oid)
        self.assertEqual(len(manifest["files"]), 1)
        entry = manifest["files"][0]
        self.assertEqual(entry["destination_path"], "src/clean.py")
        self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(entry["blob_oid"], self.git("rev-parse", oid + ":src/clean.py", repo=self.source).stdout.decode().strip())

    def test_refuses_dirty_destination_outside_gates(self):
        self.allow_source("src/clean.py")
        self.source_file("src/clean.py", b"pass\n")
        oid = self.source_commit()
        self.write("VERSION", b"dirty\n")
        proc = self.migrate(oid)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse((self.repo / "src").exists())
        self.assertFalse((self.repo / "gates/migration-manifest.json").exists())

    def test_refuses_untracked_destination_outside_gates(self):
        self.source_file("src/clean.py", b"pass\n")
        oid = self.source_commit()
        self.write("untracked.txt", b"untracked\n")
        self.assertNotEqual(self.migrate(oid).returncode, 0)

    def test_refuses_ignored_destination_outside_gates(self):
        self.source_file("src/clean.py", b"pass\n")
        oid = self.source_commit()
        self.write(".gitignore", b"ignored-data/\n")
        self.commit("Synthetic ignore rule", [".gitignore"])
        self.write("ignored-data/uncommitted.txt", b"uncommitted\n")
        self.assertNotEqual(self.migrate(oid).returncode, 0)

    def test_allows_ignored_venv_and_bytecode_only(self):
        self.allow_source("src/clean.py")
        self.source_file("src/clean.py", b"pass\n")
        oid = self.source_commit()
        self.write(".gitignore", b".venv/\nbin/\n.models\n__pycache__/\nignored-data/\n")
        self.commit("Synthetic ignore rules", [".gitignore"])
        self.write(".venv/pyvenv.cfg", b"home = synthetic\n")
        self.write("bin/tool.exe", b"MZ")
        self.write(".models/cache/config.yaml", b"synthetic: 1\n")
        self.write("scripts/__pycache__/tool.cpython-311.pyc", b"\x00")
        proc = self.migrate(oid)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        # a nested .venv is not the checkout's own environment
        (self.repo / "src/clean.py").unlink()
        (self.repo / "gates/migration-manifest.json").unlink()
        self.write("src/.venv/data.txt", b"uncommitted\n")
        self.write(".gitignore", b".venv/\n__pycache__/\n")
        self.commit("Synthetic ignore rules 2", [".gitignore"])
        self.assertNotEqual(self.migrate(oid).returncode, 0)

    def test_allows_dirty_gates_only(self):
        self.allow_source("src/clean.py")
        self.source_file("src/clean.py", b"pass\n")
        oid = self.source_commit()
        self.write("gates/work-in-progress.py", b"pass\n")
        proc = self.migrate(oid)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())

    def test_refuses_source_head_mismatch(self):
        self.allow_source("src/clean.py")
        self.source_file("src/clean.py", b"pass\n")
        old = self.source_commit()
        self.source_file("src/clean.py", b"changed\n")
        self.source_commit()
        self.assertNotEqual(self.migrate(old).returncode, 0)

    def test_refuses_dirty_copied_source_worktree(self):
        self.allow_source("src/clean.py")
        self.source_file("src/clean.py", b"pass\n")
        oid = self.source_commit()
        self.source_file("src/clean.py", b"dirty\n")
        self.assertNotEqual(self.migrate(oid).returncode, 0)
        self.assertFalse((self.repo / "src").exists())

    def test_refuses_source_index_changes_even_if_worktree_restored(self):
        self.allow_source("src/clean.py")
        self.source_file("src/clean.py", b"pass\n")
        oid = self.source_commit()
        self.source_file("src/clean.py", b"staged change\n")
        self.git("add", "src/clean.py", repo=self.source)
        self.source_file("src/clean.py", b"pass\n")
        self.assertNotEqual(self.migrate(oid).returncode, 0)

    def test_refuses_unpinned_revision_name(self):
        self.source_file("src/clean.py", b"pass\n")
        self.source_commit()
        self.assertNotEqual(self.migrate("HEAD").returncode, 0)

    def test_skips_media_even_if_individually_approved(self):
        self.source_file("assets/synthetic.wav", b"RIFF\0\0\0\0WAVE")
        oid = self.source_commit()
        blob = self.git("rev-parse", oid + ":assets/synthetic.wav", repo=self.source).stdout.decode().strip()
        self.policy["approved_paths"]["assets/synthetic.wav"] = {"blob_oid": blob, "reason": "Synthetic fixture"}
        self.write_policy()
        self.commit("Synthetic type approval")
        proc = self.migrate(oid)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        self.assertFalse((self.repo / "assets").exists())
        self.assertEqual(json.loads(proc.stdout)["skipped_counts"]["BINARY_OR_MEDIA"], 1)

    def test_skips_guard_rejected_text(self):
        self.allow_source("src/unsafe.py")
        self.source_file("src/unsafe.py", self.literal.encode())
        oid = self.source_commit()
        proc = self.migrate(oid)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        self.assertFalse((self.repo / "src/unsafe.py").exists())
        self.assertNotIn(self.literal, (proc.stdout + proc.stderr).decode())

    def test_missing_denylist_fails_before_writing(self):
        self.allow_source("src/clean.py")
        self.source_file("src/clean.py", b"pass\n")
        oid = self.source_commit()
        self.deny.unlink()
        self.assertNotEqual(self.migrate(oid).returncode, 0)
        self.assertFalse((self.repo / "src").exists())

    def test_never_overwrites_existing_file(self):
        self.allow_source("src/clean.py")
        self.source_file("src/clean.py", b"pass\n")
        oid = self.source_commit()
        self.write("src/clean.py", b"destination owned\n")
        self.commit("Synthetic existing destination", ["src/clean.py"])
        self.assertNotEqual(self.migrate(oid).returncode, 0)
        self.assertEqual((self.repo / "src/clean.py").read_bytes(), b"destination owned\n")


if __name__ == "__main__":
    import unittest
    unittest.main()
