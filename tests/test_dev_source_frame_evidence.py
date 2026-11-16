import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.dev_source_frame_evidence import extract


ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / "bin" / "ffmpeg.exe"


@unittest.skipUnless(FFMPEG.is_file(), "bundled ffmpeg is required for this CPU-only fixture")
class SourceFrameEvidenceTest(unittest.TestCase):
    def test_filename_pts_is_png_packet_pts_on_generated_video(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".state") as temporary:
            base = Path(temporary)
            source = base / "fixture.mkv"
            generated = subprocess.run(
                [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-threads", "1", "-f", "lavfi",
                 "-i", "testsrc=size=64x48:rate=2:duration=3", "-threads", "1", "-c:v", "ffv1", str(source)],
                capture_output=True, text=True, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(source.drive.upper(), "C:")
            output = base / "evidence"
            manifest_path = extract(str(source), digest, str(FFMPEG), [1.0], str(output))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frame = manifest["frames"][0]
            self.assertTrue(frame["filename_pts_identifies_saved_frame"])
            self.assertIsInstance(frame["filename_pts_microseconds"], int)
            self.assertEqual(frame["filename_pts_microseconds"], 1_000_000)
            reference = subprocess.run(
                [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(source),
                 "-threads", "1", "-filter_threads", "1", "-vf", r"select=eq(n\,2)",
                 "-frames:v", "1", "-fps_mode", "passthrough", "-vcodec", "png", "-f", "image2pipe", "pipe:1"],
                capture_output=True, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.assertEqual((output / frame["file"]).read_bytes(), reference.stdout)
            self.assertGreater((output / frame["file"]).stat().st_size, 0)
            self.assertEqual(manifest["source_sha256_before"], manifest["source_sha256_after"])
            self.assertTrue(manifest["source_unchanged"])

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            path = Path(temporary) / "already-there"
            path.mkdir()
            with self.assertRaises(FileExistsError):
                extract("C:/unused.mkv", "0" * 64, str(FFMPEG), [1.0], str(path))


if __name__ == "__main__":
    unittest.main()
