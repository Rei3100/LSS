from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from lss.core.session import (
    InputChangedError,
    UnsupportedChannelLayoutError,
    ensure_input_unchanged,
    is_completed,
    iter_input_files,
    mark_completed,
    validate_input,
)
from lss.core import session


def _write_wav(path, channels: int, sr: int = 44100, seconds: float = 0.1):
    frames = int(sr * seconds)
    wave = np.zeros((frames, channels), dtype="float32")
    sf.write(path, wave, sr)


def test_validate_input_accepts_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_wav(path, channels=2)
    validate_input(path)  # 例外を出さないこと


def test_validate_input_rejects_empty_audio(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.empty((0, 2)), 44100)
    with pytest.raises(ValueError, match="音声サンプルがありません"):
        validate_input(path)


def test_validate_input_rejects_51_surround(tmp_path):
    path = tmp_path / "surround.wav"
    _write_wav(path, channels=6)
    try:
        validate_input(path)
        assert False, "UnsupportedChannelLayoutErrorが送出されるはず"
    except UnsupportedChannelLayoutError as exc:
        assert "6" in str(exc)


def test_mark_completed_then_is_completed_true(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path))
    output = tmp_path / "結果.flac"
    output.write_bytes(b"output")
    mark_completed("hash1", "recipe1", source_path=tmp_path / "input.wav", outputs=[output])
    assert is_completed("hash1", "recipe1", tmp_path / "input.wav") is True


def test_is_completed_false_before_mark(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path))
    assert is_completed("hash1", "recipe1", tmp_path / "input.wav") is False


def test_input_change_is_detected_by_content_not_only_size(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"before")
    from lss.core.session import input_sha256_of

    expected = input_sha256_of(path)
    path.write_bytes(b"after!")

    with pytest.raises(InputChangedError, match="変更されました"):
        ensure_input_unchanged(path, expected)


def test_input_sha256_reports_byte_progress_and_ignores_consumer_failure(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"a" * ((1 << 20) + 7))
    events = []
    from lss.core.session import input_sha256_of

    digest = input_sha256_of(path, progress=lambda completed, total: events.append((completed, total)))

    assert len(digest) == 64
    assert events == [
        (0, (1 << 20) + 7),
        (1 << 20, (1 << 20) + 7),
        ((1 << 20) + 7, (1 << 20) + 7),
    ]
    assert input_sha256_of(
        path, progress=lambda *a: (_ for _ in ()).throw(RuntimeError("表示失敗")),
    ) == digest


def test_mark_completed_reports_each_output_hash_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    outputs = [tmp_path / "first.flac", tmp_path / "second.flac"]
    for output in outputs:
        output.write_bytes(b"output")
    events = []

    mark_completed(
        "hash1", "recipe1", source_path=tmp_path / "input.wav", outputs=outputs,
        progress=lambda path, completed, total: events.append((path, completed, total)),
    )

    assert events == [
        (outputs[0], 0, 6), (outputs[0], 6, 6),
        (outputs[1], 0, 6), (outputs[1], 6, 6),
    ]


def test_mark_completed_is_idempotent(tmp_path, monkeypatch):
    from lss.core.session import completion_ledger_path
    from lss.harness.tsv_io import load_tsv

    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path))
    output = tmp_path / "結果.flac"
    output.write_bytes(b"output")
    mark_completed("hash1", "recipe1", source_path=tmp_path / "input.wav", outputs=[output])
    mark_completed("hash1", "recipe1", source_path=tmp_path / "input.wav", outputs=[output])
    _, rows = load_tsv(completion_ledger_path())
    assert len(rows) == 1


def test_completion_time_keeps_same_day_order(tmp_path, monkeypatch):
    from lss.core.session import completion_ledger_path
    from lss.harness.tsv_io import load_tsv

    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path))
    output = tmp_path / "結果.flac"
    output.write_bytes(b"output")
    mark_completed("hash1", "recipe1", source_path=tmp_path / "input.wav", outputs=[output])

    completed_at = load_tsv(completion_ledger_path())[1][0]["completed_at"]
    parsed = datetime.fromisoformat(completed_at)
    assert "T" in completed_at
    assert parsed.tzinfo is not None


@pytest.mark.parametrize("change", ["delete", "modify", "move_source"])
def test_completion_requires_unchanged_outputs_at_same_source(tmp_path, monkeypatch, change):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path))
    source = tmp_path / "入力.wav"
    output = tmp_path / "結果.flac"
    output.write_bytes(b"original")
    mark_completed("hash1", "recipe1", source_path=source, outputs=[output])
    if change == "delete":
        output.unlink()
    elif change == "modify":
        output.write_bytes(b"modified")  # 同じサイズでも検知する。
    else:
        source = tmp_path / "別フォルダ" / source.name
    assert not is_completed("hash1", "recipe1", source)


def test_empty_output_list_cannot_be_marked_completed(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="出力がありません"):
        mark_completed("hash1", "recipe1", source_path=tmp_path / "input.wav", outputs=[])


def _zero_result(reason="no_wanted_score"):
    return {
        "reason": reason,
        "validation_result": {
            "validator_id": "bgm-semantic-gate",
            "validator_revision": "r1",
            "hard_gates_passed": True,
            "evidence": {"wanted_frames": 0},
        },
    }


def test_validated_zero_result_is_durable_and_counts_as_completed(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    source = tmp_path / "source.wav"

    session.mark_zero_result(
        "a" * 64, "bgm-recipe-r1", source_path=source,
        zero_result=_zero_result("no_wanted_score"),
    )

    assert session.is_completed("a" * 64, "bgm-recipe-r1", source)
    assert session.completed_zero_result("a" * 64, "bgm-recipe-r1", source) == _zero_result(
        "no_wanted_score"
    )
    assert session.check_completed_outputs("a" * 64, "bgm-recipe-r1", source) == {}


@pytest.mark.parametrize(
    "zero_result,message",
    [
        ({"reason": "unknown", "validation_result": {}}, "reasonが不正"),
        ({"reason": "no_wanted_score", "validation_result": {}}, "validator_id"),
        ({
            "reason": "no_wanted_score",
            "validation_result": {
                "validator_id": "validator", "validator_revision": "r1",
                "hard_gates_passed": False,
            },
        }, "hard gate通過証拠"),
    ],
)
def test_unvalidated_zero_result_cannot_be_completed(tmp_path, zero_result, message):
    with pytest.raises(ValueError, match=message):
        session.mark_zero_result(
            "a" * 64, "recipe", source_path=tmp_path / "source.wav",
            zero_result=zero_result, state_root=tmp_path / "state",
        )


def test_legacy_completion_row_does_not_skip_and_is_preserved(tmp_path, monkeypatch):
    from lss.core.session import completion_ledger_path
    from lss.harness.tsv_io import load_tsv, save_tsv

    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path))
    legacy = {"input_sha256": "hash1", "recipe_id": "recipe1", "completed_at": "2026-08-14"}
    save_tsv(completion_ledger_path(), list(legacy), [legacy])
    source = tmp_path / "入力.wav"
    assert not is_completed("hash1", "recipe1", source)
    output = tmp_path / "結果.flac"
    output.write_bytes(b"output")
    mark_completed("hash1", "recipe1", source_path=source, outputs=[output])
    assert is_completed("hash1", "recipe1", source)
    _, rows = load_tsv(completion_ledger_path())
    assert rows[0]["completed_at"] == "2026-08-14"


def test_iter_input_files_expands_folder_recursively(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    f1 = tmp_path / "a.flac"
    f2 = sub / "b.wav"
    f1.write_bytes(b"")
    f2.write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")  # 対象外拡張子

    result = iter_input_files([tmp_path])

    assert set(result) == {f1.resolve(), f2.resolve()}


def test_iter_input_files_dedupes(tmp_path):
    f1 = tmp_path / "a.flac"
    f1.write_bytes(b"")

    result = iter_input_files([f1, tmp_path])

    assert result.count(f1.resolve()) == 1


def test_folder_registration_excludes_generated_stems_but_explicit_file_is_allowed(tmp_path):
    from mutagen.flac import FLAC

    source = tmp_path / "source.flac"
    generated = tmp_path / "stem.flac"
    _write_wav(source, channels=2)
    _write_wav(generated, channels=2)
    tags = FLAC(generated)
    tags["LSS_SOURCE_TYPE"] = "separated"
    tags.save()
    assert iter_input_files([tmp_path]) == [source.resolve()]
    assert iter_input_files([generated]) == [generated.resolve()]


def test_folder_registration_includes_common_ffmpeg_audio_extensions(tmp_path):
    folder = tmp_path / "inputs"
    folder.mkdir()
    for suffix in (".aac", ".aiff", ".opus", ".wma"):
        (folder / f"input{suffix}").write_bytes(b"placeholder")

    assert iter_input_files([folder]) == [
        (folder / "input.aac").resolve(),
        (folder / "input.aiff").resolve(),
        (folder / "input.opus").resolve(),
        (folder / "input.wma").resolve(),
    ]


def _publication_paths(tmp_path, names=("one", "two")):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    pairs = []
    for name in names:
        staged = output_dir / f".lss-{name}.tmp"
        final = output_dir / f"{name}.flac"
        staged.write_bytes(f"validated-{name}".encode())
        pairs.append((staged, final))
    return pairs


def test_reconcile_completes_interrupted_multi_output_publication(tmp_path):
    state_root = tmp_path / "state"
    pairs = _publication_paths(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    journal = session.prepare_publication(
        "a" * 64, "recipe-1", source_path=source,
        outputs=[final for _, final in pairs], staged=pairs, state_root=state_root,
    )
    pairs[0][0].rename(pairs[0][1])

    result = session.reconcile_publications(state_root)

    assert result == [{"status": "completed", "journal": str(journal)}]
    assert all(final.is_file() for _, final in pairs)
    assert not journal.exists()
    assert session.is_completed("a" * 64, "recipe-1", source, state_root=state_root)


def test_forced_process_exit_between_renames_is_reconciled(tmp_path):
    state_root = tmp_path / "state"
    pairs = _publication_paths(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    code = (
        "import os,sys; from pathlib import Path; from lss.core import session; "
        "state=Path(sys.argv[1]); source=Path(sys.argv[2]); "
        "pairs=[(Path(sys.argv[3]),Path(sys.argv[4])),(Path(sys.argv[5]),Path(sys.argv[6]))]; "
        "session.prepare_publication('a'*64,'recipe-1',source_path=source,"
        "outputs=[p[1] for p in pairs],staged=pairs,state_root=state); "
        "pairs[0][0].rename(pairs[0][1]); os._exit(73)"
    )

    crashed = subprocess.run(
        [sys.executable, "-c", code, str(state_root), str(source),
         str(pairs[0][0]), str(pairs[0][1]), str(pairs[1][0]), str(pairs[1][1])],
        # Hang guard only. The crash itself is asserted by exit code 73; a cold interpreter start plus the package
        # import can exceed 10 s on a machine loaded by parallel validation lanes.
        cwd=Path(__file__).resolve().parents[1], timeout=120,
    )

    assert crashed.returncode == 73
    assert pairs[0][1].is_file() and pairs[1][0].is_file()
    assert session.reconcile_publications(state_root)[0]["status"] == "completed"
    assert all(final.is_file() for _, final in pairs)
    assert session.is_completed("a" * 64, "recipe-1", source, state_root=state_root)


def test_reconcile_rolls_back_partial_publication_when_candidate_is_missing(tmp_path):
    state_root = tmp_path / "state"
    pairs = _publication_paths(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    session.prepare_publication(
        "a" * 64, "recipe-1", source_path=source,
        outputs=[final for _, final in pairs], staged=pairs, state_root=state_root,
    )
    pairs[0][0].rename(pairs[0][1])
    pairs[1][0].unlink()

    result = session.reconcile_publications(state_root)

    assert result[0]["status"] == "rolled_back"
    assert not pairs[0][1].exists()
    assert not pairs[1][1].exists()
    assert not session.is_completed("a" * 64, "recipe-1", source, state_root=state_root)
    assert Path(result[0]["journal"]).parent == state_root / "publication_failures"


@pytest.mark.parametrize("boundary", ["read_journal", "read_audio", "rename", "ledger", "remove_journal"])
def test_reconcile_retries_io_failure_without_losing_validated_audio(tmp_path, monkeypatch, boundary):
    state_root = tmp_path / "state"
    pairs = _publication_paths(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    journal = session.prepare_publication(
        "a" * 64, "recipe-1", source_path=source,
        outputs=[final for _, final in pairs], staged=pairs, state_root=state_root,
    )
    expected = [temporary.read_bytes() for temporary, _ in pairs]
    pairs[0][0].rename(pairs[0][1])
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise PermissionError("一時的なI/O障害")

        if boundary == "read_journal":
            patch.setattr(session, "_load_publication", fail)
        elif boundary == "read_audio":
            patch.setattr(session, "input_sha256_of", fail)
        elif boundary == "rename":
            patch.setattr(Path, "rename", fail)
        elif boundary == "ledger":
            patch.setattr(session, "save_tsv", fail)
        else:
            unlink = Path.unlink
            def fail_journal(path, *args, **kwargs):
                if path == journal:
                    fail()
                return unlink(path, *args, **kwargs)
            patch.setattr(Path, "unlink", fail_journal)
        for _ in range(2):
            result = session.reconcile_publications(state_root)
            assert result[0]["status"] == "pending"
            assert journal.exists()
            assert [(final if final.exists() else temporary).read_bytes()
                    for temporary, final in pairs] == expected

    assert session.reconcile_publications(state_root)[0]["status"] == "completed"
    assert not journal.exists()
    assert [final.read_bytes() for _, final in pairs] == expected
    assert session.is_completed("a" * 64, "recipe-1", source, state_root=state_root)
    assert session.reconcile_publications(state_root) == []


def test_reconcile_never_deletes_reused_known_good_output_on_rollback(tmp_path):
    state_root = tmp_path / "state"
    pairs = _publication_paths(tmp_path, names=("new",))
    reused = tmp_path / "outputs" / "reused.flac"
    reused.write_bytes(b"known good")
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    session.prepare_publication(
        "a" * 64, "recipe-1", source_path=source,
        outputs=[reused, pairs[0][1]], staged=pairs, state_root=state_root,
    )
    pairs[0][0].unlink()

    result = session.reconcile_publications(state_root)

    assert result[0]["status"] == "rolled_back"
    assert reused.read_bytes() == b"known good"


def test_reconcile_preserves_unrecognized_final_file(tmp_path):
    state_root = tmp_path / "state"
    pairs = _publication_paths(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    session.prepare_publication(
        "a" * 64, "recipe-1", source_path=source,
        outputs=[final for _, final in pairs], staged=pairs, state_root=state_root,
    )
    pairs[0][1].write_bytes(b"not this transaction")

    result = session.reconcile_publications(state_root)

    assert result[0]["status"] == "rolled_back"
    assert pairs[0][1].read_bytes() == b"not this transaction"


def test_corrupt_publication_journal_is_quarantined_without_touching_outputs(tmp_path):
    state_root = tmp_path / "state"
    root = session.publication_dir(state_root)
    root.mkdir(parents=True)
    journal = root / "corrupt.json"
    journal.write_text("not json", encoding="utf-8")
    unrelated = tmp_path / "important.flac"
    unrelated.write_bytes(b"preserve")

    result = session.reconcile_publications(state_root)

    assert result[0]["status"] == "failed"
    assert unrelated.read_bytes() == b"preserve"
    assert not journal.exists()


def test_prepare_publication_rejects_staging_outside_final_directory(tmp_path):
    state_root = tmp_path / "state"
    staged = tmp_path / ".lss-output.tmp"
    staged.write_bytes(b"candidate")
    final_dir = tmp_path / "elsewhere"
    final_dir.mkdir()

    with pytest.raises(ValueError, match="安全な領域"):
        session.prepare_publication(
            "a" * 64, "recipe-1", source_path=tmp_path / "source.wav",
            outputs=[final_dir / "output.flac"], staged=[(staged, final_dir / "output.flac")],
            state_root=state_root,
        )


def test_cleanup_stale_work_preserves_unowned_directories_even_with_old_managed_names(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    root = session.work_dir()
    old_inference = root / "0123456789ab-old"
    old_decode = root / "decode-old"
    recent = root / "decode-recent"
    unrelated = root / "user-data"
    for path in (old_inference, old_decode, recent, unrelated):
        path.mkdir()
        (path / "data").write_bytes(b"work")
    old_time = 1000.0
    for path in (old_inference, old_decode):
        os.utime(path, (old_time, old_time))

    removed = session.cleanup_stale_work(older_than_seconds=100, now=1200.0)

    assert removed == []
    assert old_inference.exists() and old_decode.exists()
    assert recent.exists() and unrelated.exists()


def test_cleanup_stale_work_never_deletes_managed_name_regular_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    managed_name = session.work_dir() / "decode-not-a-directory"
    managed_name.write_bytes(b"preserve")
    os.utime(managed_name, (1000.0, 1000.0))

    assert session.cleanup_stale_work(older_than_seconds=100, now=1200.0) == []
    assert managed_name.read_bytes() == b"preserve"
