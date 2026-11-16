from lss.output import tags


def _scan_stub():
    return {
        "samplerate": 44100,
        "dynamic_range_db": 12.3,
        "track_gain_db": -4.56,
        "sample_peak_linear": 0.987654,
        "loudness_range_lu": 5.67,
        "clipped_samples": 0,
        "true_peak_position_sample": 44100 * 65,  # 1分5秒地点
        "plr_db": 9.87,
    }


def test_inherited_tags_copies_known_keys_and_title_track():
    source = {"ARTIST": "A", "ALBUM": "B", "TITLE": "曲名", "TRACK": "3", "UNKNOWN": "x"}
    result = tags.inherited_tags(source)
    assert result == {"ARTIST": "A", "ALBUM": "B", "TITLE": "曲名", "TRACK": "3"}


def test_inherited_tags_skips_missing_keys():
    result = tags.inherited_tags({"ARTIST": "A"})
    assert result == {"ARTIST": "A"}


def test_inherited_tags_preserves_multiple_values():
    source = {"ARTIST": ["A", "B"], "GENRE": ["Rock", "Live"]}

    assert tags.inherited_tags(source) == source


def test_recalculated_tags_has_all_d8_keys():
    result = tags.recalculated_tags(_scan_stub())
    expected_keys = {
        "DYNAMIC RANGE", "REPLAYGAIN_ALGORITHM", "REPLAYGAIN_TRACK_GAIN",
        "REPLAYGAIN_TRACK_PEAK", "REPLAYGAIN_TRACK_RANGE",
        "TRUEPEAK_SCANNER_CLIPPED_SAMPLES_TRACK", "TRUEPEAK_SCANNER_PEAK_POSITION",
        "TRUEPEAK_SCANNER_TRACK_PLR",
    }
    assert set(result.keys()) == expected_keys
    assert result["REPLAYGAIN_ALGORITHM"] == "ITU-R BS.1770"
    assert result["REPLAYGAIN_TRACK_GAIN"] == "-4.56 dB"
    assert result["REPLAYGAIN_TRACK_RANGE"] == "5.67 LU"
    assert result["TRUEPEAK_SCANNER_TRACK_PLR"] == "9.87 LU"
    assert result["TRUEPEAK_SCANNER_PEAK_POSITION"] == "1:05.000"


def test_new_tags_default_source_type():
    assert tags.new_tags("Instrumental") == {"LSS_STEM": "Instrumental", "LSS_SOURCE_TYPE": "separated"}


def test_provenance_tags_has_all_7_items():
    result = tags.provenance_tags(
        recipe_id="becruily_deux_inst", model="becruily_deux_inst", model_sha256="abc123",
        processed_at="2026-08-14T00:00:00+00:00", input_sha256="def456", score="sdr=2.14", version="0.1.0",
    )
    assert len(result) == 7
    assert result["LSS_MODEL_SHA256"] == "abc123"


def test_provenance_tags_records_machine_readable_runtime_revision():
    result = tags.provenance_tags(
        recipe_id="r", model="m", model_sha256="model-sha", processed_at="p",
        input_sha256="input-sha", score="score", version="v",
        config_sha256="config-sha",
        runtime_revision={"runtime_id": "runtime-1", "device": "cuda:0"},
    )

    assert result["LSS_CONFIG_SHA256"] == "config-sha"
    assert result["LSS_RUNTIME_ID"] == "runtime-1"
    assert result["LSS_RUNTIME_REVISION"] == '{"device":"cuda:0","runtime_id":"runtime-1"}'


def test_build_tags_title_track_never_overwritten_by_other_categories():
    source = {"TITLE": "元のタイトル", "TRACK": "1", "ARTIST": "A"}
    result = tags.build_tags(
        source_tags=source, stem="Vocals", scan=_scan_stub(),
        provenance=tags.provenance_tags(
            recipe_id="r", model="m", model_sha256="s", processed_at="p",
            input_sha256="i", score="sc", version="v",
        ),
    )
    assert result["TITLE"] == "元のタイトル"
    assert result["TRACK"] == "1"
    assert result["LSS_STEM"] == "Vocals"
    assert result["ARTIST"] == "A"
