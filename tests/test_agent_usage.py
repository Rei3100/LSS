"""scripts/agent_usage.py on a synthetic Claude transcript and Codex rollout: token sums, idle gaps before/after
the keep-alive cutoff, and the cost-equivalent shares."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "agent_usage.py"


@pytest.fixture()
def au():
    spec = importlib.util.spec_from_file_location("agent_usage_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf8")


def _asst(ts, mid, i, cw, cr, o):
    return {"type": "assistant", "timestamp": ts, "gitBranch": "feat/x",
            "message": {"id": mid, "model": "claude-test",
                        "usage": {"input_tokens": i, "cache_creation_input_tokens": cw,
                                  "cache_read_input_tokens": cr, "output_tokens": o}, "content": []}}


def _user(ts):
    return {"type": "user", "timestamp": ts, "gitBranch": "feat/x", "message": {"content": "x"}}


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    proj = tmp_path / ".claude" / "projects" / "C--work-LSS"
    _write(proj / "sess1.jsonl", [
        _user("2026-10-20T09:59:00Z"),
        _asst("2026-10-20T10:00:00Z", "m1", 10, 1000, 2000, 20),
        _asst("2026-10-20T10:00:01Z", "m1", 10, 1000, 2000, 20),  # streamed duplicate of m1
        _user("2026-10-20T12:30:00Z"),
        _asst("2026-10-20T12:31:00Z", "m2", 5, 500, 4000, 10),   # gap 1: starts before the cutoff
        _user("2026-10-24T12:00:00Z"),
        _asst("2026-10-24T12:01:00Z", "m3", 5, 800, 0, 10),      # gap 2: starts before the cutoff (10-20 12:31)
        _user("2026-10-24T15:00:00Z"),
        _asst("2026-10-24T15:01:00Z", "m4", 5, 300, 0, 10),      # gap 3: starts after the cutoff
    ])
    _write(proj / "sess1" / "subagents" / "agent-1.jsonl", [
        _asst("2026-10-20T10:05:00Z", "s1", 1, 400, 100, 40),
    ])
    _write(tmp_path / ".codex" / "sessions" / "2026" / "10" / "21" / "rollout-a.jsonl", [
        {"timestamp": "2026-10-21T08:00:00Z", "type": "session_meta",
         "payload": {"id": "cx1", "timestamp": "2026-10-21T08:00:00Z", "source": "cli", "git": {"branch": "feat/x"}}},
        {"timestamp": "2026-10-21T08:00:01Z", "type": "turn_context", "payload": {"turn_id": "t1", "model": "gpt-test", "effort": "high"}},
        {"timestamp": "2026-10-21T08:00:02Z", "type": "token_usage_record",
         "payload": {"turn_id": "t1", "response_id": "r1",
                     "usage": {"input_tokens": 1000, "cached_input_tokens": 400, "cache_write_input_tokens": 0,
                               "output_tokens": 50, "reasoning_output_tokens": 5}}},
        {"timestamp": "2026-10-21T08:00:03Z", "type": "token_usage_record",  # same response id: counted once
         "payload": {"turn_id": "t1", "response_id": "r1",
                     "usage": {"input_tokens": 1000, "cached_input_tokens": 400, "cache_write_input_tokens": 0,
                               "output_tokens": 50, "reasoning_output_tokens": 5}}},
    ])
    return tmp_path


def _run(au):
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    end = datetime(2026, 10, 30, tzinfo=timezone.utc)
    u = au.Usage(start, end)
    u.process_claude()
    u.process_codex()
    cutoff = datetime.fromisoformat("2026-10-23T03:39:19+09:00")
    return au.build(u, cutoff, 30)


def test_token_sums(au, home):
    res = _run(au)
    c = res["claude"]
    assert c["sessions_main"] == 1 and c["subagent_transcripts"] == 1
    assert c["tokens_by_role"]["main"] == dict(input=25, cache_write=2600, cache_read=6000, output=50, reasoning=0, msgs=4)
    assert c["tokens_by_role"]["subagent"]["cache_write"] == 400
    assert c["tokens_total"]["cache_write"] == 3000 and c["tokens_total"]["msgs"] == 5
    x = res["codex"]["tokens_total"]
    assert (x["input"], x["cache_read"], x["output"], x["msgs"]) == (600, 400, 50, 1)


def test_gaps_split_at_cutoff(au, home):
    g = _run(au)["idle_gaps"]
    assert g["claude"]["gaps_gt_1h"] == 3
    assert g["claude"]["before"]["gaps"] == 2 and g["claude"]["before"]["cache_write_after_gap"] == 1300
    assert g["claude"]["after"]["gaps"] == 1 and g["claude"]["after"]["cache_write_after_gap"] == 300
    last = g["list"][-1]
    assert last["start_jst"] == "2026-10-24 21:01" and last["sid"] == "sess1" and last["gap_h"] == 2.98
    assert last["cache_write_first_msg"] == 300


def test_cost_shares(au, home):
    ce = _run(au)["cost_equivalent"]["claude"]
    main = 25 + 6000 * 0.1 + 2600 * 2.0 + 50 * 5.0
    sub = 1 + 100 * 0.1 + 400 * 1.25 + 40 * 5.0
    assert ce["units"] == round(main + sub)
    assert ce["by_role_units"] == {"main": round(main), "subagent": round(sub)}
    assert ce["share"]["cache_write"] == pytest.approx((2600 * 2.0 + 400 * 1.25) / (main + sub), abs=1e-4)
    assert ce["share"]["output"] == pytest.approx((50 * 5.0 + 40 * 5.0) / (main + sub), abs=1e-4)
    assert sum(ce["share"].values()) == pytest.approx(1.0, abs=1e-3)


def test_text_report_is_compact_and_json_written(au, home, capsys, tmp_path):
    out = tmp_path / "out.json"
    rc = au.main(["--since", "2026-10-01", "--days", "9999", "--json", str(out), "--cutoff", "2026-10-23T03:39:19+09:00"])
    text = capsys.readouterr().out
    assert rc == 0 and len(text.splitlines()) <= 60
    assert "before 2 gaps" in text and "after 1 gaps" in text
    assert json.loads(out.read_text(encoding="utf8"))["idle_gaps"]["cutoff_jst"] == "2026-10-23 03:39"


def test_commands_are_masked(au):
    c = au.norm_cmd("curl --token abcdef0123456789abcdef0123456789 https://x && echo AKIAABCDEFGHIJKLMNOPQRST")  # lss-secret-ok: synthetic placeholders for the masking test
    assert "abcdef0123456789" not in c and "AKIAABCDEFGHIJKLMNOPQRST" not in c  # lss-secret-ok: same synthetic placeholders
