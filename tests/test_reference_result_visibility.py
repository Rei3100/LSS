from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from lss.output import results, result_report

TYPES = ("released_recording_edit", "released_recording_with_additional_material",
         "released_recording_component_variant")


def related(kind, title="<script>参照録音</script>"):
    return {"relation": kind, "display_title": title, "reference_sha256": "r" * 64,
            "start_seconds": 1., "end_seconds": 31., "source_start_seconds": 61., "source_end_seconds": 91.,
            "extraction": "normal_decision_kept", "regions": [{"kind": "unexplained_music", "start_seconds": 29., "end_seconds": 31.}]}


def test_related_versions_are_visible_without_becoming_already_available_or_renaming_assets():
    exact = {**related("exact_released_recording", "同じ録音"), "reference_sha256": "e" * 64}
    semantic = {"references": {"known_spans": [{"display_title": "指紋候補だけ"}]},
                "reference_relations": {"state": "checked", "occurrences": [exact],
                                        "related_occurrences": [related(t) for t in TYPES] + [related("unknown")]}}
    original = deepcopy(semantic)
    value = results.project(disposition="PUBLISHED", outputs=["kept.flac"], semantic_record=semantic)
    assert value["available_count"] == 1 and value["related_recording_count"] == 3
    assert [r["relation"] for r in value["related_recordings"]] == list(TYPES)
    assert all(r["extraction"] == "normal_decision_kept" for r in value["related_recordings"])
    assert value["related_named_count"] == 0 and value["assets"] == []
    assert value["paths"] == ["kept.flac"] and semantic == original
    page = result_report.render(Path("episode.mkv"), "bgm", value)
    assert "手元の既発売の録音と同じだった区間：1件" in page
    assert "01:01–01:31" in page and "参照録音：&lt;script&gt;" in page and "<script>" not in page
    assert "区間全体の公式曲名を示すものではありません" in page
    assert all(label in page for label in ("編集のある版", "追加の音楽を含む版", "音楽の構成要素が異なる版"))


@pytest.mark.parametrize("state", [None, "not_checked"])
def test_unchecked_candidate_relations_are_not_asserted_as_recognized(state):
    semantic = {"reference_relations": {"state": state, "related_occurrences": [related(TYPES[0])]}}
    value = results.project(disposition="PUBLISHED", semantic_record=semantic)
    assert value["related_recording_count"] is None and value["related_recordings"] == []
    assert "参照録音：" not in result_report.render(Path("episode.mkv"), "bgm", value)


def test_checked_zero_and_unchecked_stay_distinct_in_session_snapshot():
    from lss.ui.web.session import _result_view

    checked = results.project(disposition="PUBLISHED", semantic_record={"reference_relations": {"state": "checked"}})
    unknown = results.project(disposition="PUBLISHED")
    assert _result_view(checked)["related_recording_count"] == 0
    assert _result_view(unknown)["related_recording_count"] is None
    assert _result_view(checked)["available_count"] == 0
    assert _result_view(unknown)["available_count"] is None
    assert "related_recordings" not in _result_view(checked)
    assert "同じだと確定した区間はありませんでした" in result_report.render(Path("episode.mkv"), "bgm", checked)


@pytest.mark.parametrize("semantic", [None, {},
    {"reference_relations": {"state": "not_checked", "error": "OSError", "occurrences": [related("exact_released_recording")]}},
    {"references": {"known_spans": [{"start_seconds": 0., "end_seconds": 5.}]}}])
def test_unverified_reference_count_stays_unknown_through_the_session_snapshot(semantic):
    from lss.ui.web.session import _result_view

    value = results.project(disposition="PUBLISHED", semantic_record=semantic)
    assert value["available_count"] is None
    assert _result_view(value)["available_count"] is None
    # A JSON snapshot must carry null, not a fabricated negative observation.
    assert json.loads(json.dumps(_result_view(value)))["available_count"] is None
    page = result_report.render(Path("episode.mkv"), "bgm", value)
    assert "同じだと確定した区間はありませんでした" not in page
    assert "既存の照合・重複の記録：0区間" not in page


def test_actual_bridge_presents_related_and_exact_counts_separately_without_a_window():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the actual bridge JavaScript check")
    bridge = Path(__file__).resolve().parents[1] / "lss/ui/web/bridge.js"
    script = r'''
const fs = require('fs'), vm = require('vm');
const ctx = {inputs: [{related_recording_count:14, available_count:0},
                     {related_recording_count:3, available_count:2, report_path:'result.html'},
                     {related_recording_count:null, available_count:0}]};
vm.runInNewContext(`let Component=class {
 renderVals(){return {ctxItems:[],rows:[{key:'one',outputs:[{key:'bgm',file:'kept.flac'}],wave:[]}]};}
};\n` + fs.readFileSync(process.argv[1], 'utf8') + `\n
globalThis.answer=inputs.map(result=>{
 const c=Object.create(Component.prototype);
 c.state={queue:[],st:{},tst:{one:{bgm:{k:'done',n:1,files:['kept.flac'],result}}}};
 c._lss={}; c.song=()=>({dir:'',zero:false});
 return c.renderVals().rows[0].outputs.map(x=>x.file).join(' | ');
});`,ctx);
process.stdout.write(JSON.stringify(ctx.answer));
'''
    completed = subprocess.run([node, "-e", script, str(bridge)], capture_output=True, check=False,
                               timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    only_related, both, unchecked = json.loads(completed.stdout)
    assert "関連する 14 区間" in only_related and "作っていません" not in only_related
    assert "結果票" not in only_related
    assert "関連する 3 区間" in both and "同じ 2 曲は作っていません" in both and "結果票" not in both
    assert "関連する" not in unchecked and "作っていません" not in unchecked
