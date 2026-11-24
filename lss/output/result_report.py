"""Regenerable local result view, projected from Product truth. This file is never an authority."""
from __future__ import annotations

import html
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from lss.output.filenames import component

LABELS = {"PUBLISHED": "保存しました", "REUSED": "保存済みの音声を再利用しました",
          "REBUILT": "欠けていた音声を、保存済みの記録から推論せずに作り直しました",
          "RESTORED": "前の保存結果へ戻しました",
          "IMPROVED_REPLACED": "検証済みの改善へ更新しました", "HELD": "判断を保留しています",
          "ZERO": "対象なしを確認しました", "UNSUPPORTED": "現在の対応範囲外です", "FAILED": "処理に失敗しました"}


def _tempo(ratio) -> str:
    """The other occurrence's tempo against this cue, when it plays the recording faster or slower."""
    if ratio is None or abs(float(ratio) - 1.0) < 0.005:
        return ""
    return f"・テンポ{(float(ratio) - 1.0) * 100:+.1f}%"


def render(source: Path, target: str, projection: dict) -> str:
    escape = html.escape
    rows = []
    assets = {item["path"]: item for item in projection.get("assets") or []}
    for value in projection.get("paths") or []:
        path = Path(value)
        item = assets.get(value) or {}
        title = item.get("title") or path.stem
        rows.append(f'<li><a href="{escape(path.resolve().as_uri(), quote=True)}">{escape(title)}</a>'
                    f'<br><small>{escape(path.name)}</small></li>')
    known = projection.get("known") or []
    future = projection.get("future_improvement") or []
    note = ("保留した区間は非公開のまま保存しています。追加の根拠で改善できる可能性はありますが、改善は保証していません。"
            if future else "")
    kind = "BGM" if target == "bgm" else "伴奏"
    known_note = (f"既存の照合・重複の記録：{len(known)}区間。" if projection.get("known_evidence_available")
                  else "照合・重複の記録は現在確認できません。")
    available = [item for item in known if item.get("basis") == "reference_relation_decision"]
    if projection.get("reference_check") == "checked":
        known_note = (f"手元の既発売の録音と同じだった区間：{len(available)}件（重複する音声は作っていません）。"
                      if available else "手元の既発売の録音と同じだと確定した区間はありませんでした。")
        if projection.get("related_named_count"):
            known_note += f"既発売の録音の別の版と確かめた{projection['related_named_count']}件は「関連版」と仮題を付けています。"
    elif projection.get("reference_check") == "not_checked":
        known_note = "既発売の録音との照合はできませんでした（照合できない区間は既発売とみなさず、通常どおり判断しています）。"

    def clock(value):
        value = max(0, int(value or 0))
        return f"{value // 60:02d}:{value % 60:02d}"
    available_rows = "".join(
        f'<li>{escape(clock(item.get("source_start_seconds")))}–{escape(clock(item.get("source_end_seconds")))}　'
        f'{escape(item.get("display_title") or "曲名の記録が無い既発売の録音")}</li>' for item in available)
    relation_labels = {"released_recording_edit": "編集のある版",
                       "released_recording_with_additional_material": "追加の音楽を含む版",
                       "released_recording_component_variant": "音楽の構成要素が異なる版"}
    related_rows = "".join(
        f'<li>{escape(clock(item.get("source_start_seconds")))}–{escape(clock(item.get("source_end_seconds")))}　'
        f'参照録音：{escape(item.get("display_title") or "曲名の記録なし")}'
        f'<br><small>{escape(relation_labels[item["relation"]])}</small></li>'
        for item in projection.get("related_recordings") or [] if item.get("relation") in relation_labels)
    asset_rows = "".join(
        f'<li>{escape(Path(item["path"]).stem)}<br><small>{item.get("members")}回の使用から作成・'
        f'複数回で補った部分 {round(item.get("consensus_seconds") or 0)}秒</small></li>'
        for item in projection.get("recurrence_assets") or []
        if item.get("state") in ("published", "reused", "kept_previous") and item.get("path"))
    recurrence_rows = "".join(
        f'<li>{escape(item["label"])}　{escape(Path(item["path"]).stem)}{"（同じ録音の中で最も長い）" if item.get("longest") else ""}'
        f'<br><small>同じ録音：' + escape("、".join(
            f'{other.get("source_name") or "別の入力"} {clock(other.get("start_seconds"))}'
            f'（{round(other.get("duration_seconds") or 0)}秒{_tempo(other.get("tempo_ratio"))}）'
            for other in item["others"])) + '</small></li>'
        for item in projection.get("recurrence") or [])
    recurrence_note = ""
    if target == "bgm" and projection.get("cue_memory_state") == "checked":
        recurrence_note = (f"前に保存した曲と同じ録音：{projection['recurring_count']}曲。同じ録音には同じ番号（劇伴#）と"
                           "同じ仮題を付け、前の保存先の曲名もそろえました（音声は変えていません）。"
                           if projection.get("recurring_count") else "前に保存した曲と同じ録音はありませんでした。")
        assets = [item for item in projection.get("recurrence_assets") or []
                  if item.get("state") in ("published", "reused", "kept_previous") and item.get("path")]
        if assets:
            recurrence_note += (f"同じ録音が3回以上流れた部分は、各回で台詞などが重なっていない回を優先して1曲にまとめた"
                                f"「統合版」を{len(assets)}曲保存しています（各話の曲はそのまま残します）。")
    elif target == "bgm" and projection.get("cue_memory_state"):
        recurrence_note = "前に保存した曲との照合はできませんでした（曲名と音声は照合なしの場合と同じです）。"

    def count(value):
        return "未確認" if value is None else f"{value}件"
    counts = f"公開音声 {projection['published_count']}件"
    if projection.get("rebuilt_count"):
        counts += f"（うち作り直し {projection['rebuilt_count']}件）"
    if target == "bgm":
        counts += (f"／保留 {count(projection.get('held_count'))}／"
                   f"対応範囲外 {count(projection.get('unsupported_count'))}")
    return ('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="lss-result-view" content="snapshot-v1">'
            f'<meta name="lss-source-sha256" content="{escape(str(projection.get("source_sha256") or ""), quote=True)}">'
            '<title>LSS 保存結果</title><style>body{font-family:system-ui;max-width:850px;margin:40px auto;padding:0 24px;'
            'line-height:1.8}h1{font-size:26px}li{margin:20px 0}small{color:#555}a{color:#174ca0}</style>'
            f'<h1>{escape(kind)}：{escape(LABELS.get(projection["state"], projection["state"]))}</h1>'
            f'<p>入力：{escape(Path(source).name)}</p>'
            f'<p>{escape(counts)}</p><ul>{"".join(rows)}</ul>'
            f'<p>{escape(known_note)}公式曲名は、確認された名称があるときだけ使います。</p>'
            + (f'<ul>{available_rows}</ul>' if available_rows else '')
            + ('<p>手元の録音に関連する区間です。編集や別の音楽などを含むため、同じ既発売録音として扱わず、'
               '抽出・保存の判断を続けています。参照録音の曲名は、この区間全体の公式曲名を示すものではありません。</p>'
               f'<ul>{related_rows}</ul>' if related_rows else '')
            + (f'<p>{escape(recurrence_note)}</p>' if recurrence_note else '')
            + (f'<ul>{recurrence_rows}</ul>' if recurrence_rows else '')
            + (f'<ul>{asset_rows}</ul>' if asset_rows else '') +
            f'<p>{escape(note)}</p><p>音声の値は変更せず、対応する再生ソフトが使う音量情報を付けています。'
            '「仮題」は音声から得た暫定的な記述です。区切り未確定の表示もそのまま保持しています。</p>'
            f'<p>置換履歴：{projection.get("replacement_history_count", 0)}回。以前の音声と来歴は既存の履歴管理に残っています。</p>'
            '<small>この結果票はLSSの保存状態から作った表示用の記録です。公開判断や同一性の正本ではありません。</small></html>')


def path_for(source: Path, target: str, projection: dict) -> Path:
    source = Path(source)
    paths = projection.get("paths") or []
    directory = Path(paths[0]).parent if paths else source.parent / "LSS"
    source_key = projection.get("source_sha256") or hashlib.sha256(str(source.resolve()).encode()).hexdigest()
    return directory / (component(f"{source.stem} - {'BGM' if target == 'bgm' else 'Instrumental'} 結果 [{source_key[:8]}]") + ".html")


def save(source: Path, target: str, projection: dict) -> dict:
    """Expose the result next to saved assets, reachable via the existing COMPLETE saved-folder action.

    Only a result that saved audio has a place next to it. Held, absent, unsupported or failed results
    create no folder or file in the user's tree (r4 contract: no placeholder); COMPLETE shows them.
    """
    if not projection.get("paths"):
        return dict(projection)
    path = path_for(source, target, projection)
    directory = path.parent
    content = render(source, target, projection).encode("utf-8")
    temporary = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.read_bytes() == content:
            return {**projection, "report_path": str(path)}
        if path.is_file():
            previous = path.read_text(encoding="utf-8", errors="replace")
            marker = f'<meta name="lss-source-sha256" content="{projection.get("source_sha256") or ""}">'
            if '<meta name="lss-result-view" content="snapshot-v1">' not in previous or marker not in previous:
                raise OSError("同じ結果票の名前に別のファイルがあります。既存ファイルを保護しました")
        with NamedTemporaryFile(dir=directory, prefix=".lss-result-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return {**projection, "report_path": str(path)}
    except OSError as exc:
        # A readout failure does not falsify the already authoritative audio publication/completion.
        return {**projection, "report_error": str(exc)}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
