"""番組のBGM区間と、手元の既発売・参照録音との関係を型付きで決める（重複抽出を避け、編集・追加・別版・未知は守る）。

契約は `docs/BGM.md` の「Known OSTとIdentity」「Occurrence」と `docs/PRODUCT.md` の work-avoidance。

v1は「参照の波形が混合の一部（2秒ごとに15%）を説明する」ことが区間を通して続けば `EXACT` にしていた。
これは参照が**鳴っている**証拠であって、区間の音楽が**参照だけでできている**証拠ではない。そのため、
登録していない別の曲が重なった区間や、短く差し替えた番組の編集まで既発売とみなして消していた
（`.state/validation/reference-aware-r4-independent-promotion-20261007/`）。v2は次の考え方で決め直す。

1. 安い候補の取り出し：`identity.match`（ランドマーク指紋→波形）で、参照録音ごとの整列（ずれ）を得る。
   指紋・埋め込み・名前・検索の点数は候補を出すだけで、関係の根拠にしない。
2. 音楽の説明の会計（1秒のセルごと、前後0.5秒を含む2秒の窓）：分離した**Music stem**を、整列した参照
   （16tapのFIRで音量・EQ・位相の違いを許す）と、台詞・効果音のstem（Musicへの漏れ）とで同時に説明する。
   - `joint`：説明できたエネルギーの割合。
   - `novelty`：同時の説明でも残る、参照にも台詞・効果音にも無い時間周波数成分の割合（参照に無い音高）。
     重ねられた別の曲・差し替えた部分はここに出る。分離の誤差・マスタリングの差・台詞の漏れは出にくい。
   - `absence`（v3）：整列した参照の成分のうち、番組の音（Music＋台詞・効果音のstem）に無い時間周波数成分の
     割合（フレームごとに音量を合わせてから測る。番組でのダッキングは欠けとみなさない）。ドラム・ギター・
     一つの層を抜いた版（番組だけの版）はここに出る。音量・EQの違いだけの抜粋は出にくい。
   セルは「参照で説明し尽くせる（supported）」「参照は鳴っているが別の素材がある（additional）」
   「参照の成分の一部が無い（missing_component）」「説明できない音楽（unexplained）」「判断できない
   （uncertain）」「小さすぎる（quiet）」に分ける。
3. 領域の地図：同じ録音の別のずれ（切り貼り）は、説明できないセルの範囲を参照全体と相互相関で探して
   確かめる（局所の整列）。セルごとに、どの録音のどの位置か・追加の素材か・未知の音楽かを並べる。
4. 型付きの関係（区間＝一つのOccurrenceごと）：
   - `EXACT`：一つの録音の一つのずれが、区間の音楽を途切れず説明し尽くす。台詞の山で測れないセルは少数だけ許し、
     説明できない音楽のセルを内側に含めない。区間の端の8秒以上の別の音楽は分けて通常の判定へ戻す。
     EXACTは「公開しない」判断なので、測定の分解能の外は通さない（v4）：端の吸収は判断できない・小さい・同じ録音の
     同じずれで説明できるセルだけ（説明できない音楽・追加の素材・別のずれのセルは吸収しない＝短い別の端があれば
     区間全体をEXACTにしない）。novelty は中央値・続く4セルの平均・どの1セルでも厳しい上限（-18dBの重ねを通さない）、
     台詞・効果音のstemだけが説明した割合にも上限を置く。満たせない既発売の抜粋は消さず通常の公開へ戻る。
     v5：台詞・効果音の無いセルでは、参照と漏れで説明できない残り（1 − joint）にも上限を置く（実の分離器が弱めた
     -18dBの重ねは novelty に出ないが、残りにはその音量のまま出る）。
   - `EDIT`：同じ録音の別のずれが要る（切り貼り）、または録音の使用の内側に番組固有の素材がある。
   - `ADDITIONAL`：参照は鳴っているが、参照で説明できない音楽の素材が重なる（重ね・追加の層・別ミックス）。
   - `COMPONENT`（v3）：同じ録音の構造（整列）は続くが、参照の成分の一部が番組の音に無い（ドラムレス・
     ギターレス・編成を減らした版など。追加の層を伴うこともある）。番組だけの版なので既発売にしない。
   - `RELATED`：波形では同じ録音と言えないが、和声の並びが強く対応する（別アレンジ・別演奏の候補）。
   - `UNKNOWN`：上のどれでもない。
   どの層も「一致しない」を「劇伴が無い」「未発売」の根拠にしない。
   どの成分が違うか（足された・欠けた成分の打撃／調波の割合と帯域）は、`COMPONENT`・`ADDITIONAL` の
   区間の記録へ残す（楽器の名前は言わない。音の性質の記述）。
5. 製品の動作：`EXACT` だけを既発売（`KNOWN_RELEASED_MUSIC`）にして重複の抽出・公開をしない。
   他の関係は従来の判定のまま（消さない）。領域の地図は、参照を使った再構成の材料として記録へ残す。

判断は音声の証拠だけで決め、参照の名前（タグ・ファイル名）は記録へ写すだけにする。
参照やstemを読めない・照合できない時は全区間を `UNKNOWN` のまま返し、何も消さない。
しきい値はDevelopment段階の値（`.state/validation/reference-intelligence-r1/`）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from lss.bgm import identity, versions

SCHEMA_VERSION = 4
METHOD = "reference-relations-region-accounting-v4"

EXACT = "exact_released_recording"
EDIT = "released_recording_edit"
ADDITIONAL = "released_recording_with_additional_material"
COMPONENT = "released_recording_component_variant"
RELATED = "related_version_candidate"
UNKNOWN = "unknown"
STATES = (EXACT, EDIT, ADDITIONAL, COMPONENT, RELATED, UNKNOWN)

CELL_SUPPORTED = "supported"
CELL_ADDITIONAL = "additional_material"
CELL_MISSING = "missing_component"
CELL_UNEXPLAINED = "unexplained_music"
CELL_UNCERTAIN = "uncertain"
CELL_QUIET = "quiet"

EDGE_REFERENCE = "reference_recording_edge"
REASON_EXACT = "exact_released_recording"
REASON_EDIT = "released_recording_edited_by_program"

SR = identity.SAMPLE_RATE
TAPS = 16
_HALF = TAPS // 2
_NFFT, _HOP = 1024, 256
_BAND = (int(100 * _NFFT / SR), int(5000 * _NFFT / SR))
# 区間の音楽のうち、主の録音が現れるセルの割合の下限（これ未満は区間全体の関係を言わない）。
PRESENT_MIN_FRACTION = 0.3
# 区間の音楽のうち、参照の成分の一部が無いセルがこの割合以上なら、成分の違う版（COMPONENT）とする。
COMPONENT_MIN_FRACTION = 0.2
# 成分の欠けを言うのに要る、続くセルの数（窓が編集点・替わり目を跨ぐと前後2セルに欠けが出る）。
MISSING_MIN_RUN = 3

# 参照のタグから記録へ写す項目（再生ソフトの再生回数などは写さない）。
METADATA_FIELDS = ("title", "version_info", "artist", "album", "tracknumber", "discnumber", "date", "composer")


class ReferenceRelationsError(ValueError):
    """宣言か材料が契約に合わない。"""


@dataclass(frozen=True)
class RelationRecipe:
    """関係の判定の宣言。どれかが変われば識別子が変わる。"""

    supported_joint: float          # セルを参照で説明し尽くせるとみなす同時説明の割合
    present_joint: float            # 参照が鳴っているとみなす同時説明の割合（これ未満は説明できない音楽）
    contradict_novelty: float       # 参照に無い成分の割合がこれ以上のセルは参照だけでは説明できない
    exact_max_novelty: float        # EXACTの範囲の novelty の中央値の上限
    exact_max_local_novelty: float  # EXACTの範囲の、続く exact_local_cells セルの novelty の平均の最大値の上限
    exact_max_interference_share: float  # EXACTの範囲で、台詞・効果音のstemだけが説明した割合（joint − 参照だけのjoint）の中央値の上限
    exact_max_joint_drop: float     # EXACTの範囲の、中央値のjointからこれ以上下がるセル（台詞・効果音が大きいセルを除く）を許さない
    exact_max_cell_novelty: float   # EXACTの範囲の、どの1セルの novelty の上限（端の短い差し替えを平均に埋もれさせない）
    excuse_interference_ratio: float  # 台詞・効果音のstemのエネルギーがMusicのこの割合以上のセルは、jointの落ちを台詞の山とみなす
    exact_local_cells: int          # 局所の novelty を測る、続くセルの数（短い差し替え・部分的な重ねを平均に埋もれさせない）
    contradict_absence: float       # 参照の成分が番組に無い割合がこれ以上のセルは、成分の違う版
    exact_max_absence: float        # EXACTの範囲の absence の中央値の上限
    exact_min_joint: float          # EXACTの範囲の joint の中央値の下限
    exact_min_cells: int            # EXACTに要る、参照で説明し尽くせるセルの数（秒）
    exact_clean_interference: float  # 台詞・効果音のstemのエネルギーがMusicのこの割合未満のセルを「台詞の無いセル」とする
    exact_max_clean_residual: float  # EXACTの範囲の台詞の無いセルで、参照と漏れで説明できない割合（1 − joint）の中央値の上限
    exact_max_local_clean_residual: float  # 同じく、続く exact_local_cells セルの平均の最大値の上限
    exact_clean_taps: int           # 残りを測る時の参照のFIRの長さ（放送・配信のEQの違いを説明し尽くせる長さ）
    max_uncertain_run: int          # EXACTの内側で許す、続けて判断できないセルの数
    max_uncertain_fraction: float   # EXACTの内側で許す、判断できないセルの割合
    edge_absorb_cells: int          # 区間の端で、EXACTへ含める音楽のセルの数（フェード・cueの境目の揺れ）
    split_min_seconds: int          # 区間の端の別の音楽を、別の区間として分ける最短の長さ
    quiet_db: float                 # 区間の大きいセル（90パーセンタイル）からこのdB以上小さいセルは判断しない
    local_min_ncc: float            # 局所の整列を候補にする正規化相互相関
    local_max_runs: int             # 一つの区間で局所の整列を探す範囲の数の上限
    related_min_z: float            # 和声の対応を別版の候補とする、集まりの中での外れ具合
    nomination_max_references: int = 4
    nomination_max_windows: int = 32
    nomination_top_offsets: int = 3
    nomination_min_waveform: float = 0.15
    # 対応領域が一曲として扱えない長さ（`cue_scope` の maximum_cue_seconds）を超える区間では、主の録音の続く範囲の
    # 外側にある長い音楽を、同じ録音の偶然の一致が点在していても別の区間として分ける（OP/EDと隣の曲）。
    long_split_seconds: int = 150
    long_split_max_support: float = 0.25
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise ReferenceRelationsError("未対応のschemaです")
        if not 0.0 < self.present_joint < self.supported_joint < 1.0:
            raise ReferenceRelationsError("0 < present_joint < supported_joint < 1 である必要があります")
        if not 0.0 < self.exact_max_novelty < self.contradict_novelty < 1.0:
            raise ReferenceRelationsError("0 < exact_max_novelty < contradict_novelty < 1 である必要があります")
        if not 0.0 < self.exact_max_absence < self.contradict_absence < 1.0:
            raise ReferenceRelationsError("0 < exact_max_absence < contradict_absence < 1 である必要があります")
        if not self.supported_joint <= self.exact_min_joint < 1.0:
            raise ReferenceRelationsError("exact_min_jointはsupported_joint以上1未満である必要があります")
        if not 0.0 < self.exact_max_joint_drop < 1.0 or not 0.0 < self.excuse_interference_ratio:
            raise ReferenceRelationsError("exact_max_joint_drop・excuse_interference_ratioが不正です")
        if not 0.0 <= self.exact_max_interference_share < 1.0:
            raise ReferenceRelationsError("exact_max_interference_shareは0以上1未満である必要があります")
        if not self.exact_max_novelty <= self.exact_max_local_novelty <= self.exact_max_cell_novelty < self.contradict_novelty:
            raise ReferenceRelationsError("exact_max_novelty <= exact_max_local_novelty <= exact_max_cell_novelty < contradict_novelty である必要があります")
        if (not 0.0 < self.exact_clean_interference < self.excuse_interference_ratio
                or not 0.0 < self.exact_max_clean_residual <= self.exact_max_local_clean_residual < 1.0):
            raise ReferenceRelationsError("0 < exact_clean_interference < excuse_interference_ratio、"
                                          "0 < exact_max_clean_residual <= exact_max_local_clean_residual < 1 である必要があります")
        for name, low, high in (("exact_local_cells", 1, 60), ("exact_clean_taps", TAPS, 512), ("exact_min_cells", 2, 120), ("max_uncertain_run", 0, 10),
                                ("edge_absorb_cells", 0, 10), ("split_min_seconds", 2, 120),
                                ("long_split_seconds", 10, 3600),
                                ("local_max_runs", 0, 64), ("nomination_max_references", 0, 4),
                                ("nomination_max_windows", 0, 32), ("nomination_top_offsets", 1, 3)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ReferenceRelationsError(f"{name}は{low}以上{high}以下の整数である必要があります")
        if not 0.0 <= self.long_split_max_support < 0.5:
            raise ReferenceRelationsError("long_split_max_support は0以上0.5未満である必要があります")
        if not 0.0 <= self.max_uncertain_fraction < 1.0:
            raise ReferenceRelationsError("max_uncertain_fractionは0以上1未満である必要があります")
        if not -80.0 <= self.quiet_db < 0.0:
            raise ReferenceRelationsError("quiet_dbは-80以上0未満である必要があります")
        if not 0.0 < self.local_min_ncc < 1.0:
            raise ReferenceRelationsError("local_min_nccは0より大きく1未満である必要があります")
        if not 0.15 <= self.nomination_min_waveform < 1.0:
            raise ReferenceRelationsError("nomination_min_waveform must be at least 0.15 and below 1")
        if not 0.0 < self.related_min_z <= 100.0:
            raise ReferenceRelationsError("related_min_zは0より大きく100以下である必要があります")

    def to_dict(self) -> dict:
        from lss.bgm import reference_stereo
        return {**asdict(self), "stereo_accounting": reference_stereo.policy()}

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps({**self.to_dict(), "identity_matching": identity.matching_settings_id(),
                           "versions_method": versions.METHOD_VERSION, "taps": TAPS,
                           "present_min_fraction": PRESENT_MIN_FRACTION, "alignment_limit": 8,
                           "component_min_fraction": COMPONENT_MIN_FRACTION, "absence": ABSENCE_METHOD,
                           "missing_min_run": MISSING_MIN_RUN,
                           "absence_min_frame_share": ABSENCE_MIN_FRAME_SHARE,
                           "spectrum": [_NFFT, _HOP, list(_BAND)]},
                          sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-reference-relations-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def default_recipe() -> RelationRecipe:
    """Development段階の宣言（`.state/validation/reference-intelligence-r1/`）。

    - 実番組（corpus-a 第1話）の既発売の抜粋の内側のセルは joint 0.72〜0.98・novelty 0.00〜0.15（台詞の漏れを
      同時に説明した後）。合成の台詞入りの抜粋は joint 0.9以上・novelty 0.05以下。
    - 登録した録音へ別の曲を重ねると novelty の中央値は 等音量0.43〜0.47・-6dB 0.16、差し替えたセルは1.0。
      v3の中央値0.06の上限は -6dB以上の重ねしか止めず、-12dB（0.035）は分解能の外として通していた
      （独立検証で既発売として消された）。v4は上限を下げ、分解能の外は通さない側へ倒した
      （`EXACT_MAX_NOVELTY` の註釈と `.state/validation/ri-nondestructive-claude-r1/`）。
    - 別のずれ・別の曲の境目は窓が跨ぐので、端の2セルまでは区間の揺れとして含める。
    """
    return RelationRecipe(supported_joint=0.5, present_joint=0.3, contradict_novelty=0.25,
                          exact_max_novelty=EXACT_MAX_NOVELTY, exact_max_local_novelty=EXACT_MAX_LOCAL_NOVELTY,
                          exact_max_cell_novelty=EXACT_MAX_CELL_NOVELTY,
                          exact_max_joint_drop=EXACT_MAX_JOINT_DROP, excuse_interference_ratio=EXCUSE_INTERFERENCE_RATIO,
                          exact_max_interference_share=EXACT_MAX_INTERFERENCE_SHARE,
                          exact_local_cells=EXACT_LOCAL_CELLS, contradict_absence=CONTRADICT_ABSENCE,
                          exact_max_absence=EXACT_MAX_ABSENCE, exact_min_joint=0.6, exact_min_cells=6,
                          exact_clean_interference=EXACT_CLEAN_INTERFERENCE,
                          exact_max_clean_residual=EXACT_MAX_CLEAN_RESIDUAL,
                          exact_max_local_clean_residual=EXACT_MAX_LOCAL_CLEAN_RESIDUAL,
                          exact_clean_taps=EXACT_CLEAN_TAPS,
                          max_uncertain_run=2, max_uncertain_fraction=0.2, edge_absorb_cells=2,
                          split_min_seconds=8, quiet_db=-30.0, local_min_ncc=0.5, local_max_runs=8,
                          related_min_z=versions.SEGMENT_MIN_Z)


def reference_metadata(path: Path) -> dict:
    """参照録音のタグから、決めた項目だけを読む。無いものは作らない（ファイル名は名前としては使わない）。"""
    try:
        import mutagen

        tags = mutagen.File(str(path))
        tags = getattr(tags, "tags", None)
    except Exception:  # noqa: BLE001 - タグが読めないことは関係の判定に影響させない
        tags = None
    found = {}
    if tags is not None:
        for name in METADATA_FIELDS:
            try:
                value = tags.get(name)
            except Exception:  # noqa: BLE001
                value = None
            if isinstance(value, list):
                value = value[0] if value else None
            if isinstance(value, str) and value.strip():
                found[name] = value.strip()
    # LSS自身が書いた出力（LSS_* のタグ）は、利用者が持つ既発売の録音の写しではない（自分自身と一致させない）。
    lss_output = False
    if tags is not None:
        try:
            lss_output = any("lss_" in str(key).lower() for key in tags.keys())
        except Exception:  # noqa: BLE001
            lss_output = False
    title = found.get("title")
    display = None
    if title:
        display = f"{title} [{found['version_info']}]" if found.get("version_info") else title
    return {"tags": found, "display_title": display, "title_source": "reference_file_tags" if display else None,
            "file_name": Path(path).name, "lss_output": lss_output}


@dataclass
class ReferenceLibrary:
    """一つのrunで使う参照録音の索引と、参照ごとの長さ・タグ。"""

    index: identity.ReferenceIndex
    metadata: dict[str, dict] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)
    chroma_cache: Path | None = None
    _chroma: dict[str, np.ndarray] | None = field(default=None, repr=False)

    @classmethod
    def build(cls, paths: Iterable[Path], *, cache_dir: Path | None = None,
              loader=identity.decode_mono, metadata: dict[str, dict] | None = None) -> "ReferenceLibrary":
        paths = [Path(p) for p in paths]
        index = identity.ReferenceIndex.build(paths, loader=loader, cache_dir=cache_dir)
        found, durations = {}, {}
        for reference, path in zip(index.references, paths):
            found[reference.sha256] = (metadata or {}).get(reference.sha256) or reference_metadata(path)
            durations[reference.sha256] = round(reference.frames * identity.FRAME_SECONDS, 3)
        return cls(index, found, durations, Path(cache_dir) if cache_dir is not None else None)

    def owner(self, sha256: str) -> int:
        for number, reference in enumerate(self.index.references):
            if reference.sha256 == sha256:
                return number
        raise KeyError(sha256)

    def samples(self, sha256: str) -> np.ndarray:
        return self.index.samples(self.owner(sha256))

    def chroma(self) -> dict[str, np.ndarray]:
        """参照ごとのクロマ（和声の並び）。内容SHA-256と手法の版でcacheする（再生成できる）。"""
        if self._chroma is None:
            found = {}
            for number, reference in enumerate(self.index.references):
                path = (self.chroma_cache / f"{reference.sha256}-chroma-v{versions.METHOD_VERSION}.npy"
                        if self.chroma_cache is not None else None)
                values = None
                if path is not None:
                    try:
                        values = np.load(path)
                    except (OSError, ValueError):
                        values = None
                if values is None:
                    values = versions.chroma(self.index.samples(number))
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_name(f".{path.name}.tmp.npy")
                        np.save(temporary, values)
                        temporary.replace(path)
                found[reference.sha256] = values
            self._chroma = found
        return self._chroma


# --- 音の会計（セルごとの測定） ------------------------------------------------

def _window(signal: np.ndarray, start: int, size: int) -> np.ndarray:
    """範囲外は無音で埋めた区間（float64）。"""
    out = np.zeros(max(0, size), dtype=np.float64)
    lo, hi = max(start, 0), min(start + size, signal.size)
    if hi > lo:
        out[lo - start:hi - start] = signal[lo:hi]
    return out


def _power(x: np.ndarray) -> np.ndarray:
    frames = sliding_window_view(x, _NFFT)[::_HOP] * np.hanning(_NFFT)
    return (np.abs(np.fft.rfft(frames, axis=1)) ** 2)[:, _BAND[0]:_BAND[1]]


def _novelty(y: np.ndarray, predicted: np.ndarray, interference: np.ndarray) -> float:
    """yのうち、予測（参照と漏れ）の4倍以上かつ台詞・効果音stemより大きい時間周波数成分の割合。"""
    m, p, o = _power(y), _power(predicted), _power(interference)
    total = float(m.sum())
    if total <= 0.0:
        return 0.0
    novel = (m > 4.0 * p) & (m > o)
    return float(m[novel].sum() / total)


ABSENCE_METHOD = "band-and-frame-gain-matched-reference-power-over-4x-episode-v3"
# 音量を合わせる時に、番組の音楽が参照のこの割合（エネルギー）未満しか残らないフレームは測らない
# （フェード・ダッキング・曲の終わりで、成分の欠けではなく音全体が小さい）。成分を一つ抜いた版は
# 参照の大半が残る（打撃の成分を抜いた代理で0.8前後）。
ABSENCE_MIN_FRAME_SHARE = 0.3
# 値は `.state/validation/reference-intelligence-r2/`（absence-real-r2.json・evaluate-components-r2.json）の測定。
# - 同じ録音：corpus-c 第8話の参照で説明し尽くせるセル 中央値0.004・90%点0.010・最大0.032、区間の中央値は最大0.008。
#   合成の抜粋・EQ・マスタリングの違い 0.001〜0.003。
# - 打撃の成分を抜いた代理（打撃の割合13〜20%の6曲）：区間の中央値 0.017〜0.113。
# - 境界：corpus-a 第1話の既発売曲（Anime Ver.）は v2 で EXACT、absence の中央値0.028（欠けは低域・
#   半分が打撃）。放送のミックスか別のミックスかは音だけでは言えない。既発売と断定しない側（UNKNOWNの側）へ倒す。
CONTRADICT_ABSENCE = 0.04
EXACT_MAX_ABSENCE = 0.012

# --- EXACTの novelty の上限（独立検証 Gate 1：測定の分解能の外は「既発売」にしない） ---------------------------
# 値は `.state/validation/ri-nondestructive-claude-r1/`（overlay-grid-*.json・explore*.py・REPORT.md）の測定。
# EXACTは「消す」判断なので、重ねた別の曲が分解能の外にある時は消さない側へ倒す。
# - 既発売の抜粋＋登録していない別の曲（3録音×3録音、Music stem＝混合）の novelty の中央値：-12dB 0.026〜0.040、
#   -15dB 0.012〜0.017、-18dB 0.004〜0.007、-21dB 0.001〜0.003、-24dB 以下は 0.001以下。
#   続く4セルの平均の最大値は -18dB 0.011〜0.019、-21dB 0.004〜0.009、-24dB 0.002〜0.004。
# - 台詞の漏れの無い既発売の抜粋は 0.000（セルごとの最大 0.000）。台詞を重ねた番組の実の分離（Music stem）では既発売の
#   抜粋でも中央値 0.0046〜0.0048（分離の誤差。-18dBの重ね 0.008）、実番組（corpus-a 第1話）のEXACT候補は 0.004 と 0.039。
#   実際のpackagedの分離器は小さい重ねを弱める（-12dBの重ねの中央値 0.012）。
# 上限を -18dB の重ねの最小値（0.004）の下の 0.003、局所を 0.010 とする。これを満たす既発売の抜粋だけがEXACTになり、
# -18dB より強い重ねと、台詞の漏れが多く満たせない既発売の抜粋は、消さずに公開へ戻る（冗長な公開は許す）。
# 分解能の限界は基準レベルの約 -21dB（エネルギーで0.8%）。これより弱い重ねは測れない（EXACTを通す）。
EXACT_MAX_NOVELTY = 0.003
# 続く4セルの novelty の平均の最大値：区間の一部だけの重ね・端の短い差し替え（約0.1秒以上の別の音）は中央値に埋もれる。
EXACT_MAX_LOCAL_NOVELTY = 0.01
EXACT_LOCAL_CELLS = 4
# どの1セルでも：端の 0.25秒の差し替えが窓（2秒）に入ったセルは 0.02 になる（平均には埋もれる）。台詞の漏れの無い既発売の抜粋は 0.000。
EXACT_MAX_CELL_NOVELTY = 0.015
# 台詞・効果音のstemを同時に回帰するのは、Musicへの漏れを引くため。重ねた別の曲が非音楽のstemにも入っていると、その回帰が
# 重ねた音を「説明」してしまう（独立検証の joint-accounting 反例：説明の半分を非音楽のstemが担う）。実の分離の既発売の抜粋で
# 非音楽のstemが説明した割合は 中央値0.00〜0.02・90%点0.047・最大0.127。中央値が0.1を超えるものはEXACTにしない。
EXACT_MAX_INTERFERENCE_SHARE = 0.1
# 端や内側の 0.25〜0.5秒の差し替えは、参照のその時刻のスペクトルと重なるため novelty には出にくい（0.01）が、そのセルの
# joint は中央値から 0.2〜0.4 落ちる（台詞の漏れの無い抜粋の全セルが 1.000、台詞を重ねた実の分離でも内側は中央値から
# 0.07 以内）。中央値から 0.1 以上落ちるセルがあれば、台詞・効果音が大きい（Musicの半分以上）セルを除き、EXACTにしない。
EXACT_MAX_JOINT_DROP = 0.1
EXCUSE_INTERFERENCE_RATIO = 0.5
# --- EXACTの「台詞の無いセルの残り」の上限（v5。`.state/validation/integrated-r2-claude/` の relation-cells.json） -------
# novelty は「参照より4倍以上大きい時間周波数成分」の割合なので、実際のpackagedの分離器が弱めた小さい重ねは閾値の下に
# 隠れる（corpus-c の抜粋へ別の曲を -18dB で重ねた実の分離：novelty の中央値 0.002＝既発売の抜粋と区別できず EXACT として
# 消していた）。一方、台詞・効果音の無いセルでは、同じ録音なら参照（FIR）が Music stem をほぼ完全に説明する：
# 実の分離の既発売の抜粋で 1 − joint の中央値 0.0000（-60dB以下）、合成の台詞入り番組の台詞の無いセルも 0.0000。
# 重ねた別の曲はその音量のまま残る：-18dB 0.013（-18.9dB）、-15dB 0.024、-12dB 0.043。
# 台詞の無いセル（台詞・効果音のstemが Music の5%未満）の残りの中央値を 0.005（-23dB）、続く4セルの平均を 0.008 までに
# 限る。符号化・マスタリングの差の余地を残し、-18dB の重ねを4dB以上の余裕で止める。台詞の無いセルが無い区間には
# 掛からない（台詞の山の下では分解能が約 -12dB しかなく、novelty の上限がそのまま守る＝冗長な公開へ倒れる）。
EXACT_CLEAN_INTERFERENCE = 0.05
EXACT_MAX_CLEAN_RESIDUAL = 0.005
EXACT_MAX_LOCAL_CLEAN_RESIDUAL = 0.008
# 残りは参照を128tap（11.6ms）のFIRで合わせて測る。16tapでは放送・配信のEQ（低域・高域の棚）を説明しきれず、同じ録音でも
# 残りが 0.026 になる（合成：低域-0.6・高域-0.3の棚）。64tap 0.00003・256tap 0.0000001。重ねた別の曲はFIRを伸ばしても
# 説明できない（-20dB：16tap 0.0097・256tap 0.0092、実の分離の-18dB：0.0134→0.0127）。台詞・効果音のstemは従来の16tap。
EXACT_CLEAN_TAPS = 128


_FREQS = np.arange(_BAND[0], _BAND[1]) * SR / _NFFT
# 1/3オクターブの帯（100Hz〜5kHz）。マスタリング・放送のEQの違いは帯の中で滑らかなので、帯ごとに直す。
_THIRDS = np.searchsorted(100.0 * 2 ** (np.arange(0, 18) / 3.0), _FREQS, side="right")


def _equalized(m: np.ndarray, p: np.ndarray, others: np.ndarray) -> np.ndarray:
    """参照の予測を、帯ごとの静的な音量の違い（EQ・マスタリング・放送の低域の処理）だけ番組の音へ合わせる。

    帯ごとに、時間方向の中央値で「番組の音（Music＋他のstem）÷参照」を取り、下げる向きだけ直す。
    中央値なので、打撃のように一部のフレームにだけある成分の欠けはそのまま残る（EQとはみなさない）。
    """
    total = m + others
    ratio = np.where(p > 1e-12, total / np.maximum(p, 1e-30), np.nan)
    gains = np.ones(p.shape[1])
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 参照の無い帯（全てNaN）は直さない
        per_bin = np.nanmedian(ratio, axis=0)
    for band in np.unique(_THIRDS):
        inside = _THIRDS == band
        values = per_bin[inside]
        values = values[np.isfinite(values)]
        if values.size:
            gains[inside] = float(np.clip(np.median(values), 0.0, 1.0))
    return p * gains[None, :]


def _gated(m: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """フレームごとに参照の予測の音量を番組のMusicへ合わせ（下げるだけ）、測れるフレームを選ぶ。"""
    frame_m, frame_p = m.sum(axis=1), p.sum(axis=1)
    share = frame_m / np.maximum(frame_p, 1e-20)
    measured = share >= ABSENCE_MIN_FRAME_SHARE
    return p * np.minimum(1.0, share)[:, None], measured


def _absence(y: np.ndarray, reference_part: np.ndarray, interference: np.ndarray) -> float:
    """整列した参照の成分のうち、番組の音（Music＋台詞・効果音）に無い時間周波数成分の割合。

    フレームごとに参照の予測の音量を番組のMusicへ合わせる（下げるだけ。番組でのダッキング・音量の変化を
    欠けとみなさない）。番組の音楽が参照の大半を残すフレームだけで、番組の音の4倍以上ある参照の成分を
    「無い」とする。測れるフレームが無ければ0（欠けの証拠なし）。
    """
    m, p, o = _power(y), _power(reference_part), _power(interference)
    p, measured = _gated(m, _equalized(m, p, o))
    total = float(p[measured].sum())
    if total <= 0.0:
        return 0.0
    absent = (p > 4.0 * (m + o)) & measured[:, None]
    return float(p[absent].sum() / total)


def _solve(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """最小二乗の係数（正規方程式。数値の安定のため対角へごく小さい値を足す）。"""
    gram = design.T @ design
    gram[np.diag_indices_from(gram)] += 1e-9 * (np.trace(gram) / gram.shape[0] + 1e-12)
    try:
        return np.linalg.solve(gram, design.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(design, y, rcond=None)[0]


@dataclass
class _Cells:
    """一つの整列の、セルごとの測定値。"""

    joint: np.ndarray
    novelty: np.ndarray
    inside: np.ndarray
    absence: np.ndarray
    joint_reference: np.ndarray    # 参照だけで説明できた割合（台詞・効果音のstemの回帰なし）
    interference: np.ndarray       # 台詞・効果音のstemのエネルギー ÷ Music stemのエネルギー
    scoped: bool = False
    reference_ranges: list[tuple[int, int] | None] | None = None


@dataclass(frozen=True)
class _Alignment:
    sha256: str
    offset: float              # 参照の時刻 = 番組の時刻 + offset（標本単位へ詰めた値）
    origin: str                # "fingerprint_exact" / "fingerprint" / "local_search"
    scope: tuple[int, int] | None = None


def _reference_input_bounds(first: int, shift: int, size: int) -> tuple[int, int]:
    """Half-open reference-sample range consumed by one measure() cell."""
    start = first + shift - _HALF
    return start, start + size + TAPS - 1


def _reference_ranges_conflict(left: tuple[int, int] | None,
                               right: tuple[int, int] | None) -> bool:
    """Unknown bounds conservatively conflict; known half-open ranges need overlap."""
    if left is None or right is None:
        return True
    return max(left[0], right[0]) < min(left[1], right[1])


class _Evidence:
    """区間の判定に使う、11025Hz・monoの信号（Music stem、台詞・効果音のstem）。"""

    def __init__(self, music: np.ndarray, interference: list[np.ndarray]):
        self.music = music
        self.interference = [x for x in interference if x is not None]
        self.total = (np.sum([_window(x, 0, music.size) for x in self.interference], axis=0)
                      if self.interference else np.zeros(music.size))

    def cell_energy(self, cells: list[int]) -> np.ndarray:
        values = []
        for cell in cells:
            y = _window(self.music, int(round((cell - 0.5) * SR)), 2 * SR)
            values.append(float(y @ y))
        return np.asarray(values)

    def measure(self, reference: np.ndarray, offset: float, cells: list[int]) -> _Cells:
        shift = int(round(offset * SR))
        full = 2 * SR
        # 測る範囲（cellsの最初の秒の頭〜最後の秒の終わり）。端のセルの窓は範囲の外へ0.5秒はみ出すが、はみ出した先は
        # この区間の音ではない（隣の音楽・次の台詞）ので、窓を範囲に切り詰める。切り詰めた窓が短すぎる時は切らない。
        lo, hi = cells[0] * SR, (cells[-1] + 1) * SR
        joint, novelty, inside, absence, joint_reference, interference, reference_ranges = [], [], [], [], [], [], []
        for cell in cells:
            first = int(round((cell - 0.5) * SR))
            size = full
            if min(first + full, hi) - max(first, lo) >= SR:
                size = min(first + full, hi) - max(first, lo)
                first = max(first, lo)
            y = _window(self.music, first, size)
            energy = float(y @ y)
            reference_start, reference_stop = _reference_input_bounds(first, shift, size)
            reference_ranges.append((reference_start, reference_stop))
            x = _window(reference, reference_start, reference_stop - reference_start)
            within = (first + shift >= -SR // 2) and (first + shift + size <= reference.size + SR // 2)
            inside.append(bool(within))
            if energy < 1e-9 * size:
                joint.append(np.nan)
                novelty.append(np.nan)
                absence.append(np.nan)
                joint_reference.append(np.nan)
                interference.append(np.nan)
                continue
            has_reference = bool(np.any(x))
            regressors = [x] if has_reference else []
            regressors += [_window(s, first - _HALF, size + TAPS - 1) for s in self.interference]
            other = _window(self.total, first, size)
            interference.append(float(other @ other) / energy)
            if regressors and has_reference:
                design = np.hstack([sliding_window_view(r, TAPS) for r in regressors])
                coefficients = _solve(design, y)
                predicted = design @ coefficients
                residual = y - predicted
                joint.append(max(0.0, 1.0 - float(residual @ residual) / energy))
                if self.interference:
                    reference_only = design[:, :TAPS]
                    leftover = y - reference_only @ _solve(reference_only, y)
                    joint_reference.append(max(0.0, 1.0 - float(leftover @ leftover) / energy))
                else:
                    joint_reference.append(joint[-1])
                absence.append(_absence(y, design[:, :TAPS] @ coefficients[:TAPS], other))
            else:
                predicted = np.zeros_like(y)
                joint.append(0.0)
                joint_reference.append(0.0)
                absence.append(np.nan)
            novelty.append(_novelty(y, predicted, other))
        return _Cells(np.asarray(joint, float), np.asarray(novelty, float), np.asarray(inside, bool),
                      np.asarray(absence, float), np.asarray(joint_reference, float),
                      np.asarray(interference, float), reference_ranges=reference_ranges)

    def clean_residual(self, reference: np.ndarray, offset: float, cells: list[int], span: tuple[int, int],
                       taps: int, max_interference: float) -> list[float]:
        """台詞・効果音の無いセルだけで、参照（長いFIR）と漏れ（16tap）で説明できない割合（1 − joint）。

        窓は `measure` と同じく、区間 [span[0], span[1]) の秒の外へはみ出す分を切り詰める。"""
        shift = int(round(offset * SR))
        full = 2 * SR
        lo, hi = span[0] * SR, span[1] * SR
        half = taps // 2
        values = []
        for cell in cells:
            first = int(round((cell - 0.5) * SR))
            size = full
            if min(first + full, hi) - max(first, lo) >= SR:
                size = min(first + full, hi) - max(first, lo)
                first = max(first, lo)
            y = _window(self.music, first, size)
            energy = float(y @ y)
            if energy < 1e-9 * size:
                continue
            other = _window(self.total, first, size)
            if float(other @ other) / energy >= max_interference:
                continue
            x = _window(reference, first + shift - half, size + taps - 1)
            if not np.any(x):
                continue
            regressors = [sliding_window_view(x, taps)]
            regressors += [sliding_window_view(_window(s, first - _HALF, size + TAPS - 1), TAPS)
                           for s in self.interference]
            design = np.hstack(regressors)
            residual = y - design @ _solve(design, y)
            values.append(max(0.0, float(residual @ residual) / energy))
        return values

    def character(self, reference: np.ndarray, offset: float, cells: list[int], labels: list[str]) -> dict:
        """足された成分（番組にあって参照に無い）と欠けた成分（参照にあって番組に無い）の音の性質。

        打撃／調波は時間方向・周波数方向の中央値の比（HPSSの考え方）で分ける。帯域は100–300Hz・
        300–2000Hz・2–5kHz。楽器の名前は言わない（音の性質の記述だけ）。
        """
        from scipy.ndimage import median_filter

        shift = int(round(offset * SR))
        size = 2 * SR
        sums = {"added": np.zeros(4), "missing": np.zeros(4)}
        totals = {"added": 0.0, "missing": 0.0, "music": 0.0, "reference": 0.0}
        freqs = np.arange(_BAND[0], _BAND[1]) * SR / _NFFT
        bands = [(freqs < 300), (freqs >= 300) & (freqs < 2000), (freqs >= 2000)]
        for cell, label in zip(cells, labels):
            if label == CELL_QUIET:
                continue
            first = int(round((cell - 0.5) * SR))
            y = _window(self.music, first, size)
            x = _window(reference, first + shift - _HALF, size + TAPS - 1)
            if not np.any(x) or float(y @ y) < 1e-9 * size:
                continue
            regressors = [x] + [_window(s, first - _HALF, size + TAPS - 1) for s in self.interference]
            design = np.hstack([sliding_window_view(r, TAPS) for r in regressors])
            coefficients = _solve(design, y)
            part = design[:, :TAPS] @ coefficients[:TAPS]
            other = _window(self.total, first, size)
            m, p, o = _power(y), _power(part), _power(other)
            p_scaled, measured = _gated(m, _equalized(m, p, o))
            totals["music"] += float(m.sum())
            totals["reference"] += float(p_scaled[measured].sum())
            for kind, power, mask in (("added", m, (m > 4.0 * p) & (m > o)),
                                      ("missing", p_scaled, (p_scaled > 4.0 * (m + o)) & measured[:, None])):
                if not mask.any():
                    continue
                harmonic = median_filter(power, size=(9, 1), mode="nearest")
                percussive = median_filter(power, size=(1, 9), mode="nearest")
                share = percussive ** 2 / np.maximum(harmonic ** 2 + percussive ** 2, 1e-30)
                energy = power * mask
                totals[kind] += float(energy.sum())
                sums[kind][0] += float((energy * share).sum())
                for number, band in enumerate(bands, start=1):
                    sums[kind][number] += float(energy[:, band].sum())
        result = {}
        for kind, base in (("added", "music"), ("missing", "reference")):
            energy = totals[kind]
            if totals[base] <= 0.0:
                result[kind] = None
                continue
            result[kind] = {"energy_share": round(energy / totals[base], 3)}
            if energy > 0.0:
                result[kind].update(
                    percussive_share=round(sums[kind][0] / energy, 3),
                    bands={name: round(sums[kind][number] / energy, 3) for number, name in
                           enumerate(("low_100_300hz", "mid_300_2000hz", "high_2000_5000hz"), start=1)})
        return result


def _fine_offset(evidence: _Evidence, reference: np.ndarray, offset: float, span: tuple[float, float]) -> float:
    """フレーム単位のずれを、照合した範囲のMusic stemで標本単位へ一度だけ詰める。"""
    first, last = int(span[0] * SR), int(min(span[1], span[0] + 20.0) * SR)
    shift = int(round(offset * SR))
    y = _window(evidence.music, first, last - first)
    x = _window(reference, first + shift, last - first)
    if y.size < identity._BLOCK_SAMPLES or not np.any(x) or not np.any(y):
        return float(offset)
    return float(offset - identity._fine_lag(y, x) / SR)


def _local_search(music: np.ndarray, reference: np.ndarray, start: float, end: float,
                  min_ncc: float, limit: int = 3) -> list[tuple[float, float]]:
    """番組の [start, end) のMusic stemが、参照のどこに最もよく重なるか（正規化相互相関の上位）。

    返り値は（offset = 参照の時刻 - 番組の時刻, 相関）。参照より長い・無音の窓は探さない。
    """
    y = _window(music, int(round(start * SR)), int(round((end - start) * SR)))
    n = y.size
    norm = float(np.sqrt(y @ y))
    if n < SR or norm <= 0.0 or reference.size < n:
        return []
    size = 1 << int(np.ceil(np.log2(reference.size + n)))
    corr = np.fft.irfft(np.fft.rfft(reference, size) * np.conj(np.fft.rfft(y, size)), size)[:reference.size - n + 1]
    cumulative = np.concatenate([[0.0], np.cumsum(reference.astype(np.float64) ** 2)])
    energy = np.sqrt(np.maximum(cumulative[n:] - cumulative[:-n], 0.0))
    ncc = np.abs(corr) / np.maximum(energy * norm, 1e-12)
    found: list[tuple[int, float]] = []
    for lag in np.argsort(-ncc)[:5000]:
        value = float(ncc[lag])
        if value < min_ncc or len(found) >= limit:
            break
        if all(abs(int(lag) - other) > SR // 2 for other, _ in found):
            found.append((int(lag), value))
    return [(lag / SR - start, value) for lag, value in found]


def _alignments(matches: list[identity.RecordingMatch], start: float, end: float,
                limit: int = 8) -> list[tuple[_Alignment, tuple]]:
    """区間に掛かる照合の整列。波形まで確かめた整列は全部、指紋だけの整列は票の多い順に上限まで。"""
    found: dict[tuple[str, float], tuple[_Alignment, tuple, int]] = {}
    for match in matches:
        if min(end, match.query_end_seconds) - max(start, match.query_start_seconds) <= 0:
            continue
        key = (match.reference_sha256, round(match.offset_seconds, 1))
        exact = match.identity_state == "known_exact"
        current = found.get(key)
        if current is None or (exact and current[0].origin != "fingerprint_exact"):
            found[key] = (_Alignment(match.reference_sha256, float(match.offset_seconds),
                                     "fingerprint_exact" if exact else "fingerprint"),
                          (float(match.query_start_seconds), float(match.query_end_seconds)), int(match.votes))
    exact = [key for key in found if found[key][0].origin == "fingerprint_exact"]
    rest = sorted((key for key in found if key not in exact), key=lambda k: (-found[k][2], k))
    chosen = sorted(exact + rest[:max(0, limit - len(exact))])
    return [found[key][:2] for key in chosen]


def _runs(mask) -> list[tuple[int, int]]:
    """真が続く範囲（開始, 終了の次）。"""
    runs, start = [], None
    for index, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


class _Relator:
    def __init__(self, library: ReferenceLibrary, matches: list[identity.RecordingMatch], recipe: RelationRecipe,
                 evidence: _Evidence | None, music_for_chroma: np.ndarray | None):
        self.library, self.matches, self.recipe = library, matches, recipe
        self.evidence, self.chroma_music = evidence, music_for_chroma
        self.unavailable: dict[str, str] = {}

    def _samples(self, sha256: str) -> np.ndarray | None:
        """波形の確認に使う参照の音声。重い音源が無く取り直せなければNone（その整列は根拠にしない）。"""
        from lss.bgm.reference_store import ReferenceMediaUnavailable

        if sha256 in self.unavailable:
            return None
        try:
            samples = self.library.samples(sha256)
        except (ReferenceMediaUnavailable, OSError, identity.DecodeError) as exc:
            self.unavailable[sha256] = type(exc).__name__
            return None
        missing = getattr(self.library.index.loader, "unavailable", None) or {}
        if sha256 in missing:
            self.unavailable[sha256] = "ReferenceMediaUnavailable"
            return None
        return samples

    def reference_view(self, sha256: str) -> dict:
        from lss.bgm import release_availability
        meta = self.library.metadata.get(sha256) or {}
        return {"reference_sha256": sha256, "display_title": meta.get("display_title"),
                "title_source": meta.get("title_source"), "tags": meta.get("tags") or {},
                "file_name": meta.get("file_name"), "duration_seconds": self.library.durations.get(sha256),
                "independent_distribution": release_availability.lookup(sha256, meta)}

    def _label(self, cells: _Cells, index: int) -> str:
        recipe = self.recipe
        joint, novelty, absence = cells.joint[index], cells.novelty[index], cells.absence[index]
        if not np.isfinite(joint):
            return CELL_QUIET
        missing = bool(np.isfinite(absence) and absence >= recipe.contradict_absence)
        if cells.inside[index] and joint >= recipe.supported_joint and novelty < recipe.contradict_novelty:
            # 参照で説明し尽くせても、参照の成分の一部が番組に無ければ同じ版ではない（抜いた版）。
            return CELL_MISSING if missing else CELL_SUPPORTED
        if cells.inside[index] and joint >= recipe.present_joint and novelty >= recipe.contradict_novelty:
            return CELL_ADDITIONAL
        if joint < recipe.present_joint or not cells.inside[index]:
            return CELL_UNEXPLAINED
        return CELL_UNCERTAIN

    def _nominate(self, cells, candidates, quiet):
        """Bounded opportunity search; every observation remains inside its two-second scope."""
        recipe = self.recipe
        # Candidates retain the original fingerprint vote order, before offset collapse/limit.
        raw = sorted((m for m in self.matches
                      if m.query_end_seconds > cells[0] and m.query_start_seconds < cells[-1] + 1),
                     key=lambda m: (-m.votes, m.reference_sha256, m.offset_seconds))
        references = list(dict.fromkeys(m.reference_sha256 for m in raw))[:recipe.nomination_max_references]
        starts = list(range(cells[0], cells[-1], 2))
        selected = ([starts[i] for i in np.linspace(0, len(starts) - 1,
                     min(len(starts), recipe.nomination_max_windows), dtype=int)] if starts else [])
        report = {"method": "bounded_scoped_local_v1", "window_seconds": 2,
                  "reference_budget": recipe.nomination_max_references,
                  "windows_per_reference_budget": recipe.nomination_max_windows,
                  "offsets_per_window_budget": recipe.nomination_top_offsets,
                  "references": len(references), "searched_windows": 0, "waveform_checks": 0,
                  "accepted_scopes": 0, "budget_exhausted": len(selected) < len(starts)
                  or len(set(m.reference_sha256 for m in raw)) > len(references)}
        measured = []
        for sha256 in references:
            reference = self._samples(sha256)
            if reference is None:
                continue
            for first in selected:
                if all(quiet[c - cells[0]] for c in (first, first + 1)):
                    continue
                report["searched_windows"] += 1
                for offset, _ in _local_search(self.evidence.music, reference, first, first + 2,
                                               recipe.local_min_ncc, recipe.nomination_top_offsets):
                    report["waveform_checks"] += 1
                    _, values, _ = identity.waveform_blocks(
                        self.evidence.music, reference, first * SR, (first + 2) * SR,
                        int(round(offset * SR)), fine=False, block_samples=2 * SR)
                    if not values.size or float(np.min(values)) < recipe.nomination_min_waveform:
                        continue
                    scope = [first, first + 1]
                    observed = self.evidence.measure(reference, offset, scope)
                    # Zero joint/false inside means unobserved music, never quiet or supported.
                    full = _Cells(np.zeros(len(cells)), np.ones(len(cells)), np.zeros(len(cells), bool),
                                  np.full(len(cells), np.nan), np.zeros(len(cells)),
                                  np.full(len(cells), np.nan), scoped=True,
                                  reference_ranges=[None] * len(cells))
                    where = slice(first - cells[0], first - cells[0] + 2)
                    for name in ("joint", "novelty", "inside", "absence", "joint_reference", "interference"):
                        getattr(full, name)[where] = getattr(observed, name)
                    full.reference_ranges[where] = (
                        list(observed.reference_ranges) if observed.reference_ranges is not None
                        else [None] * len(observed.joint))
                    measured.append((_Alignment(sha256, offset, "scoped_local_nomination", (first, first + 2)), full))
                    report["accepted_scopes"] += 1
                    break
        return measured, report

    def relate(self, start: int, end: int, depth: int = 0) -> list[tuple[int, int, dict]]:
        """[start, end) を関係ごとの片へ。返り値は（開始, 終了, 関係）。"""
        recipe = self.recipe
        cells = list(range(start, end))
        candidates = _alignments(self.matches, start, end)
        if self.evidence is None or len(cells) < 2 or not candidates or depth > 8:
            return [(start, end, self.unrelated(start, end, candidates))]
        energy = self.evidence.cell_energy(cells)
        loud = float(np.percentile(energy, 90)) if energy.size else 0.0
        if not any(self._samples(a.sha256) is not None for a, _ in candidates):
            relation = self.unrelated(start, end, candidates)
            relation["basis"] = "reference_waveform_unavailable"
            return [(start, end, relation)]
        quiet = energy < loud * 10 ** (recipe.quiet_db / 10.0)

        measured: list[tuple[_Alignment, _Cells]] = []
        for alignment, span in candidates:
            reference = self._samples(alignment.sha256)
            if reference is None:
                continue
            offset = _fine_offset(self.evidence, reference, alignment.offset, span)
            if any(a.sha256 == alignment.sha256 and abs(a.offset - offset) <= 0.05 for a, _ in measured):
                continue
            measured.append((_Alignment(alignment.sha256, offset, alignment.origin),
                             self.evidence.measure(reference, offset, cells)))

        def labels_of(cells_of: _Cells) -> list[str]:
            labels = [CELL_QUIET if quiet[i] else self._label(cells_of, i) for i in range(len(cells))]
            # 欠けは続く時だけ言う。2セルまでの欠けは、窓（2秒）が曲の替わり目・編集点を跨いだ跡とみなす。
            for first, last in ([] if cells_of.scoped else _runs([label == CELL_MISSING for label in labels])):
                if last - first < MISSING_MIN_RUN:
                    labels[first:last] = [CELL_SUPPORTED] * (last - first)
            return labels

        def strength(item):
            labels = labels_of(item[1])
            supported = [i for i, label in enumerate(labels) if label == CELL_SUPPORTED]
            return (len(supported), labels.count(CELL_ADDITIONAL) + labels.count(CELL_MISSING),
                    float(np.nansum(item[1].joint[supported])) if supported else 0.0,
                    item[0].origin == "fingerprint_exact", -abs(item[0].offset), item[0].sha256)

        measured.sort(key=strength, reverse=True)
        nomination = None

        def cell_map_for(items, scoped: bool):
            """Make the ordinary or scoped map from already measured evidence."""
            per = [(a, c, labels_of(c)) for a, c in items]
            main_alignment, _, main_cell_labels = per[0]
            mapping: list[tuple[str, int | None]] = []
            for i in range(len(cells)):
                missing_ranges = [(a.sha256, c.reference_ranges[i]
                                   if c.reference_ranges is not None else None)
                                  for a, c, labels in per
                                  if c.scoped and labels[i] == CELL_MISSING]

                def contradicted(alignment, evidence_cells):
                    candidate_range = (evidence_cells.reference_ranges[i]
                                       if evidence_cells.reference_ranges is not None else None)
                    return any(sha256 == alignment.sha256
                               and _reference_ranges_conflict(missing_range, candidate_range)
                               for sha256, missing_range in missing_ranges)

                if main_cell_labels[i] == CELL_SUPPORTED and not contradicted(main_alignment, per[0][1]):
                    mapping.append((CELL_SUPPORTED, 0))
                    continue
                if quiet[i]:
                    mapping.append((CELL_QUIET, None))
                    continue
                options = [(n, a, c) for n, (a, c, labels) in enumerate(per)
                           if n > 0 and labels[i] == CELL_SUPPORTED and not contradicted(a, c)]
                if options:
                    options.sort(key=lambda o: (o[1].sha256 == main_alignment.sha256,
                                                float(o[2].joint[i])), reverse=True)
                    mapping.append((CELL_SUPPORTED, options[0][0]))
                    continue
                if scoped:
                    observed = [(n, labels[i]) for n, (_, _, labels) in enumerate(per)
                                if labels[i] in (CELL_ADDITIONAL, CELL_MISSING)]
                    if observed:
                        observed.sort(key=lambda item: (per[item[0]][1].scoped
                                                        and item[1] == CELL_MISSING), reverse=True)
                        mapping.append((observed[0][1], None))
                        continue
                mapping.append((main_cell_labels[i], None))
            return mapping, per

        def enough(items):
            # The ordinary route needs a sufficient individual alignment below.
            # A union of short fragments cannot satisfy that prerequisite and
            # must not suppress the scoped fallback before the early return.
            for _, observed in items:
                labels = labels_of(observed)
                present = sum(label in (CELL_SUPPORTED, CELL_ADDITIONAL, CELL_MISSING) for label in labels)
                unexplained = _runs([label == CELL_UNEXPLAINED for label in labels])
                # Presence in one alignment is not accounting for other music.
                # A short unexplained run at a cue boundary can be an ordinary
                # edge transition; interior runs remain an opportunity for
                # bounded local observations. Keep fully present overlays and
                # component relations on the ordinary path.
                interior_unexplained = any(
                    not ((first == 0 and last <= recipe.edge_absorb_cells)
                         or (last == len(labels) and len(labels) - first <= recipe.edge_absorb_cells))
                    for first, last in unexplained)
                if (present >= recipe.exact_min_cells
                        and present >= PRESENT_MIN_FRACTION * int((~quiet).sum())
                        and not interior_unexplained):
                    return True
            # Preserve an ordinary donor only when the exact downstream clean
            # region policy would already accept it. Presence by itself still
            # cannot suppress exploration of an unexplained span.
            ordinary_map, ordinary_per = cell_map_for(items, scoped=False)
            ordinary_main = items[0][0]
            ordinary_same = {n for n, (a, _, _) in enumerate(ordinary_per)
                             if a.sha256 == ordinary_main.sha256}
            ordinary_present = sum(1 for label, owner in ordinary_map
                                   if (label == CELL_SUPPORTED and owner in ordinary_same)
                                   or label in (CELL_ADDITIONAL, CELL_MISSING))
            ordinary_music = sum(1 for label, _ in ordinary_map if label != CELL_QUIET)
            if (ordinary_present < recipe.exact_min_cells
                    or ordinary_present < PRESENT_MIN_FRACTION * ordinary_music):
                return False
            ordinary_regions = self._regions(cells, ordinary_map, ordinary_per)
            ordinary_main, ordinary_cells = items[0]
            ordinary_labels = labels_of(ordinary_cells)
            ordinary_coverage = self._coverage(cells, ordinary_map, ordinary_cells, quiet)
            ordinary_view = {**self._alignment_view(ordinary_main, ordinary_labels),
                             **self.reference_view(ordinary_main.sha256)}
            ordinary_relation = self._non_exact(ordinary_map, ordinary_per, ordinary_main,
                                                ordinary_view, ordinary_coverage, ordinary_regions)
            # A clean region cannot authorize suppression when the ordinary
            # relation class itself is outside reconstruction's authority.
            if ordinary_relation["state"] not in (EDIT, COMPONENT):
                return False
            from lss.bgm import reference_reconstruction
            donor_regions = reference_reconstruction.clean_donor_regions(ordinary_regions, start, end)
            return bool(donor_regions["donors"])
        if not enough(measured) and recipe.nomination_max_windows and recipe.nomination_max_references:
            local, nomination = self._nominate(cells, candidates, quiet)
            measured.extend(local)
            measured.sort(key=strength, reverse=True)
        main, main_cells = measured[0]
        main_labels = labels_of(main_cells)
        if (main_labels.count(CELL_SUPPORTED) + main_labels.count(CELL_ADDITIONAL)
                + main_labels.count(CELL_MISSING)) < recipe.exact_min_cells and nomination is None:
            return [(start, end, self.unrelated(start, end, candidates, measured=True))]

        # 同じ録音の別の位置（切り貼り・差し替え）を、主のずれで説明できない範囲から探す。
        recordings = sorted({a.sha256 for a, _ in measured[:4]})
        others_labels = [labels_of(c) for _, c in measured[1:]]
        # An alternate offset accounts only for the cells it actually supports.
        # One supported cell must not suppress search over an entire remaining
        # run: TV edits can use three or more offsets inside that run.
        unexplained = [label in (CELL_UNEXPLAINED, CELL_ADDITIONAL, CELL_MISSING)
                       and not any(labels[i] == CELL_SUPPORTED for labels in others_labels)
                       for i, label in enumerate(main_labels)]
        for run_start, run_end in ([] if nomination is not None else _runs(unexplained)[:recipe.local_max_runs]):
            window_start, window_end = cells[run_start] - 0.5, cells[run_end - 1] + 1.5
            for sha256 in recordings:
                reference = self._samples(sha256)
                if reference is None:
                    continue
                for offset, _ in _local_search(self.evidence.music, reference, window_start, window_end,
                                               recipe.local_min_ncc):
                    if any(a.sha256 == sha256 and abs(a.offset - offset) <= 0.25 for a, _ in measured):
                        continue
                    measured.append((_Alignment(sha256, float(offset), "local_search"),
                                     self.evidence.measure(reference, float(offset), cells)))

        # セルごとの地図：主のずれが説明し尽くせればそれ、無理なら別の整列（同じ録音を先に）。
        cell_map, per_alignment = cell_map_for(measured, scoped=nomination is not None)

        regions = self._regions(cells, cell_map, per_alignment)
        coverage = self._coverage(cells, cell_map, main_cells, quiet)
        # 録音が「現れる」と言うには、主の録音で説明し尽くせるか、参照が鳴っているセルが、最短の長さ以上かつ
        # 区間の音楽の一定の割合以上要る。数秒だけの一致で、区間全体をその録音の版とは言わない。
        same_recording = {n for n, (a, _) in enumerate(measured) if a.sha256 == main.sha256}
        present = sum(1 for label, owner in cell_map
                      if (label == CELL_SUPPORTED and owner in same_recording)
                      or label in (CELL_ADDITIONAL, CELL_MISSING))
        music_cells = sum(1 for label, _ in cell_map if label != CELL_QUIET)
        if present < recipe.exact_min_cells or present < PRESENT_MIN_FRACTION * music_cells:
            relation = self.unrelated(start, end, candidates, measured=True)
            relation["partial_reference_regions"] = [r for r in regions if r.get("reference_sha256")]
            if nomination is not None:
                relation["local_nomination"] = nomination
                relation["regions"] = regions
            return [(start, end, relation)]
        main_view = {**self._alignment_view(main, main_labels), **self.reference_view(main.sha256)}
        if nomination is not None:
            relation = self._non_exact(cell_map, per_alignment, main, main_view, coverage, regions)
            relation["local_nomination"] = nomination
            relation["donor_reconstructable"] = False
            relation["authority_hold_reasons"] = ["scoped_local_observations_require_full_accounting"]
            cover = relation.get("released_cover")
            if cover is not None:
                cover["complete"] = False
                cover["reasons"].append("scoped_local_observations_require_full_accounting")
            return [(start, end, relation)]

        def non_exact(**medians):
            relation = self._non_exact(cell_map, per_alignment, main, main_view, coverage, regions, **medians)
            if relation["state"] in (COMPONENT, ADDITIONAL):
                reference = self._samples(main.sha256)
                if reference is not None:
                    # 主のずれが番組の音楽を説明しているセル（説明し尽くせる・欠け・追加の素材）だけで性質を測る。
                    # 別のずれ・別の曲・説明できないセルを「足された成分」に数えない。
                    measured_labels = [label if owner in (0, None) and label in (CELL_SUPPORTED, CELL_MISSING,
                                                                                   CELL_ADDITIONAL)
                                       else CELL_QUIET for label, owner in cell_map]
                    relation["components"] = self.evidence.character(reference, main.offset, cells,
                                                                     measured_labels)
            return [(start, end, relation)]

        run = self._main_run(cell_map)
        if run is None:
            return non_exact()
        first, last = run
        music = [i for i, (label, _) in enumerate(cell_map) if label != CELL_QUIET]
        before = [i for i in music if i < first]
        after = [i for i in music if i > last]
        same = {n for n, (a, _, _) in enumerate(per_alignment) if a.sha256 == main.sha256}

        def foreign(indexes) -> bool:
            """端の残りが、主の録音とは無関係な音楽（別のcue）か。成分の欠けた同じ録音は無関係ではない。"""
            return not any((cell_map[i][0] == CELL_SUPPORTED and cell_map[i][1] in same)
                           or cell_map[i][0] == CELL_MISSING for i in indexes)

        def absorbable(indexes) -> bool:
            """端の残りをEXACTへ含めてよいか。含めてよいのは、同じ録音の同じずれで説明できるセル・判断できない
            セル・小さいセル（フェードや窓が跨ぐ揺れ）だけ。説明できない音楽・追加の素材・成分の欠け・別の録音／別の
            ずれのセルは、EXACTの内側へ吸収しない（無かったことにしない）。"""
            return all(cell_map[i][0] in (CELL_QUIET, CELL_UNCERTAIN) or cell_map[i] == (CELL_SUPPORTED, 0)
                       for i in indexes)

        absorb_before = len(before) <= recipe.edge_absorb_cells and absorbable(before)
        absorb_after = len(after) <= recipe.edge_absorb_cells and absorbable(after)
        split_before = (first >= recipe.split_min_seconds and not absorb_before and foreign(range(0, first)))
        split_after = (len(cells) - 1 - last >= recipe.split_min_seconds and not absorb_after
                       and foreign(range(last + 1, len(cells))))

        def sparse(indexes) -> bool:
            """長い区間の端の音楽に、主の録音が点在する偶然の一致（数秒）しか無いか。"""
            music_cells = [i for i in indexes if cell_map[i][0] != CELL_QUIET]
            held = sum(1 for i in music_cells if (cell_map[i][0] == CELL_SUPPORTED and cell_map[i][1] in same)
                       or cell_map[i][0] == CELL_MISSING)
            return bool(music_cells) and held <= recipe.long_split_max_support * len(music_cells)

        def carved():
            """一曲として扱えない長い区間を、主の録音の続く範囲の端で分けた片。分けられなければ None。

            端の外側は改めて関係を測る（別の曲は別の区間）。主の範囲もそれだけで測り直すので、録音そのものなら
            EXACTの厳しさで判断される。分けるだけで、どの片も消さない。"""
            if len(cells) <= recipe.long_split_seconds:
                return None
            cut_before = first >= recipe.split_min_seconds and not absorb_before and sparse(range(0, first))
            cut_after = (len(cells) - 1 - last >= recipe.split_min_seconds and not absorb_after
                         and sparse(range(last + 1, len(cells))))
            if not (cut_before or cut_after):
                return None
            low = start + (first if cut_before else 0)
            high = start + (last + 1 if cut_after else len(cells))
            pieces = self.relate(start, low, depth + 1) if low > start else []
            pieces += self.relate(low, high, depth + 1)
            return pieces + (self.relate(high, end, depth + 1) if high < end else [])
        inner = range(first, last + 1)
        supported = [i for i in inner if cell_map[i] == (CELL_SUPPORTED, 0)]
        judged = [i for i in inner if cell_map[i][0] != CELL_QUIET]
        joint_median = float(np.median(main_cells.joint[supported])) if supported else 0.0
        novelty_median = float(np.nanmedian(main_cells.novelty[judged])) if judged else 1.0
        absences = main_cells.absence[judged] if judged else np.array([1.0])
        absence_median = float(np.nanmedian(absences)) if np.isfinite(absences).any() else 1.0
        shares = (main_cells.joint[supported] - main_cells.joint_reference[supported]) if supported else np.zeros(0)
        interference_share = float(np.nanmedian(shares)) if shares.size and np.isfinite(shares).any() else 0.0
        medians = {"joint_median": joint_median, "novelty_median": novelty_median,
                   "absence_median": absence_median}
        if not (split_before or absorb_before) or not (split_after or absorb_after):
            # 録音の使用の内側（または分けるには短い端）に、主のずれで説明できない音楽がある＝番組の版。
            return carved() or non_exact(**medians)
        piece_first = first if split_before else 0
        piece_last = last if split_after else len(cells) - 1
        # 分けた境目のセルは窓が別の音楽へ跨ぐので、局所の測定から外す（境目は参照の波形で裏付けた境目）。
        local_first, local_last = piece_first + int(split_before), piece_last - int(split_after)
        local_novelty = self._local_novelty(main_cells.novelty, cell_map, local_first, local_last)
        cell_novelty = self._local_novelty(main_cells.novelty, cell_map, local_first, local_last, width=1)
        dropped = [i for i in range(local_first, local_last + 1)
                   if cell_map[i][0] != CELL_QUIET and np.isfinite(main_cells.joint[i])
                   and main_cells.joint[i] < joint_median - recipe.exact_max_joint_drop
                   and not main_cells.interference[i] >= recipe.excuse_interference_ratio]
        clean_residual, local_clean_residual = self._clean_residual(
            main, [cells[i] for i in supported if local_first <= i <= local_last], (start, end))
        if (len(supported) < recipe.exact_min_cells or joint_median < recipe.exact_min_joint or dropped
                or clean_residual > recipe.exact_max_clean_residual
                or local_clean_residual > recipe.exact_max_local_clean_residual
                or novelty_median > recipe.exact_max_novelty or local_novelty > recipe.exact_max_local_novelty
                or cell_novelty > recipe.exact_max_cell_novelty
                or interference_share > recipe.exact_max_interference_share
                or absence_median > recipe.exact_max_absence):
            return carved() or non_exact(**medians, local_novelty=local_novelty)
        piece_start, piece_end = start + piece_first, start + piece_last + 1
        absorbed = [cells[i] for i in range(piece_first, piece_last + 1)
                    if i not in inner and cell_map[i][0] != CELL_QUIET]
        also = []
        for n, (alignment, _, labels) in enumerate(per_alignment):
            if n == 0 or alignment.sha256 == main.sha256:
                continue
            share = float(np.mean([labels[i] == CELL_SUPPORTED for i in supported]))
            if share >= 0.9:
                also.append({**self._alignment_view(alignment, labels), **self.reference_view(alignment.sha256),
                             "support_fraction": round(share, 3)})
        reference_start = piece_start + main.offset
        reference_end = piece_end + main.offset
        duration = self.library.durations[main.sha256]
        exact = {"state": EXACT, "basis": "one_alignment_accounts_for_all_music", "reference": main_view,
                 "also_matching_references": also,
                 "reference_range_seconds": [round(max(0.0, reference_start), 3),
                                             round(min(duration, reference_end), 3)],
                 "starts_at_reference_start": bool(reference_start <= 1.0),
                 "ends_at_reference_end": bool(reference_end >= duration - 1.0),
                 "coverage": {**coverage, "exact_cells": len(supported), "median_joint": round(joint_median, 3),
                              "median_novelty": round(novelty_median, 3),
                              "median_absence": round(absence_median, 3),
                              "max_local_novelty": round(local_novelty, 4),
                              "interference_share": round(interference_share, 4), "absorbed_edge_cells": absorbed,
                              "clean_residual": round(clean_residual, 4),
                              "max_local_clean_residual": round(local_clean_residual, 4)},
                 "regions": [dict(r) for r in regions
                             if r["end_seconds"] > piece_start and r["start_seconds"] < piece_end]}
        pieces = []
        if piece_start > start:
            pieces += self.relate(start, piece_start, depth + 1)
        pieces.append((piece_start, piece_end, exact))
        if piece_end < end:
            pieces += self.relate(piece_end, end, depth + 1)
        return pieces

    def _clean_residual(self, main: _Alignment, cells: list[int], span: tuple[int, int]) -> tuple[float, float]:
        """主のずれで説明し尽くせるセル（分けた境目のセルを除く）のうち台詞・効果音の無いセルの、説明できない割合の
        中央値と、続く exact_local_cells セルの平均の最大値。台詞の無いセルが無ければ (0, 0)（この測定では何も言えない）。"""
        reference = self._samples(main.sha256)
        if reference is None or self.evidence is None or not cells:
            return 0.0, 0.0
        values = self.evidence.clean_residual(reference, main.offset, cells, span, self.recipe.exact_clean_taps,
                                              self.recipe.exact_clean_interference)
        if not values:
            return 0.0, 0.0
        width = min(self.recipe.exact_local_cells, len(values))
        return (float(np.median(values)),
                float(max(np.mean(values[k:k + width]) for k in range(len(values) - width + 1))))

    def _local_novelty(self, novelty: np.ndarray, cell_map, first: int, last: int, width: int | None = None) -> float:
        """[first, last] の判断したセルの novelty の、続く `width`（既定 exact_local_cells）セルの平均の最大値。"""
        values = [float(novelty[i]) for i in range(first, last + 1)
                  if cell_map[i][0] != CELL_QUIET and np.isfinite(novelty[i])]
        if not values:
            return 1.0
        width = min(width or self.recipe.exact_local_cells, len(values))
        return float(max(np.mean(values[k:k + width]) for k in range(len(values) - width + 1)))

    def _main_run(self, cell_map: list[tuple[str, int | None]]) -> tuple[int, int] | None:
        """主のずれで説明し尽くせるセルが（判断できない・小さいセルを制限内で挟んで）続く最長の範囲。"""
        recipe = self.recipe
        best = None
        count = len(cell_map)
        i = 0
        while i < count:
            if cell_map[i] != (CELL_SUPPORTED, 0):
                i += 1
                continue
            first = last = i
            uncertain_run = 0
            j = i + 1
            while j < count:
                label = cell_map[j][0]
                if cell_map[j] == (CELL_SUPPORTED, 0):
                    last, uncertain_run = j, 0
                elif label == CELL_QUIET:
                    pass
                elif label == CELL_UNCERTAIN and uncertain_run < recipe.max_uncertain_run:
                    uncertain_run += 1
                else:
                    break
                j += 1
            span = last - first + 1
            inside_uncertain = sum(1 for k in range(first, last + 1) if cell_map[k][0] == CELL_UNCERTAIN)
            if inside_uncertain <= recipe.max_uncertain_fraction * span:
                key = (span, -first)
                if best is None or key > best[0]:
                    best = (key, (first, last))
            i = last + 1
        return None if best is None else best[1]

    def _non_exact(self, cell_map, per_alignment, main, main_view, coverage, regions, *,
                   joint_median=None, novelty_median=None, absence_median=None, local_novelty=None) -> dict:
        """EXACTにできない、参照の録音が現れる区間の関係（COMPONENT・EDIT・ADDITIONAL）。"""
        same = [n for n, (a, _, _) in enumerate(per_alignment) if n > 0 and a.sha256 == main.sha256]
        used_same = sorted({owner for label, owner in cell_map if label == CELL_SUPPORTED and owner in same})
        labels = [label for label, _ in cell_map]
        additional = labels.count(CELL_ADDITIONAL)
        unexplained = labels.count(CELL_UNEXPLAINED)
        others = [{**self._alignment_view(per_alignment[n][0], per_alignment[n][2]),
                   "cells": sum(1 for _, owner in cell_map if owner == n)} for n in used_same]
        music_cells = [i for i, label in enumerate(labels) if label != CELL_QUIET]
        reconstructable = bool(music_cells) and all(labels[i] == CELL_SUPPORTED for i in music_cells)
        relation = {"reference": main_view, "other_alignments": others, "coverage": dict(coverage),
                    "regions": regions, "donor_reconstructable": reconstructable}
        if joint_median is not None:
            relation["coverage"].update(median_joint=round(joint_median, 3), median_novelty=round(novelty_median, 3),
                                        median_absence=round(absence_median, 3))
            if local_novelty is not None:
                relation["coverage"]["max_local_novelty"] = round(local_novelty, 4)
        main_supported = sum(1 for item in cell_map if item == (CELL_SUPPORTED, 0))
        missing = labels.count(CELL_MISSING)
        # 成分の欠けは、主の整列が番組の音楽を説明しているセル（主のずれで説明し尽くせる・欠けのある）でだけ言える。
        # 切り貼りの別の部分（別のずれ）が説明するセルで、主のずれの参照が鳴らないのは欠けではない。
        explained = [i for i in music_cells
                     if cell_map[i] == (CELL_SUPPORTED, 0) or labels[i] == CELL_MISSING]
        absences = [coverage["main_absence"][i] for i in explained
                    if i < len(coverage.get("main_absence") or []) and coverage["main_absence"][i] is not None]
        absence_level = float(np.median(absences)) if absences else 0.0
        relation["coverage"]["median_absence_music"] = round(absence_level, 3)
        component = ((missing >= 2 and missing >= COMPONENT_MIN_FRACTION * len(music_cells))
                     or (absence_level > self.recipe.exact_max_absence
                         and main_supported + missing >= self.recipe.exact_min_cells))
        if component:
            # 同じ録音の構造は続くが、参照の成分の一部が番組の音に無い：番組だけの版（抜いた版）。
            basis = ("reference_components_absent_and_material_added" if additional >= 2
                     else "reference_components_absent_in_episode")
            return {"state": COMPONENT, "basis": basis, "structure_edit": bool(used_same and main_supported >= 2),
                    **relation}
        if used_same and main_supported >= 2:
            return {"state": EDIT, "basis": "multiple_alignments_of_one_recording",
                    "released_cover": self._released_cover(cell_map, per_alignment, main, regions), **relation}
        if main_supported >= self.recipe.exact_min_cells and unexplained > additional:
            # 録音の使用は確かめられた（主のずれで説明し尽くせる範囲がある）が、その内側に説明できない音楽がある。
            return {"state": EDIT, "basis": "episode_local_material_inside_recording_use", **relation}
        # 参照は鳴っているが、それだけでは区間の音楽を説明できない（重ね・追加の層・別ミックス）。
        return {"state": ADDITIONAL, "basis": "reference_with_material_it_does_not_explain", **relation}

    def _released_cover(self, cell_map, per_alignment, main, regions) -> dict:
        """切り貼り（短くした・途中から・繋ぎ直した）の使用が、一つの録音の部分だけでできているか。

        できていれば、番組の編集を施した既発売の録音そのもの（`BGM_AND_AVAILABILITY.md`）。EXACTと同じ厳しさで
        見る：音楽のセルはすべて同じ録音のどれかのずれで説明し尽くせる（別の曲・追加の素材・成分の欠け・説明できない
        音楽のセルは一つも無い）。各部分の内側の novelty・absence はEXACTの上限以下（重ねた別の音を通さない）。
        編集点の前後1セルは窓が二つの位置を跨ぐので、局所の測定から外す（説明し尽くせることは要る）。
        """
        recipe = self.recipe
        same = {n for n, (a, _, _) in enumerate(per_alignment) if a.sha256 == main.sha256}
        music = [i for i, (label, _) in enumerate(cell_map) if label != CELL_QUIET]
        recording = [i for i in music if cell_map[i][0] == CELL_SUPPORTED and cell_map[i][1] in same]
        uncertain = [i for i in music if cell_map[i][0] == CELL_UNCERTAIN]
        foreign = [i for i in music if i not in recording and i not in uncertain]
        owners = [cell_map[i][1] for i in range(len(cell_map))]
        # 編集点の継ぎ目：同じ録音の別のずれ二つに挟まれた2セルまでの外れは、窓が二つの位置を跨いだ跡。
        seams = []
        for first, last in _runs([i in foreign for i in range(len(cell_map))]):
            left, right = first - 1, last
            if (last - first <= MISSING_MIN_RUN - 1 and left >= 0 and right < len(cell_map)
                    and left in recording and right in recording and owners[left] != owners[right]):
                seams += list(range(first, last))
        foreign = [i for i in foreign if i not in seams]
        longest = max((b - a for a, b in _runs([cell_map[i][0] == CELL_UNCERTAIN for i in range(len(cell_map))])),
                      default=0)
        inner = [i for i in recording
                 if all(owners[j] == owners[i] for j in (i - 1, i + 1) if 0 <= j < len(owners)
                        and cell_map[j][0] != CELL_QUIET)]
        novelty = np.array([per_alignment[owners[i]][1].novelty[i] for i in inner], dtype=float)
        absence = np.array([per_alignment[owners[i]][1].absence[i] for i in inner], dtype=float)
        novelty_median = float(np.nanmedian(novelty)) if np.isfinite(novelty).any() else 1.0
        novelty_max = float(np.nanmax(novelty)) if np.isfinite(novelty).any() else 1.0
        absence_median = float(np.nanmedian(absence)) if np.isfinite(absence).any() else 1.0
        pieces = len({owners[i] for i in recording})
        reasons = []
        if foreign:
            reasons.append("material_outside_the_recording")
        if len(recording) < recipe.exact_min_cells:
            reasons.append("too_few_recording_cells")
        if longest > recipe.max_uncertain_run or len(uncertain) > recipe.max_uncertain_fraction * max(1, len(music)):
            reasons.append("too_much_undecided")
        if len(inner) < recipe.exact_min_cells:
            reasons.append("too_few_cells_inside_pieces")
        if novelty_median > recipe.exact_max_novelty or novelty_max > recipe.exact_max_cell_novelty:
            reasons.append("material_overlaid_on_recording")
        if absence_median > recipe.exact_max_absence:
            reasons.append("recording_components_absent")
        # Cut edits have the same unknown-overlay risk as a contiguous EXACT
        # excerpt. Novelty alone can miss a quiet added cue. Reuse the clean
        # residual gate on each piece, excluding its edit seams; never average
        # a dirty piece away with a longer clean piece or another offset.
        residuals = []
        if not reasons:
            origin = int(regions[0]["start_seconds"])
            span = (origin, int(regions[-1]["end_seconds"]))
            for owner in sorted(set(owners[i] for i in inner)):
                for first, last in _runs([i in inner and owners[i] == owner for i in range(len(cell_map))]):
                    median, local = self._clean_residual(per_alignment[owner][0],
                                                        [origin+i for i in range(first, last)], span)
                    residuals.append({"start_seconds": origin+first, "end_seconds": origin+last,
                                      "median": median, "max_local": local})
            if any(r["median"] > recipe.exact_max_clean_residual
                   or r["max_local"] > recipe.exact_max_local_clean_residual for r in residuals):
                reasons.append("unexplained_clean_music_in_edit_piece")
        return {"complete": not reasons, "reasons": reasons, "recording_cells": len(recording),
                "uncertain_cells": len(uncertain), "foreign_cells": len(foreign), "seam_cells": len(seams),
                "pieces": pieces,
                "inner_cells": len(inner), "clean_residual_pieces": residuals,
                "median_novelty": round(novelty_median, 4),
                "max_cell_novelty": round(novelty_max, 4), "median_absence": round(absence_median, 4)}

    @staticmethod
    def _alignment_view(alignment: _Alignment, labels: list[str]) -> dict:
        return {"reference_sha256": alignment.sha256, "offset_seconds": round(alignment.offset, 3),
                "origin": alignment.origin, "exact_match": alignment.origin == "fingerprint_exact",
                "supported_cells": labels.count(CELL_SUPPORTED),
                **({"scope_seconds": list(alignment.scope)} if alignment.scope is not None else {})}

    @staticmethod
    def _coverage(cells, cell_map, main_cells: _Cells, quiet: np.ndarray) -> dict:
        labels = [label for label, _ in cell_map]
        return {"cells": len(cells), "quiet_cells": int(quiet.sum()),
                "supported_cells": labels.count(CELL_SUPPORTED),
                "main_supported_cells": sum(1 for item in cell_map if item == (CELL_SUPPORTED, 0)),
                "additional_cells": labels.count(CELL_ADDITIONAL),
                "missing_component_cells": labels.count(CELL_MISSING),
                "unexplained_cells": labels.count(CELL_UNEXPLAINED),
                "uncertain_cells": labels.count(CELL_UNCERTAIN),
                "main_joint": [None if not np.isfinite(v) else round(float(v), 3) for v in main_cells.joint],
                "main_novelty": [None if not np.isfinite(v) else round(float(v), 3) for v in main_cells.novelty],
                "main_absence": [None if not np.isfinite(v) else round(float(v), 3) for v in main_cells.absence]}

    @staticmethod
    def _regions(cells, cell_map, per_alignment) -> list[dict]:
        """続くセルを（種類, 整列）ごとにまとめた領域の地図。参照の位置は番組の時刻＋offset。"""
        merged: list[list] = []
        for index, key in enumerate(cell_map):
            if merged and merged[-1][0] == key:
                merged[-1][2] = cells[index] + 1
            else:
                merged.append([key, cells[index], cells[index] + 1])
        result = []
        for (label, owner), first, last in merged:
            region = {"start_seconds": float(first), "end_seconds": float(last), "kind": label}
            if owner is not None:
                alignment, measured = per_alignment[owner][0], per_alignment[owner][1]
                span = slice(cells.index(first), cells.index(last - 1) + 1)
                joint, novelty, absence = measured.joint[span], measured.novelty[span], measured.absence[span]
                region.update({"reference_sha256": alignment.sha256, "offset_seconds": round(alignment.offset, 3),
                               "origin": alignment.origin,
                               "reference_range_seconds": [round(first + alignment.offset, 3),
                                                           round(last + alignment.offset, 3)]})
                if np.isfinite(joint).any():
                    # この領域を説明した整列での会計（台詞・効果音の漏れを差し引いた後）。再構成の計画が使う。
                    # 端の1秒は窓が隣の領域（編集点・別の音）へ跨ぐので、最大値は内側で測る。
                    inner = novelty[1:-1] if novelty.size > 3 else novelty
                    region.update(median_joint=round(float(np.nanmedian(joint)), 3),
                                  median_novelty=round(float(np.nanmedian(novelty)), 3),
                                  max_novelty=round(float(np.nanmax(inner)) if np.isfinite(inner).any() else 1.0, 3))
                    inner_absence = absence[1:-1] if absence.size > 3 else absence
                    region.update(median_absence=round(float(np.nanmedian(absence)), 3)
                                  if np.isfinite(absence).any() else None,
                                  max_absence=round(float(np.nanmax(inner_absence)), 3)
                                  if np.isfinite(inner_absence).any() else None)
            result.append(region)
        return result

    def unrelated(self, start: int, end: int, alignments, measured: bool = False) -> dict:
        """参照で説明できる音楽が無い区間。和声の対応があれば別版の候補、無ければUNKNOWN。"""
        fingerprint = [{"reference_sha256": a.sha256, "offset_seconds": round(a.offset, 3),
                        "display_title": (self.library.metadata.get(a.sha256) or {}).get("display_title"),
                        "exact_match_nearby": a.origin == "fingerprint_exact"}
                       for a, _ in alignments]
        related = []
        if self.chroma_music is not None and end - start >= versions.MIN_SEGMENT_SECONDS:
            segment = versions.chroma(self.chroma_music[int(start * SR):int(end * SR)])
            references = self.library.chroma()
            if len(references) >= versions.MIN_COHORT + 1:
                for item in versions.segment_relations(segment, references):
                    if item["z"] >= self.recipe.related_min_z:
                        related.append({**self.reference_view(item["reference_id"]), "score": item["score"],
                                        "z": item["z"], "evidence": "shared_harmonic_material"})
        if related:
            return {"state": RELATED, "basis": "shared_harmonic_material_without_waveform_identity",
                    "candidates": related, "fingerprint_candidates": fingerprint}
        basis = "reference_does_not_account_for_music" if measured else "no_waveform_aligned_reference"
        return {"state": UNKNOWN, "basis": basis, "fingerprint_candidates": fingerprint}


def _edge(kind_piece: dict, other: dict | None) -> dict:
    view = {"kind": EDGE_REFERENCE}
    for relation in (kind_piece, other):
        if relation and relation.get("state") == EXACT:
            view["reference_sha256"] = relation["reference"]["reference_sha256"]
    return view


def relate_segments(segments: list[dict], *, query: np.ndarray | None = None, library: ReferenceLibrary,
                    matches: list[identity.RecordingMatch], recipe: RelationRecipe | None = None,
                    music: np.ndarray | None = None, interference: list[np.ndarray] | None = None) -> list[dict]:
    """`cue_semantics.segment` の区間を関係ごとの片へ分け、各片に `reference_relation` を付けた複製を返す。

    `music` は分離したMusic stem、`interference` は台詞・効果音のstem（いずれも11025Hz・mono）。
    Music stemが無い時は音楽の会計ができないので、EXACTを出さない（参照の存在だけで消さない）。
    EXACTの範囲で分けた境目は `reference_recording_edge`。区間の外側の端は元のまま。
    入力の区間は変更しない。同じ入力からは同じ結果を返す。`query`（混合）は互換のために受け取るだけ。
    """
    recipe = recipe or default_recipe()
    recipe.validate()
    evidence = _Evidence(music, list(interference or [])) if music is not None else None
    relator = _Relator(library, matches, recipe, evidence, music)
    result = []
    for item in segments:
        # Measure the complete cells covering a fractional boundary. Rounded
        # endpoints formerly discarded or extended the fine source interval.
        # A subset of measured coverage never extends identity authority.
        original_start, original_end = float(item["start_seconds"]), float(item["end_seconds"])
        start, end = int(np.floor(original_start)), int(np.ceil(original_end))
        pieces = relator.relate(start, end)
        pieces = [(max(original_start, a), min(original_end, b), r) for a, b, r in pieces
                  if max(original_start, a) < min(original_end, b)]
        core = item.get("core_seconds", [item["start_seconds"], item["end_seconds"]])
        for number, (piece_start, piece_end, relation) in enumerate(pieces):
            piece = dict(item)
            piece["start_seconds"], piece["end_seconds"] = float(piece_start), float(piece_end)
            piece["core_seconds"] = [float(max(core[0], piece_start)), float(min(core[1], piece_end))]
            if piece["core_seconds"][1] <= piece["core_seconds"][0]:
                piece["core_seconds"] = [float(piece_start), float(piece_end)]
            if number > 0:
                piece["start_edge"] = _edge(relation, pieces[number - 1][2])
            if number < len(pieces) - 1:
                piece["end_edge"] = _edge(relation, pieces[number + 1][2])
            piece["candidates"] = [c for c in item.get("candidates", [])
                                   if min(c[1], piece_end) - max(c[0], piece_start) > 0]
            if len(pieces) > 1:
                piece["split_from"] = [float(item["start_seconds"]), float(item["end_seconds"])]
            piece["reference_relation"] = relation
            if original_start != start or original_end != end:
                relation = dict(relation)
                relation["measured_scope_seconds"] = [start, end]
                relation = restrict_relation_span(relation, piece_start, piece_end, parent_span=[start, end])
                piece["reference_relation"] = relation
            result.append(piece)
    return result


def restrict_relation_span(relation: dict, start: float, end: float, *, parent_span=None) -> dict:
    """Clip relation geometry to analysis time without mutating shared parent provenance.

    The measured scope records where the relation decision was made; clipping the geometry
    does not recompute that analysis.
    """
    restricted = dict(relation)
    if "measured_scope_seconds" in restricted:
        restricted["measured_scope_seconds"] = list(restricted["measured_scope_seconds"])
    elif parent_span is not None:
        restricted["measured_scope_seconds"] = list(parent_span)
    for key in ("regions", "partial_reference_regions"):
        if key not in relation or relation[key] is None:
            continue
        clipped = []
        for region in relation[key]:
            lo = max(float(start), float(region["start_seconds"]))
            hi = min(float(end), float(region["end_seconds"]))
            if lo >= hi:
                continue
            view = {**region, "start_seconds": lo, "end_seconds": hi}
            offset = region.get("offset_seconds")
            if region.get("reference_range_seconds") is not None and offset is not None:
                view["reference_range_seconds"] = [lo + offset, hi + offset]
            clipped.append(view)
        restricted[key] = clipped
    if restricted.get("state") == EXACT and restricted.get("reference"):
        reference = restricted["reference"]
        offset = reference["offset_seconds"]
        duration = reference.get("duration_seconds", float("inf"))
        restricted["reference_range_seconds"] = [max(0., start + offset), min(duration, end + offset)]
        restricted["starts_at_reference_start"] = bool(start + offset <= 1.)
        restricted["ends_at_reference_end"] = bool(end + offset >= duration - 1.)
    return restricted


def released_use(relation: dict | None) -> bool:
    """既発売の録音そのものの使用か：EXACT、または一つの録音の部分だけでできた切り貼り（短くした・途中から）。"""
    relation = relation or {}
    return relation.get("state") == EXACT or (relation.get("state") == EDIT
                                              and bool((relation.get("released_cover") or {}).get("complete")))


def apply(decided: list[dict], known_class: str, *, reason: str = REASON_EXACT) -> list[dict]:
    """判定済みの区間のうち既発売の録音そのものの使用（`released_use`）だけを既発売にする（重複の抽出・公開を
    しない）。配布の受領が無ければ抽出を残す。他は変えない。"""
    result = []
    for item in decided:
        relation = item.get("reference_relation") or {}
        from lss.bgm import release_availability
        reference = relation.get("reference") or {}
        receipt = reference.get("independent_distribution")
        released = released_use(relation)
        if released and release_availability.available(receipt, reference.get("reference_sha256")):
            item = {**item, "acoustic_class": item["semantic_class"], "acoustic_reasons": item["reasons"],
                    "semantic_class": known_class,
                    "reasons": [reason if relation.get("state") == EXACT else REASON_EDIT]}
        elif released:
            item = {**item, "independent_availability": "unknown_preserve_extraction"}
        result.append(item)
    return result


def occurrences(decided: list[dict], *, time_offset: float = 0.0) -> list[dict]:
    """既発売の録音の使用箇所（EXACTの片と、組み立て直すと既発売の一続きの範囲と実質同じになったcue）。

    同じ録音の同じずれで続く片は一つにまとめる。
    """
    found: list[dict] = []
    for item in sorted(decided, key=lambda x: x["start_seconds"]):
        relation = item.get("reference_relation") or {}
        available = item.get("independently_available")
        if available and relation.get("reference"):
            reference = relation["reference"]
            same = reference.get("reference_sha256") == available["reference_sha256"]
            found.append({
                "start_seconds": item["start_seconds"], "end_seconds": item["end_seconds"],
                "source_start_seconds": round(item["start_seconds"] + time_offset, 3),
                "source_end_seconds": round(item["end_seconds"] + time_offset, 3),
                "relation": relation["state"], "basis": "reconstruction_materially_equals_release",
                "reference_sha256": available["reference_sha256"],
                "display_title": reference.get("display_title") if same else None,
                "offset_seconds": available["offset_seconds"],
                "reference_range_seconds": list(available["reference_range_seconds"]),
                "unique_seconds": available["unique_seconds"],
                "extraction": "not_created_independent_recording_available",
            })
            continue
        from lss.bgm import release_availability
        reference = relation.get("reference") or {}
        if not released_use(relation) or not release_availability.available(
                reference.get("independent_distribution"),reference.get("reference_sha256")):
            continue
        reference = relation["reference"]
        if relation["state"] == EDIT:
            # 番組が短くした・途中から流した・繋ぎ直した既発売の録音：どの部分を使ったかを記録する。
            pieces = [r for r in relation.get("regions", []) if r.get("reference_sha256") == reference["reference_sha256"]]
            found.append({
                "start_seconds": item["start_seconds"], "end_seconds": item["end_seconds"],
                "source_start_seconds": round(item["start_seconds"] + time_offset, 3),
                "source_end_seconds": round(item["end_seconds"] + time_offset, 3),
                "relation": EDIT, "basis": "program_edit_of_released_recording",
                "reference_sha256": reference["reference_sha256"],
                "display_title": reference.get("display_title"), "title_source": reference.get("title_source"),
                "tags": reference.get("tags") or {}, "file_name": reference.get("file_name"),
                "offset_seconds": reference["offset_seconds"],
                "reference_pieces": [{"source_start_seconds": round(r["start_seconds"] + time_offset, 3),
                                      "source_end_seconds": round(r["end_seconds"] + time_offset, 3),
                                      "reference_range_seconds": r.get("reference_range_seconds")} for r in pieces],
                "reference_duration_seconds": reference.get("duration_seconds"),
                "extraction": "not_created_independent_recording_available",
            })
            continue
        entry = {
            "start_seconds": item["start_seconds"], "end_seconds": item["end_seconds"],
            "source_start_seconds": round(item["start_seconds"] + time_offset, 3),
            "source_end_seconds": round(item["end_seconds"] + time_offset, 3),
            "relation": EXACT, "reference_sha256": reference["reference_sha256"],
            "display_title": reference.get("display_title"), "title_source": reference.get("title_source"),
            "tags": reference.get("tags") or {}, "file_name": reference.get("file_name"),
            "offset_seconds": reference["offset_seconds"],
            "reference_range_seconds": list(relation["reference_range_seconds"]),
            "reference_duration_seconds": reference.get("duration_seconds"),
            "starts_at_reference_start": relation["starts_at_reference_start"],
            "ends_at_reference_end": relation["ends_at_reference_end"],
            "also_matching_references": [a["reference_sha256"] for a in relation.get("also_matching_references", [])],
            "extraction": "not_created_independent_recording_available",
        }
        if found:
            last = found[-1]
            if (last["reference_sha256"] == entry["reference_sha256"]
                    and abs(last["offset_seconds"] - entry["offset_seconds"]) <= 1.0 / SR
                    and abs(entry["start_seconds"] - last["end_seconds"]) <= 1e-6
                    and abs(entry["reference_range_seconds"][0]-last["reference_range_seconds"][1]) <= 1.0 / SR):
                last["end_seconds"], last["source_end_seconds"] = entry["end_seconds"], entry["source_end_seconds"]
                last["reference_range_seconds"][1] = entry["reference_range_seconds"][1]
                last["ends_at_reference_end"] = entry["ends_at_reference_end"]
                last["pieces"] = last.get("pieces", 1) + 1
                continue
        found.append(entry)
    return found


def related_occurrences(decided: list[dict], *, time_offset: float = 0.0) -> list[dict]:
    """EXACT以外で参照の録音が現れる区間（EDIT・ADDITIONAL・COMPONENT）と、その領域の地図（来歴・再構成の材料）。"""
    found = []
    for item in sorted(decided, key=lambda x: x["start_seconds"]):
        relation = item.get("reference_relation") or {}
        if relation.get("state") not in (EDIT, ADDITIONAL, COMPONENT) or REASON_EDIT in (item.get("reasons") or ()):
            continue
        found.append({
            "start_seconds": item["start_seconds"], "end_seconds": item["end_seconds"],
            "source_start_seconds": round(item["start_seconds"] + time_offset, 3),
            "source_end_seconds": round(item["end_seconds"] + time_offset, 3),
            "relation": relation["state"], "basis": relation.get("basis"),
            "reference_sha256": relation["reference"]["reference_sha256"],
            "display_title": relation["reference"].get("display_title"),
            "donor_reconstructable": bool(relation.get("donor_reconstructable")),
            **({"local_nomination": dict(relation["local_nomination"])}
               if relation.get("local_nomination") is not None else {}),
            **({"authority_hold_reasons": list(relation["authority_hold_reasons"])}
               if relation.get("authority_hold_reasons") else {}),
            **({"measured_scope_seconds": list(relation["measured_scope_seconds"])}
               if relation.get("measured_scope_seconds") is not None else {}),
            **({"components": relation["components"]} if relation.get("components") else {}),
            "regions": [{**region, "source_start_seconds": round(region["start_seconds"] + time_offset, 3),
                         "source_end_seconds": round(region["end_seconds"] + time_offset, 3)}
                        for region in relation.get("regions", [])],
            "extraction": "normal_decision_kept",
        })
    return found


def summary(decided: list[dict]) -> dict:
    """話ごとの分類：関係ごとの区間数と秒数。"""
    counts = {state: {"segments": 0, "seconds": 0.0} for state in STATES}
    for item in decided:
        state = (item.get("reference_relation") or {}).get("state", UNKNOWN)
        counts[state]["segments"] += 1
        counts[state]["seconds"] = round(counts[state]["seconds"] + item["end_seconds"] - item["start_seconds"], 3)
    return counts


def compact(relation: dict | None) -> dict | None:
    """公開cueの来歴へ入れる、関係の要約（セルごとの値は入れない。領域の地図は入れる）。"""
    if not relation:
        return None
    view = {"state": relation["state"], "basis": relation.get("basis")}
    if relation.get("stereo_accounting") is not None:
        view["stereo_accounting"] = relation["stereo_accounting"]
    if relation.get("mono_state") is not None:
        view["mono_state"] = relation["mono_state"]
    if relation.get("local_nomination") is not None:
        view["local_nomination"] = dict(relation["local_nomination"])
    if relation.get("authority_hold_reasons"):
        view["authority_hold_reasons"] = list(relation["authority_hold_reasons"])
    if relation.get("measured_scope_seconds") is not None:
        view["measured_scope_seconds"] = list(relation["measured_scope_seconds"])
    if relation.get("reference"):
        reference = relation["reference"]
        view["reference"] = {name: reference.get(name) for name in
                             ("reference_sha256", "display_title", "title_source", "offset_seconds", "independent_distribution")}
    if relation.get("other_alignments"):
        view["other_offsets_seconds"] = [item["offset_seconds"] for item in relation["other_alignments"]]
    if relation.get("candidates"):
        view["candidates"] = [{name: item.get(name) for name in ("reference_sha256", "display_title", "z")}
                              for item in relation["candidates"]]
    if relation.get("regions"):
        view["regions"] = [{name: region[name] for name in
                            ("start_seconds", "end_seconds", "kind", "reference_sha256", "offset_seconds",
                             "reference_range_seconds") if name in region}
                           for region in relation["regions"]]
    if "donor_reconstructable" in relation:
        view["donor_reconstructable"] = bool(relation["donor_reconstructable"])
    if relation.get("components"):
        view["components"] = relation["components"]
    return view


def identity_state(relation: dict | None, *, references_registered: bool) -> str:
    """公開cueの来歴の `identity_state`。参照が無い時と、関係が無い時は従来の値。"""
    if not references_registered:
        return "unresolved_no_registered_reference"
    state = (relation or {}).get("state", UNKNOWN)
    if state == EDIT:
        return "released_recording_edit"
    if state == ADDITIONAL:
        return "released_recording_with_additional_material"
    if state == COMPONENT:
        return "released_recording_component_variant"
    if state == RELATED:
        return "related_version_candidate"
    if state == EXACT:
        return "known_exact"
    return "unresolved_no_registered_reference_match"


@dataclass
class Gate:
    """一つの話の参照の照合結果。`relate` で区間を分け、`record` で記録の本文を作る。"""

    library: ReferenceLibrary
    query: np.ndarray
    matches: list[identity.RecordingMatch]
    music: np.ndarray | None
    interference: list[np.ndarray]
    recipe: RelationRecipe
    timing: dict
    music_path: Path | None = None

    @classmethod
    def prepare(cls, analysis: Path, reference_paths: list[Path] = (), *, cache_dir: Path | None,
                music: Path | None = None, interference: Iterable[Path] = (),
                recipe: RelationRecipe | None = None, loader=identity.decode_mono,
                metadata: dict[str, dict] | None = None, library: ReferenceLibrary | None = None) -> "Gate":
        """`library` を渡すと、参照の知識（`reference_store`）から作った索引をそのまま使う。"""
        import time

        recipe = recipe or default_recipe()
        recipe.validate()
        clock = time.perf_counter()
        if library is None:
            library = ReferenceLibrary.build(reference_paths, cache_dir=cache_dir, loader=loader, metadata=metadata)
        timing = {"index_seconds": round(time.perf_counter() - clock, 2)}
        clock = time.perf_counter()
        query = loader(Path(analysis))
        music_samples = loader(Path(music)) if music is not None else None
        others = [loader(Path(path)) for path in interference]
        timing["decode_seconds"] = round(time.perf_counter() - clock, 2)
        clock = time.perf_counter()
        matches = identity.match(query, library.index, collapse=False)
        timing["match_seconds"] = round(time.perf_counter() - clock, 2)
        return cls(library, query, matches, music_samples, others, recipe, timing,
                   Path(music) if music is not None else None)

    def known_spans(self) -> list[dict]:
        return [{"start_seconds": round(m.query_start_seconds, 3), "end_seconds": round(m.query_end_seconds, 3),
                 "reference": Path(m.reference_path).name, "reference_sha256": m.reference_sha256,
                 "offset_seconds": m.offset_seconds}
                for m in self.matches if m.identity_state == "known_exact"]

    def relate(self, segments: list[dict]) -> list[dict]:
        import time

        clock = time.perf_counter()
        result = relate_segments(segments, query=self.query, library=self.library, matches=self.matches,
                                 recipe=self.recipe, music=self.music, interference=self.interference)
        from lss.bgm import reference_stereo
        result = reference_stereo.guard(result, music=self.music_path, library=self.library, recipe=self.recipe)
        self.timing["relation_seconds"] = round(time.perf_counter() - clock, 2)
        return result

    def record(self, decided: list[dict], *, time_offset: float = 0.0) -> dict:
        return {"state": "checked", "recipe": self.recipe.to_dict(), "recipe_id": self.recipe.recipe_id,
                "reference_count": len(self.library.index.references),
                "episode_classification": summary(decided),
                "occurrences": occurrences(decided, time_offset=time_offset),
                "related_occurrences": related_occurrences(decided, time_offset=time_offset),
                "timing": dict(self.timing)}


def not_checked(error: BaseException, recipe: RelationRecipe | None = None) -> dict:
    """照合できなかった記録。関係はすべてUNKNOWNのままにする（何も消さない）。"""
    recipe = recipe or default_recipe()
    return {"state": "not_checked", "error": type(error).__name__, "recipe_id": recipe.recipe_id,
            "episode_classification": None, "occurrences": [], "related_occurrences": []}
