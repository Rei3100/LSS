"""候補モデルの採否フィルタ F1-F4（D-6①・T310 spec §6 Step3）。

F1: 匿名HTTPSで直接ダウンロードできること
F2: 手動のYAML編集・手動パッチを要さないこと
F3: ライセンス旗が立っていないこと（既定=無効。理由は check_f3_license_flag のdocstring）
F4: LSSの推論経路（D-2の第一/第二）で実機動作すること
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import requests
import soundfile as sf

from lss.paths import app_root, resource_root

# use_torch_checkpoint等、手動編集を要求する既知のYAMLキー（将来増える前提でlistにする）。
MANUAL_YAML_KEYS: tuple[str, ...] = ("use_torch_checkpoint",)

# F3で「保留」に倒す文字列パターン（空/不明/条件付き共有・限定配布）。
_CONDITIONAL_LICENSE_MARKERS: tuple[str, ...] = (
    "限定", "専用", "無し", "条件付", "premium", "proprietary",
)

_LOGIN_URL_MARKERS: tuple[str, ...] = ("login", "signin", "sign-in", "accounts.google", "/auth")


def check_f1_anonymous_download(distributor_url: str, timeout: float = 10.0) -> tuple[bool, bool]:
    """HEADリクエストでdl_ok, auth_requiredを返す。curl相当（requests.head, allow_redirects=True）。

    401/403/ログイン誘導リダイレクト→auth_required=True。
    到達不能（例外・不明URL等）はdl_ok=False, auth_required=Falseとして返す
    （呼び出し側で状態=候補のまま保留させ、個別確認を促す）。
    """
    try:
        resp = requests.head(distributor_url, allow_redirects=True, timeout=timeout)
    except requests.RequestException:
        return False, False

    if resp.status_code in (401, 403):
        return False, True

    final_url = resp.url.lower()
    if any(marker in final_url for marker in _LOGIN_URL_MARKERS):
        return False, True

    if 200 <= resp.status_code < 400:
        return True, False

    return False, False


def check_f2_needs_manual_yaml(config_path: Path) -> bool:
    """YAML内にuse_torch_checkpoint等の手動編集要求キーがあるかを走査。

    既知キー集合はMANUAL_YAML_KEYS（将来増える前提でlistにする）。
    """
    text = Path(config_path).read_text(encoding="utf-8")
    return any(key in text for key in MANUAL_YAML_KEYS)


def check_f3_license_flag(license_name: str, redistribute: bool = False) -> str:
    """ライセンス旗の判定。既定（redistribute=False）は常に'候補'＝ライセンスでは落とさない。

    本人裁定2026-08-14（T326）: LSSは個人使用専用で、モデル重みを第三者へ再配布しない
    （ADR-0001 D9①・GitHubリモートも既定=持たない）。再配布しない限り第三者モデルの配布条件は
    採否の阻却要因にならないため、ライセンス未記載（HuggingFaceの多くのRoformerモデルが該当）を
    理由に'保留'へ倒す旧仕様を撤回する。旧仕様は候補7件を全て保留に固定し、Reference上位の
    現行モデルが1件も評価対象にならないという実害を出していた。

    redistribute=True は「例外的にGitHub等で公開・再配布する」場合の判定用に旧ロジックを残したもの。
    その時だけライセンス条項の有無を採否に効かせる（空/不明/条件付き→'保留'）。
    """
    if not redistribute:
        return "候補"
    name = (license_name or "").strip()
    if not name or name == "不明":
        return "保留"
    if any(marker in name for marker in _CONDITIONAL_LICENSE_MARKERS):
        return "保留"
    return "候補"


def check_f4_infers_on_path(
    model_id: str,
    infer_tool: str,
    sample_audio: Path,
    *,
    arch: str = "",
    ckpt_path: Path | None = None,
    config_path: Path | None = None,
    store_dir: Path | None = None,
) -> bool:
    """D-2第一(audio-separator)→ダメなら第二(MSST)の順に1曲流し、例外を捕捉。

    成功=True、例外=False（呼び出し側で状態=除外に倒す）。
    MSST経路は`arch`(config.training.model_type相当)・`ckpt_path`・`config_path`が要る
    （T326実測: これらを渡さない旧実装ではproc_folderが必ず失敗しており、MSST側は事実上
    未検証のままだった）。取得は`fetch.ensure_model_files`が担当する。
    """
    # 出力先は必ずリポ配下（既定）。試験曲の親フォルダは利用者の領域なので書き込まない（T326実測で是正）。
    out = store_dir or (app_root() / ".f4_probe" / model_id)
    out.mkdir(parents=True, exist_ok=True)
    if infer_tool == "audio-separator":
        attempt = lambda attempt_dir: _try_audio_separator(model_id, sample_audio, attempt_dir)
    elif infer_tool == "msst":
        attempt = lambda attempt_dir: _try_msst(
            arch or model_id, sample_audio, ckpt_path, config_path, attempt_dir,
        )
    else:
        return False
    try:
        with TemporaryDirectory(prefix="f4-", dir=out) as attempt_dir:
            attempt(Path(attempt_dir))
        return True
    except Exception:
        return False


def _require_audio_outputs(paths: list[Path], store_dir: Path) -> None:
    if not paths:
        raise FileNotFoundError(f"分離出力がありません: {store_dir}")
    store_root = store_dir.resolve()
    for path in paths:
        resolved = (path if path.is_absolute() else store_dir / path).resolve()
        if not resolved.is_relative_to(store_root):
            raise ValueError(f"分離出力が作業領域の外にあります: {resolved}")
        info = sf.info(resolved)
        if info.frames <= 0 or info.channels <= 0 or info.samplerate <= 0:
            raise ValueError(f"分離出力が空か不正です: {resolved}")


def _try_audio_separator(model_id: str, sample_audio: Path, store_dir: Path) -> None:
    from audio_separator.separator import Separator

    separator = Separator(output_dir=str(store_dir))
    separator.load_model(model_filename=model_id)
    outputs = separator.separate(str(sample_audio))
    if not isinstance(outputs, list) or not all(isinstance(path, str) for path in outputs):
        raise ValueError("audio-separatorから不正な出力一覧が返りました")
    _require_audio_outputs([Path(path) for path in outputs], store_dir)


def _try_msst(
    arch: str,
    sample_audio: Path,
    ckpt_path: Path | None,
    config_path: Path | None,
    store_dir: Path,
) -> None:
    import sys

    if ckpt_path is None or config_path is None:
        raise ValueError("MSST経路にはckpt_pathとconfig_pathが必要です（台帳のconfig_url列から取得する）")

    msst_root = resource_root() / "vendor" / "msst"
    if str(msst_root) not in sys.path:
        sys.path.insert(0, str(msst_root))
    from inference import proc_folder  # type: ignore

    store_dir.mkdir(parents=True, exist_ok=True)
    proc_folder({
        "model_type": arch,
        "config_path": str(config_path),
        "start_check_point": str(ckpt_path),
        "input_folder": str(sample_audio.parent),
        "store_dir": str(store_dir),
    })
    outputs = [
        path for path in store_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
    ]
    _require_audio_outputs(outputs, store_dir)
