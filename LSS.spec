# -*- mode: python ; coding: utf-8 -*-
# 薄い凍結launcher/core案を実測するonedir構成。配備方式そのものは製品決定ではない。
# EXE(exclude_binaries=True)+COLLECTの型を使い、provider/model/stateは独立境界に保つ。
#
# worker役は同じexeへ`--worker`引数で分岐させる(lss_gui.py)。
# 現在workerが実装しているtorch/MSST経路を本体へ含める。未実装のaudio-separator経路と
# ONNX Runtime全体は、利用不能なコードと依存を配布物へ先行同梱しない。

from PyInstaller.utils.hooks import collect_data_files

datas = [
    ("vendor/msst", "vendor/msst"),  # sys.path挿入で生ソースのままimportする(静的解析させない)
    ("bin/ffmpeg.exe", "bin"),
    ("ledger", "ledger"),
    ("assets", "assets"),
    # 画面は凍結Claude Designのページそのもの（lss/ui/web）。原本とランタイムを同梱する。
    ("design/reference/LSS App.dc.html", "design/reference"),
    ("lss/ui/web/runtime.js", "lss/ui/web"),
    ("lss/ui/web/bridge.js", "lss/ui/web"),
    ("lss/ui/web/tray_menu.html", "lss/ui/web"),
    ("VERSION", "."),
    # PYZ内のmoduleには物理パスがないため、実行コードと同じsourceをRuntimeRevision用に保持する。
    ("lss/harness/infer.py", "runtime-sources/lss/harness"),
    # BGM候補Recipeも実装bytesをidentityへ含める。episode-only製品入口が参照する
    # exact sourceをmoduleと同じ相対位置へ置き、packaged実行で来歴を欠落させない。
    ("lss/validation/episode_discovery.py", "lss/validation"),
    ("lss/validation/activity_evidence.py", "lss/validation"),
    ("lss/bgm/activity.py", "lss/bgm"),
    ("lss/bgm/candidate_store.py", "lss/bgm"),
    # Optional source-bound visual text: the WinRT bridge must exist beside its module.
    ("lss/bgm/frame_text.py", "lss/bgm"),
    ("lss/bgm/source_visual.py", "lss/bgm"),
    ("lss/bgm/source_visual_job.py", "lss/bgm"),
    ("lss/bgm/source_visual_backfill.py", "lss/bgm"),
    ("lss/bgm/windows_frame_text.ps1", "lss/bgm"),
    # Pinned yt-dlp release-signing keys, read beside provider.py by pinned_keys(); without them the verified install is HELD.
    ("lss/acquisition/keys/yt-dlp.asc", "lss/acquisition/keys"),
    ("lss/output/writer.py", "lss/output"),
    ("lss/output/finalizer.py", "lss/output"),
    ("lss/output/scan.py", "lss/output"),
    ("lss/output/tags.py", "lss/output"),
    ("lss/core/audio_input.py", "lss/core"),
    ("lss/core/audio_timeline.py", "lss/core"),
    ("lss/core/session.py", "lss/core"),
    ("lss/separation/selective.py", "lss/separation"),
    ("lss/separation/regions.py", "lss/separation"),
    # 単話BGM候補（SFC解析＋episode_cues）。候補の中身を決める実装と解析器のadapterのbytesを
    # 識別へ含めるので、実行コードと同じsourceを同梱する。解析器のmodel実装はvendorのまま読む。
    ("lss/bgm/episode_cues.py", "lss/bgm"),
    ("lss/bgm/cue_boundaries.py", "lss/bgm"),
    ("lss/bgm/second_pass.py", "lss/bgm"),
    ("lss/bgm/sfc_analysis.py", "runtime-sources/lss/bgm"),
    ("lss/bgm/gpu_execution.py", "runtime-sources/lss/bgm"),
    ("vendor/sfc_locoformer", "vendor/sfc_locoformer"),
    # 歌唱回収（効果音の出力の候補区間だけを4分類へ通す）。採否の実装bytesはRecipeへ、4分類のadapterと
    # 同梱実装（公開MIT層、upstreamのbytesのまま）は4分類の版（analyzer_id）へ入る。
    ("lss/bgm/vocal_recovery.py", "lss/bgm"),
    ("lss/bgm/four_stem_analysis.py", "runtime-sources/lss/bgm"),
    ("vendor/facing_the_music", "vendor/facing_the_music"),
    # 意味判定（曲単位への分け直しと公開の判定）と公開cueの組み立て。bytesがRecipeへ入る。
    ("lss/bgm/cue_semantics.py", "lss/bgm"),
    ("lss/bgm/cue_formation.py", "lss/bgm"),
    ("lss/bgm/stereo_continuity.py", "lss/bgm"),
    ("lss/bgm/memory_support.py", "lss/bgm"),
    ("lss/bgm/memory_refresh.py", "lss/bgm"),
    ("lss/bgm/recording_support.py", "lss/bgm"),
    ("lss/bgm/occurrence_support.py", "lss/bgm"),
    ("lss/bgm/music_corroboration.py", "lss/bgm"),
    ("lss/bgm/rendered_content.py", "lss/bgm"),
    ("lss/bgm/episode_usage.py", "lss/bgm"),
    ("lss/bgm/cue_publication.py", "lss/bgm"),
    ("lss/bgm/cue_representation.py", "lss/bgm"),
    ("lss/bgm/dialogue_residual.py", "lss/bgm"),
    ("lss/bgm/published_duplicates.py", "lss/bgm"),
    ("lss/bgm/cue_scope.py", "lss/bgm"),
    ("lss/bgm/short_cues.py", "lss/bgm"),
    ("lss/bgm/supported_scope.py", "lss/bgm"),
    ("lss/bgm/cue_readiness.py", "lss/bgm"),
    ("lss/bgm/current_best.py", "lss/bgm"),
    # Frozen scene-name worker and native runtime. The Python Transformers backend remains source-only.
    ("lss/bgm/scene_job.py", "lss/bgm"),
    ("lss/bgm/scene_runtime.py", "lss/bgm"),
    ("lss/bgm/scene_native.py", "lss/bgm"),
    ("lss/bgm/scene_models.py", "lss/bgm"),
    ("lss/bgm/scene_names.py", "lss/bgm"),
    # 録音照合。指紋の保存物の識別子（fingerprint_settings_id）が自分のsource bytesを読むので実体を置く。
    ("lss/bgm/identity.py", "lss/bgm"),
    # 参照録音との関係（同じ録音の連続した抜粋・編集・別版の候補・未知）。参照がある時のrunの識別子へ
    # 実装の内容hashが入る（`application._bgm_reference_policy`）。
    ("lss/bgm/reference_relations.py", "lss/bgm"),
    ("lss/bgm/versions.py", "lss/bgm"),
    ("lss/bgm/reference_reconstruction.py", "lss/bgm"),
    ("lss/bgm/release_availability.py", "lss/bgm"),
    # 音響イベントのタグ付け（PANNs CNN14）。adapterのbytesがタグ付けの版（analyzer_id）へ入る。
    # model実装はadapterの中に同じ演算で書いてあり、重みは初回に固定sha256で取得する。
    ("lss/bgm/event_tagging.py", "runtime-sources/lss/bgm"),
    # 欲しい音（言葉で指定した音）。言葉の分離（SAM Audio）の同梱実装はupstreamのbytesのまま読み、
    # adapter・文のencoder・要求の読み方と経路の実装のbytesが抽出の版とRecipeへ入るので、同じsourceを置く。
    ("vendor/sam_audio", "vendor/sam_audio"),
    ("lss/sound/sam_audio.py", "runtime-sources/lss/sound"),
    ("lss/sound/t5_text.py", "runtime-sources/lss/sound"),
    # 言葉の時刻の検出（PE-A-Frame）と、語彙に無い日本語の読み（訳と意味の確かめ）も同じく版とRecipeへ入る。
    ("lss/sound/pe_frame.py", "runtime-sources/lss/sound"),
    ("lss/sound/translate.py", "runtime-sources/lss/sound"),
    ("lss/sound/xling.py", "runtime-sources/lss/sound"),
    ("lss/sound/interpret.py", "runtime-sources/lss/sound"),
    ("lss/sound/intent.py", "runtime-sources/lss/sound"),
    ("lss/sound/lexicon.py", "runtime-sources/lss/sound"),
    ("lss/sound/plan.py", "runtime-sources/lss/sound"),
    ("lss/sound/extract.py", "runtime-sources/lss/sound"),
    ("lss/sound/verify.py", "runtime-sources/lss/sound"),
    ("lss/sound/job.py", "runtime-sources/lss/sound"),
    ("lss/sound/audioset_relations.py", "runtime-sources/lss/sound"),
]
# Numbaは凍結moduleの相対`__file__`にも物理sourceを要求する。librosaの
# cache=True関数を初回importできるよう、実行bytecodeと同じsourceを保持する。
datas += collect_data_files("librosa", include_py_files=True)
datas += collect_data_files("certifi", includes=["cacert.pem"])
binaries = []
hiddenimports = [
    "lss.bgm.source_visual",
    # The scene worker dynamically selects the pinned native bridge.
    "lss.bgm.scene_job",
    "lss.bgm.scene_runtime",
    "lss.bgm.scene_native",
    "lss.bgm.scene_models",
    "lss.bgm.scene_names",
    # WebView2ホスト（pywebview の WinForms/EdgeChromium 経路と pythonnet）
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "clr",
    # vendor/msstはdataとして同梱するため、そこから動的に読み込まれる実行依存を列挙する。
    "torch",
    "torchaudio",
    "librosa",
    "soundfile",
    "mutagen",
    "omegaconf",
    "ml_collections",
    "beartype",
    "rotary_embedding_torch",
    "einops",
    "tqdm",
    "yaml",
    # vendor/sfc_locoformerはdataとして同梱するため、そこから読み込まれる実行依存を列挙する。
    "einops.layers.torch",
    "packaging.version",
    "spafe.fbanks.bark_fbanks",
    "spafe.utils.converters",
    "scipy.signal",
    "torchaudio.transforms",
    "torchaudio.functional.functional",
    "torch.utils.checkpoint",
    # vendor/facing_the_musicはdataとして同梱するため、そこから読み込まれる実行依存を列挙する。
    "torch.nn.modules.activation",
    "torch.nn.modules.rnn",
    "torch.backends.cuda",
    "torchaudio.functional",
]

# torch/librosaのPyInstaller hookは、インストール済みというだけで推論に未使用の巨大な
# 任意依存まで収集する。LSSの実行経路とvendor/msstから参照されないものは配布しない。
excludes = [
    # 画面はWebView2へ移った。Qtは配布物に含めない（LV-UI-001）。
    "PySide6",
    "shiboken6",
    "audio_separator",
    "onnx",
    "onnxruntime",
    "sklearn",
    "torchvision",
    "pytest",
    "_pytest",
    "tensorboard",
    # 場面の命名のTransformers経路はソースからだけ動く（scene_runtime）。開発のvenvに入っていても配布しない。
    "transformers",
    "bitsandbytes",
    "accelerate",
    "tokenizers",
    "safetensors",
]

a = Analysis(
    ['lss_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LSS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon='assets/app.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='LSS',
    contents_directory='_internal',
)
