#Requires -Version 5.1
<#
.SYNOPSIS
    LSS をローカルで PyInstaller ビルドする(単一exe化ガイド=onedir・T322)。
.DESCRIPTION
    既存アイコンの確認→PyInstallerビルド→構造スモークテストを行う。
    完了後は deploy-to-freesoft.ps1 -BuildOutput 'dist\LSS' で配備する。
#>

param([double]$StorageGB = 12, [switch]$StorageManaged)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$icon = Join-Path $root "assets\app.ico"
Set-Location $root

if (-not (Test-Path $python)) {
    throw "仮想環境が見つかりません: $python。先に setup-venv.ps1 を実行してください。"
}

if (-not $StorageManaged) {
    # New build payloads go to the data SSD. Existing output stays untouched;
    # the scheduled layout mover handles its verified relocation.
    $pathsTool = Join-Path $root 'scripts\storage_paths.py'
    if (Test-Path -LiteralPath $pathsTool) {
        & $python $pathsTool prepare-build --repo $root
        if ($LASTEXITCODE -ne 0) { throw 'Cannot prepare policy-owned build paths' }
    }
    # Reserve peak growth before any build output; the suspended-child supervisor owns the entire process tree.
    & $python (Join-Path $root "scripts\storage_budget.py") run --required-gb $StorageGB --protect $root `
        --protect (Join-Path $root "build") --protect (Join-Path $root "dist") --archive (Join-Path $root "build") --purpose "LSS build" -- `
        powershell -NoProfile -File $PSCommandPath -StorageGB $StorageGB -StorageManaged
    exit $LASTEXITCODE
}
if (-not $env:LSS_STORAGE_RESERVATION) { throw "StorageManaged requires the storage reservation supervisor" }

Write-Host "[build] アイコン確認"
if (-not (Test-Path $icon)) {
    throw "アプリアイコンが見つかりません: $icon"
}

Write-Host "[build] 依存バージョン記録"
& $python -m pip freeze | Out-File -Encoding utf8 build-manifest.txt
if ($LASTEXITCODE -ne 0) {
    throw "依存バージョンの記録に失敗しました（終了コード: $LASTEXITCODE）"
}

Write-Host "[build] ビルド（spec 経由・数分〜数十分かかる場合あり）"
& $python -m PyInstaller LSS.spec --clean --noconfirm
if ($LASTEXITCODE -ne 0) {
    throw "PyInstallerビルドに失敗しました（終了コード: $LASTEXITCODE）"
}

$dist = Join-Path $root "dist\LSS"
$need = @("LSS.exe", "_internal\webview\lib\Microsoft.Web.WebView2.Core.dll", "_internal\design\reference\LSS App.dc.html", "_internal\assets\web\react.production.min.js", "_internal\assets\system\tray-ready-dark-f0.ico", "_internal\lss\ui\web\tray_menu.html","_internal\vendor\msst", "_internal\vendor\msst\models\bs_roformer\mel_band_roformer.py", "_internal\vendor\msst\models\bs_roformer\bs_roformer.py", "_internal\bin\ffmpeg.exe", "_internal\ledger\models.tsv", "_internal\lss\validation\episode_discovery.py", "_internal\lss\validation\activity_evidence.py", "_internal\lss\bgm\activity.py", "_internal\lss\bgm\candidate_store.py", "_internal\lss\output\writer.py", "_internal\lss\core\audio_input.py", "_internal\lss\core\audio_timeline.py", "_internal\lss\core\session.py", "_internal\lss\bgm\episode_cues.py", "_internal\lss\bgm\cue_boundaries.py", "_internal\lss\bgm\second_pass.py", "_internal\runtime-sources\lss\bgm\sfc_analysis.py", "_internal\vendor\sfc_locoformer\spectral_feature_compression\core\model\bslocoformer.py", "_internal\lss\bgm\vocal_recovery.py", "_internal\runtime-sources\lss\bgm\four_stem_analysis.py", "_internal\lss\bgm\cue_semantics.py", "_internal\lss\bgm\cue_publication.py", "_internal\lss\bgm\cue_representation.py", "_internal\lss\bgm\dialogue_residual.py", "_internal\lss\bgm\published_duplicates.py", "_internal\lss\bgm\cue_scope.py", "_internal\lss\bgm\short_cues.py", "_internal\lss\bgm\supported_scope.py", "_internal\lss\bgm\cue_readiness.py", "_internal\lss\bgm\current_best.py", "_internal\lss\bgm\identity.py", "_internal\lss\bgm\reference_relations.py", "_internal\lss\bgm\versions.py", "_internal\lss\bgm\reference_reconstruction.py","_internal\runtime-sources\lss\bgm\event_tagging.py", "_internal\vendor\facing_the_music\core\models\e2e\bandit\bandsplit.py", "_internal\vendor\facing_the_music\LICENSE","_internal\librosa\core\notation.py", "_internal\librosa\core\audio.py")
$need += @("_internal\runtime-sources\lss\bgm\gpu_execution.py")
$need += @("_internal\lss\bgm\scene_job.py", "_internal\lss\bgm\scene_runtime.py",
           "_internal\lss\bgm\scene_native.py", "_internal\lss\bgm\scene_models.py",
           "_internal\lss\bgm\scene_names.py")
$need += @("_internal\lss\bgm\cue_formation.py", "_internal\lss\bgm\stereo_continuity.py", "_internal\lss\bgm\recording_support.py",
           "_internal\lss\bgm\occurrence_support.py", "_internal\lss\bgm\episode_usage.py",
           "_internal\lss\bgm\release_availability.py")
# 伴奏の選択的な原音合成（Recipeのpostprocessingがこのsourceのbytesをidentityへ読む）と、
# 欲しい音（SOUND）の実行sourceとSAM Audio同梱実装。欠けると該当ジョブが実行時に失敗する。
$need += @("_internal\lss\separation\selective.py", "_internal\lss\separation\regions.py", "_internal\vendor\sam_audio",
           "_internal\runtime-sources\lss\sound\sam_audio.py", "_internal\runtime-sources\lss\sound\t5_text.py",
           "_internal\runtime-sources\lss\sound\pe_frame.py", "_internal\runtime-sources\lss\sound\translate.py",
           "_internal\runtime-sources\lss\sound\xling.py", "_internal\runtime-sources\lss\sound\interpret.py",
           "_internal\runtime-sources\lss\sound\intent.py", "_internal\runtime-sources\lss\sound\lexicon.py",
           "_internal\runtime-sources\lss\sound\plan.py", "_internal\runtime-sources\lss\sound\extract.py",
           "_internal\runtime-sources\lss\sound\verify.py", "_internal\runtime-sources\lss\sound\job.py",
           "_internal\runtime-sources\lss\sound\audioset_relations.py")
$need += @("_internal\lss\bgm\memory_support.py", "_internal\lss\bgm\music_corroboration.py",
           "_internal\lss\bgm\rendered_content.py")
$need += @("_internal\lss\bgm\memory_refresh.py")
foreach ($r in $need) {
    $p = Join-Path $dist $r
    if (-not (Test-Path $p)) { throw "MISSING: $p" }
}
Write-Host "[build] structural smoke OK"

Write-Host "[build] runtime source identity check"
& $python (Join-Path $root "scripts\check_runtime_sources.py") $dist --source $root --require-data-files
if ($LASTEXITCODE -ne 0) {
    throw "Runtime source resources are missing or differ from the source tree."
}

Write-Host "[build] OK -> $dist"
Write-Host "[build] deploy with: powershell -File $root\deploy-to-freesoft.ps1 -BuildOutput '$dist'"
