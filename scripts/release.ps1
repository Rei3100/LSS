#Requires -Version 5.1
<#
.SYNOPSIS
    コミット済みのHEADをcandidateとしてビルドし、隔離受入に合格したものだけを配備する。
.DESCRIPTION
    1. HEADを別の作業ツリーへ取り出す（未コミットの変更をビルドへ混ぜない）。
       改行を変換しない。providerとadapterのbytesが変わると実行環境IDが変わり、
       同じ処理の過去の成果を再利用できなくなるため。
    2. build.ps1 でcandidateを作り（構造確認を含む）、同梱したproviderとadapterが
       ソースと同じbytesかを確かめる（scripts/check_runtime_sources.py）。
    3. 配備版受入をcandidateの実行物に対して隔離環境で行う。不合格なら配備先に触れない。
    4. deploy-to-freesoft.ps1 -KeepPrevious で昇格する。直前版は退避に残る。
    5. 配備先に対して同じ受入を行い、合格なら -Finalize、不合格なら -Rollback で既知良好版へ戻す。
    LSSが起動している間は開始しない（配備スクリプトは実行中のLSSを停止するため）。

    powershell -File scripts\release.ps1
#>
param(
    [string]$Worktree = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "lss-build"),
    [string]$DeployBase = $env:LSS_DEPLOY_BASE
)

$ErrorActionPreference = "Stop"
if (-not $DeployBase) { throw "Set LSS_DEPLOY_BASE or pass -DeployBase." }
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"
$deploy = Join-Path $root "deploy-to-freesoft.ps1"
$liveExe = Join-Path $DeployBase "LSS\LSS.exe"

function Invoke-Acceptance([string]$Executable) {
    $env:LSS_RUN_DEPLOYED_ACCEPTANCE = "1"
    $env:LSS_ACCEPTANCE_EXECUTABLE = $Executable
    try {
        # pytestの出力を関数の戻り値へ混ぜない。混ざると戻り値が配列になり、不合格でも真と判定される。
        & $python -m pytest -q -p no:cacheprovider (Join-Path $root "tests\test_deployed_backend_acceptance.py") | Out-Host
        return ($LASTEXITCODE -eq 0)
    } finally {
        Remove-Item Env:LSS_RUN_DEPLOYED_ACCEPTANCE, Env:LSS_ACCEPTANCE_EXECUTABLE -ErrorAction SilentlyContinue
    }
}

function Remove-Worktree {
    $junction = Join-Path $Worktree ".venv"
    if (Test-Path -LiteralPath $junction) {
        # 本体の.venvへのjunction。リンクだけを外し、リンク先には降りない。
        [IO.Directory]::Delete($junction, $false)
    }
    if (Test-Path -LiteralPath $Worktree) {
        & git -C $root -c core.longpaths=true worktree remove --force $Worktree
    }
    & git -C $root worktree prune
}

if (Get-Process LSS -ErrorAction SilentlyContinue) {
    throw "LSSが起動しています。終了してから実行してください"
}
if (Test-Path -LiteralPath $Worktree) {
    throw "作業ツリーの置き場所が既に使われています: $Worktree"
}

Write-Host "[release] HEADを作業ツリーへ取り出す"
& git -C $root -c core.longpaths=true -c core.autocrlf=false worktree add --detach $Worktree HEAD
if ($LASTEXITCODE -ne 0) { throw "作業ツリーを作れません" }
try {
    New-Item -ItemType Junction -Path (Join-Path $Worktree ".venv") -Target (Join-Path $root ".venv") | Out-Null
    # Gitが追跡しないvendorのコードを写す。__pycache__は持ち込まない。
    $ignored = & git -C $root ls-files --others --ignored --exclude-standard -- vendor
    foreach ($relative in $ignored) {
        if ($relative -match "__pycache__") { continue }
        $destination = Join-Path $Worktree $relative
        New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $root $relative) -Destination $destination
    }

    Write-Host "[release] candidateをビルド"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Worktree "build.ps1")
    if ($LASTEXITCODE -ne 0) { throw "ビルドに失敗しました。配備先には触れていません" }
    $candidate = Join-Path $Worktree "dist\LSS"
    & $python (Join-Path $root "scripts\check_runtime_sources.py") $candidate
    if ($LASTEXITCODE -ne 0) { throw "candidateのproviderかadapterがソースと違います。配備先には触れていません" }

    Write-Host "[release] candidateを隔離して受入"
    if (-not (Invoke-Acceptance (Join-Path $candidate "LSS.exe"))) {
        throw "candidateが受入に不合格です。配備先には触れていません"
    }

    Write-Host "[release] 受入待ちとして昇格"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $deploy -BuildOutput $candidate -DeployBase $DeployBase -KeepPrevious
    if ($LASTEXITCODE -ne 0) { throw "配置に失敗しました（配置スクリプトが既知良好版へ戻しています）" }

    Write-Host "[release] 配備先で受入"
    if (Invoke-Acceptance $liveExe) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $deploy -DeployBase $DeployBase -Finalize
        if ($LASTEXITCODE -ne 0) { throw "受入済み配置の確定に失敗しました" }
        Write-Host "[release] 完了: 配備先はcandidateに置き換わりました"
    } else {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $deploy -DeployBase $DeployBase -Rollback
        throw "配備先の受入に不合格のため既知良好版へ戻しました"
    }
} finally {
    Remove-Worktree
}
