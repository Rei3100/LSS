# deploy-to-freesoft.ps1
# ビルド成果物を配備先（-DeployBase または環境変数 LSS_DEPLOY_BASE の下の LSS\）へ配置する（N07）。
# 注意（N11・T297確定）: このリポでは自動更新の配信元にGitHub等の外部サービスを使わない。
# 開発機と実行機は同一PCであり、本スクリプトの再実行＝配置の上書き＝更新そのものが唯一の更新経路。
# アプリ本体に「起動時に最新版を確認する」自己更新コードを実装しないこと。
#
# 使い方: <ビルドコマンドで dist/ 等へ成果物を生成した後> powershell -File deploy-to-freesoft.ps1
# 呼び出し元（ビルドスクリプト）で -BuildOutput にビルド成果物のディレクトリを渡す想定。

# 受入を挟む配置（scripts/release.ps1）:
#   -KeepPrevious  昇格後も直前のランタイムを退避に残し「受入待ち」にする
#   -Finalize      受入に合格した配置を確定し、退避を消す
#   -Rollback      受入待ちの配置を直前の既知良好版へ戻す（可変データは保つ）
param(
    [string]$BuildOutput = "",
    [string]$TargetName = "LSS",
    [string]$DeployBase = $env:LSS_DEPLOY_BASE,
    [switch]$KeepPrevious,
    [switch]$Finalize,
    [switch]$Rollback,
    [int]$SceneStopSeconds = 60
)
if ([string]::IsNullOrWhiteSpace($DeployBase)) { throw "配備先が未指定です。-DeployBase か環境変数 LSS_DEPLOY_BASE で指定してください" }
$acceptanceMode = $Finalize -or $Rollback
if ($Finalize -and $Rollback) { throw "-Finalizeと-Rollbackは同時に指定できません" }

if ([string]::IsNullOrWhiteSpace($BuildOutput)) {
    $BuildOutput = Join-Path $PSScriptRoot "dist\LSS"
}
if ($TargetName -ne [IO.Path]::GetFileName($TargetName) -or
    $TargetName.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0) {
    throw "配置名が不正です: $TargetName"
}
$buildRoot = [IO.Path]::GetFullPath($BuildOutput)
$deployBasePath = [IO.Path]::GetFullPath($DeployBase)
$deployRoot = Join-Path $deployBasePath $TargetName
$targetExe = [IO.Path]::GetFullPath((Join-Path $deployRoot "$TargetName.exe"))

function Test-PathContains([string]$Parent, [string]$Child) {
    $parentWithSeparator = $Parent.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $childWithSeparator = $Child.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    return $childWithSeparator.StartsWith($parentWithSeparator, [StringComparison]::OrdinalIgnoreCase)
}

if (-not $acceptanceMode) {
if (-not (Test-Path -LiteralPath $buildRoot -PathType Container)) {
    Write-Error "ビルド成果物が見つかりません: $BuildOutput（先にビルドしてください）"
    exit 1
}
if ((Test-PathContains $buildRoot $deployRoot) -or (Test-PathContains $deployRoot $buildRoot)) {
    throw "ビルド成果物と配置先は同一または親子のパスにできません: build=$buildRoot deploy=$deployRoot"
}
$buildExe = Join-Path $buildRoot "$TargetName.exe"
if (-not (Test-Path -LiteralPath $buildExe -PathType Leaf)) {
    throw "ビルド成果物に実行ファイルがありません: $buildExe"
}
}

$stageName = ".$TargetName.staging"
$previousName = ".$TargetName.previous"
$journalName = ".$TargetName-deploy.json"
$stageRoot = [IO.Path]::GetFullPath((Join-Path $deployBasePath $stageName))
$previousRoot = [IO.Path]::GetFullPath((Join-Path $deployBasePath $previousName))
$journalPath = [IO.Path]::GetFullPath((Join-Path $deployBasePath $journalName))
$mutableNames = @("ledger", "models", "scene-native", ".state", "ui_state.json", "Downloads", "Backups")
$runtimeManifestName = ".lss-runtime-manifest.json"
$legacyRuntimeNames = @("$TargetName.exe", "_internal", $runtimeManifestName)

function Assert-ManagedPath([string]$Path, [string]$ExpectedName) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if ([IO.Path]::GetDirectoryName($resolved) -ne $deployBasePath -or
        [IO.Path]::GetFileName($resolved) -ne $ExpectedName) {
        throw "配置作業パスが管理範囲外です: $resolved"
    }
}

function Test-IsReparsePoint([IO.FileSystemInfo]$Item) {
    return (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
}

function Assert-NoReparseAncestors([string]$Path) {
    $cursor = [IO.Path]::GetFullPath($Path)
    while (-not (Test-Path -LiteralPath $cursor)) {
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ([string]::IsNullOrEmpty($parent) -or $parent -eq $cursor) { return }
        $cursor = $parent
    }
    $item = Get-Item -LiteralPath $cursor -Force
    while ($null -ne $item) {
        if (Test-IsReparsePoint $item) {
            throw "reparse pointを経由するパスは配置に使えません: $($item.FullName)"
        }
        $item = $item.Parent
    }
}

function Assert-NoReparseTree([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $pending = [Collections.Generic.Stack[IO.DirectoryInfo]]::new()
    $rootItem = Get-Item -LiteralPath $Path -Force
    if (Test-IsReparsePoint $rootItem) {
        throw "reparse pointを配置・削除対象にできません: $($rootItem.FullName)"
    }
    if ($rootItem -is [IO.DirectoryInfo]) { $pending.Push($rootItem) }
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($entry in Get-ChildItem -LiteralPath $directory.FullName -Force) {
            if (Test-IsReparsePoint $entry) {
                throw "reparse pointを配置・削除対象にできません: $($entry.FullName)"
            }
            if ($entry -is [IO.DirectoryInfo]) { $pending.Push($entry) }
        }
    }
}

# 保全対象（可変・未知）エントリ内のreparse point（junction等）は不透明なリンクとして扱う。
# ランタイムエントリ・ルート自身はreparse pointを拒否し、保全エントリは列挙せず（リンク先に触れず）リンクごと移す。
function Assert-NoReparseRuntime([string]$Root, [string[]]$RuntimeEntries) {
    if (-not (Test-Path -LiteralPath $Root)) { return }
    $rootItem = Get-Item -LiteralPath $Root -Force
    if (Test-IsReparsePoint $rootItem) {
        throw "reparse pointを配置・削除対象にできません: $($rootItem.FullName)"
    }
    if ($rootItem -isnot [IO.DirectoryInfo]) { return }
    foreach ($entry in Get-ChildItem -LiteralPath $Root -Force) {
        if ($RuntimeEntries -contains $entry.Name) { Assert-NoReparseTree $entry.FullName }
    }
}

function Get-PathRelativeTo([string]$Root, [string]$FullPath) {
    $rootWithSeparator = [IO.Path]::GetFullPath($Root).TrimEnd(
        [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    $resolved = [IO.Path]::GetFullPath($FullPath)
    if (-not $resolved.StartsWith($rootWithSeparator, [StringComparison]::OrdinalIgnoreCase)) {
        throw "パスが基準ディレクトリ配下ではありません: root=$Root path=$FullPath"
    }
    return $resolved.Substring($rootWithSeparator.Length)
}

function Get-FileSha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace("-", "")
    } finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Remove-ManagedDirectory([string]$Path, [string]$ExpectedName) {
    Assert-ManagedPath $Path $ExpectedName
    if (Test-Path -LiteralPath $Path) {
        Assert-NoReparseTree $Path
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Get-KnownRuntimeEntries([string]$Root) {
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { return @() }
    $manifestPath = Join-Path $Root $runtimeManifestName
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        return @($legacyRuntimeNames | Where-Object { Test-Path -LiteralPath (Join-Path $Root $_) })
    }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    } catch {
        throw "旧配置のランタイムmanifestが壊れています: $manifestPath"
    }
    if ($manifest.version -ne 1 -or $null -eq $manifest.runtime_entries) {
        throw "旧配置のランタイムmanifestが不正です: $manifestPath"
    }
    $entries = @($manifest.runtime_entries | ForEach-Object { [string]$_ })
    foreach ($name in $entries) {
        if ([string]::IsNullOrWhiteSpace($name) -or $name -ne [IO.Path]::GetFileName($name)) {
            throw "旧配置のランタイムmanifestに不正な名前があります: $name"
        }
    }
    return @($entries + $runtimeManifestName | Select-Object -Unique)
}

function Write-RuntimeManifest([string]$Root, [string[]]$RuntimeEntries) {
    $payload = [ordered]@{
        version = 1
        runtime_entries = @($RuntimeEntries | Where-Object { $_ -ne $runtimeManifestName })
    } | ConvertTo-Json -Depth 3
    [IO.File]::WriteAllText(
        (Join-Path $Root $runtimeManifestName), $payload + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

function Write-DeployJournal([string]$Phase, [string[]]$RuntimeEntries) {
    $payload = [ordered]@{
        version = 1
        phase = $Phase
        target = $deployRoot
        staging = $stageRoot
        previous = $previousRoot
        runtime_entries = @($RuntimeEntries)
    } | ConvertTo-Json -Depth 4
    $temporary = "$journalPath.tmp"
    try {
        [IO.File]::WriteAllText($temporary, $payload + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        $stream = [IO.File]::Open($temporary, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        try { $stream.Flush($true) } finally { $stream.Dispose() }
        Move-Item -LiteralPath $temporary -Destination $journalPath -Force
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
}

function Assert-PreservedEntriesCanMove([string]$From, [string]$To, [string[]]$KnownRuntimeEntries) {
    if (-not (Test-Path -LiteralPath $From -PathType Container)) { return }
    Assert-NoReparseRuntime $From $KnownRuntimeEntries
    Assert-NoReparseRuntime $To @(Get-KnownRuntimeEntries $To)
    foreach ($entry in Get-ChildItem -LiteralPath $From -Force) {
        if ($KnownRuntimeEntries -contains $entry.Name) { continue }
        $destination = Join-Path $To $entry.Name
        if (Test-Path -LiteralPath $destination) {
            throw "新ランタイムと旧ユーザーデータの名前が衝突します。旧配置を保持して停止します: $($entry.Name)"
        }
    }
}

function Move-PreservedEntries([string]$From, [string]$To, [string[]]$KnownRuntimeEntries) {
    Assert-PreservedEntriesCanMove $From $To $KnownRuntimeEntries
    foreach ($entry in Get-ChildItem -LiteralPath $From -Force) {
        if ($KnownRuntimeEntries -contains $entry.Name) { continue }
        Move-Item -LiteralPath $entry.FullName -Destination (Join-Path $To $entry.Name)
    }
}

function Recover-InterruptedDeploy {
    if (-not (Test-Path -LiteralPath $journalPath -PathType Leaf)) { return }
    Assert-NoReparseTree $journalPath
    Assert-NoReparseTree $stageRoot
    try {
        $journal = Get-Content -LiteralPath $journalPath -Raw | ConvertFrom-Json
    } catch {
        throw "配置ジャーナルが壊れているため自動復旧できません: $journalPath"
    }
    if ($journal.version -ne 1 -or
        [IO.Path]::GetFullPath([string]$journal.target) -ne $deployRoot -or
        [IO.Path]::GetFullPath([string]$journal.staging) -ne $stageRoot -or
        [IO.Path]::GetFullPath([string]$journal.previous) -ne $previousRoot) {
        throw "配置ジャーナルの対象が現在の配置先と一致しません: $journalPath"
    }
    $runtimeEntries = @($journal.runtime_entries | ForEach-Object { [string]$_ })
    Assert-NoReparseRuntime $deployRoot @($runtimeEntries + (Get-KnownRuntimeEntries $deployRoot) | Select-Object -Unique)
    Assert-NoReparseRuntime $previousRoot @(Get-KnownRuntimeEntries $previousRoot)
    if (Test-Path -LiteralPath $previousRoot -PathType Container) {
        if (Test-Path -LiteralPath $deployRoot -PathType Container) {
            Move-PreservedEntries $deployRoot $previousRoot $runtimeEntries
            Remove-ManagedDirectory $deployRoot $TargetName
        }
        Move-Item -LiteralPath $previousRoot -Destination $deployRoot
    } elseif (-not (Test-Path -LiteralPath $deployRoot -PathType Container)) {
        throw "既知良好な旧配置を復元できません: $previousRoot"
    }
    Remove-ManagedDirectory $stageRoot $stageName
    Remove-Item -LiteralPath $journalPath -Force
    Write-Output "[deploy] 中断された前回配置を既知良好版へ復旧しました"
}

function Get-TargetProcesses([switch]$SceneWorkersOnly) {
    @(Get-CimInstance Win32_Process -Filter "Name='$TargetName.exe'" -ErrorAction SilentlyContinue | Where-Object {
        $isTarget = $false
        try { $isTarget = [IO.Path]::GetFullPath([string]$_.ExecutablePath) -eq $targetExe } catch { }
        $isTarget -and (-not $SceneWorkersOnly -or [string]$_.CommandLine -like "*--scene-naming*")
    })
}

# 場面名ワーカー（LSS.exe --scene-naming*）へ停止要求を書き、区切り（キュー間・ダウンロード塊間・待機中）で
# 自ら終了するのを上限付きで待つ。途中のダウンロード(.part)は再開用に残る。論理は lss/bgm/scene_stop.py と同じ。
function Request-SceneWorkerStop {
    if (-not (Test-Path -LiteralPath $deployRoot -PathType Container)) { return }
    $nativeDir = Join-Path $deployRoot "scene-native"
    New-Item -ItemType Directory -Path $nativeDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $nativeDir "stop-request") -Value "requested_at=$((Get-Date).ToString('o')) by=deploy" -Encoding ASCII
    $deadline = (Get-Date).AddSeconds($SceneStopSeconds)
    while ((Get-TargetProcesses -SceneWorkersOnly).Count -gt 0) {
        if ((Get-Date) -ge $deadline) {
            Write-Output "[deploy] 場面名ワーカーが${SceneStopSeconds}秒以内に止まらないため強制停止します"
            return
        }
        Start-Sleep -Milliseconds 500
    }
}

function Clear-SceneWorkerStop {
    $request = Join-Path (Join-Path $deployRoot "scene-native") "stop-request"
    if (Test-Path -LiteralPath $request -PathType Leaf) { Remove-Item -LiteralPath $request -Force -ErrorAction SilentlyContinue }
}

function Stop-TargetProcesses {
    Request-SceneWorkerStop
    $running = Get-TargetProcesses
    if ($running.Count -eq 0) { return }
    Write-Output "[deploy] 実行中の$TargetName.exeを停止します（$($running.Count)プロセス）"
    $running | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Milliseconds 200
        if ((Get-TargetProcesses).Count -eq 0) { return }
    }
    throw "$TargetName.exeを停止できません"
}

New-Item -ItemType Directory -Path $deployBasePath -Force | Out-Null
if ($acceptanceMode) {
    Assert-NoReparseAncestors $deployBasePath
    Assert-ManagedPath $deployRoot $TargetName
    Assert-ManagedPath $stageRoot $stageName
    Assert-ManagedPath $previousRoot $previousName
    if (-not (Test-Path -LiteralPath $journalPath -PathType Leaf)) {
        throw "受入待ちの配置がありません: $journalPath"
    }
    $pending = Get-Content -LiteralPath $journalPath -Raw | ConvertFrom-Json
    if ($pending.phase -ne "awaiting_acceptance") {
        throw "受入待ちではない配置ジャーナルです（phase=$($pending.phase)）。通常の配置で復旧してください"
    }
    if ($Rollback) {
        # 受入待ちの新版（場面名ワーカーを含む）を止めてからファイルを戻す。
        try {
            Stop-TargetProcesses
            Recover-InterruptedDeploy
        } finally {
            Clear-SceneWorkerStop
        }
        Write-Output "[deploy] 受入待ちの配置を既知良好版へ戻しました"
    } else {
        # 確定は、配置したEXEが利用者と同じlink配置の状態でBGMを実際に作れた記録があるときだけ
        # （scripts/post_deploy_bgm_check.py）。隔離した空の状態での受入だけで確定した76b88f4は、
        # .state\work のjunctionで全BGMジョブが失敗した。
        $bgmReceiptPath = Join-Path $deployBasePath ".$TargetName-postdeploy-bgm.json"
        if (-not (Test-Path -LiteralPath $bgmReceiptPath -PathType Leaf)) {
            throw "配置後のBGM確認がありません: python scripts/post_deploy_bgm_check.py --source <動画> を先に実行してください"
        }
        $bgmReceipt = Get-Content -LiteralPath $bgmReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $installedSha = (Get-FileSha256 $targetExe).ToLowerInvariant()
        if ($bgmReceipt.status -ne "PASS" -or ([string]$bgmReceipt.exe_sha256).ToLowerInvariant() -ne $installedSha) {
            throw "配置後のBGM確認が合格していないか、別のEXEの記録です（status=$($bgmReceipt.status)）"
        }
        Remove-ManagedDirectory $previousRoot $previousName
        Remove-Item -LiteralPath $journalPath -Force
        Write-Output "[deploy] 受入済みの配置を確定しました"
    }
    exit 0
}
Assert-NoReparseAncestors $buildRoot
Assert-NoReparseAncestors $deployBasePath
Assert-NoReparseTree $buildRoot
Assert-ManagedPath $deployRoot $TargetName
Assert-ManagedPath $stageRoot $stageName
Assert-ManagedPath $previousRoot $previousName

Recover-InterruptedDeploy
if (Test-Path -LiteralPath $previousRoot) {
    throw "配置退避先が残っているため安全に続行できません: $previousRoot"
}
Remove-ManagedDirectory $stageRoot $stageName

$runtimeEntries = @(
    Get-ChildItem -LiteralPath $buildRoot -Force |
        Where-Object { $mutableNames -notcontains $_.Name } |
        ForEach-Object { $_.Name }
)
if ($runtimeEntries.Count -eq 0) {
    throw "配置可能なランタイム成果物がありません: $buildRoot"
}

$hadPrevious = Test-Path -LiteralPath $deployRoot -PathType Container
try {
    New-Item -ItemType Directory -Path $stageRoot | Out-Null
    foreach ($entry in Get-ChildItem -LiteralPath $buildRoot -Force) {
        if ($mutableNames -contains $entry.Name) { continue }
        if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "ビルド成果物のreparse pointは配置できません: $($entry.FullName)"
        }
        Copy-Item -LiteralPath $entry.FullName -Destination $stageRoot -Recurse
    }

    # コピー完了だけではなく、全ファイルがビルド成果物と同一であることを確認してから停止・切替する。
    $buildFiles = @(
        Get-ChildItem -LiteralPath $buildRoot -File -Recurse -Force |
            Where-Object {
                $relative = Get-PathRelativeTo $buildRoot $_.FullName
                $mutableNames -notcontains $relative.Split([IO.Path]::DirectorySeparatorChar)[0]
            }
    )
    foreach ($file in $buildFiles) {
        if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "ビルド成果物のreparse pointは配置できません: $($file.FullName)"
        }
        $relative = Get-PathRelativeTo $buildRoot $file.FullName
        $stagedFile = Join-Path $stageRoot $relative
        if (-not (Test-Path -LiteralPath $stagedFile -PathType Leaf) -or
            (Get-FileSha256 $file.FullName) -ne (Get-FileSha256 $stagedFile)) {
            throw "ステージ済み成果物の検証に失敗しました: $relative"
        }
    }
    $stagedFiles = @(Get-ChildItem -LiteralPath $stageRoot -File -Recurse -Force)
    if ($stagedFiles.Count -ne $buildFiles.Count) {
        throw "ステージ済み成果物のファイル数が一致しません"
    }
    Write-RuntimeManifest $stageRoot $runtimeEntries
    $runtimeEntries = @($runtimeEntries + $runtimeManifestName | Select-Object -Unique)
    $oldRuntimeEntries = @(Get-KnownRuntimeEntries $deployRoot)
    if ($hadPrevious) {
        Assert-PreservedEntriesCanMove $deployRoot $stageRoot $oldRuntimeEntries
    }
    Write-DeployJournal "stage_ready" $runtimeEntries

    if ($env:LSS_DEPLOY_TEST_FAIL_PHASE -eq "after_stage") {
        throw "試験用配置失敗: after_stage"
    }

    # exe自身とworker役を、検証済みステージが完成してから停止する。場面名ワーカーには先に停止要求を出し、
    # 上限付きで自発終了を待ってから残りを強制停止する。
    Stop-TargetProcesses

    if ($hadPrevious) {
        Assert-NoReparseRuntime $deployRoot $oldRuntimeEntries
        Move-Item -LiteralPath $deployRoot -Destination $previousRoot
        Write-DeployJournal "previous_saved" $runtimeEntries
    }
    Move-Item -LiteralPath $stageRoot -Destination $deployRoot
    Write-DeployJournal "target_promoted" $runtimeEntries

    if ($env:LSS_DEPLOY_TEST_FAIL_PHASE -eq "after_promote") {
        throw "試験用配置失敗: after_promote"
    }

    if ($hadPrevious) {
        # 新ランタイムに存在しない可変台帳・モデル・状態・未知のユーザーファイルを移す。
        Move-PreservedEntries $previousRoot $deployRoot $oldRuntimeEntries
        Write-DeployJournal "mutable_preserved" $runtimeEntries
        if ($KeepPrevious) {
            # 配備後の受入が済むまで、直前のランタイムを即時に戻せる形で残す。
            Write-DeployJournal "awaiting_acceptance" $runtimeEntries
        } else {
            Remove-ManagedDirectory $previousRoot $previousName
        }
    }
    if (-not ($KeepPrevious -and $hadPrevious)) {
        Remove-Item -LiteralPath $journalPath -Force
    }
    Clear-SceneWorkerStop
} catch {
    # 通常のコピー/切替失敗はこの場で既知良好版へ戻す。強制終了時は次回実行冒頭で同じ復旧を行う。
    if ($hadPrevious -and (Test-Path -LiteralPath $previousRoot -PathType Container)) {
        if (Test-Path -LiteralPath $deployRoot -PathType Container) {
            Move-PreservedEntries $deployRoot $previousRoot $runtimeEntries
            Remove-ManagedDirectory $deployRoot $TargetName
        }
        Move-Item -LiteralPath $previousRoot -Destination $deployRoot
    } elseif (-not $hadPrevious -and (Test-Path -LiteralPath $deployRoot -PathType Container)) {
        Remove-ManagedDirectory $deployRoot $TargetName
    }
    Remove-ManagedDirectory $stageRoot $stageName
    if (Test-Path -LiteralPath $journalPath) { Remove-Item -LiteralPath $journalPath -Force }
    Clear-SceneWorkerStop
    throw
}

if ($KeepPrevious -and $hadPrevious) {
    Write-Output "配置: $buildRoot -> $deployRoot（受入待ち。直前版を $previousRoot に保持）"
} else {
    Write-Output "配置: $buildRoot -> $deployRoot（検証済みステージから昇格、可変データを保全）"
}
