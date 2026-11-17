#Requires -Version 5.1
# Build an exact candidate commit into a detached worktree that may live on another drive (C: is near its
# free-space floor). Same steps as earlier candidate builds: worktree without EOL conversion, shared .venv junction,
# ignored vendor files copied, build.ps1, provider/adapter source check. Never deploys.
param([Parameter(Mandatory = $true)][string]$Commit,
      [Parameter(Mandatory = $true)][string]$Worktree,
      [switch]$Plan, [switch]$Admitted)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$common = & git -C $root rev-parse --git-common-dir
if ($LASTEXITCODE -ne 0) { throw 'Cannot discover shared Git root' }
if (-not [IO.Path]::IsPathRooted($common)) { $common = Join-Path $root $common }
$sharedRoot = Split-Path ([IO.Path]::GetFullPath($common)) -Parent
$python = Join-Path $sharedRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Shared Python missing: $python" }
$exactCommit = & git -C $root rev-parse --verify "$Commit`^{commit}"
if ($LASTEXITCODE -ne 0 -or $exactCommit -notmatch '^[0-9a-f]{40}$') { throw 'Commit must resolve before preparation' }
$Worktree = [IO.Path]::GetFullPath($Worktree)
$receiptPath = $Worktree + '.build-receipt.json'
$buildData = & $python -c "import sys; sys.path.insert(0, sys.argv[1]); import storage_paths,storage_tiers; print(storage_paths.build_directory(sys.argv[2], storage_tiers.load_policy()))" (Join-Path $root 'scripts') $Worktree
if ($LASTEXITCODE -ne 0 -or -not $buildData) { throw 'Cannot resolve policy-owned build data before reservation' }
if (Test-Path -LiteralPath $Worktree) { throw "Worktree path already exists: $Worktree" }
if (Test-Path -LiteralPath $receiptPath) { throw "Existing build receipt retained: $receiptPath" }
Push-Location $root
try {
    & $python -c "import sys; from scripts.dev_hot_paths import require_hot; [require_hot(p, 'candidate preparation') for p in sys.argv[1:]]" $root $Worktree $python $buildData
    if ($LASTEXITCODE -ne 0) { throw 'Candidate preparation requires verified fast-volume paths' }
} finally { Pop-Location }
if ($Plan) {
    [ordered]@{source_root=$root; shared_root=$sharedRoot; python=$python; commit=$exactCommit; worktree=$Worktree;
               receipt=$receiptPath; build_data=$buildData; required_gb=12; mutates_install=$false; reserves_before_preparation=$true} | ConvertTo-Json
    exit 0
}
if (-not $Admitted) {
    # Reserve capacity and own the entire preparation/build tree, including ignored vendor copies.
    & $python (Join-Path $root 'scripts\storage_budget.py') run --required-gb 12 --protect $Worktree --protect $buildData --protect $receiptPath `
        --protect (Join-Path $Worktree 'build') --protect (Join-Path $Worktree 'dist') --purpose 'Candidate preparation and build' -- `
        powershell -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -Commit $exactCommit -Worktree $Worktree -Admitted
    exit $LASTEXITCODE
}
if (-not $env:LSS_STORAGE_RESERVATION) { throw 'Admitted preparation requires a live storage supervisor' }
& $python -c "import sys; sys.path.insert(0,sys.argv[1]); import storage_budget; c=storage_budget.reserve(12,paths=sys.argv[2:]); c.__enter__(); c.__exit__(None,None,None)" (Join-Path $root 'scripts') $Worktree $buildData $receiptPath
if ($LASTEXITCODE -ne 0) { throw 'Preparation reservation is stale or does not cover the resolved output paths' }
$record = [ordered]@{commit=$exactCommit; source_root=$root; worktree=$Worktree; python=$python; status='running'; phase='prepare'; started=(Get-Date).ToUniversalTime().ToString('o')}
New-Item -ItemType Directory -Force -Path (Split-Path $receiptPath -Parent) | Out-Null
$claim = [IO.File]::Open($receiptPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
$claim.Dispose()
function Save-BuildReceipt {
    $temporary = $receiptPath + '.tmp'
    [IO.File]::WriteAllText($temporary, ($record | ConvertTo-Json -Depth 6), (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $receiptPath -Force
}
try {
Save-BuildReceipt
& git -C $root -c core.longpaths=true -c core.autocrlf=false worktree add --detach $Worktree $exactCommit
if ($LASTEXITCODE -ne 0) { throw "git worktree add failed" }
New-Item -ItemType Junction -Path (Join-Path $Worktree ".venv") -Target (Join-Path $sharedRoot ".venv") | Out-Null
# bin/ffmpeg.exe is fetched, not tracked, and LSS.spec packages it.
$ignored = & git -C $root ls-files --others --ignored --exclude-standard -- vendor bin
foreach ($relative in $ignored) {
    if ($relative -match "__pycache__") { continue }
    $destination = Join-Path $Worktree $relative
    New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $root $relative) -Destination $destination
}
& $python (Join-Path $root 'scripts\storage_paths.py') prepare-build --repo $Worktree
if ($LASTEXITCODE -ne 0) { throw 'Cannot prepare candidate outputs on the data SSD' }
$record.phase = 'build'; Save-BuildReceipt
# The integrated supervisor owns capacity even when building an older commit
# whose build script predates per-volume accounting.
& $python (Join-Path $root 'scripts\storage_budget.py') run --required-gb 12 --protect $Worktree `
    --protect (Join-Path $Worktree 'build') --protect (Join-Path $Worktree 'dist') --purpose 'Candidate build' -- `
    powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Worktree "build.ps1")
if ($LASTEXITCODE -ne 0) { throw "build failed" }
$record.phase = 'runtime-source-check'; Save-BuildReceipt
$candidate = Join-Path $Worktree "dist\LSS"
& $python (Join-Path $Worktree "scripts\check_runtime_sources.py") $candidate --require-data-files
if ($LASTEXITCODE -ne 0) { throw "candidate provider/adapter differs from source" }
$exe = Join-Path $candidate "LSS.exe"
Write-Host "[candidate] OK -> $candidate"
$exeHash = & $python -c "import hashlib,sys; print(hashlib.file_digest(open(sys.argv[1], 'rb'), 'sha256').hexdigest())" $exe
if ($LASTEXITCODE -ne 0 -or $exeHash -notmatch '^[a-f0-9]{64}$') { throw "candidate EXE hash failed" }
Write-Host ("[candidate] commit {0} exe sha256 {1}" -f (& git -C $Worktree rev-parse HEAD), $exeHash)
$record.exe_sha256 = $exeHash
$record.candidate = $candidate
$record.status = 'complete'
$record.phase = 'complete'
Save-BuildReceipt
} catch {
    $record.status = 'failed'
    $record.error = $_.Exception.Message
    Save-BuildReceipt
    throw
}
