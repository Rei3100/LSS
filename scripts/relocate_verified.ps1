#Requires -Version 7
# Verified cross-volume relocation: copy -> per-file SHA-256 compare -> remove original -> junction at original path.
# Usage: relocate_verified.ps1 -Source <dir> -Dest <dir> -Receipt <json>
param([Parameter(Mandatory)][string]$Source, [Parameter(Mandatory)][string]$Dest, [Parameter(Mandatory)][string]$Receipt)
$ErrorActionPreference = 'Stop'
$src = (Resolve-Path -LiteralPath $Source).Path
$item = Get-Item -LiteralPath $src -Force
if ($item.LinkType) { throw "Source is already a link: $src" }
if (Test-Path -LiteralPath $Dest) { throw "Destination exists: $Dest" }
# Active consumers: any process whose executable lives under the source.
$busy = @(Get-Process | Where-Object { try { $_.Path -and $_.Path.StartsWith($src, [StringComparison]::OrdinalIgnoreCase) } catch { $false } })
if ($busy.Count) { throw "Active process under source: $($busy | ForEach-Object Id)" }
$freeBefore = (Get-PSDrive C).Free
function Get-Manifest([string]$root) {
    $rows = @{}
    Get-ChildItem -LiteralPath $root -Recurse -File -Force | ForEach-Object {
        $rel = $_.FullName.Substring($root.Length).TrimStart('\')
        $rows[$rel] = @{ bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower(); mtime = $_.LastWriteTimeUtc.ToString('o') }
    }
    return $rows
}
$before = Get-Manifest $src
New-Item -ItemType Directory -Path (Split-Path $Dest -Parent) -Force | Out-Null
& robocopy $src $Dest /E /XJ /COPY:DAT /DCOPY:DAT /R:2 /W:2 /NFL /NDL /NP /NJH | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed: $LASTEXITCODE" }
$after = Get-Manifest ((Resolve-Path -LiteralPath $Dest).Path)
$mismatch = @()
foreach ($k in $before.Keys) {
    if (-not $after.ContainsKey($k) -or $after[$k].sha256 -ne $before[$k].sha256 -or $after[$k].bytes -ne $before[$k].bytes) { $mismatch += $k }
}
if ($after.Count -ne $before.Count) { $mismatch += "<count $($before.Count) vs $($after.Count)>" }
$bytes = ($before.Values | Measure-Object -Property bytes -Sum).Sum
$record = [ordered]@{
    type = 'lss_verified_relocation'; owner = 'claude-integration'; utc = (Get-Date).ToUniversalTime().ToString('o')
    source = $src; destination = $Dest; files = $before.Count; bytes = $bytes; mismatches = $mismatch
    status = $(if ($mismatch.Count) { 'COPY_MISMATCH_ORIGINAL_KEPT' } else { 'VERIFIED_COPY' })
    files_sha256 = $before
}
if ($mismatch.Count) { $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Receipt -Encoding utf8; throw "Mismatch; original kept" }
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Dest '.lss-relocation-manifest.json') -Encoding utf8
Remove-Item -LiteralPath $src -Recurse -Force
New-Item -ItemType Junction -Path $src -Target $Dest | Out-Null
$probe = Get-ChildItem -LiteralPath $src -Recurse -File -Force | Where-Object { $_.Name -ne '.lss-relocation-manifest.json' } | Measure-Object
if ($probe.Count -ne $before.Count) { throw "Junction probe count mismatch: $($probe.Count)" }
$record.status = 'RELOCATED_VERIFIED_JUNCTION'
$record.free_c_before = $freeBefore; $record.free_c_after = (Get-PSDrive C).Free
$record.Remove('files_sha256'); $record.manifest = (Join-Path $Dest '.lss-relocation-manifest.json')
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Receipt -Encoding utf8
Write-Host "[relocated] $src -> $Dest files=$($before.Count) bytes=$bytes"
