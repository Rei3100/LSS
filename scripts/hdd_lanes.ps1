#Requires -Version 5.1
# Active lanes whose worktree is physically on the HDD (read-only, dot-sourced by dev_hygiene.ps1).
#   . scripts\hdd_lanes.ps1; Get-HddActiveLanes -Root <checkout> -MainRoot <main checkout> -HotRoot 'C:\' -Hours 48
# A lane is a branch checked out in a worktree whose tip commit is younger than -Hours. "On the HDD" means the
# worktree's volume root differs from -HotRoot and the path is not a junction/symlink to a directory on the hot volume
# (a promoted worktree keeps a junction at its old path). One git call (for-each-ref knows every branch's tip time and
# worktree path) and one attribute read per candidate: no directory walk, no size scan, no per-worktree git process.
function Get-HddActiveLanes {
    param([Parameter(Mandatory)][string]$Root, [Parameter(Mandatory)][string]$MainRoot, [Parameter(Mandatory)][string]$HotRoot, [int]$Hours = 48)
    $cut = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $Hours * 3600
    $main = ($MainRoot -replace '/', '\').TrimEnd('\')
    foreach ($line in @(& git -C $Root for-each-ref '--format=%(committerdate:unix)|%(refname:short)|%(worktreepath)' refs/heads 2>$null)) {
        $parts = ([string]$line) -split '\|', 3
        if ($parts.Count -lt 3 -or -not $parts[2] -or -not ($parts[0] -match '^\d+$') -or [int64]$parts[0] -lt $cut) { continue }
        $path = ($parts[2] -replace '/', '\').TrimEnd('\')
        if ($path -ieq $main -or [IO.Path]::GetPathRoot($path) -ieq $HotRoot) { continue }
        $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        if (-not $item) { continue }
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            $target = [string]($item.Target | Select-Object -First 1)
            if ($target -and ([IO.Path]::GetPathRoot(($target -replace '^\\\\\?\\', '')) -ieq $HotRoot)) { continue }
        }
        [pscustomobject]@{ Path = $path; Branch = $parts[1]; AgeHours = [math]::Round(([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [int64]$parts[0]) / 3600, 1) }
    }
}
