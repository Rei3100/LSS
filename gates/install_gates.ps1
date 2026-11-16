# Install the integrated gates, never the worktree's uncommitted gates.
[CmdletBinding(SupportsShouldProcess=$true)]
param(
    [string]$Ref = 'origin/main',
    [string]$Repository = (Split-Path $PSScriptRoot -Parent),
    [string]$Target = $(if ($env:LSS_PRIVATE_ROOT) { Join-Path $env:LSS_PRIVATE_ROOT 'bin' }),
    [string]$Python = (Join-Path $Repository '.venv\Scripts\python.exe'),
    [string]$NormalUserSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
)
$ErrorActionPreference = 'Stop'
if (-not $Target) { throw 'LSS_PRIVATE_ROOT or -Target is required' }
$targetPath = [IO.Path]::GetFullPath($Target)
$pythonPath = [IO.Path]::GetFullPath($Python)
if ($pythonPath -match '["%\r\n]' -or $targetPath -eq [IO.Path]::GetPathRoot($targetPath)) {
    throw 'Unsafe target or Python path'
}
$plan = [ordered]@{ source_ref=$Ref; repository=$Repository; target=$targetPath;
    python=$pythonPath; source='gates/'; manifest='gates-manifest.json';
    acl='Administrators and SYSTEM: FullControl; normal user: ReadAndExecute';
    normal_user_sid=$NormalUserSid; elevation_required=$true; mutates=(-not $WhatIfPreference) }
if ($WhatIfPreference) { $plan | ConvertTo-Json; return }
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Elevation required; no files were changed'
}
if (-not $PSCmdlet.ShouldProcess($targetPath, 'Install integrated gate copy and protect ACL')) { return }
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw 'Pinned Python is missing' }
# Reject links at the destination and every existing ancestor before any write.
$cursor = $targetPath
while ($cursor) {
    if (Test-Path -LiteralPath $cursor) {
        if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw 'Destination has a reparse ancestor'
        }
    }
    $parent = [IO.Path]::GetDirectoryName($cursor)
    if ($parent -eq $cursor) { break }
    $cursor = $parent
}
$commit = (& git -C $Repository rev-parse --verify --end-of-options "$Ref`^{commit}").Trim()
if ($LASTEXITCODE -ne 0 -or $commit -notmatch '^[0-9a-f]{40}([0-9a-f]{24})?$') { throw 'Cannot resolve ref' }
$entries = @(& git -C $Repository ls-tree -r $commit -- gates/)
if ($LASTEXITCODE -ne 0 -or $entries.Count -eq 0) { throw 'Missing gates tree' }
$files = @()
foreach ($entry in $entries) {
    if ($entry -notmatch '^100(?:644|755) blob ([0-9a-f]+)\t(gates/[A-Za-z0-9_./-]+)$') { throw 'Unsupported gate entry' }
    $relative = $Matches[2].Substring(6)
    if ($relative.Split('/') -contains '..') { throw 'Unsafe gate path' }
    $files += @{path=$relative; blob=$Matches[1]}
}
if (Test-Path -LiteralPath $targetPath) {
    foreach ($item in Get-ChildItem -LiteralPath $targetPath -Recurse -Force) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Existing target has a reparse point' }
        if (-not $item.PSIsContainer) {
            $relative = [IO.Path]::GetRelativePath($targetPath, $item.FullName).Replace('\', '/')
            if ($relative -ne 'gates-manifest.json' -and $relative -notin $files.path) {
                throw 'Target contains a stale or unknown file; retain it for administrator review'
            }
        }
    }
} else { New-Item -ItemType Directory -Path $targetPath | Out-Null }
# Protect the directory before adding content. Do not inherit write access.
function Protect-GatePath([string]$Path, [bool]$Directory) {
    $acl = if ($Directory) { [Security.AccessControl.DirectorySecurity]::new() } else { [Security.AccessControl.FileSecurity]::new() }
    $acl.SetAccessRuleProtection($true, $false)
    $admin = [Security.Principal.SecurityIdentifier]::new('S-1-5-32-544')
    $acl.SetOwner($admin)
    $inherit = if ($Directory) { [Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit' } else { [Security.AccessControl.InheritanceFlags]::None }
    foreach ($pair in @(@('S-1-5-32-544','FullControl'), @('S-1-5-18','FullControl'), @($NormalUserSid,'ReadAndExecute'))) {
        $rule = [Security.AccessControl.FileSystemAccessRule]::new(
            [Security.Principal.SecurityIdentifier]::new($pair[0]), [Security.AccessControl.FileSystemRights]$pair[1],
            $inherit, [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow)
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}
Protect-GatePath $targetPath $true
$manifest = [ordered]@{schema_version=1; source_commit=$commit; python=$pythonPath; files=@()}
foreach ($file in $files) {
    $destination = Join-Path $targetPath $file.path
    New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
    $start = [Diagnostics.ProcessStartInfo]::new('git')
    foreach ($arg in @('-C', $Repository, 'cat-file', 'blob', $file.blob)) { $start.ArgumentList.Add($arg) }
    $start.UseShellExecute = $false; $start.CreateNoWindow = $true; $start.RedirectStandardOutput = $true
    $process = [Diagnostics.Process]::Start($start)
    try {
        $stream = [IO.File]::Create($destination)
        try { $process.StandardOutput.BaseStream.CopyTo($stream) } finally { $stream.Dispose() }
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { throw 'Cannot read gate blob' }
    } finally { $process.Dispose() }
    if ($file.path -in @('lss-gh.cmd', 'deploy-gated.cmd')) {
        $text = [IO.File]::ReadAllText($destination).Replace('__LSS_GATE_PYTHON__', $pythonPath)
        [IO.File]::WriteAllText($destination, $text, [Text.UTF8Encoding]::new($false))
    }
    $manifest.files += @{path=$file.path; source_blob=$file.blob; sha256=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $targetPath 'gates-manifest.json') -Encoding utf8
foreach ($item in Get-ChildItem -LiteralPath $targetPath -Recurse -Force) { Protect-GatePath $item.FullName $item.PSIsContainer }
Protect-GatePath $targetPath $true
$manifest | ConvertTo-Json -Depth 6
