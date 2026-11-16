# -Pin re-pins the sha256 of every stable owner to its current content (after an intended edit or a merge) and then checks.
param([string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path, [switch]$Pin, [string]$IndexSnapshot = '')
$ErrorActionPreference = 'Stop'
$errors = New-Object 'System.Collections.Generic.List[string]'
if ($IndexSnapshot -and $Pin) { throw '-Pin cannot change a staged snapshot' }
$indexFiles = @{}
$indexPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
if ($IndexSnapshot) {
  $snapshot = Get-Content -LiteralPath $IndexSnapshot -Raw -Encoding UTF8 | ConvertFrom-Json
  if ($snapshot.schema_version -ne 1) { throw 'Unknown staged snapshot schema' }
  foreach ($p in $snapshot.paths) {
    $parts = ([string]$p).Split('/')
    for ($i=1; $i -le $parts.Count; $i++) { [void]$indexPaths.Add(($parts[0..($i-1)] -join '/')) }
  }
  foreach ($property in $snapshot.files.PSObject.Properties) { $indexFiles[$property.Name] = $property.Value }
}
function Repo-Relative([string]$Path) {
  $prefix = [IO.Path]::GetFullPath($RepoRoot).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
  $absolute = [IO.Path]::GetFullPath($Path)
  if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Path outside repository: $Path" }
  return $absolute.Substring($prefix.Length).Replace('\','/')
}
function Test-RepoPath([string]$Path) {
  if ($IndexSnapshot) { return $indexPaths.Contains((Repo-Relative $Path)) }
  return Test-Path -LiteralPath $Path
}
function Read-RepoText([string]$Path) {
  if (-not $IndexSnapshot) { return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 }
  $rel = Repo-Relative $Path
  if (-not $indexFiles.ContainsKey($rel)) { throw "Staged text unavailable: $rel" }
  return [string]$indexFiles[$rel].text
}
function Get-RepoHash([string]$Path) {
  if ($IndexSnapshot) { return [string]$indexFiles[(Repo-Relative $Path)].sha256 }
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Get-RepoLength([string]$Path) {
  if ($IndexSnapshot) { return [long]$indexFiles[(Repo-Relative $Path)].bytes }
  return (Get-Item -LiteralPath $Path).Length
}
$manifestPath = Join-Path $RepoRoot 'docs\DOCS_MANIFEST.json'
if (-not (Test-RepoPath $manifestPath)) { throw 'DOCS_MANIFEST.json missing' }
$m = Read-RepoText $manifestPath | ConvertFrom-Json
if ($m.schema_version -ne 2) { $errors.Add('Unknown docs manifest schema') }
if ($Pin) {
  # Replace only the changed hash strings in the manifest text: ConvertTo-Json (PowerShell 5.1) reformats the whole
  # file and escapes characters, which turned a one-hash re-pin into a 200-line diff.
  $text = [IO.File]::ReadAllText($manifestPath)
  foreach ($o in $m.owners) {
    $f = Join-Path $RepoRoot (([string]$o.path).Replace('/','\'))
    if ($o.class -notin @('mutable','research','historical_design_record') -and (Test-RepoPath $f)) {
      $hash = (Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($o.sha256 -and $o.sha256 -ne $hash) { $text = $text.Replace([string]$o.sha256, $hash) }
      $o.sha256 = $hash
    }
  }
  [IO.File]::WriteAllText($manifestPath, $text, (New-Object Text.UTF8Encoding($false)))
  Write-Host '[LSS docs] pinned current content of stable owners'
}
if ($m.owner_count -ne ($m.owners.Count + 1)) { $errors.Add('Manifest owner_count does not include all installed documents') }
$seen = @{}
foreach ($o in $m.owners) {
  $rel = ([string]$o.path).Replace('/','\')
  if ($rel -match '(^|\\)\.\.(\\|$)' -or [IO.Path]::IsPathRooted($rel) -or $seen.ContainsKey($rel)) {
    $errors.Add("Unsafe or duplicate owner: $rel"); continue
  }
  $seen[$rel] = $true
  $target = Join-Path $RepoRoot $rel
  if (-not (Test-RepoPath $target)) { $errors.Add("Missing owner: $rel"); continue }
  if ((Get-RepoLength $target) -lt 30) { $errors.Add("Empty owner: $rel") }
  if ($o.class -notin @('mutable','research','historical_design_record')) {
    $actual = (Get-RepoHash $target)
    if ($actual -ne ([string]$o.sha256).ToLowerInvariant()) { $errors.Add("Drift from installed stable owner: $rel") }
  }
}
if (((Read-RepoText (Join-Path $RepoRoot 'AGENTS.md')).TrimEnd([char[]]"`r`n").Split("`n")).Count -gt 100) { $errors.Add('AGENTS entry point grew beyond 100 lines') }
$checks = @(
  @('docs\product\BGM_AND_AVAILABILITY.md','choir','independent','EXACT'),
  @('docs\product\PRODUCT_INTENT.md','single','autonom','user','2ch','What would betray this intent','named from the music','scene event'),
  @('AGENTS.md','2ch','song-like','Sound alone never invents a scene','LSS-owned results'),
  @('docs\dev\AGENT_PLAYBOOK.md','Intent first','betray'),
  # Text regression only: these anchors do not certify an agent's task selection or behavior.
  @('.agents\skills\lss-dev-loop\SKILL.md','Choose the usable result before the task','Reuse accumulated successes as recipe components','Keep the same requested outcome across negative trials','A free file or a cheap independent bug'),
  @('.agents\skills\lss-local-draft\SKILL.md','delegation unit, not the Product objective'),
  @('AGENTS.md','selects a usable Product capability before an execution task'),
  @('docs\engine\RECOVERY_AND_REFERENCES.md','unexplained','donor','UNKNOWN'),
  @('docs\product\IDENTITY_AND_CURRENTBEST.md','CurrentBest','tag-only','history'),
  @('docs\engine\LISTENING_AND_PRESENTATION.md','SESSION COMPLETE','START','metadata','the **scene** the music played in','is named from the music','manner of the work''s own official titles','LSS-owned results are the current LSS''s best','Sound alone never invents a scene'),
  @('docs\dev\CORPUS_AND_EVIDENCE.md','corpus-a','corpus-b','corpus-c','corpus-d','LSS_MEDIA_ROOT','Holdout')
)
foreach ($c in $checks) {
  $p=Join-Path $RepoRoot $c[0]
  if (-not (Test-RepoPath $p)) {
    if ($c[0].StartsWith('.agents\')) { $errors.Add("Missing required development skill: $($c[0])") }
    continue
  }
  $t=Read-RepoText $p
  for ($i=1;$i -lt $c.Count;$i++) { if ($t.IndexOf($c[$i],[StringComparison]::OrdinalIgnoreCase) -lt 0) { $errors.Add("Required text anchor absent: $($c[0]) -> $($c[$i])") } }
}
# Superseded naming wording must not come back: titles come from the scene, never from acoustic traits.
foreach ($o in $m.owners) {
  $rel = ([string]$o.path).Replace('/','\')
  if (-not $rel.EndsWith('.md')) { continue }
  $p = Join-Path $RepoRoot $rel
  if (-not (Test-RepoPath $p)) { continue }
  $t = Read-RepoText $p
  if ($t -match '(?i)scene\s*/\s*music traits|scene or musical traits|title[^.\n]{0,60}(tempo|mood|instrument)s? (is|are) fine') { $errors.Add("Superseded naming wording in: $rel") }
}
# AGENTS.md loads into every Claude/Codex session: keep it small and free of incident history (vendor guidance:
# concise, only what applies every session; history belongs in git).
$agentsText = Read-RepoText (Join-Path $RepoRoot 'AGENTS.md')
if ($agentsText.Length -gt 9000) { $errors.Add("AGENTS.md is $($agentsText.Length) chars (budget 9000): move detail to its owner document") }
if ($agentsText -match '\(user, 20\d\d|「') { $errors.Add('AGENTS.md carries incident history or quotes: state the rule, leave the story to git') }
# STATE.md is read most sessions: keep it a short navigation page (history belongs in validation/INDEX.md and CANDIDATE_LEDGER.md).
$statePath = Join-Path $RepoRoot 'docs\current\STATE.md'
if (Test-RepoPath $statePath) {
  $stateLen = (Read-RepoText $statePath).Length
  if ($stateLen -gt 6000) { $errors.Add("STATE.md is $stateLen chars (budget 6000): move history to docs/validation/CANDIDATE_LEDGER.md") }
}
# Backticked repo-relative .py/.ps1/.md paths in owner docs must exist (untracked-but-present counts).
# Skipped: .state/, drive/absolute/worktree paths, globs, braces, placeholders, `branch:path` references and Evidence roots on other branches.
foreach ($o in $m.owners) {
  $rel = ([string]$o.path).Replace('/','\')
  if (-not $rel.EndsWith('.md') -or $o.class -eq 'historical_design_record') { continue }
  $p = Join-Path $RepoRoot $rel
  if (-not (Test-RepoPath $p)) { continue }
  $t = Read-RepoText $p
  foreach ($mm in [regex]::Matches($t, '`([A-Za-z0-9_./-]+\.(?:py|ps1|md))`')) {
    $ref = $mm.Groups[1].Value
    if ($ref -notmatch '/' -or $ref -match '^(\.state/|\./|\.\./)' -or $ref -match '(^|/)(worktrees?|release-evidence|[a-z]+-evidence)/') { continue }
    if (-not (Test-RepoPath (Join-Path $RepoRoot $ref.Split('/')[0]))) { continue }  # first segment must be a repo directory
    if (-not (Test-RepoPath (Join-Path $RepoRoot $ref.Replace('/','\')))) { $errors.Add("Dead path reference in ${rel}: $ref") }
  }
}
# Recurrence guard: an agent-process rule that a mechanism enforces for every agent it applies to lives only in that
# mechanism (docstring, block message). scripts/hooks/RULES.json registers such rules with phrases of the old rule text;
# an owned document that carries one of those phrases restates the rule and fails here.
$rulesPath = Join-Path $RepoRoot 'scripts\hooks\RULES.json'
if (-not (Test-RepoPath $rulesPath)) { $errors.Add('scripts/hooks/RULES.json missing') }
else {
  $reg = Read-RepoText $rulesPath | ConvertFrom-Json
  foreach ($rule in $reg.rules) {
    foreach ($mech in $rule.mechanisms) {
      if (([string]$mech).StartsWith('agent-kit:')) {
        # A path inside the user-level agent kit repository (AGENT_KIT_ROOT, default %USERPROFILE%\src\agent-kit); on a machine without the kit the path is not checked.
        $kitRoot = if ($env:AGENT_KIT_ROOT) { $env:AGENT_KIT_ROOT } else { Join-Path $env:USERPROFILE 'src\agent-kit' }
        $kitRel = ([string]$mech).Substring('agent-kit:'.Length).Trim().Replace('/','\')
        if ((Test-Path -LiteralPath $kitRoot -PathType Container) -and -not (Test-Path -LiteralPath (Join-Path $kitRoot $kitRel))) { $errors.Add("RULES.json rule $($rule.id): agent-kit mechanism missing: $mech") }
        continue
      }
      if (-not (Test-RepoPath (Join-Path $RepoRoot ([string]$mech).Replace('/','\')))) { $errors.Add("RULES.json rule $($rule.id): mechanism missing: $mech") }
    }
    $uncovered = @($rule.applies_to | Where-Object { $rule.enforced_for -notcontains $_ })
    if ($uncovered.Count -gt 0) { continue }   # a rule some agent is not covered for keeps its one Codex/other-side line in the docs
    foreach ($o in $m.owners) {
      if (-not ([string]$o.path).EndsWith('.md') -or $o.class -eq 'historical_design_record') { continue }
      $dp = Join-Path $RepoRoot ([string]$o.path).Replace('/','\')
      if (-not (Test-RepoPath $dp)) { continue }
      $dt = Read-RepoText $dp
      foreach ($phrase in $rule.not_restated) {
        if ($dt.IndexOf([string]$phrase, [StringComparison]::OrdinalIgnoreCase) -ge 0) { $errors.Add("Rule restated in $($o.path) but enforced by a mechanism ($($rule.id): $(($rule.mechanisms | Select-Object -First 1))): '$phrase'") }
      }
    }
  }
}
# An absolute active-model path in the agent playbook must agree with policy.
# Prefer a policy-key reference so changing physical placement has one owner.
$storagePolicyPath = Join-Path $RepoRoot 'scripts\storage_policy.json'
$playbookPath = Join-Path $RepoRoot 'docs\dev\AGENT_PLAYBOOK.md'
if ((Test-RepoPath $storagePolicyPath) -and (Test-RepoPath $playbookPath)) {
  $storagePolicy = Read-RepoText $storagePolicyPath | ConvertFrom-Json
  $playbookText = Read-RepoText $playbookPath
  foreach ($match in [regex]::Matches($playbookText, 'Active local-AI weights[^\r\n]*?`([A-Z]:[\\/][^`]+)`')) {
    $documented = $match.Groups[1].Value.Replace('\\','\').Replace('/','\').TrimEnd('\')
    $configured = ([string]$storagePolicy.local_ai_hot_models).Replace('/','\').TrimEnd('\')
    if ($documented -ine $configured) { $errors.Add("Agent playbook active-model path differs from storage policy: $documented != $configured") }
  }
}
# CHECK (warning only): the generated STATE.md status block is stale.
if (-not $IndexSnapshot) {
$pyExe = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pyExe)) { $common = (& git -C $RepoRoot rev-parse --git-common-dir 2>$null); if ($common) { if (-not [IO.Path]::IsPathRooted($common)) { $common = Join-Path $RepoRoot $common }; $pyExe = Join-Path (Split-Path ([IO.Path]::GetFullPath($common)) -Parent) '.venv\Scripts\python.exe' } }
$stateStatus = Join-Path $RepoRoot 'scripts\state_status.py'
if ((Test-Path -LiteralPath $pyExe) -and (Test-Path -LiteralPath $stateStatus) -and (Test-Path -LiteralPath $statePath)) {
  & $pyExe $stateStatus --check --root $RepoRoot 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Host '[LSS docs] CHECK: STATE.md generated status block is stale: python scripts/state_status.py --write' -ForegroundColor Yellow }
}
}
if ($errors.Count -gt 0) {
  Write-Host '[LSS docs] FAIL' -ForegroundColor Red
  $errors | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
  exit 1
}
Write-Host "[LSS docs] PASS: $($seen.Count) owned documents; key semantic regressions present." -ForegroundColor Green
exit 0
