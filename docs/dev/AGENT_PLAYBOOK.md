# Development | Agent Operating Contract

## Recover narrow truth
Read `AGENTS.md` and follow only the owners for the active question. Inspect live code/tests/runtime and machine Evidence before assuming old decisions. Search prior safe implementation and rejected counterexamples; do not rebuild a working subsystem from the last prompt's vocabulary alone. Read large files (`lss/application.py`, `STATE.md`, `PORTFOLIO.md`) by `rg -n` and line range, not whole.

## Intent first
`docs/product/` states what the user wants LSS to be. Before choosing what to build, read the owner for that question and its "What would betray this intent" list in `PRODUCT_INTENT.md`; an old vision note, a summary or an easy technical source (e.g. acoustic features alone for naming) does not replace it. When a document has drifted from it, fix the document in the same commit; record code drift as a Product gap in the private portfolio. When the intent leaves a choice open, decide technically and report the decision.

## Attack Product outcomes
Choose one meaningful decision boundary, not one tiny test and not an unlimited exploration swamp. A consequential outcome usually includes implementation, real packaged behavior, a small challenger/falsifier set, durable Evidence and a clear Product-level disposition. When the local objective closes, compare the full Product opportunity set; do not automatically follow the same failure family forever. Development tooling is built only when a Product change needs it. Research spending rules: `docs/research/FRONTIER.md`.

## Mechanical work runs without an agent
Fixed-rule chores are scripts or scheduled tasks: SSD/HDD tiering and regenerable-file cleanup (`scripts/storage_tiers.py`, OS task `LSS Storage Tiers`), GPU exclusivity (`scripts/gpu_run.py`), background jobs (`scripts/hidden_run.py`), ordered job chains with one completion marker (`scripts/batch.py`), re-pinning edited docs (`powershell -File docs/scripts/check_docs.ps1 -Pin`). What a script does lives in its docstring and policy file, not here.

## Tests
The default iteration and test-selection procedure is `.agents/skills/lss-dev-loop/SKILL.md`. `scripts/dev.py` owns test arguments, parallel execution and receipts; the full local suite gates a candidate. Keep pytest temp and worktrees on C: (`docs/dev/DATA_LIFECYCLE.md`). Release continuation uses `.agents/skills/lss-release/SKILL.md`, with `docs/engine/LOCAL_RUNTIME.md` owning the release runbook. Claude discovery wrappers contain only a pointer to each shared procedure.

## Claude and Codex together
Claude and Codex are interchangeable capable engineering agents and may work at the same time.
- One trunk, `main`: it holds the deployed Product code and the shared docs and release records. A lane branches from `main`, lands back in it, and the deployed release is recorded on `main` in the same commit that updates `docs/current/`. A merged lane is tagged `archive/<branch>` and its branch deleted; its worktree is removed once clean. `python scripts/lanes.py` is the live list of lanes.
- One writer per mutable worktree, branch, installed runtime and storage scope. `main` and release candidates belong to the integration lead; `codex/*` branches and their worktrees belong to Codex until handed over.
- Stage and commit only the paths you changed; another agent's uncommitted edits in a shared checkout stay untouched.
- Never `cd` into or write another agent's worktree or Evidence root; read it if needed.
- GPU jobs run through `scripts/gpu_run.py`; background jobs through `scripts/hidden_run.py`. Never DETACHED_PROCESS or a plain `start` for console programs. Long probes keep one item's audio in memory at a time.
- Candidate handoff: a frozen candidate directory with `package-manifest.json` and `HANDOFF-TO-*.md`; findings come back as files in its `inbox-from-*` directory. No chat-only claims.
- Independent certification requires a distinct verifier when it changes a consequential promotion claim.
- Subagents get a compact brief, the expected artifact and a stopping condition; the parent integrates and judges.

## Tool and language boundary
User-facing conversation, including progress, is Japanese. Internal agent handoffs, docs and Evidence are English unless they must preserve Japanese target strings or UI.

## Evidence discipline
Exact artifact identity, test cases, denominators and falsifiers survive. No 'better score' claim without semantic wanted preservation. Distinguish deployed, independently verified, builder-frozen, prototype, observed-only, held, rejected and research-only.

## Git / user state
Preserve user changes and all important Evidence. Git dirty indicators do not decide Product worth; inspect actual ownership before any changes. Local commits are a technical concern, not a user job. Repo documentation maintenance is not permission to rewrite Product code or deploy.

## External work
Fresh official implementations and primary papers can inform hard decisions; measure actual local applicability. Public reference acquisition (released audio via yt-dlp, catalogue and reference pages) is automatic and needs no user permission. A login the user has set up for a service reaches the acquisition layer as a per-service file named by `LSS_COOKIES_FILE_<SERVICE>`, the state's `cookie-files.json`, or the folder in `LSS_GUI_COOKIE_DIR` (the layer passes yt-dlp a copy and never reads the content); list the folder in full before concluding a file is missing. With such a file, member-only released soundtracks are acquirable. Do not demand new episodes merely for holdout ceremony. No secrets, credential harvesting, DRM bypass or hosted separation dependency.

## Completion report
For agents and handoffs (English): one concise machine-readable summary and one human-agent note: changed Product capability, exact identity, best-positive and strongest-falsifier, supported/unknown boundary, state mutations, what was not deployed, resume pointer. The user-facing report follows `AGENTS.md`.
