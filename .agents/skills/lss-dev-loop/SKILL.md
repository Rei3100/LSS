---
name: lss-dev-loop
description: Develop LSS toward a usable Product capability, composing prior results into a technical route and carrying it through implementation and relevant verification. Use for LSS engineering work, and also when about to ask the user a question, about to stop or end a turn during LSS work, or about to delegate work to another model or agent. Not for questions or status-only reports.
---

# LSS development loop

## Choose the usable result before the task

For open-ended development, choose the Product capability to gain before choosing a defect, function, model, test or experiment. Read `docs/product/PRODUCT_INTENT.md` and the relevant owner. For BGM recovery, also read `docs/engine/SOURCE_TO_ASSET.md` and `docs/research/FRONTIER.md`: the route follows the music, its instrumentation and available occurrences, rather than one separator or one fixed chain. An explicit bounded user request keeps its scope; this rule does not expand a requested fix into a new project.

Use a concrete source/cue or normal-use failure to identify what prevents the requested result. In the existing working brief or progress statement, connect that capability barrier to the usable result, the existing material that can contribute, the proposed route and the decision the next step can change. Keep this concise; no new task registry, mandatory form or general evaluation platform is needed. A bug fix is the main task when it removes that barrier, not merely because it is easy to isolate and test.

Reuse accumulated successes as recipe components, not just as a list of tried technologies. Inspect the relevant actual outputs, alignments, stems and failure scopes. Decide what each can supply, what remains missing, and which new role, input, order or assembly could fill it. A negative whole-mix trial does not close an instrument-local role; a promising component does not authorize an arbitrary combination. Preserve wanted material and known falsifiers through the combined route. The technology lookup below supports this design, not only duplicate avoidance.

A function, test, trial and its minimal verification are execution units, not the Product objective. Local-draft limits split implementation, not the capability. Keep the same requested outcome across negative trials: change the technical route or establish the capability that is missing, rather than closing development at a small negative result or switching to an unrelated safe defect. Re-rank when evidence changes the best route; do not keep probing a failed method family to satisfy a trial count. State honestly when no usable improvement has yet been achieved.

When another agent owns a route, coordinate a useful capability role or handoff boundary. A free file or a cheap independent bug is not by itself a reason to select work. Use the existing independent-review agreement for nontrivial auxiliary expansions: it challenges the requested result and actual proposed commands, not merely whether the local tests can pass. Verification size, commit count and report completeness never determine the development objective.

## Before asking

Before asking the user anything, read the existing decisions: `AGENTS.md`, the owner document it routes to, and the private progress or notes the resume entry names. Check what a command, test or record can settle yourself. Ask only for a decision that is genuinely the user's (Product meaning, an irreversible outward action not delegated, a security setting); when one part needs the user, hold only that part and continue the rest. Otherwise decide, proceed, and state the decision in the report.

## Recover the implementation context

Paths below are relative to the current LSS checkout root. Use its `.venv/Scripts/python.exe` when present. A linked worktree can instead use the existing shared foundation interpreter: `git rev-parse --path-format=absolute --git-common-dir` returns the common `.git` directory; its parent owns `.venv/Scripts/python.exe`. Keep the current worktree as cwd and execute its scripts/code with that interpreter. Sharing Python does not mean running another lane's code or copying/installing another environment. Substitute this interpreter for `python` in the examples below; `scripts/dev.py doctor` reports the expected interpreter and package availability without running inference.

## Recover the next useful action

Preserve the requested outcome and ordering. Read `AGENTS.md` and only the owner it routes to for this change. Use `rg -n` and bounded ranges for large files. On a resumed task, locate the prior commit, receipt or handoff for that task before reconstructing its investigation. the private state page is navigation, not proof that a candidate is installed.

Start with `git status --short` and the relevant diff. Existing edits have owners; stage only yours. Use `python scripts/lanes.py` when lane/integration ownership affects the work. Inspect the live implementation and the exact previous result that can change the next action, rather than reading every lane, Evidence directory or historical chat.

Before choosing/reinstalling a technology, repeating an experiment or declaring a method unavailable/untried, use `python scripts/technology.py search "<technology or pattern>"` and the relevant record from `docs/research/TECHNOLOGY.md`. Check the role, input and order, not just the model name. `assets <id>` observes only declared local files; a missing path is not an authentication failure. After a trial or adoption/availability change, update that role and its durable source in `docs/research/technology_ledger.json`, regenerate the map with `technology.py map`, and run `technology.py check --freshness`. Source/model imports and new inference are unnecessary for lookup. When the chosen capability barrier has no promising untried pattern in the ledger, or a role's last external check predates a relevant release you notice, make one bounded primary-source search for that role before settling the route; register what you found but did not try in the ledger the same day, with its date, primary links and falsifier (as `external-leads-20261113` does), so a later lookup finds it. No schedule, feed or notification is involved.

For technology/research trials, use `python scripts/research_tests.py plan <profile> --pattern <variation>` and `docs/dev/RESEARCH_TESTS.md` to choose fixed cases, guards, truth scope and existing comparison paths before inventing a new benchmark. Check the selected bank assets explicitly; a plan is not a test result and missing cached outputs never trigger implicit inference. Keep BGM wanted singing separate from Instrumental unwanted singing, and controlled waveforms separate from fitted references or episode answer keys. Extend the existing native case/index only for a discriminating missing condition.

## Choose verification from the change

The test runner is `scripts/dev.py`, not a new pytest wrapper. These commands run from the checkout root:

| Decision | Existing command |
|---|---|
| Which files need testing when the scope is unclear? | `python scripts/dev.py test --changed <base-sha> --changed-list` |
| Run related local tests for the complete change | `python scripts/dev.py test --changed <base-sha>` |
| Recheck one changed failure path | `python scripts/dev.py test tests/test_example.py` (substitute the actual file or node ID) |
| Validate an intended owner-document edit | `powershell -NoProfile -ExecutionPolicy Bypass -File docs/scripts/check_docs.ps1` (process-local policy only) |

Use an explicit pre-change base when the work includes commits: on `main`, the default merge-base is HEAD and misses already committed changes. The selector includes uncommitted and untracked files too; in a shared checkout inspect why unrelated files were selected. If the exact covering test is already known, run it directly; selection preview is optional.

`--changed-list` lists targets and exits 0 without running tests. `--changed` with no targets exits **3 / NOT_RUN**, not PASS. A documentation-only change needs its document/skill checks, not an unrelated full suite. The selector traces imports and path strings; a changed dynamic or packaged boundary can require an explicit relevant test beyond its selection.

Use the runner's digest first; read the receipt's `pytest.log` around a failure instead of dumping the whole log. After a fix, rerun what covers the changed path. A candidate's full-suite requirement remains in `docs/dev/AGENT_PLAYBOOK.md`; ordinary iteration does not repeatedly rerun it. GPU/native/installed acceptance uses an explicit `--suite` from `python scripts/dev.py test --help`, not `--changed`.

For a baseline-sensitive failure, make one discriminating comparison with the installed release. If that establishes a stale harness/baseline, record the limit and return to the requested change; do not replay every ancestor or turn the task into a harness rebuild.

## Delegation

This section is the only allocation rule; other skills call it instead of keeping their own. Decide once per piece of work, in this order, and record why an earlier option is unsuitable or unavailable:

1. **Local AI** for a code or test draft that a command can judge: follow `../lss-local-draft/SKILL.md`.
2. **Codex** for independent work that a command or a review can judge, is too large for local AI, and does not need this session's context.
3. **Claude** (this session or a subagent) for design, Product decisions, review of every kept diff, and work that needs this session's context. Use subagents when the work needs them; launching and collecting them is never the user's job.

The brief states the purpose, inputs, write scope, output and completion or stop condition. Collect and check the result yourself. Codex's sandbox currently cannot run the checkout's Python or git: Codex edits and reads, and the delegating Claude runs the tests, reviews the diff and commits. Codex's sandbox also refuses writes under `.agents/`; skill edits stay with Claude.

Command examples (the configured default model is used; reviews and hard work raise the effort). Git Bash:

```bash
task_worktree="$(git rev-parse --show-toplevel)"; task_run="$LSS_PRIVATE_ROOT/run"; task_job='job'
codex exec -c 'model_reasoning_effort="high"' -c 'windows.sandbox="elevated"'   -C "$task_worktree" --json -o "$task_run/$task_job.last.md" - < "$task_run/$task_job.txt" > "$task_run/$task_job.events.jsonl"
```

PowerShell 7 (Windows PowerShell 5.1 turns Japanese briefs into `?`):

```powershell
$taskWorktree = git rev-parse --show-toplevel; $taskRun = "$env:LSS_PRIVATE_ROOT/run"; $taskJob = 'job'
Get-Content -Raw -LiteralPath "$taskRun/$taskJob.txt" | codex exec -c 'model_reasoning_effort="high"' -c 'windows.sandbox="elevated"' -C $taskWorktree --json -o "$taskRun/$taskJob.last.md" - > "$taskRun/$taskJob.events.jsonl"
```

Record the session ID (the `thread.started` event) and the start and end HEAD with the job. Continue an interrupted job with `codex exec resume <id>`. Review uses `codex exec review --base origin/main`.

Handing ownership to another engine: first stop the old owner's running or waiting work and its automatic resume and confirm it stopped; record owner, HEAD and stop point; only then does the new owner check them and start. Development state, GPU and deployment go through the existing locks; a label or a rejected push is not a lock.

## At a break point

A break point is reached when a merge is scheduled, a Draft PR is pushed with its stop reason, or a measured trial is closed as not adopted. Record the result, how it was checked, the owner, HEAD and stop point. Then check the stop conditions:

1. No work that can proceed autonomously remains; the rest waits for the user's decision or listening.
2. The same failure repeats without new information; try an independent route to the same goal instead of repeating the fix.
3. A real usage limit is reached on a route: hold that route's work safely. Codex may be used up. Claude must not hit its limit, because a stopped Claude needs a user message: before the limit, hand the next independent piece to Codex and collect it from the completion notice.
4. A user message arrived: answer it, and stop unless it says to continue.
5. The context is near compaction: right after a break point, compact and continue; in the middle of a step, push and write the note first.

When none applies, continue with the top item of `python scripts/whatsnext.py`. The user using the Product or GPU is not a stop: defer only GPU work and continue implementation, CPU tests, review and documents. When no safe work remains, save the concrete blocker and wait; do not create work to spend usage.

## Continue and close

Close a step with its actual evidence, but keep the capability objective open when a passing fix or a rejected trial has not delivered the requested usable result. A handoff or isolated commit alone does not make a normal-use capability available. Reports distinguish the execution result, Product integration and the remaining capability barrier.

Existing execution helpers and ownership rules live in `docs/dev/AGENT_PLAYBOOK.md` (mechanical work, background/GPU jobs, coexistence). Use their existing docstrings for arguments instead of making a new launcher, timer or scheduler.

For long owned step chains, use `scripts/batch.py` and retain the queue path as the attempt identity. `batch.py status <queue> --json` distinguishes terminal results, a live OS owner and UNKNOWN/stopped attempts; an unfinished receipt is not running proof. Prefer `after_queues` for new queue dependencies so a stopped producer does not leave consumers waiting for a marker forever. Failed/old attempts are preserved; inspect completed steps and use a new reviewed attempt path for a deliberate retry. Direct background work can use `hidden_run.py --receipt <path>` and `--status <path>`. Command arguments are not put in that launch receipt. Read the helper docstrings for resource and dependency settings.

For a release request, continue with `../lss-release/SKILL.md`; a development fix alone does not start a release. Keep the next action attached to the user's outcome. Finish with the bounded commit and relevant Evidence required by `AGENTS.md`. If interrupted, record the exact commit/artifact, receipt path, unresolved boundary and next command in the task's existing handoff or Evidence note; no separate task registry is needed.
