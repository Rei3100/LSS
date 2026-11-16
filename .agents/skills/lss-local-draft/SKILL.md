---
name: lss-local-draft
description: 'Default first drafter for implementation and test code that a command can judge: hand it to the local Strata coder (one function, one test, or several steps through DeepSeek Harness) so the cloud model spends its tokens on design, review and Product decisions. The ledger closes a job kind only when it measurably does not pay.'
---

# Local drafting through Strata

The allocation order between local AI, Codex and Claude is decided in the Delegation section of `../lss-dev-loop/SKILL.md`; this skill covers the local-AI route.

Strata is the local coder model (`strata_dev.py`, engine/model under `%LSS_WORK_ROOT%\local-ai-*`). It pays only when the cloud model would otherwise have written the code itself **and** a command can judge the result. A draft the cloud must repair costs more than writing it, so the ledger below closes a kind that stops paying.

Paths are relative to the checkout root; use its `.venv/Scripts/python.exe`.

## Decide in this order

Delegation is the default for code and tests; the cloud model keeps design, Product decisions, review of every kept diff and anything no command can judge.

The Product capability is selected through `../lss-dev-loop/SKILL.md` before drafting tasks. A small function or test is a delegation unit, not the Product objective: keep the full chosen outcome, split its implementation as needed, and integrate the pieces through the actual Product route. Judge availability and draft size do not select the development goal. Model setup remains proportionate to that goal and coordinated with its owner; local drafting does not become a separate setup objective.

1. **Can a command judge it?** An existing test, a test written first (itself a `test` job), or a script that exits 0 only when the result is right. With a judge, hand the first draft to Strata: one function or test (`code`/`test`, `strata_task.py`), or a change that needs a few read/edit/run steps or several functions or files in one workspace (`harness`, below). Without a judge, write the judge first, then delegate; an edit so small that the spec is as long as the code (a few lines) is written directly.
2. **Is the kind open?** `python scripts/strata_ledger.py gate code` (or `test`, `harness`). Exit 1 = the last window of that kind did not pay: write it yourself. Exit 0 = hand it over. A kind on probation (fewer than six records) is open and is used, not avoided: the ledger learns only from real jobs. `python scripts/strata_ledger.py report` shows the numbers.
3. **Size.** Prefer many small judged tasks to one large one: a function or test under about 60 lines per task, a harness job limited to what fits in its 32k context (about 6k prompt tokens of files read per step). Split larger work into judged pieces and delegate each piece; design stays with the cloud.

4. **Does the spec cost as much as the code?** The saving is the drafted output only. Code whose contract is the design you are still deciding (tests of semantics defined minutes ago, glue between several new pieces) needs a spec about as long as the code plus a review read, so it saves nothing: write it yourself. Code whose contract fits in a few sentences and whose body is several times longer (a parser, a fixture builder, tests of behaviour that already exists) is where Strata pays.

Start the job in the background and keep working on the next piece (design, the next judge, review of the previous diff). Delegating and then waiting idle costs more than writing it.

**A busy GPU is not a reason to change who drafts.** `strata_task.py` waits for the GPU lease by itself, so a Product run, a GPU test or an installed-EXE scoring run only delays the draft. Decide by the questions above, then by timing: when other work can proceed meanwhile (the usual case: you are waiting on that same GPU job), queue the draft and continue; write it yourself only when the result blocks the next step, nothing else can proceed, and the GPU stays busy longer than writing it would take. Long GPU chains (several installed-episode runs, about 17 minutes each) are the time to queue judged drafts behind them, not to absorb those drafts into the cloud model. Work too large for Strata's context goes to the next option in that Delegation section.

## Run

Write `tasks.jsonl` (fields in the `scripts/strata_task.py` docstring: `id, file, function, spec, test, context_files`, or `test_file`/`test_function` for a test). Start it in the background and keep working on something else; the run waits for the GPU lease on its own, so it never fights a GPU test or Product run, and `dev_quiet_mode` defers it (exit 3) when GPU work is not allowed:

    python scripts/hidden_run.py --log .cache/local-ai/<job>.log --protect %LSS_WORK_ROOT%/local-ai-runtimes/strata --protect %LSS_WORK_ROOT%/local-ai-models/strata/coder --protect <abs out dir> -- <abs python> scripts/strata_task.py <tasks.jsonl> --repo . --out <abs out dir>

(`--protect` widens the launcher's storage reservation so the lease inside `gpu_run` nests; without it the run fails with `nested expansion refused`. Use absolute paths. Copy `.state/strata-pin.json` and `.state/strata-development.json` into a fresh worktree's `.state` first. Measured 2026-11-04: model load plus a ~800-token function 148 s, a ~2500-token test 255 s; both passed on attempt 1, and the test draft needed a tidy-up of redundant imports.)

A passing draft stays in the working tree, unstaged. Read the diff before keeping it (the judge says it passes, not that it is the code you want); a failed draft restores the file byte for byte, so nobody edits a file while a task is drafting it (the restore erases those edits). Cut tasks so each task's judging command can pass with that task's draft alone: two functions judged by one test are one task, not two. Every run appends one row per task to `.state/strata-ledger.jsonl` of the main checkout (`strata_ledger.LEDGER`, shared by all worktrees).

## After a failure

Do the job yourself once. Do not re-spec and retry more than once. Three failures of a kind inside its window close it; `python scripts/strata_ledger.py reopen <kind> "<what changed: engine, model, prompt, task shape>"` starts a new window only with a recorded change.

## Setup state

`python scripts/strata_dev.py status` shows whether the server and model are installed. The coder model files live in `%LSS_WORK_ROOT%\local-ai-models\strata\coder` (about 58 GB, public download, SHA-256 pinned in `strata_dev.PROFILES`); `python scripts/strata_dev.py install --profile coder` restores them and may be run without asking. Measured limits: `release-evidence/strata-capability-20261031/RESULTS.md` (11 of 11 bounded functions, 8k context, about 120 s each).

## Harness: several steps in one small workspace (`harness` kind)

For a job that needs a few read/edit/run steps, not one function, DeepSeek Harness (`dsh`, local, own `DSH_HOME` under `%LSS_WORK_ROOT%\local-ai-runtimes\dsh`) drives Strata as the model:

1. Start Strata (`strata_dev.py --profile coder run` through `hidden_run.py`). The machine policy sets the context per model in `launch_profiles` of the main checkout's `.state/strata-development.json` (Coder 128k, int8 KV, MTP off; dsh needs at least 32k), which every worktree reads; `strata_dev.py --profile coder plan` shows `launch-iq1_m-128k`. `--profile general` runs the general Qwen3.8-Flash-Next IQ2_XS for development work that is not code (reading and summarising Evidence, papers and logs, Japanese reports, dsh research jobs; measured 2026-11-05: an Evidence summary job PASS in 330 s, read its prose for a stray Chinese token and for design decisions it must not make). `--profile qwen36` runs Qwen3.6-35B-A3B (MTP GGUF) the same way, but at 128k its experts need about 19 GiB of host RAM and on this 31 GB machine it falls below the memory floor while loading (2026-11-05). The model loads in about a minute; `strata_dev.py stop` releases the GPU when done. `strata_task.py` takes the same `--profile`; `--no-ledger` keeps comparisons and replays out of the ledger.
2. `python scripts/dsh_local.py run --workspace <dir under C:/Users/<you>> --judge "<test command>" "<task>"`. The workspace has to be under the user profile (dsh's sandbox cannot get write access under `%LSS_C_WORK_ROOT%`). The judge decides PASS and the run is appended to the ledger as kind `harness` (`strata_ledger.py gate harness`).
3. Measured 2026-11-04 on tiny tasks (implement one function until a given test passes): 60 s and 62 s, PASS. Real multi-file LSS work is unmeasured; the ledger window decides whether it pays. Do not run it beside GPU tests or image generation: Strata holds the GPU lease while it serves.

`setup` installs the pinned dsh (`0.2.1-rc.5`, the version the image-generation setup also runs) with npm; the image setup under `%LOCAL_IMAGE_AI_ROOT%` is read only.

## Measured so far (2026-11-04, ledger `.state/strata-ledger.jsonl`)

`code` 1/1, `test` 6/6, `harness` 3/3 (2026-11-05 on Strata v0.2.10, Coder at 128k: the 10-task comparison passed 10/10 on the first attempt in 434 s with 13 GB RAM left free; dsh with the served 128k context and an 8192-token reply budget wrote 5 judged tests in 355 s). A draft always gets a read before it is kept. All kinds are open or on probation. Measurements: `release-evidence/local-coder-compare-20261105/README.md`.
