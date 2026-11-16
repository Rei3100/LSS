---
name: lss-release
description: Prepare or resume an LSS release from exact package identities and existing acceptance receipts. Use for candidate build, freeze, independent verification, deployment or rollback requests, not ordinary development changes.
---

# LSS release continuation

Paths are relative to the current LSS checkout root. Read the Deployment and Release runbook sections of `docs/engine/LOCAL_RUNTIME.md` for the maintained commands and `docs/dev/AGENT_PLAYBOOK.md` for lane ownership. This skill chooses where to resume; the runbook owns the release procedure.

## Establish the existing stage

Locate the task's existing `HANDOFF-TO-*.md`, `package-manifest.json`, verifier inbox/verdict and check receipts. Use the private state page and `.state/releases/current-product.json` to locate the installed release, then verify its actual artifact identity when the decision depends on deployment. A branch name, a PASS marker or the newest directory alone does not establish a stage.

| Evidence for the requested artifact | Next action |
|---|---|
| No package for the intended integrated commit | Build that exact commit using `scripts/build_candidate_offdrive.ps1` on SSD, following the runbook |
| Built package, no valid freeze | Freeze using `scripts/freeze_candidate.py`; resolve its integration gaps or document specific exclusions |
| Frozen bytes and manifest, missing packaged receipts | Run the relevant builder checks on isolated copies/state clones |
| Builder checks complete, no independent verdict for these bytes | Hand off the frozen identity and changed surfaces to a distinct verifier |
| Independent PASS for these bytes and supported scope | Proceed to the authorized deployment stage with rollback retained |
| Candidate already installed with identity established | Complete only missing installed-path acceptance and release records |
| FAIL, HOLD or contradictory identities | Resolve the concrete blocker; preserve the prior known-good release |

Reuse a receipt only when artifact identity, relevant inputs/state, check scope and result match the current claim. Changed bytes require a new candidate; an old verdict does not certify it. Do not run workers from the frozen tree or modify it to make a check pass. Use the runbook's single installed-release comparison for stale A/B baselines, not a replay of intermediate candidates.

`scripts/release.ps1` is an older build/accept/deploy pipeline: it mutates the install and removes its build worktree. It does not implement the current freeze/independent-verifier handoff. Use the maintained runbook rather than invoking it as a harmless release-preparation shortcut.

## Finish the requested stage

The request and existing conversation determine permission to deploy, rollback, finalize or remove anything. A skill invocation or an independent PASS grants no new permission. Carry authorized preparation through to a concrete candidate before requesting any missing deployment authorization; do not ask again for authorization already given.

Run installed checks through the existing hidden/isolated Product-path harnesses. Preserve mutable user state and known-good rollback. Record the source commit, EXE/package hashes, verifier verdict, installed receipt, supported/unknown scope and rollback pointer in the existing release Evidence and release record; update the affected current/validation facts in the same commit. Stop when the requested stage is achieved or its concrete blocker is recorded, not at a newly invented certification stage.
