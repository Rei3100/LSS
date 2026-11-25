# Gate entry points

These standard-library entry points implement design sections 6.4, 7.4, 7.5 and
8.3. They are procedural safeguards. They do not isolate arbitrary programs
running under the same Windows account, authenticate receipt authors, or replace
GitHub rulesets. A verdict means only that the stated evidence passed these checks.

## Installing the integrated copy

Use PowerShell 7. Preview without elevation or mutation:

```powershell
./gates/install_gates.ps1 -WhatIf
```

An administrator runs the same script without `-WhatIf` after integration. The
default source is `origin/main:gates/` and the destination is
`$env:LSS_PRIVATE_ROOT/bin`. `-Ref`, `-Repository`, `-Target`, `-Python` and
`-NormalUserSid` are explicit installation inputs. The normal SID defaults to the
invoking account; with alternate administrator credentials, supply the ordinary
user's SID. The installer never elevates itself.

The installer resolves the ref to a commit, reads Git blobs as bytes, substitutes
the absolute `-Python` path into the two `.cmd` templates, and records source blob
IDs and installed SHA-256 values in `gates-manifest.json`. Source `.cmd` files are
templates; installed wrappers call the pinned interpreter with `-I -B` and the
adjacent pinned Python module. No Python packages are installed.

The target and its contents have protected ACLs: Administrators and SYSTEM have
FullControl; the normal SID has ReadAndExecute. Ownership is Administrators.
Reparse destinations and non-regular Git entries are refused. Updates overwrite
only the integrated file set; a stale or unknown destination file blocks the
update for administrator review. Installation is not transactional: a failure
requires administrator inspection before using the copy. The manifest does not
hash itself. A future environment checker must verify every installed hash and
the ACL against the resolved integrated source, allowing only the documented
interpreter substitution. Actual write-denial and independent worktree-copy
behavior require plan T15 on the real machine.

## GitHub publication

Set `LSS_REPO_ROOT` to the intended checkout. Origin and every effective push URL
must be the HTTPS or SSH `.git` URL of `LSS_GITHUB_REPOSITORY` (`owner/name`, set on
the machine, never in this tree). Branch-bound operations require the assigned
`codex/...` or `claude/...` branch in `LSS_AGENT_BRANCH` to be checked out. Issue
operations, `pr-comment` and `comment-delete` also work from main without that
variable. Set
`LSS_PUBLISH_DENYLIST` to the private nonempty literal list, or use the public
guard's default. Missing configuration, guard errors and command failures block.
No denylist content or rejected body is printed.

The installed `lss-gh.cmd` accepts exactly these operations:

| Operation | Inputs and scope |
| --- | --- |
| `issue-create` | `--title`, `--body-file`, repeated optional `--label`; prints the new issue URL from gh |
| `issue-comment` | `--issue N`, `--body-file`, optional `--allow-canary-test`; fixed repository |
| `issue-close` | `--issue N`, `--reason-file`; posts the reason, then closes after posting succeeds |
| `issue-edit` | `--issue N`, repeated `--add-label` / `--remove-label`; at least one label change, no title/body edits |
| `pr-comment` | `--pr N`, `--body-file`; fixed repository, any head branch |
| `pr-create` | `--title`, `--body-file`, optional `--draft`; assigned branch to main in the fixed repository |
| `pr-ready` / `pr-draft` | `--pr N`; changes draft state only for the assigned branch's PR in the fixed repository |
| `pr-close` | `--pr N`, `--reason-file`; assigned branch's PR only; posts the reason before closing |
| `comment-delete` | `--comment-id ID`; deletes one issue/PR comment in the fixed repository |
| `push` | No extra arguments; exact checked HEAD to the same assigned branch on origin |
| `thread-reply` | `--thread PRRT_...`, `--reason-file`; current HEAD's thread in the fixed repository |
| `thread-resolve` | Same inputs; posts the reason first, then resolves only after posting succeeds |
| `review-record` | `--pr`, `--reason-file`; JSON record posted to the assigned branch's current PR |
| `ci-rerun` | `--run-id`; requires the fixed repository's pull_request run for current HEAD and branch |

The new issue, PR and comment selectors accept positive ASCII numeric IDs only. Labels
match `[a-z0-9:_-]{1,40}` and pass the same outgoing inspection as titles and bodies.
`--allow-canary-test` accepts only one complete canary token as the entire issue
comment body, without whitespace. Only that body's single `CANARY` finding is
tolerated; private literal hits and every other finding still block.

Review-record JSON requires `base_sha`, `review_head_sha`, `diff_digest`, `engine`,
`complete`, `summary`, `findings_p0p1` and `unresolved_p0p1`.
`review_head_sha` must equal current HEAD; the README describes the field checks.
This transports the record; CI owns base/digest invalidation and independent
engine policy. Replies and resolution require a nonempty recorded reason.
The text file is read once (maximum 1 MiB), inspected using the existing public
guard's Reporter and private literals, then sent via stdin. Titles have the same
1 MiB limit. Titles, labels, branch names and outgoing API payloads are also
inspected. Commit publication scans
every ancestor of HEAD, including commit metadata, through `public_guard.scan`.
It conservatively refuses an ancestor that is not public-safe.

Place `--dry-run` before the operation. It prints one JSON object per planned
write containing the exact `argv` array and `stdin` snapshot. No planned write is
executed. Local Git inspection and necessary read-only GitHub identity queries
still run; therefore PR state, thread/review/run dry runs need GitHub access in deployment.
No shell command strings are constructed. Arbitrary commands, options, force,
main pushes, ref deletion, alternate repositories and no-verify are refused.
Generic edits and canary exceptions outside `issue-comment` are refused.

The public guard pre-push hook permits main and the documented agent branches,
with full content and ancestor inspection. This gate preserves the hook.
T3 must verify the narrow allow,
worktree behavior and cross-repository rejection; T13 must verify raw-write
denials and synthetic private text rejection with fake executables. No test of
this implementation performs a real GitHub write or network request.

## Deployment evidence contract

`deploy-gated.cmd --check-only` evaluates without launching anything and prints
one JSON verdict with `conditions` and named `failures`. Exit 0 means PASS and
exit 1 means BLOCK. Without `--check-only`, a full pass allows one launch
attempt. The gate rechecks HEAD, the remote-tracking commit and checkout
cleanliness immediately before invoking the hidden launcher. If deployment
raises an error after a full pass, the verdict is BLOCK and retains the
preflight `conditions` and `failures`, with `deployment: REFUSED` and a
`deployment_error` describing the refusal. A successful launch reports
`deployment: LAUNCHED`. Use its launch receipt and the runbook's installed
acceptance to establish actual deployment. The command preserves `-KeepPrevious`.

`--state <json>` selects an explicit state file. The default is
`$LSS_PRIVATE_ROOT/run/deploy-gate.json`. Each path in that file is an absolute
path string. A relative path fails the condition that reads it and
`failure_details` names the key. State and receipts stay private. The state
contains these path references:

| Key | Existing evidence or normalized receipt |
| --- | --- |
| `repo` | Checkout whose locally recorded origin/main identifies the target |
| `manifest` | Unmodified `freeze_candidate.py` package-manifest.json |
| `build_receipt` | Unmodified off-drive build .build-receipt.json |
| `packaged_receipt` | Real machine package checks and builder identity |
| `verification_receipt` | Independent verifier verdict |
| `answer_key` | Exact answer-key file used in comparison |
| `answer_key_receipt` | Non-regression against the installed release |
| `current_product` | Existing current-product.json release record |
| `rollback_receipt` | Successful startup check for the current release retained as rollback |
| `idle_receipt` | Fresh observation of product, GPU jobs and PC usage |
| `deploy_base` | Deployment base passed explicitly to the runbook script |
| `deploy_log`, `launch_receipt` | Hidden launch output and exclusive launch receipt |

The native freeze and build formats are consumed directly. Other heterogeneous
existing receipts need a normalized JSON summary referring to their actual
Evidence; this gate does not generate or invent PASS receipts. The live
receipt-to-summary integration remains to be verified in T17. Every candidate
receipt requires `status: PASS` and exact `source_commit`, `exe_sha256` and
`tree_sha256` from the freeze manifest. Fields of the wrong type fail the
condition that reads them. Additional fields are below.

| Condition | Exact check |
| --- | --- |
| `integrated_main` | Source commit equals the commit resolved by local `git rev-parse --verify refs/remotes/origin/main^{commit}`. The gate uses the locally recorded tip. |
| `frozen_candidate` | Native frozen type; all package file names, sizes, hashes and tree digest match; no links; EXE hash matches; native build status/phase complete and commit, EXE hash and candidate directory match |
| `packaged_acceptance` | Candidate receipt has `scope: real-machine-package` |
| `independent_verification` | Nonempty packaged `builder` and verification `verifier` strings differ |
| `answer_key_non_regression` | Candidate receipt has `baseline_exe_sha256` equal to current-product hash, exact `answer_key_sha256`, `non_regression: true`, nonempty `user_benefit`, and `change_kind`; timing-only additionally requires `pcm_identical: true` |
| `rollback_ready` | Current-product `deployed_executable` exists and hashes to its `exe_sha256`; rollback receipt has same `exe`, `exe_sha256`, `status: PASS` and `startup_pass: true` |
| `machine_idle` | Candidate receipt's numeric Unix-seconds `observed_at` is at most 60 seconds old, `product_running: false`, integer `other_gpu_jobs: 0`, `user_heavy_use: false` |
| `checkout_head` | HEAD equals the candidate source commit |
| `checkout_clean` | `git status --porcelain --untracked-files=all` returns no entries |
| `deploy_base` | Deployment base is an existing directory |
| `deploy_log` | Parent directory exists and an existing log is a file |
| `launch_receipt` | Parent directory exists, the receipt path has no existing entry, and it resolves to a different path from the log |

The freshness check uses local time. Receipt observations are not a process or
GPU lock; a job starting after observation remains a race. Build/verifier names
are an accountable record, not cryptographic proof of independence. Package
checks rehash bytes; they do not enforce freeze ACLs. T17 must remove each real
receipt in turn, observe refusal, then use valid receipts for an authorized real
hidden deployment and independent installed verification. T3 must establish the
wrapper permission behavior. These machine-dependent claims remain deferred.
