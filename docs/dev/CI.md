# Public CI and local environment inspection

Integration, local deployment and Product promotion remain separate gates. CI
certifies its deterministic checks and review evidence; it makes no audio-quality
or deployment claim. Configure only the workflow's `result` job as required in S3.

`scripts/change_class.py` reads `.github/change-classes.yml` from the base commit.
The file is JSON, a YAML subset, to keep runtime tools standard-library-only.
Overlapping rules take the highest class. A covers non-Product documentation and
test files, B covers Product code, and C covers the mechanism and Product intent.
Unknown paths are C at this build's explicit request (the design's default was B).
D describes irreversible operations and is never an automatic diff classification.
Renames classify both old and new paths. The digest hashes sorted B/C paths and
their binary, full-index patches from the base, including deleted files.

The workflow handles PR updates, submitted/dismissed reviews and main pushes.
Each deciding script and its imports run from a separate `_trusted` checkout:
the PR base SHA for PR events, or the pushed SHA for main pushes. `LSS_CI_REPO`
points to the head checkout, used only for Git objects, files and diffs as data.
The head checkout runs first so its cleanup cannot remove `_trusted`.
Classification, publication and documentation checks always run. B/C add Linux
and Windows tests through `python scripts/dev.py test --ci -j 1 -m "not local"`.
The CI mode reuses the existing selector, runner, temporary state and receipts,
includes `tests/gates`, and omits machine storage admission on hosted runners.
It cannot run acceptance/native suites or change-based selection. `result` uses
`always()` and accepts only `success` for every job needed by event and class;
unneeded jobs may be skipped. Main pushes do not require PR forms or reviews.
Draft B/C reviews fail. There are no workflow path filters, artifact uploads,
`pull_request_target` events or repository secrets. Only the explicit review
fetch uses the ephemeral `github.token`, with read-only permissions.

All action references use full commit pins, including the trusted checkout.
`test_actions_are_pinned_before_first_push` rejects placeholders and floating refs.

The public guard scans committed content and generic PR/commit text. Its public
sentinel file is not the private word list: the private scan stays local. The
existing guard's token patterns run; a separate gitleaks integration remains S3
work, including a pinned distribution and synthetic detection checks.

## Review evidence contract

Review code that handles original media and secrets, UNKNOWN versus ABSENT,
rollback preservation and P/D/T state-root isolation. Report actionable P0/P1
findings against the reviewed commit. For mechanism changes, check fail-closed
classification and result aggregation, bot identity and receipt freshness,
thread resolution and implementation-engine provenance. Integration never
authorizes deployment or Product promotion.

`gates/review_gate.py` accepts `{"pr": {...}, "policy": {...}}` on stdin and
emits a Boolean and fixed reasons. The trusted `.github/review-policy.json`
explicitly selects `mode: "bot"` or `mode: "local-records"`; missing or unknown
modes fail closed. The current policy selects local records. Bot mode requires
a configured bot login, node ID, app database ID and completion regex. Its review
must be a submitted PR review by the matching bot and app, have a recognized
completion body, and identify the reviewed commit. Null configuration, missing
REST app attribution and dismissed reviews block. Ordinary comments cannot
count as bot reviews. S3 validates attribution against actual GitHub responses.

In bot mode, each receipt binds `base_sha`, `review_head_sha` and `diff_digest`.
The API collector accepts one fenced `lss-review-receipt` JSON object in PR
comments per review, additionally keyed by integer `review_id`. No duplicate receipt is chosen
automatically. Comments provide bookkeeping, never bot identity. For example:

```lss-review-receipt
{"review_id": 123, "base_sha": "<base>", "review_head_sha": "<reviewed-head>", "diff_digest": "<B/C digest>"}
```

S3 establishes the receipt adapter with the real bot; the bot is not assumed to
emit this custom format. In local-records mode, every implementation engine
requires a current independent record from another engine: Claude requires
Codex, Codex requires Claude, and human requires either. Mixed Claude/Codex
implementations require both records. Each fenced `lss-review-record` JSON
comment binds `base_sha`, `review_head_sha` and `diff_digest`, and contains
`engine` (`"claude"` or `"codex"`), `complete: true`, a nonempty `summary`,
`findings_p0p1` (integer >= 0) and `unresolved_p0p1` (integer equal to zero).
Booleans are not integer counts. Zero findings is a valid completed review.
The same account can write these records: this is an omission check, not
forgery resistance or proof of author independence.

Both modes require the same base and B/C digest, and a reviewed head equal to
the current head or a Git ancestor followed only by class-A changes. Product
changes and base movement require another review. Every unresolved P0/P1 thread
blocks; unresolved threads without a recognized priority also block conservatively. Rejected findings must
have their reason written in the thread before resolution. The collector checks
resolution, but cannot establish the semantic adequacy of that reason.

PR bodies use `## Checked`, `## Not checked`, `## Rollback` and
`## Implementation engines` headings with nonempty contents. C additionally
requires `## Review` with the mechanism risks and finding dispositions. Write the
engine list as comma-separated `claude`, `codex`, `human` values. Current owner
labels do not change this provenance. In bot mode, Codex implementations,
including mixed ones, also require a fenced `lss-review-record` JSON comment
containing the same base/head/digest, `engine: "claude"`, `complete: true` and a
nonempty `summary`.
Its author identity is not independent proof: it records a separate-context
review. Posting it or resolving threads requires rerunning CI on the current
head through the separately owned gated GitHub wrapper. Comment/thread events do
not automatically rerun this workflow.

The collector paginates reviews, comments and GraphQL review threads. A thread
whose comment list exceeds the collection bound fails closed. API errors and
head/base races fail without logging raw responses. These tests cannot establish
that rulesets reject a forged `result` context: S3 must require the GitHub Actions
app and intended workflow/job, enable strict base freshness, then try counterfeit
statuses/check runs and human reviews (T10), A/B/base/dismissal and zero-findings
reviews (T11), and Claude/Codex/mixed provenance and owner swaps (T12).

## Read-only state tools

Run `python scripts/env_check.py` for status codes only. It parses project JSON
and TOML settings, checks isolated hook registrations in both agents, allow/deny
presence, administrative deny restoration, Codex forbidden rules, explicit
environment-root resolution and state junction drift. Missing optional settings
are `UNKNOWN`; installed broken settings are `FAIL`. It neither executes hooks
nor reads global profiles, sandbox logs or credentials. It never prints environment
values. Runtime hook injection, ACLs, trust, real sandbox enforcement, state guard
behavior and GitHub administrative facts remain `UNKNOWN` pending their own checks.

The installer writes `bin/gates-manifest.json` under `LSS_PRIVATE_ROOT` with
`schema_version: 1` and a `files` array of `path`, `source_blob` and `sha256`.
Every listed file is checked against its installed hash and
`origin/main:gates/<path>` when available. For `lss-gh.cmd` and `deploy-gated.cmd`,
the upstream comparison substitutes `__LSS_GATE_PYTHON__` with the manifest's
`python` value, exactly as installation does. Paths cannot escape `bin`.
A missing manifest is `UNKNOWN`; malformed manifests and mismatches are `FAIL`.
No installer or copy is invoked. `origin/main` must already exist locally; the
tool never fetches it. The same reference supplies CI job-presence monitoring.

`python scripts/whatsnext.py` reports checkout identity/dirty state, all registered
worktrees' dirty states, leading private resume headings, GPU holder metadata,
free disk space and the environment summary. The only private note it opens is
`env-build/PROGRESS.md` below the explicit private root; prose is never returned.
The GPU helper's metadata is not proof of a live lease, so availability remains
`UNKNOWN`. Product usage, deployed version and Evidence resolution also remain
`UNKNOWN` until the S2 P/D/T and single-Evidence bindings are supplied. It creates
no state roots or junctions. `--github` is the sole opt-in to read-only `gh` PR and
assigned-Issue queries, and requires the `gh` executable. Default operation is
offline. This build never used that flag or contacted GitHub.
