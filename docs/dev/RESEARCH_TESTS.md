# Development | Fixed Research Tests

Use this entry when trying a technology, input role, order or assembly. The goal is to choose a stable, relevant comparison from existing cases, expose missing conditions and reuse cached outputs. The case index is `ledger/research_tests.json`; native corpus schemas and evaluators remain authoritative. `QUALITY_AND_DECISIONS.md` owns claim strength; `CORPUS_AND_EVIDENCE.md` owns the episode corpus and answer keys.

## Choose the question before the model

Run with the current/shared repository Python from the active checkout:

```text
python scripts/research_tests.py list
python scripts/research_tests.py plan bgm-separation --pattern order --pattern context
python scripts/research_tests.py check --bank instrumental-controls --hash-assets
```

| Profile | Decision / fixed starting point |
|---|---|
| `bgm-separation` | Music cleanup: controlled clean/no-op, quiet score, wanted singing, speech/SFX and target absence; separate short condition and explosion-context banks |
| `instrumental` | Accompaniment wanted, lead singing unwanted: three songs with accompaniment-only, singing mix, vocals-only and quiet-accompaniment variants |
| `recovery-order` | Guide, residual and instrument assemblies: controlled guards plus the historical approximate-reference bank; separate their denominators |
| `occurrence` | Preserve single-occurrence usefulness; consult pinned real/synthetic splicing experiments for actual overlap, wrong donor and time mapping; common guards do not establish recurrence coverage |
| `discovery-identity` | Usage recall, boundaries, released redundancy and version identity; controlled guards support diagnostics, full episodes use the existing answer-key protocol |
| `scene-naming` | Scene observation versus inference, identity-wide naming; unit fixtures are locators, no registered cross-work semantic oracle is claimed |
| `runtime-lineage` | Provenance, revisions, permissions, interruption/reuse; use the real package when the changed boundary depends on it |

Default output is compact; `--json` before the subcommand exposes full records. `plan` fixes the selected bank fingerprint, IDs, conditions, truth classes and comparison constraints in JSON. `--out <new-file>` retains that selection without overwriting an earlier attempt. It returns `SELECTED_NOT_EXECUTED`, never PASS. Naming/runtime profiles with no indexed audio bank return `NO_REGISTERED_AUDIO_CASES`; their existing tests and missing semantic/package coverage remain visible. `check` returns nonzero for unknown IDs, changed definitions, native/index case mismatch or missing assets. Default observation checks file presence; `--hash-assets` reads only the chosen bank's declared files. It does not infer or repair a cache.

## Fixed populations and their limits

The index retains existing case IDs, original-definition SHA-256, source/target/cached artifact identities where available, and the existing comparison entry. Case IDs are bank-qualified; repeated views of one source are not independent samples.

| Bank | Truth / use |
|---|---|
| `bgm-guards` | Existing spatial plan, with each row's truth retained. Nine controlled component guards are the core screen. Three released identity intervals have uncertain waveform applicability and stay diagnostic; core separation selection excludes them. Native spatial recipes are examples, not arbitrary-model adapters. |
| `instrumental-controls` | Known additive accompaniment/singing components. Twelve variants from three songs, including absence. This target removes singing; BGM's singing guard retains it. |
| `condition-6s` | Actual manifest has seven rows: speech three, animal three, explosion one. Tagger-screened non-music may contain music. Generic `sound_effect` was not collected. |
| `explosion-context` | Six separate events embedded at 12 seconds in 30 seconds of music. Events were obtained through Musicless and may retain original BGM, favouring that method. Keep this population apart from the six-second bank. |
| `headroom-42` | Approximate fitted released-reference truth and exploratory thresholds. Historical `heldout` labels remain visible, but exploration already exposed these rows; they do not establish untouched holdout. Cached inputs are not all indexed here; follow the native manifest and repaired judge. |

Episode answer keys answer usage/completeness/identity questions, not sample-accurate clean-stem quality. corpus-a episode 1 is the focus; 137 and 145–152 check generalization. Complete, partial and unknown lists keep their different absence authority. A real unreleased cue without truth remains useful for Development diagnostics; it is never converted into fabricated clean truth.

## Vary a hypothesis, keep the comparison stable

The `--pattern` vocabulary comes from the supplied guides, current Product contracts and prior trial records:

| Pattern | Variable / required invariant |
|---|---|
| `model` | Checkpoint/architecture; same input, intended output role, context and scoring |
| `order` | Guide/cleaner/instrument order; same component models and returned stems |
| `stem` | Whole mix, dirty named stem or removed side; retain parent identity and assembly |
| `transform` | Gain, speed/pitch, shift or polarity; undo transforms and check level, timeline, native stereo |
| `assembly` | Mean/min/max/mask/fill; same cached passes, no scoring-set oracle stem choice |
| `context` | Short/padded/long input; score the same inner window so context is not a hidden advantage |
| `recurrence` | Single/aligned/wrong/misaligned donor; distinguish same recording and actual overlapping musical time |
| `reference` | None/partial/exact/wrong-master/later arrival; preserve unknown wanted spans |
| `representation` | Faithful extraction or labelled Restoration; un-ducking/generation is not extraction success |

These are selectable variation descriptions, not an evaluated Cartesian product or a compulsory experiment queue. Use `technology.py` for prior results and pinned execution code. In particular, the old condition bench chose the best output stem on its scoring set; new experiments fix the intended output role before scoring. A baseline's context or output role cannot silently change with the candidate. Freeze a useful small Development screen first; add a discriminating case for the specific hypothesis rather than repeatedly running all banks. Confirmation and Product adoption retain their separate requirements.

## Replay one cached comparison

`compare-instrumental` deliberately supports one existing bank and measurement path. The checked example protocol is `release-evidence/research-tests-20261104/cached-instrumental-protocol.json`:

```text
python scripts/research_tests.py compare-instrumental release-evidence/research-tests-20261104/cached-instrumental-protocol.json --out <new-result.json>
```

For a new candidate, retain a new protocol before measuring. It fixes `schema_version: 1`, `bank_sha256` from the selected Instrumental bank, and `baseline`/`candidate`, each containing a method `identity`, `target: Instrumental`, and an `outputs` map keyed by every fixed case ID. Every output records `path`, `sha256`, `input_sha256` and `encoding` (`float-wav` or `lss-flac`). Keep an execution receipt/Recipe identity as the method provenance; a declared identity alone is not execution certification. Changed/missing/extra cases, wrong targets, wrong input/output hashes and unsupported serialization are refused.

The comparator reuses `instrumental_quality` and `probe_instrumental_repeat.comparison`. It reads one case at a time, preserves full native stereo/rate/length, decodes recorded FLAC headroom, and applies no fitting, alignment, resampling or normalization. Target-absent energy, wanted gain, uncorrected error and stereo components remain separate. Consumed artifacts and implementation identities are fixed and rechecked before publication. No missing output triggers inference. Results are `DEVELOPMENT_DIAGNOSTIC`, with per-case vectors and regressions; no synthetic overall winner or Product promotion follows.

Other bank evaluators retain their native formats. BGM singing can use `song_component_comparison`; waveform boundaries use `bgm_corpus` / `evaluate_bgm_discovery`; exact per-case inference variants use `variant_receipt`. Do not make one universal adapter a prerequisite for trying a useful method.

The real-mix comparison entry is `scripts/prepare_bgm_quality.py --repo <foundation-checkout> --out <new-protocol.json>`, followed by `scripts/compare_bgm_quality.py <protocol.json> --out <new-result.json>`. Preparation freezes all 42 existing `headroom-42` rows, cached historical Y10 and original-mix Musicless outputs, reference/attribution snapshots, implementation hashes and runtime versions. It never infers missing outputs. The comparator reuses native source/reference masks and STFT conventions and `stereo_quality`, with fixed three-second windows plus the partial tail. It reports wanted retention, band loss, proxy speech/effects residue, stereo damage and local failures that whole-row means hide. Band loss is not proof of an individual instrument loss; source-fitted reference error is not proven non-music. Existing native split labels survive but every row is reused Development data, never untouched holdout. Historical Y10 is not the installed EXE, and standalone Musicless is not a dialogue-first or post-Y10 recipe. The result rejects blanket replacement on recorded diagnostic harm or holds it; it does not promote a Product default. Real recurrence joins, subjective naturalness and independent instrument truth remain unresolved. The first measured comparison and native parity check live in `release-evidence/real-quality-20261105/`.

## Extend the same owning index

After a useful new case or variation, retain its native definition and artifact identities, add its locator/conditions/truth/limits to the same bank or a justified new bank, and run the targeted `check`. New data changes a bank fingerprint and requires a new comparison protocol; do not replace a failing case or denominator in place to improve a result. New methods/results update the technology ledger rather than making a second technology scoreboard.

Known gaps remain visible: generic isolated tonal/transient SFX in the short condition bank, dedicated wanted choir/vocalise components, broader native sample rates and long-form contexts, and cross-work scene/name semantic truth. Unit guards and historical observations can support iteration; they do not silently close those acoustic or semantic gaps. Source media stays local and unchanged; original surround/manual-listening advice in supplied references does not change the supported stereo/autonomous/no-listening boundary.
