# Development | Real-Program Corpus and Evidence

Corpus is a set of complementary questions, not an episode freshness leaderboard.

All development and test media live outside the repository under the media root (`LSS_MEDIA_ROOT`), in `TestCorpus\`: `Sources\Development\<work>\` holds the episodes, `References\<work>\` the released recordings and OP/ED, `Acquisition\` and `Work\` acquisition and scratch. Works are named here by internal id (`corpus-a` … `corpus-d`); the table that maps an id to a folder is private (`LSS_PRIVATE_ROOT`). List the work folder before assuming an episode is missing. Development uses the media already there, reused across methods; the user curates the corpus. A cue whose other occurrences are not in the corpus is worked as a single occurrence. The corpus splits into `Sources\Development\` (development, comparison, debugging) and `Sources\Holdout\` (final evaluation only; nothing is tuned on it); when Holdout is empty it is re-designated before any final claim, keeping contiguous episodes together (measured on corpus-b: Development = 1+2, Holdout = 6+7 keeps 16 of 17 candidates; distant pairs keep 1-4). Acquisition leftovers such as `*.temp.mkv` are not inputs.

| Corpus | Main role | Hazard / useful positive |
|---|---|---|
| corpus-a | primary real BGM world: released + unreleased, recurrence, cries/action/tonal SFX | a fan-maintained per-episode usage list and episode-local timing alignment are useful *positive anchors*, not absence truth |
| corpus-b | reference-rich alignment, identity/coverage controls | easy known-clean references must not become a requirement for all media |
| corpus-c | partial-reference, musical voice, mixed/unreleased component and discovery adversary | one insert song is a real non-obvious release discovery; edge overlays challenge whole-span EXACT |
| corpus-d | sparse-reference robustness, dense transformation/device/action SFX and cross-work relation | a song from a related film soundtrack is a candidate relation, not a preverified exact TV recording |

## corpus-a answer key (per-episode usage lists)
A public fan-maintained site records the BGM played in each episode, in one of three coverage tiers: **complete** (every usage listed), **partial** (listed usages are positive anchors only; unlisted is UNKNOWN), **not yet recorded** (just aired; no answer key, LSS must still work). For corpus-a, episodes **1-152 are complete**. The lists are stored offline under the media root and development reads the local copy without re-fetching; local episode video exists only for ep1, 137 and 145-153, so occurrences outside these are not available to development. Every row links an album id, so release status comes from which album: one id is the released anime soundtrack, another is the unreleased score numbering; ep1 is 14 + 4. Classify every other album id before scoring with it. Episodes after 152 are checked on the page when used. The album pages are kept too (per track its number, title, length and every listed episode use). They are development answer keys only; LSS must work without them.

For a complete episode the list is the answer key, usage by usage:
- **Released** usage (the page names an album/track): LSS identifies it and does not publish a redundant extraction (`../product/BGM_AND_AVAILABILITY.md`); the occurrence and identity are recorded. Publishing it anyway is a failure.
- **Unreleased** usage: LSS publishes it as one cue (not split, not started mid-track) with a clean result. Missing it is a failure.
- Music LSS publishes that the list lacks is a suspected false positive judged on the audio, never removed merely for being unlisted.

**Answer-key episodes: corpus-a 1, 137 and 145-152.** Episode 1 is the focus: it is the simplest and most basic case, 18 usages of which only 4 are unreleased, so it mainly tests "released, therefore not extracted"; 137 onward carry many more unreleased cues and test extraction. Current scores and lanes are private development state. Each change is checked on the other answer-key episodes, not tuned to one.

Do not claim no soundtrack merely because a directly named OST was not found. Existing TestCorpus clips/episodes may be reused for development, comparison by measurement, debugging and adversarial regression.

## Fixed research comparisons

Technology, role, order and assembly trials start at `RESEARCH_TESTS.md` and `scripts/research_tests.py`: the versioned index selects existing case IDs, truth classes, guards and variation patterns, exposes missing assets/conditions, and points to native evaluators. One cached Instrumental comparison path reuses existing measurements without inference. Selection is not execution; condition mixtures, fitted references, episode answer keys and unit fixtures keep separate denominators and authority.

## Evidence classes
- Full actual episode Source input.
- Released/reference track or excerpt.
- Controlled mixture and known target region.
- Real packaged Product output and durable state.
- Synthetic relation/render controls.

Do not combine these populations in a denominator. A synthetic reference-reconstruction counterexample is not itself a real packaged episode counterexample. Name actual truth quality and unresolved scope.

## Identity anchors
Exact candidate, EXE and verdict identities change; they live in private development state and must be re-verified against Git and Evidence. Do not copy them into this stable contract.
