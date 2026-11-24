# Engine | Release Intelligence and Occurrence Memory

## Two optional amplifiers
**Released References:** discover independently available recordings; align waveform and specific components; decide whether extraction is redundant or whether a donor can recover a truly unavailable edit.
**Series Memory:** retain directly observed source-occurrence relations and candidates for reuse/re-evaluation; a single occurrence must still work normally.

They can cooperate but must not become hidden hard dependencies of each other or the normal no-reference path.

## Release discovery means opportunity search
Look beyond an obvious work-named OST: theme singles with score tracks, movie soundtracks, franchise collections, compilations, digital-only releases, bonus tracks, predecessor/successor works and non-obvious titles. Work aliases, franchise, composer, chronology and official metadata generate candidates; verified audio establishes relation. Use existing Product Acquisition service rather than a new downloading stack. Acquiring released reference audio (yt-dlp through the acquisition layer) is authorized standing by the user, in the Product and in development: nobody asks before it (user, 2026-10-19). Public access, bounded concurrency, retries and reproducible content identity only; no hidden cookies or credentials.

## Relations need type, scope and evidence
Record EXACT_SAME_RECORDING, EDIT/REORDER, ALTERNATE_MIX, COMPONENT_VARIANT, ARRANGEMENT/PERFORMANCE, MOTIF_RELATED, ADDITIONAL_MATERIAL or UNKNOWN with timing/coverage/channel evidence. A high fingerprint score, title or matching four-second subregion cannot authorize deleting the other 28 seconds.

## Adversarial obligations
- Mixed music: the reference accounts for the dominant corpus-c music but a low-level corpus-b cue remains wanted.
- Edge edit: final two seconds contain another cue, even if the majority is a known release.
- Different component balance: drumless/guitarless score is not full recording EXACT.
- A reconstruction map sees unexplained music: donor plan must not overwrite it.
- Sparse or failed external acquisition: local no-reference path remains valid.

For each proposed donor, reason about *what it truly covers*, whether the requested unavailable result is still there, and whether alignment/mastering/continuity and listening utility improve. Preserve uncovered regions or HOLD the unsupported reconstruction. A clean output identical to a released available master is generally not new LSS value.

## Occurrence memory
LSS's own published cues of other sources are the table-free occurrence memory a normal user has. When a first-pass segment verifies as one of them, that member measures (never renders) where the same recording is present in this episode's own source and music; the measurement enters cue formation as a recording component, under the same rules as a released reference (head to a silence edge inside the materialized stretch, tail only with rendered corroboration, hole only without a competing recording). It can only add or reconnect this episode's own music; it is never a release, donor, EXACT or absence witness, and an episode never supports itself (`lss/bgm/memory_support.py`).

When the user names a source for BGM processing whose CurrentBest contains an unresolved cue, and whose memory now has a verified, constant-offset link to a longer resolved cue of another source (`lss/bgm/memory_refresh.py`), that request re-evaluates the named source with the memory. Memory never schedules a source the user did not name: growth of the library after another source's job only makes the opportunity visible for the next time the user names the older source. Both accepted audio identities must still be present; a consistent clock and at least six physically shared seconds nominate the opportunity, never an extension or a replacement. The normal source pipeline measures the actual recording again and CurrentBest alone chooses whether to change the public asset. Pinned rollback results are excluded. The separate token-addressed run and atomic attempt receipt cover the full-source execution: source SHA, processing/memory recipes and accepted external PCM inputs, excluding that source's own outputs. A terminal attempt records those external inputs as evaluated; an alternative cue nomination or unrelated library growth cannot repeat the same source analysis. Genuinely new verified witness PCM or changed rules can reopen it. A failed attempt is terminal for those inputs too, and the ordinary run of the same source still answers the request; an abandoned/dead-owner claim remains resumable the next time the source is named, including when the UI would otherwise reuse its completed job. The re-evaluation requires no reference curation or runtime agent.

Store compact occurrence IDs, source fingerprint plus timeline/channel revision, alignment relations, quantitative observations and their scope, unknown/held reasons, provenance, retry checkpoints, and enough information to reopen conclusions after a new episode or Recipe. Do **not** claim a cleaner CurrentBest merely because recurrence exists or nonshared energy is smaller.

## Usage index and fullest form
For a work, build the usage index of a cue first (every occurrence found) and collect the occurrences before extracting, so the cleanest stretches of each can fill the others. Instrument-changing variants stay under one cue until their relation is understood, and the target is the fullest form of the piece found across occurrences, not the span trimmed to one scene.

## Common component across occurrences
The common music of several occurrences is estimated in one joint fit over all of them: estimating per occurrence and then averaging or taking a median keeps the shrinkage of every linear fit, because each estimate comes out smaller than the music. Only the stretch where the recording truly overlaps is compared, since occurrences of one cue often play different parts of the recording; non-overlapping stretches are never treated as common. The fit extracts linearly from the mix and changes no occurrence's timeline, channel count or amplitude; music it cannot take (per-occurrence performance differences, ducked-out parts) falls into the residual, so the result is never called clean music.

## Media lifecycle
Compact identity/alignment state is durable; heavy released media may be retired only if Product-owned, safely reacquirable/regenerable and not the sole Evidence for a Product claim. User-owned and unreacquirable media stay protected.
