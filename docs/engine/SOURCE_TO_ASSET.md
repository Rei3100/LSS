# Engine | From Source to Wanted Audio

## Planner input
Semantic user target + immutable source + optional compact intelligence about previous occurrences, versions, references and known failure modes.

1. Inspect local input and establish source identity, timeline, the finished 2ch stereo channels and eligible regions.
2. Identify candidate musical events/occurrences and difficulty at source/scene/cue/region/component resolution.
3. Choose a small meaningful portfolio of routes, including original/no-op. Do not select a provider merely by a fixed stem label.
4. Evaluate and harvest useful regions/components; favor preserving the musical whole over aggressive residue removal.
5. Assemble compatible material, validate continuity/phase/level/stereo/reverb, and keep representation class explicit.
6. Let the Product decision layer determine publication/HELD/UNKNOWN and CurrentBest; finishing follows that decision.

## Source representations
When several 2ch representations of the same content exist, compare verifiable facts (lossless or lossy, codec and bitrate, sample rate and bit depth, clipping or corruption, timeline completeness, stereo integrity, reacquisition reliability) and choose by them, avoiding a silent transcode before separation. Different masters, edits, language mixes or releases are different versions: never equated as "higher quality" of one source, and kept apart from the BGM identity decision.

## Fidelity defaults
A Recipe never implicitly normalizes, limits, resamples, downmixes, changes sample count or timeline, leaves an algorithmic offset uncompensated, or serialises float output into an integer format that clips; any signal-changing transform is an explicit Recipe stage inside Recipe identity and validation. Parallel stems of one source keep a common timeline: leading or trailing silence is never trimmed per stem, and provider latency, padding and start offsets are measured, compensated and recorded. Native input/output rate, internal resampling and effective bandwidth are recorded separately (a 44.1 kHz model written back at 48 kHz has not regained its bandwidth). Mono-native models run per channel are a different capability from joint-stereo processing and need stricter stereo validation as a final 2ch renderer.

## Recipe construction
- Recipe identity records every output-affecting choice (stage graph and order, provider, checkpoint/config, parameters, segment/context/overlap, precision, channel handling, resampling, alignment, branch conditions, stopping rules, serialisation). Stage order is part of it: A then B and B then A are different Recipes.
- A later stage may read the immutable original mixture as well as prior-stage output, so an early separator cannot make wanted material unrecoverable. Residual energy is never reinjected blindly: reinjection, cross-stem recovery and alternate branches change the output and are validated for target damage, interference, artifacts, timeline and stereo fidelity.
- Iteration needs a net validated improvement and an objective stop rule trading interference reduction against target damage; running the same stage again is never a default.
- Candidates are added by failure signature, not by novelty or leaderboard: name the failure mechanism as narrowly as the Evidence allows (music sent to another stem, cross-stem recovery; missing target, original-mixture branch or target-conditioned recovery; dialogue residue, dialogue-specific cleanup; strong reverb, context diagnosis; rate or stereo regression, provider I/O diagnosis) and compare a few high-information alternatives. When the mechanism is unknown, improve the diagnosis or run a bounded experiment; models are never enumerated without bound.

## Supported mechanism families / possible compositions
- RAW/crop when already sufficient.
- Dialogue/music/SFX extraction and multi-stem analysis.
- Cascade: isolate music -> local unwanted voice cleanup -> specialist residual repair.
- Parallel harvest: original clean regions + wrong-stem music + alternate separator + occurrence donor + released donor -> coherent assembly.
- Music-in-Effects or voice-in-score recovery when intended music leaks into other stems (deployed for singing; instruments and accompaniment that a separator sent to Effects/Other are the open extension).
- Component reassembly: separate the music into instrument stems (piano, strings, bass, drums, choir, other …), clean only the stem that is dirty (e.g. an effects model before drum separation where percussive SFX collide with drums) and sum the stems back; per component, keep whichever occurrence or route is cleanest. Practitioners combine it with multi-occurrence splicing rather than choosing one. The result is judged as one piece of music (timbre, attacks, tails, balance, stereo), never by stem scores; effects models also remove orchestral hits and other musical effects.
- Local reconstruction, inpainting or generated restoration when strongly justified and labeled; never represent it as recovered studio master. Un-ducking or rebuilding the pre-mix is Restoration, a separate labelled capability, not part of extraction (`../product/PRODUCT_INTENT.md`).
- Text-queried separation of any named sound (SAM Audio), used only as a time-frequency mask on a recorded parent so the result stays a faithful subset of the source; specialists win where they exist (`WANTED_SOUND.md`).

## What must remain local
Difficulties at 2 seconds do not condemn a 32-second cue. Stem scores, embeddings and waveform similarities are clues, not semantics. Do not destroy unresolved wanted material while reducing SFX or creating an EXACT relation.

Short score and stingers need positive bounded-asset evidence rather than a blanket ten-second usefulness exclusion. The bounded-short route requires Music observations on source and separated audio, measured quiet boundaries outside the proposed body in both separated channels, and no observed source-music continuation across an ordinary edge. A source digital-silence edit is a separate physical witness. It may extend a body outward inside materialized coverage, never trim wanted samples to an interior notch. Content safety, partial-reference uncertainty, program-insert uncertainty, integrity and CurrentBest remain independent gates. Tagger non-detection is not an assertion that music is absent.

## Failure frontiers
Dense tonal/musical SFX and wanted-score weakening remain technically hard. Past negative results apply only to tested checkpoint/configuration/role/positive controls. A model's verdict is scoped to its role (whole-mix route, single component, local window, re-recovery from a discarded stem, evidence only), its position in the chain and the population; a whole-cue failure does not close a narrower role, and a load/runtime failure is not a quality verdict. A role is reopened only with a reason the earlier failure no longer applies. Continue other Product work and explore new mechanisms when justified.

## Prior rejected routes and machine facts
Each entry holds for the tested role, configuration and population only; it is a recorded disposition, not a family-wide negative, and a rejected renderer stays usable as an analyzer.
- Target machine: Windows 11, RTX 5060 Ti 16 GB, 32 GB RAM.
- Whole-mix separator replacement, none of six candidates (including the current default) adopted; the common cause is Music/SFX confusion on effect-dominant material. MRX: stereo integrity and target absence. TIGER: target absence and masking. Facing the Music 4-stem: direct replacement rejected, residual use under 0.1 dB, 4 wins and 29 losses against v9 on 42 near-truth rows. BandIt v2 / DnR v3: +0.24 dB on the failure case, but target-absent leakage and activity false positives. SFC-Locoformer: hard failures `dialogue-sfx-target-absent` and `sfx-only-target-absent`, which the default also fails.
- Mid/Side routing: M/S input alone and half-and-half with the parent failed hard gates in 18 and 15 of 30 Development cases; local-branch variants failed 9, 5 and 1 of 30 (the parent-protected one also missed the improvement condition); M/S as a detection witness added no recall.
- Related-version (same piece, different recording) recognition by chroma separates synthetic arrangements but not real references: 0 related candidates on 62 Development references. The next hypothesis is binary cross-recurrence with longer-span local alignment, not the single fixed diagonal window.
- Reference-free recurrence cannot find shares shorter than about 6 s (the window-match count of 3 s x 2 and the waveform verification of 2 s x 3 blocks set the floor; lowering the minimum does not recover them).
- A resident default model saves about 10% per song (load 2.2-4.2 s of 22.9 s for a 1 min 45 s song) and is rejected against the risk to GPU exclusivity, worker recycle and RuntimeRevision; revisit only if multi-song batches dominate and per-song cost matters.

## Performance
Reuse Source/Recipe intermediate artifacts, avoid tag-only GPU reprocessing, and choose a lower-cost route when it produces the same accepted output. GPU speed or a positive separation metric cannot compensate for losing wanted notes.
