# Research Frontier | Hypotheses, Not Product Features

The field changes; refresh only the external delta needed for an actual Product question, prefer primary papers/official implementations and useful practitioner observations, and test on the local supported Windows/GPU setup before making availability claims.

Prior technology, role, order and combination coverage has one entry: `docs/research/TECHNOLOGY.md`, backed by `docs/research/technology_ledger.json` and `python scripts/technology.py search "<pattern>"`. Consult it before intake or a new experiment; update the same scoped record after a result. This page owns research direction and practitioner method, not a second tried/untried ledger. Installation, successful execution, quality in a tested role and Product adoption are separate facts.

## Open problem families
- Wanted-score retention under heavy tonal/music-like SFX and overlapping dialogue/expressive voices.
- Region/component selection and reconstruction that survives an unknown secondary score or edge change.
- Released recording discovery beyond obvious OST names, including cross-work and partial libraries.
- Exact-versus-related recording recognition without destructive whole-span shortcuts.
- Reappearance/series donors that improve real listening quality rather than simply reducing nonshared energy.
- Reliable local scene understanding (dialogue transcription with names, subtitles, video) for scene-based naming, the scene ↔ cue map and music segment boundaries. Naming from the scene is a Product requirement, not a research option; the research is how to do it accurately on the supported Windows/GPU setup, and video stays optional for extraction itself.
- Quality evaluation that can detect wanted harm, continuity and semantically useful novelty without a hidden true master.
- Future cross-target audio component reuse and generative restoration with honest provenance.

## Recovery-method exploration (current priority)
Discover recovery methods LSS has not tried; raising adoption of existing candidates or repairing listed failures one by one is not the priority. Judge a model by what it can contribute in some role (whole cue, one instrument, the removed side, pre-processing, detection, local repair), not only by dropping it into the current route; reshape the route around a model's strengths. Non-separation clues count too: repetition/phrase structure, alignment across occurrences, stereo/space, SFX character, timbre continuity, agreement between separators. No fixed experiment list or model set; follow promising ideas found on the way, and state whether a new test changes the model or its role/input/representation/assembly. Generated material is never presented as recorded.

One practitioner already succeeds at what LSS wants (anime and film score extraction, published): he is the model to learn from, and the UVR/MSST community guide is written by experts further ahead. Reproduce his methods faithfully, every step, then go beyond them; his method is not the only answer. LSS's existing assets are used as well; both together, neither ranked above the other. What these sources cover is not re-researched externally; external research is for what they leave open (newer releases, implementation details).

### Method follows the cue's instrumentation
Every cue is different: which instruments it actually uses decides the method, the stems and the order. A drum-less cue suits Band Separation Reassembly (anything drum-like is SFX); a piano-only cue needs only the piano stem; some cues give clean drums, others do not; instrument-dropped versions occur in use. First establish what the cue contains (from its cleanest seconds, other occurrences, a released reference, LSS's instrumentation analysis), then extract only those instruments, in an order that does not lose them (piano before drums when drum removal eats attacks, bass before drums when bass leaks), and clean only where a present instrument is dirty. A recipe applied identically to every cue and judged by its average is not a reproduction of the practitioner.

### The practitioner's method (his own account)
- Base pass: an instrumental model over the clip (UVR-MDX-NET Inst 3 earlier, MB-Roformer-Inst-V2 / BS-Roformer now), instrumental only, lossless output. Songs with choir/backing vocals: a karaoke model instead (5_HP-Karaoke, MB-Ro-Kara-Aufr33-Viperx) so wanted voices stay. Crowd noise: a de-crowd model before vocal removal. Known weakness: hiss when the music gets quiet.
- Occurrence splicing (his main method): index every use of the cue; extract every occurrence; take the longest non-stretched occurrence as the timeline and sync the others to it; splice the cleanest stretch of each moment; level-match each spliced part. Scenes where only voice overlaps the cue are the most valuable (voice is removable, ambience/SFX are not). A single occurrence cannot lose its SFX; battle/action cues need many occurrences. Instrument-dropped versions appear in use; the full-instrument version is the target.
- Band Separation Reassembly: pull instruments out one by one (MVSEP models); before drums, apply the FX (SFX) model, or to whichever stem is dirty first; then extract drums; leftover SFX in drums goes with Drumsep. FX is applied to dirty stems, never to the whole clip; it can call orchestra hits SFX. Drums are the weak point (SFX recognised as drums). Combining both methods is best; SFX loudness still caps quality.
- For a first attempt he advises BS-Roformer first, then the FX model.
- His surround/front-channel sources are outside LSS's 2ch input.

### Guide technique families (role and assembly coverage in the technology ledger)
These are a design space, not an unexecuted queue. Residual boosting, speed/pitch passes, phase correction, ensembles, instrument specialists, USS and occurrence splicing already have tested cells in the rotation/spatial records. A new input, order, conditioning or assembly is a distinct pattern; an earlier cell's failure neither marks that family untried nor closes every other role.
- Re-separate the "other"/residual stem after boosting it ~20 dB, then undo the gain (residues too small for the model).
- Separate at changed speed/pitch (sample-rate reinterpretation), map back, merge the runs.
- Phase fixer: correct one instrumental with another model's instrumental as reference (about 500 Hz–5 kHz, scale ≈0.8). Demudder: re-separation that returns instrument parts lost where vocals were.
- Per-section model swap; avg/max/min ensembles chosen per problem (max for missing parts, min for bleed, at most 4–5 models); fill only the missing band from another model; left/right separated independently.
- Restoration representations (labelled Restoration, never extraction): ducking restored with a sidechain expander driven by the removed voice; a weak repeated hit rebuilt from a clean repetition of the same pattern.
- Per-instrument specialists (Gilliaan bowed strings, Becruily guitar, Demucs_ft/SCNet for drums and bass, Drumsep 5/6 on extracted drums); audio-example query separation (USS, zero-shot) for a recurring timbre; delimiter/declipper before separation on loud masters.
- Order per sound: piano before drums when drum removal eats attacks; bass before drums when bass leaks into drums.

### External leads not yet tried (primary-source search, 2026-11-13)
Found outside the ledger, read from public primary pages only, nothing downloaded or run; each is a hypothesis for an existing role, recorded in the ledger as `external-leads-20261113`. Fix the falsifier before any result.
- SAM Audio Judge — no-reference separation judge scoring recall / precision / faithfulness from (mix, output, target text) ([paper](https://arxiv.org/abs/2601.19702), [weights, gated, SAM License](https://huggingface.co/facebook/sam-audio-judge)). Role: `quality-evaluation` candidate ranking. Falsifier: on rows where Aesthetics chose the missing-music candidate, recall must rank it lower; it must not score target-absent controls high; prompt wording must not flip the ranking.
- CineAudioDB — real production mixes with dialogue/music/effects stems and sidechain ducking, 2.46 h, CC licences per film ([data](https://huggingface.co/datasets/disco-eth/cineaudiodb), [paper](https://icml.cc/virtual/2026/77327)). Role: frozen external confirmation population with real ducking; only its stereo production mixes fit the 2ch input. Not for threshold fitting.
- Common-signal extraction across language versions and NMF co-factorisation with deformed references ([AES 2010](https://www.aes.org/e-lib/browse.cfm?elib=15328), TASLP 2015 hal-01070298, [motif-guided thesis](https://theses.eurasip.org/document/audio-motif-detection-for-guided-source-separation-application-to-movie-soudtracks/)). Role: another-language 2ch mix as an identity-verified donor (never a source replacement); deformation-aware joint fit for `occurrence-memory`. No public implementation found. Falsifier: synthetic versions with 0.5 dB level and few-sample offsets must beat current dialogue removal.
- TVSM — speech/music activity detector trained on about 1600 h of real TV ([paper](https://link.springer.com/10.1186/s13636-022-00253-8), [code and weights](https://github.com/biboamy/TVSM-dataset)). Role: independent witness for `cue-formation` recall of quiet music under dialogue; never an absence claim. Falsifier: it must add answer-key regions the current detector misses without adding held negatives.
- AV-CASS — flow-matching cinematic separation with audio-only and audio-visual checkpoints, MIT code ([paper](https://arxiv.org/abs/2603.26113), [code](https://github.com/pantheon5100/AVCASS)). Role: analyser or mask source for `separator-alternatives`, outputs projected onto the original stereo. Immediate reject: no stereo, music generated on target-absent controls, or over 16 GB.
- llama.cpp audio input (Qwen3-ASR GGUF and others) ([official list](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)). Role: second ASR on the shipped native runtime for `scene-understanding`, used only where both recognisers agree. First check: whether the packaged llama.cpp build reads audio at all.
- Separation-assisted broadcast fingerprinting on the BAF dataset ([code](https://github.com/guillemcortes/ssafp), [PeakNetFP](https://arxiv.org/abs/2506.21086)). Role: external answer key for `reference-identity` recall and the choice of query representation; data licence first, false EXACT counted with unreleased controls.

### Spending rules
- Breadth before confirmation. Confirmation on another population is the gate before Product use, not a step after every lead. Confirm only a lead large enough to change what the user hears (a clear share of failure rows fixed without guard harm), on a small subset fixed before any result (about 30 rows); a lead of a few rows goes back into exploration.
- Progress is counted in new methods tried (a new model, role, input, representation or assembly with its measured outcome), not in hours of measurement. A working session that ends without a new method tried has failed.
- Never idle on a GPU job: while one runs, try the next ideas on CPU or on cached passes. A new idea's passes never wait behind a confirmation.
- Report to the user per episode and cue: what is now retained or removed (measured), by how much, and what is still wrong; exploration-bank counts are internal.

## Decision form
For an active research line, preserve question, positive control, exact tested role/model/version, observed benefit/harm, contradiction, limitations and reopening trigger. Exit as ACT / TEST / HOLD_WITH_TRIGGER / REJECT / NO_CHANGE. Historical details belong in immutable Evidence or an archive; this page stays small.

A prior unsuccessful model is not a family-wide negative. A fresh model's novelty is not proof it will help. Research that does not answer a Product decision must be re-ranked.
