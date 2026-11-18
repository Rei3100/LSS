# Engine | Quality, Claims and Promotion

## Quality is a vector
Assess wanted-music retention, speech/SFX residue, timbral integrity, melody/instruments, continuity, stereo/phase, dynamics, transients, edits/seams, semantic identity, listening usefulness (inferred from measurements; listening is optional, see below), and provenance/state reliability. No one proxy wins all dimensions.

## Evidence hierarchy by claim
1. Frozen positive and negative controls with valid measurement apparatus.
2. Exact packaged/runtime artifacts and immutable hashes/provenance.
3. Local reproducible audio/case results with truth scope labeled.
4. Independent orthogonal recomputation on consequential uncertainty.
5. External research and model judges as hints, never substituted for audio Product truth.

UNKNOWN is not NEGATIVE. A failure of one role/checkpoint does not rule out a method family. Before a negative capability claim, establish a working positive control. One clean released sample does not establish performance across full episodes.

## Evidence tiers
Evidence strength matches the weight of the decision. **Development** (learning, comparison, debugging, prioritising, reversible iteration) accepts controlled proxies, imperfect references, diagnostics, cross-model disagreement, reference-free analysis and explicitly provisional semantic judgement. **Validation** (does a candidate beat the alternative on a defined local protocol and domain) uses frozen cases, repeatable measurement, adversarial and absence cases, and stronger semantic checks where available. **Promotion** (changing a Product default, formal support of a capability, strong quality or semantic claims, irreversible or destructive operations) requires what the claim needs: independent ground truth, holdout, strong provenance, equivalent confirmation. Missing clean stems, cue sheets, same-master references or an independent semantic oracle blocks claims, never development; unknown ground truth is recorded as uncertainty, and certainty is not fabricated to settle a reversible decision.

## Four different questions
- **Acoustic:** did the measured audio actually improve for this case?
- **Semantic:** was the desired music preserved and correctly distinguished from available/other material?
- **Operational:** can the real package produce and reuse the result autonomously after interruption?
- **Adoption:** should this exact candidate replace existing Product behavior/CurrentBest?

The first three inform but do not automatically grant the fourth.

## Freeze consequential evaluation
Record candidate/source/EXE hashes, case IDs, populations, thresholds, expected decisions, positive controls and scope *before* checking candidate outcomes. Never change the denominator or redefine wantedness to make a failed candidate pass. Preserve decisive counterexamples.

## Hard gates, reporting and defaults
- **Hard gates before ranking:** a candidate failing one is never the winner whatever its average: NaN/Inf, corrupted or incomplete output, target-absent hallucination beyond contract, timeline mismatch or uncompensated start offset/length change, channel collapse, unintended downmix or resampling, serialisation clipping, silent CPU or lower-quality fallback, an established serious semantic-role failure, publication without required provenance. Gates that can be executable checks are; a contested or unobservable semantic claim is not made a hard Development gate for want of perfect ground truth.
- **Reporting:** keep the measurement vector, not only a synthetic score, and report distributions, paired differences, failure counts and the worst tail rather than mean or median alone. Catastrophic deletion of wanted music is counted separately and is a heavier failure than residue of the same size; a better mean with more such deletions is not an improvement. Every metric carries its preprocessing (implementation/version/arguments, resampling, silence handling, projection, normalisation, channel handling, inactive-source behavior, time alignment/latency compensation, level fitting), and same-named metrics from different pipelines are not one measurement.
- **Changing a default:** a candidate becomes the capability's default when it was compared with the current default on the same inputs, runtime and measurement over a fixed domain case set; it does not fall below the default on the main cases and beats it on at least one product-meaningful case; it does not regress on the recorded guard cases (target absence, singing/choir, stereo, timeline, sample rate, long form); it runs on the target machine within its resource limits; and its Recipe identity is recorded with the previous revision kept for switching and rollback. Full genre coverage, comparison with every public model or with unmeasured future candidates, and an absolute ceiling are not required: a scoped adoption with a stated scope is enough. A default chosen once and never compared is as unverified as a rejected candidate and needs the same comparison to stay.

## Listening is optional
The development agent cannot listen, and no plan, comparison, acceptance condition or hand-off requires listening. Decide from measurements that need no ears (retention, residue, misallocation, target absence, stereo, timeline, identity, residual); a dimension that cannot be measured stays UNRESOLVED, and an old listening-based statement is neither a blocker nor an excuse. Where an unmeasured result can change a decision, the candidate is held without promotion and a listening request names what it decides, the number of files, their total length and their private location; other work continues without waiting for the answer, and only the work that depends on it waits. A result that cannot change a decision is recorded as UNRESOLVED without a request. What the user heard, requested or not, is Product input and is recorded as `human-listening` Evidence. Finished audio may be offered for enjoyment. The subjective fidelity of UI design is the user's (`LISTENING_AND_PRESENTATION.md`).

## Strong falsifiers to retain
- Reference causes one wanted public cue to become zero (false whole-span EXACT).
- Unexplained local corpus-c music gets donor-overwritten in reconstruction.
- SFX cleanup improves residue but destroys wanted score/singing.
- Reused asset loses its tags/artwork; a tag-only change triggers duplicate inference.
- New uncertain run causes valid prior CurrentBest disappearance.

## Risk-proportional assurance
Use a distinct builder/verifier when uncertainty is consequential: replacing live Product, destructive semantic identities, state transitions, or material rollback risk. A new verifier needs a frozen exact candidate and narrowly targeted doubt; do not replay every prior test ritualistically. Safe isolated experiment/implementation should proceed boldly and quickly.
