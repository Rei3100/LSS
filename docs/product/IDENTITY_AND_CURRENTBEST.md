# Identity and the CurrentBest Contract

## Different things — do not conflate
- **Source:** immutable origin identity/time/channel/rate information. The original input file is read-only to LSS: never edited, normalised, retagged, trimmed or remuxed; every derived representation is a new Artifact with provenance.
- **Occurrence:** one appearance in one source and context. The occurrence (source, time span) survives any regrouping of cues; the scene evidence attached to it keeps its kind (subtitle/observed vs model transcript/detection).
- **Cue hypothesis:** possibly revisable relationship among occurrences; name is not identity. Splitting or merging cues later never loses an occurrence, a title or a saved version.
- **Recording / version relation:** same recording, edit, alternate mix or master, component variant, arrangement/re-recording, shared motif, or still unknown. Similarity is evidence of sameness, not sameness.
- **Asset:** durable wanted-audio object within a semantic target and context. An extracted span is not called the whole piece: completeness (full, fragment, intro only, missing end, loop) is recorded or UNKNOWN. A music timeline may hold overlapping cues, crossfades, medleys and a cue continuing across a title card, so one start and end per cue is not assumed.
- **Revision:** concrete audio/provenance version of an Asset.
- **Representation:** faithful extract, recovered/composed, reference-assisted reconstruction, generative restoration, listening derivative.
- **Name/path/tag:** mutable presentation, not Cue/Asset identity.
- **CurrentBest:** preferred *valid* revision for defined use/identity, not newest run or largest metric. A better region may update only that region.
- **Kinds of knowledge:** observation (what the source contains) and model estimate (tied to the model/version that produced it) are stored apart. Derived conclusions should not be counted as independent evidence (an ASR line feeding a title is one source, not two), and a retracted basis should be traceable to what used it.

Scene ↔ cue navigation in both directions and a return to the original moment follow from these records.

## Revision decisions
| Previous | New Evidence | Product result |
|---|---|---|
| Valid | uncertain candidate | retain previous CurrentBest, retain useful candidate/uncertainty |
| Valid | valid demonstrably better | adopt newer with history, rollback and justification |
| Positively invalid | local counterevidence | stop public use with recorded reason, preserve history |
| None | supported useful candidate | publish and record provenance |
| Existing unchanged audio | tag-only/source environment update | reuse audio; refresh LSS-owned tags; no duplicate `(r2)` |
| Only title changes | official title/artwork available | update presentation without new Asset/re-separation |

LSS-owned results (extracted audio, cue boundaries, provisional titles, LSS-owned tags) are the current LSS's best: when a newer LSS produces a better result it replaces them as a new revision, recorded in the lineage (a title change is a naming revision), and the previous audio and name stay recoverable through history and rollback. Names follow identity: all occurrences of one recognised recording or track identity carry the same title at any moment, distinct pieces never share a title, and a better title (a newer LSS's, or a verified official one) replaces the old one for the whole identity at once as one naming revision. Re-running the same LSS version on the same input changes neither name nor result: a replacement needs a better result, not a new run.

Completeness counts as demonstrably better: a cue whose one-cue extent was unresolved (a fragment) is replaced by a new cue of the same source that covers it without gaps, is longer and has a decided extent; an unknown extent never replaces anything.

When an Asset's PCM changes (a CurrentBest update or a partial region update), signal-derived measurements (loudness, true peak, ReplayGain) are recomputed, never copied from the old audio, and regeneration starts from the source plus recorded transforms, never from a previously processed output, so repeated updates add no generation loss.

A validated Recipe that supersedes an earlier one records the capability, applicability scope and exact prior revision it replaces; an improvement on one source class, role or failure signature is never read as global replacement. The supersession yields reprocess candidates, which are not permission to replace CurrentBest: the new run passes validation and publication like any other, and a bulk reprocess is started by the user, never by LSS on its own. An identical input already processed is recognised and not run again, but a deliberate reprocess request is never skipped as a duplicate.

While LSS is unfinished every published result is LSS-owned and replaceable (user, 2026-11-03): a CurrentBest file missing from its public path is rebuilt from its record, and a file LSS did not write that sits at that path is moved into history, never overwritten in place and never a reason to fail the run. Version history, current pointer and physical file retention are separate. A prior WAV/FLAC may retire under verified lifecycle rules without erasing its identity or the audit trail needed to reconstruct it.

## Publication transaction
Produce to a staged location, validate intended PCM/integrity and identity, write the LSS-owned tags, then atomically switch publication where supported. Failure may not destroy the previous accepted audio, tags, artwork, album ReplayGain or music-library history.

## Durable explanation
For consequential decisions record source/occurrence/regions, Recipe/provider version, donor/ref relation, evidence hashes, representation class, validity/quality dimensions, selected policy and rollback link. The metadata/UI projection must not become an independent authority.
