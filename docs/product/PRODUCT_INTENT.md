# Product Intent | What LSS Is For

**Mission:** turn wanted sound embedded in finished media into a standalone, personally useful asset. Music already available as the same recording in the same form is recognised and not duplicated; a program-specific version, edit, intro or arrangement keeps its own value. BGM is the motivating priority, not the system's limit.

The user requests a semantic result, typically by providing a source (the finished 2ch stereo mix; 5.1/7.1/Atmos, M&E or centre-channel extraction are not inputs and not research shortcuts) and choosing BGM, Instrumental or STEM (internal identifier `SOUND`; resumed on 2026-11-05 as stem delivery for the work's music, `docs/engine/WANTED_SOUND.md`) — any wanted sound named in words (a lead vocal, an instrument, dialogue, an effect, ambience, or a combination). LSS must autonomously discover, plan, generate, evaluate, select, publish and recover the wanted asset. The user must not select a checkpoint, debug prompts, judge candidates, manually acquire soundtrack libraries, clean temporary folders, or repair Product state.

The user's design vision (integrated 2026-10-18) is the base of this document; later user decisions in chat and later briefs refine or supersede it where they differ, and the newer statement is the current intent.

## Stable values
1. **Get something useful today:** single episode/occurrence is first-class; no recurrence, official soundtrack, title, lyrics or external lookup required.
2. **Improve later without rewriting history:** later sources, releases and methods may upgrade CurrentBest; preserve identity, decisions, provenance, possible rollback.
3. **Faithful evidence and listening usefulness are not identical:** a clean listening reconstruction can coexist with a faithful extract; always label representation type, and keep apart what the recording contained, what the program's mix changed (ducking, EQ, fades) and what LSS repaired. Extraction returns what the finished 2ch mix actually contained; un-ducking, restoring a level the mix never had or rebuilding the pre-mix is Restoration, a separate labelled capability that is never counted as extraction success, and a louder reference or another occurrence alone does not justify it.
4. **Audio-first semantics:** desired singing, choir, vocalise, harmonies, reverb, drums and musical effects must be judged by user intent and musical role, not by an unwanted-'voice' heuristic.
5. **Minimize unnecessary damage:** no-op, localized processing, and partially resolved output can beat aggressive whole-span removal.
6. **Personal accumulation:** each normal job should make known cues/versions/occurrences and repair opportunities more reusable without requiring human cataloguing.
7. **Local independence:** core processing is local. Source media is never sent to an external analysis, judging or separation service (a public demo or hosted separator is research evidence only), and paid per-request services are excluded; public authorized metadata/reference acquisition is optional and never an inference requirement.
8. **Know what the music is in the work:** where it played (episode, time, the scene and its events), what it is (same recording, edit, arrangement, motif) and what is still unknown. A provisional name is a song-like name synthesized from that scene and from the music itself (when the scene is unknown it is named from the music alone), and one musical identity carries one name in every occurrence and distinct pieces never share one; a newer LSS that names it better, or a verified official title, replaces the name for the whole identity at once as a recorded, recoverable revision; missing scene knowledge never blocks or delays extraction.

## What would betray this intent
Each of these can look like progress; none is.
- A provisional title that asserts a scene event the evidence does not show ("熱き決戦" with no battle in the scene), a raw measurement list as the title ("ピアノ／ニ長調／120 BPM"), or no name at all because the scene is unknown (then the name comes from the music's tempo, rhythm, instrumentation/stems, structure and direction); or one recording under different titles in different occurrences, or distinct pieces under one title.
- Everything in a `Music` stem saved as BGM, or singing/choir that a separator put in `Dialogue` discarded as "dialogue removed".
- "Not in the OST list, so no music", or a similar game/soundtrack track treated as the same recording.
- Doing nothing because there is no second episode, no reference or no network.
- Destructive processing over 90 seconds to fix one difficult second.
- A newer result silently replacing a better earlier one; a rename creating a new cue.
- An LSS-owned result (extraction, boundary, provisional title) frozen against a better result of a newer LSS.
- Asking the user to pick models, compare A/B or repair state.
- Refusing 2ch input for lack of multichannel material.
- Generated or composed sound presented as what the source contained.
- Long technical logs where the user needs: what is saved, where, and what is still unknown.
- Intermediate stems left to fill the SSD.

## The success test
A person can start from a local finished-media file, request the target, receive a truthful standalone result or a truthful scoped non-result, use it in a normal music library, close/restart LSS, and come back later to an improved revision. No engineer is present during that user journey.

## Decision ownership
User: what sound is wanted, a real preference conflict, when an expensive corpus-wide reprocess runs, and final subjective acceptance when voluntarily offered (finished audio may be offered for enjoyment; it is never a gate). Downloading public models and released reference audio is already authorized; it is never put to the user.
Product: records reprocess candidates (a validated Recipe's scoped supersession) but never starts a bulk run on its own or occupies the PC unannounced; technical discovery/routing, separation/recovery, selection, naming, storage, evidence, error handling and routine update.
Development Control Plane: task choice, independent certification when warranted, research and interpretation. No ChatGPT or coding agent becomes a runtime dependency.

## Unsupported ≠ unwanted
The deployed version may not yet reliably publish some wanted sung BGM or difficult heavy-SFX regions. That is a coverage gap, **not** permission to redefine BGM as purely instrumental or to change frozen test denominators retroactively.
