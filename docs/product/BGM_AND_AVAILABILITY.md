# BGM | Wanted Music Versus Already-Available Music

## What belongs
Wanted music playing in the scene: score, underscore, battle/action music, short cues, stingers, eyecatches, music within dialogue, musical vocalise, singing, chorus and harmonies, mixed-source score and scene-specific arrangements, and music boxes, bells and chimes playing as music (a single chime used as an effect is not). Continuity across a program boundary does not alone change musical meaning. Two pieces sounding at once, or a crossfade or medley from one piece into another, are two musical identities, never one cue.

## Why some OP/ED and soundtrack recordings need no extraction
**Independent availability** is the decisive Product question, not whether the recording has vocals, appeared during an opening/ending, or has an OST-like title.

If the *appropriate requested asset* is confirmed to be the same independently available clean recording, normally record the occurrence/identity and do not publish an inferior redundant extraction. This avoids recreating what the user already has access to.

A program usage made only of a released recording is that recording, however the program edited it: one section of it, shortened, cut and rejoined, looped, faded, or with its sections reordered. Identify it, record which parts of the release were used, and report it as "released, not extracted". Arrangements, variations and component versions stay valuable, e.g. a part muted for a while that enters later, which even a finished LSS could not rebuild from the release. Only material the release does not contain keeps extraction value (in corpus-a ep1, 14 of 18 usages are released and must end that way; only the 4 unreleased are extracted).

Do not assume every OP/ED is freely/appropriately available, or that every track in a release fits the TV recording. A program-role label alone cannot decide wantedness, and an album title alone cannot prove audio identity.

**Preserve as potentially valuable**: program-only intros/outros/bridges, partial replacements, altered mixes/masters, component-muted/additional-part versions, reprise, vocal variant, alternate performance, layered unreleased cues and any unresolved material.

## Repetition is not evidence of wanted BGM
A recording that recurs within a program or series fixes a cue's extent and where to extract it from, but it is not evidence that the recording is wanted BGM: recurring catchphrase dialogue and library stock SFX recur just as strongly. A candidate built from repetition passes `lss/bgm/content_guard.py`, which keeps it only when music exceeds both speech and effects in the candidate's measured shares and drops it when either exceeds the music; loud speech or effects alone is never a reject reason, since score under dialogue is normal. Passing means only that music dominates, never that the wanted semantic role is decided.

## EXAMPLE: correct vs incorrect
- Released opening exactly matches the user's wanted recording: identify and avoid redundant re-extraction.
- Background orchestral cue includes sung choir: choir is *wanted* score, not dialogue to delete.
- A released corpus-c track matches most of the episode but an unknown cue overlays the ending: the unexplained region is still potentially wanted; whole-span EXACT suppression is false.
- A track shares a motif with an independently available release but changes arrangement: related, not EXACT.
- Program advertisement overlaps a legitimate scene score: local role evidence must not erase the adjacent wanted music.

## Policy boundary
The policy must compare actual coverage of the *requested resulting asset*. Partial matches are localized facts. UNKNOWN remains UNKNOWN. A relation engine may propose, but may not silently promote false identity or a donor overwrite. See `../engine/RECOVERY_AND_REFERENCES.md` and `../engine/QUALITY_AND_DECISIONS.md`.
