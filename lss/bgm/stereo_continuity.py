"""Bounded positive stereo witnesses; never recording identity or absence."""
import hashlib
import math
import json
from pathlib import Path

import numpy as np

from lss.bgm import cue_semantics as cs

REVISION = "bounded-side-music-continuity-v2"


def _covered(candidates, start, end):
    cursor = start
    for a, b in sorted(candidates):
        if b <= cursor:
            continue
        if a > cursor:
            return False
        cursor = max(cursor, b)
        if cursor >= end:
            return True
    return False


def nominate(parts, candidates, evidence, recipe, *, similarity=.92):
    """Nominate integer holes only inside exactly covered candidate material."""
    plans = []
    for left, right in zip(parts, parts[1:]):
        a, b = float(left['end_seconds']), float(right['start_seconds'])
        lo, hi = map(float, (left['core_seconds'][0], right['core_seconds'][1]))
        if (not all(math.isfinite(v) for v in (a,b,lo,hi))
                or a != int(a) or b != int(b) or not 3 <= b-a <= 8
                or left['stretch'] != right['stretch']
                or left['end_edge']['kind'] != cs.EDGE_SILENCE
                or right['start_edge']['kind'] != cs.EDGE_SILENCE
                or a-lo < 8 or hi-b < 8 or not _covered(candidates,lo,hi)):
            continue
        start, end = int(math.floor(lo)), int(math.ceil(hi))
        if start < 0 or end > evidence.count:
            continue
        vectors = np.asarray(evidence.embeddings[start:end],dtype=float)
        if not np.isfinite(vectors).all():
            continue
        lhs = evidence.embeddings[start:int(a)].mean(0)
        rhs = evidence.embeddings[int(b):end].mean(0)
        norm = np.linalg.norm(lhs)*np.linalg.norm(rhs)
        cosine = float(lhs@rhs/norm) if norm else 0.
        inner, _ = cs.split_profile(evidence.embeddings,start,end,recipe.minimum_part_seconds)
        if cosine < similarity or inner >= recipe.doubt_distance:
            continue
        plans.append(dict(revision=REVISION, gap_seconds=[int(a),int(b)],
                          core_seconds=[lo,hi], stretch=left['stretch'],
                          flanks=[list(left['core_seconds']),list(right['core_seconds'])],
                          flank_similarity=cosine, joined_split_distance=float(inner),
                          doubt_distance=recipe.doubt_distance, floor_music_db=recipe.floor_music_db,
                          candidates=[list(x) for x in candidates]))
    return plans


def project(music, plan, destination):
    """Write unamplified FLOAT side evidence on an integer source grid."""
    import soundfile as sf
    a,b = plan['gap_seconds']
    if not _covered(plan['candidates'],a-8,b+8):
        raise ValueError('Projection context lacks exact candidate coverage')
    with sf.SoundFile(str(music)) as stream:
        if stream.samplerate != 44100 or stream.channels != 2:
            raise ValueError('Side witness requires 44100 Hz stereo')
        origin = max(0,a-8)
        end = min(stream.frames//44100,b+8)
        if origin > a or end < b:
            raise ValueError('Materialization does not cover gap')
        stream.seek(origin*44100)
        audio = stream.read((end-origin)*44100,dtype='float32',always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError('Nonfinite side source')
    side = (audio[:,0]-audio[:,1])*.5
    level = 10*np.log10(np.maximum(np.mean(side.reshape(-1,44100).astype(float)**2,axis=1),1e-20))
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    signal = destination/'side.wav'
    if signal.exists() or (destination/'receipt.json').exists():
        raise ValueError('Projection destination already populated')
    sf.write(str(signal),np.column_stack((side,side)),44100,subtype='FLOAT')
    receipt = {**plan,'projection':'(L-R)/2 duplicated; no gain','origin_seconds':origin,
            'range_seconds':[origin,end],'signal_path':str(signal),
            'signal_sha256':hashlib.sha256(signal.read_bytes()).hexdigest(),
            'side_db':level.tolist(),'identity_authority':'none','absence_authority':'none'}
    digest=hashlib.sha256()
    with open(music,'rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            digest.update(chunk)
    receipt.update(type='bgm_stereo_projection_receipt',plan=plan,count=end-origin,
                   side_levels_db=level.tolist(),source={'path':str(music),'sha256':digest.hexdigest()},
                   outputs={'signal':{'path':str(signal),'sha256':receipt['signal_sha256']}})
    (destination/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def _evaluate(plan, probability, embedding):
    """Authorize positive seconds only when every gap second is witnessed."""
    origin,end = plan['range_seconds']
    n = end-origin
    p = np.asarray(probability,dtype=float)
    e = np.asarray(embedding,dtype=float)
    levels = np.asarray(plan['side_db'],dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Nonfinite or invalid probability')
    if p.ndim == 2 and p.shape == (n,527):
        p = p[:,137]
    if (p.shape != (n,) or e.ndim != 2 or e.shape[0] != n or not e.shape[1]
            or levels.shape != (n,) or not all(np.isfinite(x).all() for x in (p,e,levels))
            or np.any((p<0)|(p>1))):
        raise ValueError('Malformed side witness arrays')
    a,b = plan['gap_seconds']
    if not origin <= a < b <= end or a-origin < 4 or end-b < 4:
        raise ValueError('Missing side context on source grid')
    inner,_ = cs.split_profile(e,0,n,4)
    # A short foreign cue can disappear in whole-excerpt averages. Check both
    # actual gap seams with the existing doubt threshold as well as the global
    # profile. 137 falsifier: three different piano/score excerpts passed the
    # global profile (.083-.100) but failed these local seams (.145-.188).
    def distance(left,right):
        lhs,rhs=left.mean(0),right.mean(0)
        norm=np.linalg.norm(lhs)*np.linalg.norm(rhs)
        return max(0.,1.-float(lhs@rhs/norm)) if norm else 1.
    seams=[max(distance(e[t-h:t],e[t:t+h]) for h in (2,4)) for t in (a-origin,b-origin)]
    local_stable=bool(max(seams) < plan['doubt_distance'])
    heard = bool(np.all(p[a-origin:b-origin]>=.5))
    audible = bool(np.all(levels[a-origin:b-origin]>=plan['floor_music_db']))
    stable = bool(inner < plan['doubt_distance'])
    stable = stable and bool(np.all(np.linalg.norm(e,axis=1)>0))
    accepted = heard and audible and stable and local_stable
    return {**plan,'state':'stereo_music_continuation_observed' if accepted else 'stereo_music_continuation_unconfirmed',
            'positive_seconds':list(range(a,b)) if accepted else [],
            'music_confirmed':heard,'side_audible':audible,'side_stable':stable,
            'side_local_stable':local_stable,'side_seam_distances':seams,
            'side_split_distance':float(inner),'identity_authority':'none','absence_authority':'none'}


def plan_holes(candidates,evidence,recipe):
    candidates=list(candidates)
    return nominate(cs.segment(candidates,evidence,recipe),candidates,evidence,recipe)


def evaluate(plan,projection_receipt,probabilities,embeddings,recipe):
    recipe.validate()
    if (plan.get('doubt_distance') != recipe.doubt_distance
            or plan.get('floor_music_db') != recipe.floor_music_db):
        raise ValueError('Witness recipe mismatch')
    if projection_receipt.get('plan') != plan:
        raise ValueError('Projection plan mismatch')
    result=_evaluate(projection_receipt,probabilities,embeddings)
    result.update(span=plan['gap_seconds'],positive_music_seconds=result['positive_seconds'],
                  reasons=[name for name in ('music_confirmed','side_audible','side_stable','side_local_stable') if not result[name]],
                  metrics={'side_split_distance':result['side_split_distance'],
                           'side_seam_distances':result['side_seam_distances'],
                           'flank_similarity':plan['flank_similarity'],
                           'joined_split_distance':plan['joined_split_distance']})
    return result
