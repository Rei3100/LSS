"""Optional episode occurrence memory for local score presence, never donors.

Reuse usage clocks, immutable source hashes, the established mono decoder and
recording-support displaced controls. Only the source locator and positive
observations persist. Missing source media disables this amplifier; the single
episode route remains available. A prior TV occurrence cannot prove release,
cleanliness, absence, or whole-span identity.
"""
from pathlib import Path
import hashlib
import json
import os
import re

import numpy as np
from lss.bgm import identity, recording_support as rs

POLICY = {"revision":"usage-clock-occurrence-support-v1", "reference_seconds":30.,
          "maximum_sources":4, "maximum_records":64, "maximum_tracks":12,
          "minimum_music_windows":3, "minimum_rendered_seed_seconds":3.,
          "unresolved_recording_role":"retain_candidate_without_semantic_music_authority",
          "identity_authority":"none", "donor_authority":"none", "availability_authority":"none"}


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def remember(source, source_sha256, usage, evidence, state_root):
    if not re.fullmatch(r"[0-9a-f]{64}",source_sha256):
        raise ValueError("Occurrence memory needs an immutable source hash")
    if usage.get("state") != "aligned_positive_nominations":
        return {"state":"no_verified_usage_clock"}
    rows=[]
    for row in usage.get("onsets",[]):
        at = float(row["predicted_seconds"])
        a,b=max(0,int(at)),min(evidence.count,int(at+POLICY["reference_seconds"]))
        if (row.get("role")=="score" and row.get("track_id")
                and (evidence.music_probability[a:b]>=.5).sum()>=POLICY["minimum_music_windows"]):
            rows.append(dict(row))
    if not rows:
        return {"state":"no_observed_score"}
    folder=Path(state_root)/"bgm-occurrence-support"
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/f"{source_sha256}.json"
    record={"type":"bgm_usage_occurrence_memory","schema":1,"policy":POLICY,
            "source":{"path":str(Path(source).resolve()),"sha256":source_sha256},
            "music_positive_seconds":np.flatnonzero(evidence.music_probability>=.5).tolist(),
            "usage":{**usage,"onsets":rows},"sample_rate":identity.SAMPLE_RATE}
    if target.exists():
        try:
            previous=json.loads(target.read_text('utf8'))
            if previous.get("type")!="bgm_usage_occurrence_memory" or previous.get("source",{}).get("sha256")!=source_sha256:
                return {"state":"existing_unowned_record_preserved"}
        except (OSError,ValueError,AttributeError):
            return {"state":"existing_unowned_record_preserved"}
    temporary=target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open('x',encoding='utf8') as stream:
        json.dump(record,stream,ensure_ascii=False,sort_keys=True,allow_nan=False)
    temporary.replace(target)
    return {"state":"remembered_positive_occurrence","path":str(target),"sha256":_digest(target),"tracks":len(rows)}


def nominate(query, music, usage, evidence, source_sha256, state_root):
    receipt={"policy":POLICY,"records":[],"used_sources":[],"skipped":[],
             "identity_authority":"none","absence_authority":"none","donor_authority":"none"}
    if usage.get("state")!="aligned_positive_nominations" or music is None:
        return receipt
    folder=Path(state_root)/"bgm-occurrence-support"
    if not folder.is_dir():
        return receipt
    paths=sorted(folder.glob('*.json'),key=lambda p:p.name)[:POLICY['maximum_records']]
    rate=identity.SAMPLE_RATE
    sources=0
    tracks=0
    for path in paths:
        if sources>=POLICY['maximum_sources'] or tracks>=POLICY['maximum_tracks']:break
        try:
            record=json.loads(path.read_text('utf8'))
            source=record['source']
            if (record.get('type')!='bgm_usage_occurrence_memory' or record.get('schema')!=1
                    or record.get('sample_rate')!=rate or source['sha256']==source_sha256
                    or record['usage'].get('state')!='aligned_positive_nominations'):
                continue
            pairs=[(row,prior) for row in usage.get('onsets',[]) for prior in record['usage'].get('onsets',[])
                   if row.get('role')==prior.get('role')=='score' and row.get('track_id')
                   and row['track_id']==prior.get('track_id')]
            if not pairs:continue
            if _digest(source['path'])!=source['sha256']:
                raise ValueError('occurrence_source_hash_changed')
            raw=identity.decode_mono(Path(source['path']))
            if _digest(source['path'])!=source['sha256']:
                raise ValueError('occurrence_source_changed_during_decode')
            sources+=1
            for row,prior in pairs:
                if tracks>=POLICY['maximum_tracks']:break
                tracks+=1
                first=max(0.,prior['predicted_seconds']-1.)
                last=min(len(raw)/rate,first+POLICY['reference_seconds'])
                reference=raw[round(first*rate):round(last*rate)]
                low=max(0.,row['predicted_seconds']-rs.POLICY.search_before_seconds)
                high=min(len(query)/rate,low+rs.POLICY.maximum_seconds)
                a,b=round(low*rate),round(high*rate)
                if min(len(reference),b-a)<3*rate:continue
                offsets={rs.alignment(query[a:b],reference)}
                for delta in (-2.,4.,10.,18.):
                    x=max(a,round((row['predicted_seconds']+delta)*rate));y=min(b,x+8*rate)
                    if y-x>=3*rate:offsets.add(rs.alignment(query[x:y],reference)-(x-a))
                sha=hashlib.sha256(np.ascontiguousarray(reference,dtype=np.float32).tobytes()).hexdigest()
                for offset in sorted(offsets):
                    observed=rs.observe(query[a:b],reference,rate,origin=a/rate,offset=offset)
                    if observed['state']!='component_observed':continue
                    t,f,c=rs.trace(music[a:b],reference,offset,rate)
                    heard=(f>=rs.POLICY.seed_fraction)&(f>=rs.POLICY.control_ratio*c)
                    observed_music=sum((evidence.music_probability[max(0,int(s['start_seconds'])):
                                      min(evidence.count,int(np.ceil(s['end_seconds'])))]>=.5).sum()
                                       for s in observed['spans'])
                    # The new occurrence may have the classifier omission we
                    # are recovering. Independently observed music in the
                    # aligned prior occurrence supplies the second witness.
                    prior_seconds=set(record.get('music_positive_seconds',[]))
                    prior_music=set()
                    for span in observed['spans']:
                        x=span['start_seconds']+observed['offset_seconds']+first
                        y=span['end_seconds']+observed['offset_seconds']+first
                        prior_music.update(s for s in prior_seconds if x<=s and s+1<=y)
                    if heard.sum()*rs.POLICY.hop_seconds<POLICY['minimum_rendered_seed_seconds']:
                        continue
                    music_corroborated=max(observed_music,len(prior_music))>=POLICY['minimum_music_windows']
                    observed['render_trace']={'origin':a/rate,'times':t.tolist(),'fraction':f.tolist(),'control':c.tolist()}
                    receipt['records'].append({**observed,'onset':row,'reference_sha256':sha,
                                              'reference_type':'episode_occurrence','source':source,
                                              'reference_source_span':[first,last],
                                              'music_corroborated':bool(music_corroborated),
                                              'music_presence_state':'corroborated' if music_corroborated else 'unknown_preserve_candidate',
                                              'availability_authority':'none','donor_authority':'none'})
            receipt['used_sources'].append({'source':source,'memory_path':str(path),'memory_sha256':_digest(path)})
        except (OSError,ValueError,KeyError,TypeError,AttributeError,RuntimeError) as exc:
            receipt['skipped'].append({'memory_path':str(path),'reason':type(exc).__name__})
    return receipt
