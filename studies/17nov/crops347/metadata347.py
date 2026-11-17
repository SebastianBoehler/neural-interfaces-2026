"""Source-only fixed3.5–7.5s late-crop eligibility from actual800 end markers."""
import csv
import json
import sys
from pathlib import Path
import numpy as np

DEPENDENCY=Path('/mnt/lustre/work/gehler/gfh527/ni26_direct346_v1_20261117')
sys.path.insert(0,str(DEPENDENCY))
from common346 import source,restored,BASE,sha,digest
RAW=Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/data/neural_compet/Dreyer2026Proteus/download/nm000290')
LABELS={'mi':0,'sub':1,'word':2}


def plan(d,fit,people,y):
    with np.load(BASE/'data/proteus_windows.npz') as z:meta={k:z[k].copy() for k in ('task','run','onset')}
    groups={};eligible=[];missing=[];ends=[];feedback=[];allcues=[]
    for pos,row in enumerate(fit):
        session=str(d['session'][row]);session=session if session.startswith('ses-') else 'ses-'+session
        key=(str(people[row]),session,str(meta['task'][row]),int(meta['run'][row]))
        groups.setdefault(key,[]).append((int(row),pos))
    result=[]
    for (person,session,task,run),rows in sorted(groups.items()):
        directory=RAW/f'sub-{person}'/session/'eeg';matches=list(directory.glob(f'*task-{task}_run-{run}_eeg.edf'))
        if len(matches)!=1:raise ValueError('nonunique raw source recording:'+str(matches))
        raw=matches[0];event=raw.with_name(raw.name.replace('_eeg.edf','_events.tsv'))
        with event.open() as f:events=list(csv.DictReader(f,delimiter='\t'))
        cues=[r for r in events if r['trial_type'] in LABELS];mapped=[];used=[]
        for row,pos in rows:
            official=float(meta['onset'][row]);cue=[c for c in cues if abs(float(c['onset'])+1-official)<1e-7]
            if len(cue)!=1:raise ValueError('official/source cue does not match exactly:'+str(row))
            c=cue[0];onset=float(c['onset']);used.append(onset);allcues.append(onset)
            if LABELS[c['trial_type']]!=int(y[pos]) or abs(float(c['sample'])/500-onset)>1e-8:raise ValueError('source cue label/sample mismatch')
            nextcue=min([float(x['onset']) for x in cues if float(x['onset'])>onset],default=float('inf'))
            end=[float(r['onset']) for r in events if r['trial_type']=='end_trial' and int(r['value'])==800 and onset<float(r['onset'])<nextcue]
            entry=dict(row=row,official_onset=official,cue=onset,source_class=int(y[pos]),original_start=int(round(official*120)),eligible=False)
            if len(end)==0:missing.append(dict(path=str(event),row=row,cue=onset,reason='no true800end before nextcue'))
            elif len(end)!=1:raise ValueError('multiple actual800ends for source trial')
            else:
                bound=min(end[0],onset+float(c['duration']));start=entry['original_start']+300
                if start/120<onset+3.5-1/120 or (start+480)/120>bound or (start+480)/120>nextcue:raise ValueError('fixed late crop outside actual labelled period')
                fb=[float(r['onset']) for r in events if r['trial_type']=='start_feedback' and int(r['value'])==781 and onset<float(r['onset'])<end[0]]
                if len(fb)!=1:raise ValueError('missing/multiple actual feedback transition')
                entry.update(eligible=True,late_start=start,actual_end=end[0],declared_end=onset+float(c['duration']),feedback=fb[0])
                eligible.append(row);ends.append(end[0]-onset);feedback.append(fb[0]-onset)
            mapped.append(entry)
        if len(set(used))!=len(cues) or len(rows)!=len(cues):raise ValueError('source cue/export inventory differs')
        result.append(dict(path=str(raw),events_sha256=sha(event),rows=mapped,person=person,session=session,task=task,run=run))
    eligible=np.sort(np.asarray(eligible,np.int64))
    if len(fit)!=1898 or len(eligible)!=1894 or len(missing)!=4 or set(people[fit])!={'P04','P05','P06','P07','P08','P10'}:raise ValueError('pre-registered source eligibility differs')
    summary=dict(source_people=sorted(set(people[fit])),original_rows=len(fit),extra_rows=len(eligible),
        extra_rows_sha256=digest(eligible),missing_extra=missing,true_end_range=[min(ends),max(ends)],
        feedback_range=[min(feedback),max(feedback)],fixed_late_seconds=[3.5,7.5],official_overlap_seconds=1.5,
        nominal_outside_seconds=2.5,metadata_uses_query=False)
    return result,eligible,summary
