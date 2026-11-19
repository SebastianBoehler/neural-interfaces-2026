"""Replay live fit targets in the immutable exact cap250 worker."""
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from data import permutation,sha
from geometry import propose,rotate,spectra
import replay
from benchmark_utils.streaming import fresh_model

ARMS=('stock','geometry','rotation')
ORDERS=(0,1)


def changes(a,b):
    return [[int(i),int(b[i])] for i in np.flatnonzero(a!=b)]


def instrument(model,arm,smoke):
    ft,cp=model.ft,model.cp
    assert model.routed==1 and model.r15 and not model.v6r and ft.on
    assert ft.t0==20 and ft.every==10 and ft.n_steps==5 and ft.cap==250 and len(ft.member)==4
    assert cp is not None and len(cp.sos)==8 and cp.t==0 and not cp.C
    original=ft._refit
    cache=[]
    events=[]
    def refit(t,labels,weights):
        assert not model.recB and not cp.dead and cp.stopped is None
        assert cp.t==len(cp.C)==t and model.log6['n']==t+1
        assert len(ft.lab)==t and ft.relabels==ft.refits+1
        raw = np.asarray(labels)
        w = np.asarray(weights,dtype=np.float64)
        assert raw.shape==w.shape==(t,) and np.isin(raw,(0,1,2)).all()
        assert np.isfinite(w).all() and (w>0).all()
        chosen=raw
        proposed=[]
        if arm!='stock':
            tail=spectra(cp.C[len(cache):t])
            cache.extend(tail)
            assert len(cache)==t
            suggested,hard=propose(np.asarray(cache),raw,w,check=smoke and not events)
            proposed=changes(raw,suggested)
            chosen=suggested if arm=='geometry' else rotate(raw,suggested,w,hard)
        edited=changes(raw,chosen)
        assert len(edited)==len(proposed)
        events.append(dict(age=t,raw=raw.tolist(),proposed=proposed,edited=edited,
                           weight_hash=hashlib.sha256(w.astype(np.float32).tobytes()).hexdigest(),
                           edited_weight=float(sum(w[k] for k,_ in edited))))
        original(t,labels if arm=='stock' else chosen,weights)
    ft._refit=refit
    return events


def stream(template,eeg,rows,arm,smoke=False):
    start=perf_counter()
    model=fresh_model(template)
    assert not model.v5hist and model.ft.refits==0 and model.ft.relabels==0
    assert '_refit' not in vars(model.ft)
    events=instrument(model,arm,smoke)
    labels=[]
    for r in rows.tolist():
        label=model.predict(eeg[r:r+1].to('cuda:0'))
        assert label.device.type=='cpu' and label.dtype==torch.int64 and label.shape==(1,)
        labels.append(int(label[0]))
    torch.cuda.synchronize()
    activity=replay.activity(model)
    assert activity['errors']==0 and activity['windows']==len(rows)
    assert not any(p.dead for p in (model,model.cp,model.cf))
    assert np.isin(labels,(0,1,2)).all() and len(model.v5hist)==len(rows)
    assert len(events)==activity['ft_refits']==activity['ft_relabels']
    assert activity['ft_steps']==20*len(events)
    assert [e['age'] for e in events]==list(range(20,20+10*len(events),10))
    assert (model.ft.dtheta>0)==bool(events)
    assert arm!='stock' or all(not v['edited'] for v in events)
    inner=[int(q.argmax()) for q in model.v5hist]
    del model.ft._refit
    assert not template.v5hist and template.cp.t==0 and not template.cp.C
    return dict(labels=labels,inner=inner,activity=activity,events=events,
                seconds=perf_counter()-start,rows=rows.tolist())


def returning(template,channels):
    x,streams,names=replay.read_source()
    assert names==channels.tolist()
    anchor=json.loads(Path('/normal.json').read_text())['streams']['first']['P10-3']
    result=stream(template,x,dict(streams)['P10-3'][:31],'geometry')
    assert result['activity']['recognized']==10 and not result['events']
    assert result['labels']==anchor['labels'][:31]


def main(index,arm,smoke):
    assert arm in ARMS
    assert torch.__version__=='2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    with np.load('/external.npz',allow_pickle=False) as z:
        people,days,channels=(z[k] for k in ('subject','session','ch_names'))
        x=torch.from_numpy(z['X'])
    assert x.shape==(2130,41,480) and x.dtype==torch.float32
    person=sorted(set(people))[index]
    groups=np.char.add(np.char.add(people,'|'),days)
    template=replay.load(Path('/package'),channels.tolist())
    prior,new=(json.loads(Path(f'/prior/{name}_{index}.json').read_text()) for name in ('cap','old'))
    records={}
    for order in ORDERS:
        perm=permutation(groups,None,'full',order)
        for day in sorted(set(days[people==person])):
            name=f'{person}|{day}|{order}'
            rows=perm[(people==person)&(days==day)]
            old=prior[name]
            assert rows.tolist()==old['rows']==new[name]['rows']
            out=stream(template,x,rows,arm,smoke)
            assert out['activity']==old['activity'] and out['inner']==new[name]['labels']
            assert out['labels'][:20]==old['labels'][:20]
            if arm=='stock':
                assert out['labels']==old['labels']
            records[name]=out
            if smoke and arm=='geometry':
                prefix=stream(template,x,rows[:61],arm,True)
                assert prefix['labels']==out['labels'][:61] and prefix['inner']==out['inner'][:61]
                assert [e['edited'] for e in prefix['events']]==[e['edited'] for e in out['events'][:5]]
            if smoke:
                break
        if smoke:
            break
    if smoke and index==5 and arm=='geometry':
        returning(template,channels)
    assert len(records)==(1 if smoke else 4)
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    path=Path(f'{arm}_{index}{"_smoke" if smoke else ""}.json')
    path.write_text(json.dumps(dict(source=sha(Path('source.sha256')),sessions=records))+'\n')
    path.with_suffix('.sha256').write_text(sha(path)+'\n')
    print(person,arm,len(records),'verified sessions, target changes',
          sum(len(e['edited']) for v in records.values() for e in v['events']),flush=True)


if __name__=='__main__':
    main(int(sys.argv[1]),sys.argv[2],sys.argv[3:]==['smoke'])
