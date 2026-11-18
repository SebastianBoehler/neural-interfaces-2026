"""Authenticate returning full0 and export matched label-blind route inputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from bridge_refit import session_refits
from returning_owner import HERE,LEGACY_AGES,ORDER,PROTO,ROOT,SESSIONS,construct,digest,inputs,sha,without_test_labels
from returning_stages import fresh,permutation_reset,prepare,run


def read_reference():
    key='full0|B2_pool|'
    with np.load(ROOT/'runs/aux431/full0.npz',allow_pickle=False) as z:
        three,historical,index=(z[k] for k in ('three','historical','rows'))
        age=json.loads(str(z['route']))
    with np.load(ROOT/'runs/chainout/ret_full0.npz',allow_pickle=False) as z:
        base,er=(z[key+k] for k in ('base','ER'))
    with np.load(ROOT/'runs/chainout/s245_ret.npz',allow_pickle=False) as z:
        t=z[key+'T']
    assert age==LEGACY_AGES and np.array_equal(t,historical)
    return index,{'base':base,'ER':er,'T':t},three


def equal(a,b):
    assert a[2]==b[2]
    for group,other in zip(a[:2],b[:2]):
        assert group.keys()==other.keys()
        for k in group:
            assert np.array_equal(group[k],other[k]),k


def refit_trace(ctx,layout,idx,output):
    _,tape,routing=output
    records={}
    bank={}
    for name in sorted(SESSIONS):
        rows=layout[tape['session'][layout]==name]
        age=routing[name]['age']
        tgrid=np.arange(20,min(251,len(rows),age if age is not None else len(rows)),10)
        topics=('GE','pc128','rebal','gate','em','shadow')
        local=[tape[k][rows] for k in topics]
        post,errors=session_refits(ctx.gm,*local,tgrid)
        for k,t in enumerate(tgrid):
            assert np.array_equal(np.isfinite(post[:,k]).all(1),np.arange(len(rows))<t)
            prefix=[a[:t+1] for a in local]
            rerun,_=session_refits(ctx.gm,*prefix,tgrid[:k+1])
            assert np.array_equal(rerun[:-1,k],post[:t,k])
        valid=post[np.isfinite(post).all(2)]
        assert np.allclose(valid.sum(1),1,atol=1e-12,rtol=0)
        short=name.rsplit('/',1)[-1].replace('|','-')
        records[short]=dict(ages=tgrid.tolist(),past_rows=len(valid),
                    max_shadow_reconstruction=max((max(a,b) for a,b,_ in errors),default=0.))
        bank[f'{short}|rows']=idx[rows]
        bank[f'{short}|ages']=tgrid
        bank[f'{short}|refits']=post
    return records,bank


def seal(mode,idx,layout,records,payload,opened):
    path=Path(f'{mode}.npz')
    with path.open('xb') as f:
        np.savez_compressed(f,index=idx,delivery=idx[layout],**payload)
    rec=dict(mode=mode,label_free_targets=True,scored=False,status='passed',
          source=sha(HERE/'bridge_source.sha256'),bank=sha(path),global_rows=digest(idx),
          delivery=digest(idx[layout]),stages=records,read_files={str(p):h for p,h in opened.items()})
    with Path(f'{mode}.json').open('x') as f:
        json.dump(rec,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')
    print(mode,'verified',records,flush=True)


def zero(ctx,prep,idx,stream,refs):
    _,perm,_,count9,count3=prep
    reverse=permutation_reset(perm,idx,stream)
    old=run(ctx,prep,perm,count9,None,stream)
    for name,ref in refs[1].items():
        actual=old[0][name].astype(np.float32)
        assert np.array_equal(actual,ref),(name,float(np.abs(actual-ref).max()))
    deployment=run(ctx,prep,perm,count3,None,stream)
    assert np.array_equal(deployment[0]['T'].astype(np.float32),refs[2])
    assert {s:v['age'] for s,v in deployment[2].items()}==LEGACY_AGES
    assert all(s.endswith('/'+v['owner']+s[-2:]) for s,v in deployment[2].items())
    slots={k:deployment[1][f'slot{k}'].copy() for k in (1,3)}
    equal(deployment,run(ctx,prep,perm,count3,slots,stream))
    equal(deployment,run(ctx,prep,reverse,count3,slots,stream))
    r,bank=refit_trace(ctx,perm[PROTO],idx,deployment)
    assert r['P10-2']['ages']==[20] and sum(v['past_rows'] for v in r.values())==20
    payload={**deployment[0],**deployment[1],**bank}
    record=dict(route=deployment[2],refits=r,exact_archived=True,archived_slot_copy=True,
                reverse_session_exact=True)
    return record,payload


def matched(ctx,prep,idx,stream,refs):
    with Path('zero.json').open() as f:
        parent=json.load(f)
    assert parent['status']=='passed' and parent['source']==sha(HERE/'bridge_source.sha256')
    assert parent['bank']==sha(Path('zero.npz'))
    _,perm,_,_,count3=prep
    assert parent['delivery']==digest(idx[perm[PROTO]])
    reverse=permutation_reset(perm,idx,stream)
    ms=fresh(idx,idx[perm[PROTO]])
    paths={}
    records={}
    for n,values in zip(('fresh_recording','fresh_causal'),ms):
        actor=run(ctx,prep,perm,count3,values,stream)
        equal(actor,run(ctx,prep,reverse,count3,values,stream))
        s,track=refit_trace(ctx,perm[PROTO],idx,actor)
        records[n]=dict(route=actor[2],refits=s,reverse_session_exact=True,
                        global_rows=digest(idx[perm[PROTO]]))
        paths[n]=(actor,track)
    old,new=(paths[k][0][1] for k in ('fresh_recording','fresh_causal'))
    delta=lambda k:float(np.abs(new[k]-old[k]).max())
    for k in ('mix','shadow'):
        assert delta(k)>1e-5
    expected=(.5*(new['slot1']-old['slot1'])+new['slot3']-old['slot3'])/15.
    np.testing.assert_allclose(new['mix']-old['mix'],expected,atol=1e-14,rtol=0)
    r0=paths['fresh_recording'][0][2]
    r1=paths['fresh_causal'][0][2]
    records['delta']={k:delta(k) for k in ('slot1','slot3','mix','rebal','em','shadow')}
    records['delta']['routing_ages_identical']=r0==r1
    payload={}
    active_deltas=[]
    changed=0
    for n,((outputs,tracks,_),trace) in paths.items():
        payload.update({f'{n}|{k}':v for k,v in {**outputs,**tracks,**trace}.items()})
        if n=='fresh_causal':
            for k in trace:
                if k.endswith('|refits'):
                    first=paths['fresh_recording'][1][k]
                    if first.shape==trace[k].shape and np.array_equal(np.isfinite(first),np.isfinite(trace[k])):
                        valid=np.isfinite(first).all(2)
                        if valid.any():
                            active_deltas.extend(np.abs(first[valid]-trace[k][valid]).max(1).tolist())
                            changed+=int(np.count_nonzero(first[valid].argmax(1)!=trace[k][valid].argmax(1)))
    records['delta']['refit_targets_changed']=changed
    records['delta']['max_matched_refit_abs']=max(active_deltas,default=0.)
    return records,payload


def main(mode):
    people,days,stream,idx,splits=inputs()
    refs=read_reference()
    assert np.array_equal(refs[0],idx)
    with without_test_labels(splits) as opened:
        ctx=construct(people,days,splits)
        prep=prepare(ctx,idx,stream)
    # Producer cannot access the evaluator archives and prior stage outputs.
    with without_test_labels(splits) as live:
        # The matched continuation reads only its own sealed zero JSON.
        record,payload=(zero if mode=='zero' else matched)(ctx,prep,idx,stream,refs)
    seal(mode,idx,prep[1][PROTO],record,payload,opened|live)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['zero','matched'])
    main(p.parse_args().mode)
