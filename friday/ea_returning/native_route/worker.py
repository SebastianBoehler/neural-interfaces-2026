"""Run the actual pinned RouteB stage with a shared fold view and explicit split reference."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from assets import ANCHORS, BRIDGE, PREFIXES, TAPE, load_rows, meta, package, sha, source_fit, visit_case


TOL=1e-8


def numeric(a,b):
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.abs(a-b).max())


def source_view(module,fit,case,other):
    model=module.TangentCore(fit['core'])
    def feed(x):
        p,u=[],[]
        for row in x:
            prob,z=model.step(row); delta=z-fit['m']
            p.append(prob); u.append(np.r_[delta@fit['low'],delta@fit['high']])
        return np.asarray(p),np.asarray(u)
    p,u=feed(case['raw'])
    assert model.n==len(p)
    idx=np.searchsorted(fit['held'],case['rows'])
    assert np.array_equal(fit['held'][idx],case['rows'])
    ref=fit['reference']; error=(numeric(p,ref['proba'][idx]),numeric(u,ref['view256'][idx]))
    assert max(error)<=TOL and np.array_equal(p.argmax(1),ref['proba'][idx].argmax(1)),error
    model.reset_state(); py,uy=feed(other['raw'][:31]); model.reset_state()
    px,ux=feed(case['raw'][:31])
    assert np.array_equal(px,p[:31]) and np.array_equal(ux,u[:31])
    take=np.searchsorted(fit['held'],other['rows'][:31]); assert np.array_equal(fit['held'][take],other['rows'][:31])
    assert numeric(py,ref['proba'][take])<=TOL and numeric(uy,ref['view256'][take])<=TOL
    return u,uy,dict(core_max=error[0],u_max=error[1],exact_prefix_reset=True)


def gate(case,names,at,kind):
    who=int(case[kind+'_j'][at]); ramp=float(case[kind+'_r'][at]); score=float(case[kind+'_score'][at])
    if who<0:
        assert ramp==0.
        return None
    named=str(case['names'][who]).split('/')[-1]
    assert named in names and np.isfinite(score) and 0<=ramp<=1
    return names.index(named),score,ramp


def routed(module,z,fit,mode,case,other,u,ou):
    settings={**z,**fit['params']}
    names=settings['route_names'].tolist(); own=names.index(str(case['owner']))
    if mode=='split':
        assert case['owner']==other['owner']
        settings['route_ps1_W']=settings['route_ps1_W'].copy(); settings['route_ps1_W'][own]=case['source_head_w']
        settings['route_ps1_c']=settings['route_ps1_c'].copy(); settings['route_ps1_c'][own]=case['source_head_c']
        settings['v6_tpri']=settings['v6_tpri'].copy(); settings['v6_tpri'][own]=case['personal_prior'][case['winner_r']>0][0]
        u,ou=case['pc256'],other['pc256'][:31]
    actor=module.RouteB(settings,1/16)
    actor.pri=settings['v6_tpri']; actor.kappa=float(settings['v6_tpri_kappa'])
    assert actor.n_tsem==128 and actor.g==3. and actor.own.beta==.10
    em,ts=actor.tail.em,actor.tail.tsem
    original_tail,original_ts,original_own=actor.tail.step,ts.step,actor.own.apply

    def feed(data,views,n=None):
        nonlocal at
        actor.reset_state(); n=len(views) if n is None else n
        out=[]; calls={}; latch=False
        em.features=lambda x: np.r_[data['ge'][at],em.extra]
        def tail(x,pc,mix,*extra):
            r=float(data['winner_r'][at]); wanted=data['mix'][at]
            g=ts.prior
            assert (g is None)==(r==0.)
            if r:
                assert g[0]==20*r and np.array_equal(g[1],settings['v6_tpri'][own])
                f=views[:at+1]; v=(f[-1]-f.mean(0))/np.maximum(f.std(0),1e-3)
                h=module.softmax(v[None]@settings['route_ps1_W'][own].T+settings['route_ps1_c'][own])[0]
                w=r/16; wanted=(wanted-w*data['core'][at]+3*w*h)/(1-w+3*w)
            assert numeric(mix,wanted)<=TOL and np.array_equal(pc,views[at,:128])
            if mode=='split':
                assert numeric(mix,data['mixr'][at])<=TOL
            result=original_tail(x,pc,mix,*extra)
            calls['preown']=result.copy()
            return result
        def tsem(pc,p,gate_value):
            assert np.array_equal(pc,views[at,:128]); calls['em']=p.copy()
            view=data['tsem_pc'][at] if mode=='split' else pc
            return original_ts(view,p,gate_value)
        def own_stage(p,j,r):
            assert j==own and r==data['winner_r'][at] and r>0
            result=original_own(p,j,r)
            assert result is not None
            calls['own']=numeric(result[1],data['own_logs'][at]); assert calls['own']<=TOL
            return result
        actor.tail.step,ts.step,actor.own.apply=tail,tsem,own_stage
        tape=dict(preown=[],staged=[],em=[]); errors=[]
        for at in range(n):
            calls.clear()
            gc=gate(data,names,at,'contrast'); gi=gate(data,names,at,'bio')
            a=max((g[2] for g in (gc,gi) if g is not None),default=0.)
            r=float(data['winner_r'][at]); assert a==r
            latch=latch or r>0; assert latch==bool(data['recognised'][at])
            q=actor.stepx(data['raw'][at],views[at],data['mix'][at],data['core'][at],data['eyes'][at],gc,gi)
            assert actor.last['contrast']==gc and actor.last['identity']==gi
            assert ('own' in calls)==(r>0.) and np.isfinite(q).all()
            assert actor.log['err']==actor.log['degen']==0
            if not r:
                assert np.array_equal(q,calls['preown'])
            errors.append(calls.get('own',0.))
            tape['staged'].append(q.copy()); tape['preown'].append(calls['preown']); tape['em'].append(calls['em'])
        assert actor.log['routed']==actor.log['staged']==int(np.count_nonzero(data['winner_r'][:n]>0))
        return {k:np.asarray(v) for k,v in tape.items()},max(errors)
    at=0
    full,own_error=feed(case,u)
    feed(other,ou)
    prefix,_=feed(case,u,31)
    for k,v in prefix.items():
        assert np.array_equal(v,full[k][:31]),k
    return full,own_error


def main(visit):
    assert visit in ANCHORS and torch.__version__=='2.14.1+cu130' and not torch.cuda.is_initialized()
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    torch.set_num_threads(1)
    who,day=visit.split('|'); other=who+'|'+('3' if day=='2' else '2')
    module,z=package(); x,p,d,order,eyes=load_rows(); fit=source_fit(who,p,d)
    before={k:np.array(v,copy=True) for k,v in fit['params'].items()}
    report={}; output={}
    with np.load(TAPE/'matched.npz',allow_pickle=False) as t,np.load(BRIDGE/'matched.npz',allow_pickle=False) as old:
        for pred in PREFIXES:
            c=visit_case(t,old,x,p,d,order,eyes,visit,pred); o=visit_case(t,old,x,p,d,order,eyes,other,pred)
            if pred==PREFIXES[0]:
                u,ou,view=source_view(module,fit,c,o)
            cases={mode:routed(module,z,fit,mode,c,o,u,ou) for mode in ('split','shared')}
            a,b=[cases[mode][0] for mode in ('split','shared')]
            parity={key:numeric(a[key],c[key]) for key in ('em','preown','staged')}
            n=int(np.count_nonzero(a['staged'].argmax(1)!=c['staged'].argmax(1)))
            mask=c['recognised']; activity=numeric(a['staged'][mask],b['staged'][mask])
            report[pred.rstrip('|')]=dict(legacy_parity=parity,legacy_decisions_changed=n,post_recognition_delta=activity,
                   own_log_error=max(v[1] for v in cases.values()), latched=int(mask.sum()), routed=int((c['winner_r']>0).sum()),
                   gap_ages=np.flatnonzero(mask & (c['winner_r']==0)).tolist(),exact_prefix_reset=True,
                   passed=max(parity.values())<=TOL and n==0 and activity>1e-4)
            output.update({pred+mode+'|'+k:v for mode,w in cases.items() for k,v in w[0].items()})
    assert all(np.array_equal(v,fit['params'][k]) for k,v in before.items())
    with Path('route.json').open('x') as f:
        j=dict(visit=visit,fold=fit['fold'],fit_sha=fit['fit_sha'],native_view_sha=fit['view_sha'],
               context_sha=fit['context_sha'],native_view=view,branches=report,source_sha=sha('../source.sha256'),
               no_eval_labels=True,scope='actual pinned RouteB with authenticated input tapes',
               warm_scope='unresolved omitted ages 20–27' if visit=='P10|2' else 'recognised before first refit')
        json.dump(j,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    print(json.dumps(j,sort_keys=True),flush=True)
    if not all(v['passed'] for v in report.values()):
        raise ValueError('registered legacy parity or shared-view activity failed')
    with Path('route.npz').open('xb') as f:
        np.savez_compressed(f,rows=c['rows'],**output)
    print('D-EArU passed',visit,sha('route.npz'),flush=True)


if __name__=='__main__':
    main(sys.argv[1])
