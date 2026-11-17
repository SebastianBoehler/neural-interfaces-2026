"""Source-held calibration study; exposed P02/P03 inference is prohibited."""
import json
import os
import time
from pathlib import Path
import numpy as np
from model import fit,transform
from feature_io import load
from data import metadata,labels,official,sha,digest,ORDERS
from likelihood import CONFIGS,components,ordered,guards,stream


def scheme(v):return float(np.mean([np.mean([x for o,x in v.items() if o.startswith(s)]) for s in ('within','full','runs')]))


def cells(y,p,streams,interfaces):
    names=np.char.add(np.char.add(streams,'|'),interfaces);nll=[];brier=[]
    for cell in np.unique(names):
        ix=names==cell
        if set(y[ix])!={0,1,2}:raise ValueError('heldcell missingclass')
        nll.append(np.mean([-np.log(np.clip(p[ix&(y==k),k],1e-12,1)).mean() for k in range(3)]))
        brier.append(np.mean([((p[ix&(y==k)]-np.eye(3)[k])**2).sum(-1).mean() for k in range(3)]))
    return float(np.mean(nll)),float(np.mean(brier))


def fold(held,d,people,perms,root):
    tick=time.monotonic();bank,record,manifest=load('exclude_'+held)
    rows=bank['rows'];source=bank['fit'];test=np.flatnonzero(bank['people']==held)
    if held in bank['people'][source] or set(bank['people'][source])!={f'P{i:02d}' for i in range(4,11)}-{held}:raise ValueError('held exclusion')
    if not set(people[rows[test]])=={held} or np.intersect1d(rows[test],rows[source]).size:raise ValueError('row leakage')
    yy=labels(rows[source],len(people));model=fit(bank['features'][source],yy,bank['people'][source])
    z=transform(model,bank['features'][test]);guard=guards(model,z[:17]);pred={};activity={}
    cached={mode:components(model,z,mode) for mode in ('static','semantic_translation','translation','mixture')}
    for name,(mode,eta,rho) in CONFIGS.items():
        a,r,shape=cached[mode];lp=eta*(a+rho*r);byorder={};ablation=[];marginals=[]
        for order,perm in perms.items():
            p,evidence=ordered(lp,rows[test],d,perm);pred[f'{name}::{order}']=p
            if mode=='mixture':
                p0,_=ordered(eta*a,rows[test],d,perm);ablation.append(float(np.abs(p-p0).max()))
                for ev in evidence:
                    posterior=np.exp(ev).reshape(shape);values={}
                    for axis,label in enumerate(('g','b','pi')):
                        v=posterior.sum(axis=tuple(i for i in range(3) if i!=axis))
                        values[label]=dict(l1=float(np.abs(v-1/len(v)).sum()),entropy=float(-(v*np.log(np.maximum(v,1e-300))).sum()))
                    marginals.append(values)
        activity[name]=dict(residual_class_range_max=float(np.ptp(eta*rho*r,axis=-1).max()),
            residual_permutation_range_max=float(np.ptp(eta*rho*r,axis=2).max()),
            semantic_class_range_max=float(np.ptp(eta*a,axis=-1).max()),
            ablation_probability_max=max(ablation) if ablation else None,marginals=marginals)
    with (root/f'{held}_predictions.npz').open('xb') as out:np.savez(out,rows=rows[test],**pred)
    card=dict(held=held,source_people=record['fit_people'],fit_rows_sha256=digest(rows[source]),
        feature_sha256=record['file_sha256'],manifest_sha256=manifest,guards=guard,activity=activity,
        seconds=time.monotonic()-tick,query_inference=False)
    (root/f'{held}_before_labels.json').write_text(json.dumps(card))
    return card


def main():
    root=Path(os.environ['CAL_ROOT']);started=time.monotonic();d,people,fitrows,queryrows,perms=metadata();metric=official()
    smoke=os.environ.get('CAL_SMOKE')=='1'
    if not smoke:
        prior=json.loads(Path(os.environ['CAL_SMOKE_REPORT']).read_text())
        if not prior['cost_pass'] or not prior['no_held_scores']:raise ValueError('smoke required')
    worker=os.environ.get('CAL_HELD');collect=os.environ.get('CAL_COLLECT')=='1';cards={}
    if worker and worker not in [f'P{i:02d}' for i in range(4,11)]:raise ValueError('invalid worker')
    names=[] if collect else ([worker] if worker else (['P04'] if smoke else [f'P{i:02d}' for i in range(4,11)]))
    for held in names:
        cards[held]=fold(held,d,people,perms,root);print(held,'complete banks',cards[held]['seconds'],flush=True)
    if worker:return
    if collect:
        for held in [f'P{i:02d}' for i in range(4,11)]:
            cards[held]=json.loads((root/f'{held}_before_labels.json').read_text())
            if cards[held]['held']!=held or not cards[held]['guards']['passed']:raise ValueError('incomplete fold')
    before=dict(source_sha256=sha(root/'source.sha256'),cards=cards,all_banks_before_scoring=True,query_inference=False)
    (root/'all_before_scores.json').write_text(json.dumps(before))
    if smoke:
        seconds=time.monotonic()-started;projection=2*(seconds+7*cards['P04']['seconds']*1.5+60)
        report=dict(seconds=seconds,projection_cpu_coreseconds=projection,cost_pass=projection<=7200,no_held_scores=True,guards=cards['P04']['guards'])
        (root/'smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return
    results={}
    for held,card in cards.items():
        with np.load(root/f'{held}_predictions.npz') as pz:
            rows=pz['rows'];y=labels(rows,len(people));orders={}
            if not set(people[rows])=={held}:raise ValueError('scoreheld')
            for name in CONFIGS:
                vals={}
                for order in ORDERS:
                    p=pz[f'{name}::{order}'];nll,brier=cells(y,p,d['stream'][rows],d['interface'][rows])
                    accuracy=metric.score(y,p.argmax(-1),d,np.isin(np.arange(len(people)),rows))['balanced_accuracy_cell_mean']
                    vals[order]=dict(NLL=nll,Brier=brier,accuracy=accuracy)
                    if name=='uniform' and (not np.allclose(p,1/3,atol=1e-14) or abs(nll-np.log(3))>1e-12):raise ValueError('uniform NLL')
                orders[name]={k:scheme({o:v[k] for o,v in vals.items()}) for k in ('NLL','Brier','accuracy')}
            results[held]=orders
    mean={name:{k:float(np.mean([v[name][k] for v in results.values()])) for k in ('NLL','Brier','accuracy')} for name in CONFIGS}
    report=dict(source_sha256=before['source_sha256'],before_scores_sha256=sha(root/'all_before_scores.json'),
        results=results,mean=mean,cards=cards,seconds=time.monotonic()-started,query_inference=False,
        metric_path=metric.__file__,metric_sha256=sha(metric.__file__))
    (root/'result.json').write_text(json.dumps(report));print(json.dumps(dict(mean=mean,seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
