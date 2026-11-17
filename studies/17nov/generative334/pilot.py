"""Seven source-held selection episodes, shared refit, all banks/guards before querylabels."""
import json
import os
import time
from pathlib import Path
import numpy as np
from model import fit,transform
from inference import ARMS,banks,density_stream
from guards import check,cpu_feature_readout
from feature_io import load
from data import metadata,labels,official,sha,digest,ORDERS

ETAS=(.25,1.,4.)


def run_order(model,bank,zrows,global_rows,d,perm,eta,arm):
    rank=np.empty(len(perm),int);rank[perm]=np.arange(len(perm));p=np.full((len(zrows),3),np.nan)
    sessions=d['stream'][global_rows[zrows]]
    for session in dict.fromkeys(sessions):
        local=np.flatnonzero(sessions==session);local=local[np.argsort(rank[global_rows[zrows[local]]])]
        p[local],_=density_stream(model,bank[local],eta,arm)
    if not np.isfinite(p).all():raise ValueError('missingstream')
    return p


def cell_nll(y,p,stream,interface):
    cells=np.char.add(np.char.add(stream,'|'),interface);out=[]
    for cell in np.unique(cells):
        mask=cells==cell
        if set(y[mask])!={0,1,2}:raise ValueError('sourcecellmissingclass')
        out.append(np.mean([-np.log(np.clip(p[mask&(y==k),k],1e-12,1)).mean() for k in (0,1,2)]))
    return float(np.mean(out))


def scheme(values):return float(np.mean([np.mean([v for o,v in values.items() if o.startswith(s)]) for s in ('within','full','runs')]))


def main():
    root=Path(os.environ['GEN_ROOT']);started=time.monotonic();smoke=json.loads(Path(os.environ['SMOKE_REPORT']).read_text())
    if not smoke['passed'] or not smoke['cost_pass'] or smoke['query_labels_opened']:raise ValueError('passed no-query smoke required')
    d,allpeople,fitrows,queryrows,perms=metadata();yfit=labels(fitrows,len(allpeople));selection={e:[] for e in ETAS};episodes={}
    for held in [f'P{i:02d}' for i in range(4,11)]:
        bank,record,manifest=load('exclude_'+held);x=bank['features'];rows=bank['rows'];people=bank['people']
        yy=np.full(len(x),-1,dtype=int);positions=np.searchsorted(rows,fitrows);yy[positions]=yfit
        source=bank['fit'];test=positions[people[positions]==held]
        if np.any(yy[source]<0) or held in people[source] or set(people[source])!=set([f'P{i:02d}' for i in range(4,11)])-{held}:raise ValueError('episodeheldleak')
        tick=time.monotonic();model=fit(x[source],yy[source],people[source]);z=transform(model,x[test]);density=banks(model,z,1.)
        scores={}
        for eta in ETAS:
            vals={}
            for order,perm in perms.items():
                p=run_order(model,density['mixture']*eta,test,rows,d,perm,eta,'mixture')
                vals[order]=cell_nll(yy[test],p,d['stream'][rows[test]],d['interface'][rows[test]])
            scores[eta]=scheme(vals);selection[eta].append(scores[eta])
        episodes[held]=dict(fit_people=record['fit_people'],feature_sha256=record['file_sha256'],source_rows_sha256=record['source_rows_sha256'],NLL=scores,seconds=time.monotonic()-tick)
        print('334sourceheld',held,scores,flush=True)
        del x,bank,model,density
    means={eta:float(np.mean(v)) for eta,v in selection.items()};chosen=min(ETAS,key=lambda e:(means[e],e))
    bank,record,manifest=load('source7');x=bank['features'];rows=bank['rows'];source=bank['source'];query=bank['query'];people=bank['people']
    if not np.array_equal(rows[source],fitrows) or not np.array_equal(rows[query],queryrows):raise ValueError('canonical fit/query rows mismatch')
    model=fit(x[source],yfit,people[source]);guard=check(model,x[source[:17]],chosen)
    guard['CPU_producer_fitted_readout']=cpu_feature_readout(model,bank,chosen)
    with (root/'model.npz').open('xb') as out:np.savez(out,**{k:v for k,v in model.items() if k!='people'},people=model['people'],eta=chosen)
    z=transform(model,x[query]);density=banks(model,z,chosen);predictions={}
    for arm in ARMS:
        for order,perm in perms.items():predictions[f'{arm}|{order}']=run_order(model,density[arm],query,rows,d,perm,chosen,arm)
    if set(predictions)!={f'{a}|{o}' for a in ARMS for o in ORDERS}:raise ValueError('incompletearms/orders')
    for order in ORDERS:
        if not np.allclose(predictions[f'anchor_off|{order}'],1/3,atol=1e-10) or not np.allclose(predictions[f'cyclic|{order}'],predictions[f'mixture|{order}'][:,[2,0,1]],atol=1e-9):raise ValueError('querysymmetryguard')
    with (root/'predictions.npz').open('xb') as out:np.savez(out,index=queryrows,**predictions)
    with (root/'before_labels.json').open('x') as out:json.dump(dict(chosen_eta=chosen,source_selection=means,episodes=episodes,guard=guard,
        feature_manifest_sha256=manifest,source_sha256=sha(root/'source.sha256'),model_sha256=sha(root/'model.npz'),
        predictions_sha256=sha(root/'predictions.npz'),query_labels_opened=False),out,indent=2)
    # Complete allsource selections, all6arms×8orders, reset/symmetry and savedbanks BEFORE querylabels.
    metric=official();yquery=labels(queryrows,len(allpeople));results={}
    for arm in ARMS:
        results[arm]={}
        for order in ORDERS:
            p=predictions[f'{arm}|{order}'];per={}
            for person in ('P02','P03'):
                choose=people[query]==person;mask=allpeople==person
                m=metric.score(yquery[choose],p[choose].argmax(1),d,mask)
                m['class_recalls']=[float(np.mean(p[choose].argmax(1)[yquery[choose]==k]==k)) for k in range(3)]
                per[person]=m
            results[arm][order]=per
    summary={a:{p:scheme({o:results[a][o][p]['balanced_accuracy_cell_mean'] for o in ORDERS}) for p in ('P02','P03')} for a in ARMS}
    report=dict(scope='source-only whole finite-template generative decoder;two repeatedlyexposedpeople;notcurrent-system/sealedgain',
        source_sha256=sha(root/'source.sha256'),before_labels_sha256=sha(root/'before_labels.json'),chosen_eta=chosen,source_selection=means,
        episodes=episodes,guard=guard,results=results,scheme_equal=summary,P01_used=False,query_labels_after_complete_banks=True,
        metric_path=metric.__file__,metric_sha256=sha(metric.__file__),seconds=time.monotonic()-started)
    with (root/'result.json').open('x') as out:json.dump(report,out,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
