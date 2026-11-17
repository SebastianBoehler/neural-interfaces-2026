"""No-refit extension: exact source6 frozen body/readouts, exposed P02/P03."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from common346 import source,restored,permutations,bank,extract,sha,digest,ORDERS,labels,official
from features346 import direct

ORIGINAL=Path('/mnt/lustre/work/gehler/gfh527/ni26_direct346_v1_20261117')
ORIGINAL_SHA='cc433f9b3f36dc381debb5b7a803784a67c5b397958e852cfecdbd36dc429dde'


def load_heads():
    proof=json.loads((ORIGINAL/'before_labels.json').read_text())
    if sha(ORIGINAL/'source.sha256')!=ORIGINAL_SHA or proof['source_sha256']!=ORIGINAL_SHA:raise ValueError('original346source changed')
    for line in (ORIGINAL/'source.sha256').read_text().splitlines():
        pin,name=line.split(maxsplit=1)
        if sha(ORIGINAL/name)!=pin:raise ValueError('original346 implementation changed:'+name)
    if sha(ORIGINAL/'heads.npz')!=proof['heads_sha256'] or sha(ORIGINAL/'heads_before_query.json')!=proof['source_proof_sha256']:raise ValueError('frozen readout/proof changed')
    with np.load(ORIGINAL/'heads.npz') as z:arrays={k:z[k].copy() for k in z.files}
    return arrays,proof,json.loads((ORIGINAL/'heads_before_query.json').read_text())


def head_probability(arrays,arm,x):
    z=(x-arrays[arm+'_mean'])/arrays[arm+'_scale']
    return softmax(z@arrays[arm+'_coef'].T+arrays[arm+'_intercept'],axis=1)


def main():
    root=Path(os.environ['EXTEND_ROOT']);start=time.monotonic();torch.set_num_threads(6)
    d,fit,oldheld,people,yfit=source();arrays,proof,sourceproof=load_heads();model,card=restored(d,fit,oldheld)
    if sourceproof['source_rows_sha256']!=digest(fit) or sourceproof['source_people']!=['P04','P05','P06','P07','P08','P10']:raise ValueError('source6 readout exclusion mismatch')
    query=np.flatnonzero(np.isin(people,['P02','P03']))
    if set(people[query])!={'P02','P03'} or np.intersect1d(fit,query).size or np.any(d['y'][query]!=-1):raise ValueError('actual query/body/readout exclusion failed')
    if {str(p).split('/')[-1] for p in model.fit_subjects}!={'P04','P05','P06','P07','P08','P10'}:raise ValueError('native body source6 exclusion failed')
    rows=fit[:24];latent,_=extract(model,rows);extra=direct(d['X'][rows]);head_checks={}
    for arm in ('B','C','D'):
        x=latent if arm=='B' else np.c_[latent,extra];actual=head_probability(arrays,arm,x)
        lr=LogisticRegression(C=.01,max_iter=2000,tol=1e-5,solver='lbfgs')
        lr.coef_=arrays[arm+'_coef'];lr.intercept_=arrays[arm+'_intercept'];lr.classes_=np.arange(3);lr.n_features_in_=x.shape[1]
        expected=lr.predict_proba((x-arrays[arm+'_mean'])/arrays[arm+'_scale'])
        err=float(np.max(np.abs(actual-expected)))
        if err>1e-12 or not np.array_equal(actual.argmax(1),expected.argmax(1)):raise ValueError('actual sklearn frozenhead reconstruction failed')
        for suffix,key in (('mean','scaler_mean_sha256'),('scale','scaler_scale_sha256'),('coef','coefficient_sha256')):
            if digest(arrays[arm+'_'+suffix])!=sourceproof['heads'][arm][key]:raise ValueError('originalhead/scaler identity failed')
        head_checks[arm]=err
    # This source-only real-path smoke precedes all query transforms and labels.
    (root/'source_smoke.json').write_text(json.dumps(dict(passed=True,new_fits=0,
        source_sha256=sha(root/'extension.sha256'),reconstruction=head_checks,
        source_rows_sha256=digest(rows),source_people=sourceproof['source_people'],
        held_transformed=False,held_labels_opened=False,seconds=time.monotonic()-start)))
    source_before=digest(np.concatenate([v.ravel() for k,v in sorted(arrays.items())]));state=__import__('common346').state_sha(model.net)
    perms=permutations(len(people));banks={};guards={};allrows={};zero_errors={}
    for person in ('P02','P03'):
        held=np.flatnonzero(people==person);allrows[person]=held;extra=direct(d['X'][held]);banks[person]={a:{} for a in ('A','B','C','D')};guards[person]={}
        for order,perm in perms.items():
            latent,p,guard=bank(model,d,held,perm);banks[person]['A'][order]=p;guards[person][order]=guard
            # Disabled addition preserves the original head exactly, before any query label.
            zero=p+np.zeros_like(p)
            if not np.array_equal(zero,p):raise ValueError('zero-path identity failed')
            for arm in ('B','C','D'):
                pp=head_probability(arrays,arm,latent if arm=='B' else np.c_[latent,extra])
                if not np.isfinite(pp).all() or not np.allclose(pp.sum(1),1,atol=1e-12):raise ValueError('invalid frozenhead posterior')
                banks[person][arm][order]=pp
        if all(np.array_equal(banks[person]['B'][o],banks[person]['C'][o]) for o in ORDERS):raise ValueError('direct summary inactive')
    # Cross-person session interleaving, stronger than the single-session bank's internal check.
    for person,other in (('P02','P03'),('P03','P02')):
        pp0=bank(model,d,allrows[person],perms['within0'])[1]
        bank(model,d,allrows[other],perms['within0'])
        pp1=bank(model,d,allrows[person],perms['within0'])[1]
        if not np.array_equal(pp0,pp1) or not np.array_equal(pp0,banks[person]['A']['within0']):raise ValueError('cross-person/reset guard failed')
        zero_errors[person]=0.
    if source_before!=digest(np.concatenate([v.ravel() for k,v in sorted(arrays.items())])) or state!=__import__('common346').state_sha(model.net):raise ValueError('frozen source body/head/scaler changed')
    np.savez(root/'predictions.npz',**{person+'_'+arm+'_'+order:p for person,bb in banks.items() for arm,oo in bb.items() for order,p in oo.items()},**{person+'_index':rr for person,rr in allrows.items()})
    before=dict(all_four_arms_both_people_eight_orders=True,new_fits=0,source_sha256=sha(root/'extension.sha256'),
        original_source_sha256=ORIGINAL_SHA,heads_sha256=proof['heads_sha256'],checkpoint_sha256=card['checkpoint_sha256'],
        source_rows_sha256=digest(fit),query_rows_sha256=digest(query),query_counts={p:len(r) for p,r in allrows.items()},
        source_people=sourceproof['source_people'],reconstruction=head_checks,guards=guards,zero_errors=zero_errors,
        predictions_sha256=sha(root/'predictions.npz'),held_labels_opened=False)
    (root/'before_labels.json').write_text(json.dumps(before,indent=2))
    metric=official();scores={}
    def scheme(b):return float(np.mean([np.mean([b[o] for o in ORDERS if o.startswith(s)]) for s in ('within','full','runs')]))
    for person,bb in banks.items():
        held=allrows[person];y=labels(held,len(people));scores[person]={}
        for arm,oo in bb.items():
            by={o:metric.score(y,p.argmax(1),d,people==person)['balanced_accuracy_cell_mean'] for o,p in oo.items()}
            nll={o:float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()) for o,p in oo.items()}
            scores[person][arm]=dict(accuracy=scheme(by),by_order=by,pooled_windowNLL=scheme(nll))
    result=dict(scores=scores,delta_pp={p:{a+'-'+b:100*(v[a]['accuracy']-v[b]['accuracy']) for a,b in [('C','A'),('C','B'),('C','D'),('B','A')]} for p,v in scores.items()},
        new_fits=0,source_sha256=sha(root/'extension.sha256'),before_labels_sha256=sha(root/'before_labels.json'),metric_sha256=sha(metric.__file__),
        seconds=time.monotonic()-start,scope='INDIVIDUAL-MODEL exposedP02/P03, fixedA6source6/readouts selectedafterP09;noindependent/fullsystemclaim')
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
