"""Posthoc exact official-weight error attribution around actual runtime first recognition."""
import json
import sys
from pathlib import Path

import numpy as np

from returning_assets import ROOT,sha

FULL=ROOT/'ni26_js321_20261117_returning_pair_cpu1'
PRIMARY=ROOT/'ni26_js321_20261117_returning_pair_cpu_complete1/conditional_results.json'
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def main():
    p=json.loads(PRIMARY.read_text());primary_sha=sha(PRIMARY)
    if p['status']!='COMPLETE_CONDITIONAL_ENGINEERING_SENSITIVITY_NOT_FULLSYSTEMDEV' or len(p['complete_cells'])!=16:raise ValueError('complete primary required')
    sys.path.insert(0,str(ROOT/'neural-interfaces-2026/2026-competition/tracks/bci_decoding'))
    from benchmark_utils.metrics import balanced_accuracy
    data=ROOT/'neural-interfaces-2026/data/proteus_windows.npz';perm=ROOT/'neural-interfaces-2026/runs/shuffle_perms.npz'
    with np.load(data,allow_pickle=True) as z:y=z['y'].copy();sub=z['subject'].copy();ses=z['session'].astype(str);interface=z['interface'].astype(str)
    contributions=np.zeros((2,8,2));records=[];maxerror=0.;all_three=True
    with np.load(perm) as orders:
        for i in range(16):
            who=('P04','P10')[i//8];order=ORDERS[i%8];folder=FULL/f'cell_{i}';proof=json.loads((folder/'proof.json').read_text())
            if sha(folder/'predictions.npz')!=proof['predictions_sha256']:raise ValueError('complete bank changed')
            with np.load(folder/'predictions.npz') as z:index=z['index'].copy();base=z['base_labels'].copy();js=z['js_labels'].copy()
            delivery=orders[f'{order[:-1]}|{int(order[-1])}|seen_session'];delivery=delivery[sub[delivery]=='Dreyer2026Proteus/'+who]
            if not np.array_equal(np.sort(delivery),index):raise ValueError('actual order/source coverage differs')
            post=np.zeros(len(index),bool);session_records=[]
            for row in proof['records']:
                positions=delivery[ses[delivery]==row['session']];rec=row['activation']['base']['rec']
                if any(a['rec']!=rec for a in row['activation'].values()) or len(positions)!=row['windows']:raise ValueError('actual common recognition record differs')
                selected=positions[np.arange(len(positions))>=rec] if rec>=0 else np.array([],dtype=np.int64)
                post[np.searchsorted(index,selected)]=True
                session_records.append(dict(session=row['session'],first_recognition_zero_based=rec,pre=len(positions)-len(selected),post=len(selected)))
            keys=list(zip(ses[index],interface[index]));unique=sorted(set(keys));scores={};weighted={}
            for name,pred in (('base',base),('js',js)):
                weights=np.zeros(len(index));reference=[]
                for key in unique:
                    select=np.array([v==key for v in keys]);truth=y[index[select]];estimate=pred[select];classes=np.union1d(truth,estimate)
                    all_three=all_three and np.array_equal(classes,np.arange(3))
                    for c in np.unique(truth):
                        rows=select&(y[index]==c);weights[rows]=1./(len(unique)*len(classes)*rows.sum())
                    reference.append(balanced_accuracy(truth,estimate))
                terms=weights*(pred==y[index]);scores[name]=float(terms.sum());weighted[name]=terms
                maxerror=max(maxerror,abs(scores[name]-np.mean(reference)))
            delta=100*(weighted['js']-weighted['base']);parts=np.array([delta[~post].sum(),delta[post].sum()]);contributions[i//8,i%8]=parts
            original=next(r for r in p['official_metrics'] if r['fold']==who and r['order']==order)
            actual=100*(original['metrics']['js']['balanced_accuracy_cell_mean']-original['metrics']['base']['balanced_accuracy_cell_mean'])
            error=abs(parts.sum()-actual);maxerror=max(maxerror,error)
            if error>1e-10:raise ValueError('official total attribution failed')
            records.append(dict(person=who,order=order,pre_recognition_pp=float(parts[0]),post_recognition_pp=float(parts[1]),total_pp=float(parts.sum()),sessions=session_records))
    person=contributions.mean(1);weighted=np.array([.2,.8])@person
    equal=np.array([.2,.8])@np.mean(np.stack([contributions[:,:4].mean(1),contributions[:,4:6].mean(1),contributions[:,6:8].mean(1)],1),1)
    if abs(weighted.sum()-p['returning_weighted_JS_minus_base_pp'])>1e-10 or maxerror>1e-10:raise ValueError('complete primary reconstruction differs')
    result=dict(status='READONLY_POSTHOC_RECOGNITION_ERROR_ATTRIBUTION_COMPLETE',primary_sha256=primary_sha,source_sha256=sha('source.sha256'),
                primary_uniform8_weighted_pre_pp=float(weighted[0]),primary_uniform8_weighted_post_pp=float(weighted[1]),primary_total_pp=float(weighted.sum()),
                person_uniform8_parts={who:dict(pre_pp=float(person[j,0]),post_pp=float(person[j,1])) for j,who in enumerate(('P04','P10'))},
                supplemental_equal_scheme_weighted_parts=dict(pre_pp=float(equal[0]),post_pp=float(equal[1])),
                maximum_reconstruction_error_pp=maxerror,all_cell_arm_union_classes_012=all_three,cases=records,
                scope='Actualruntimefirst-recognitionlatch/positions,NOTtrue-IDgating;officialcell/classweights;NOnewmodel/policy/evaluator;conditionalexposureunchanged',
                limitation='Associationofrecordederrorwithpost-recognition;cannotseparateC30/FBLweight/denominator/historycausesorclaimnewpolicyperformance')
    if sha(PRIMARY)!=primary_sha:raise ValueError('primary changed')
    Path('recognition_attribution.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
