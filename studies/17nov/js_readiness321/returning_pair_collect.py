"""Owning complete16cell conditional sensitivity collector; unchanged official protocol metric."""
import json
import sys
from pathlib import Path

import numpy as np

from returning_assets import ROOT,sha

CELLS=ROOT/'ni26_js321_20261117_returning_pair_full1'
GUARD=ROOT/'ni26_js321_20261117_returning_pair_guard2/output/pair_guard.json'
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def main():
    guard=json.loads(GUARD.read_text())
    if guard['status']!='actual_private_source_bridge_guards_passed_NOT_accuracy':raise ValueError('owning source guard required before any labels')
    with np.load(ROOT/'neural-interfaces-2026/data/proteus_windows.npz',allow_pickle=True) as z:
        subjects=z['subject'].copy();sessions=z['session'].astype(str);interfaces=z['interface'].copy()
    expected={p:np.flatnonzero((subjects=='Dreyer2026Proteus/'+p)&(sessions!='1')) for p in ('P04','P10')}
    predictions=[];records=[]
    for i in range(16):
        folder=CELLS/f'cell_{i}';p=json.loads((folder/'proof.json').read_text());file=folder/'predictions.npz'
        fold=('P04_s23','P10_s23')[i//8];order=ORDERS[i%8]
        if (p['status'],p['index'],p['fold'],p['order'])!=('complete_actual_conditional_pair_cell_NOT_scored',i,fold,order):raise ValueError('incomplete fixed16cell recipe')
        if p['accuracy_computed'] or p['query_labels_read'] or p['guard_sha256']!=sha(GUARD) or p['bundle_manifest_sha256']!=guard['bundle_manifest_sha256'] or p['source_sha256']!=sha(CELLS/'source.sha256') or p['predictions_sha256']!=sha(file):raise ValueError('source/predictions/guard changed')
        if len(p['records'])!=2 or any(not r['off_base_final_bitwise'] or not r['common_inner_gate_FT_bitwise'] or not r['source_immutable'] or r['owning_mix_maxabs']>1e-8 or r['raw_view_maxabs']>1e-8 or any(v['ft_over'] for v in r['activation'].values()) for r in p['records']):raise ValueError('whole actual frame/session guard failed')
        with np.load(file) as z:a={k:z[k].copy() for k in z.files}
        idx=expected[fold[:3]]
        if not np.array_equal(a['index'],idx) or len(idx)!=(377 if i//8==0 else 385) or len(set(zip(sessions[idx],interfaces[idx])))!=6:raise ValueError('complete raw metadata/canonical/six-cell coverage failed BEFORElabels')
        for arm in ('base','off','js'):
            if a[arm+'_labels'].shape!=(len(idx),) or a[arm+'_scores'].shape!=(len(idx),3) or not np.isin(a[arm+'_labels'],[0,1,2]).all():raise ValueError('complete prediction coverage/classschema failed BEFORElabels')
        if not np.array_equal(a['base_scores'],a['off_scores']) or not np.array_equal(a['base_labels'],a['off_labels']):raise ValueError('complete zerochange control failed')
        if not all(np.isfinite(v).all() for v in a.values()):raise ValueError('nonfinite complete predictions')
        predictions.append(a);records.append(dict(index=i,fold=fold,order=order,predictions_sha256=sha(file),proof_sha256=sha(folder/'proof.json')))
    # Every source/frame/wholecase guard above completes before this first label load/metric call.
    replay=Path('replay_check.txt').read_text()
    if not any(line.strip()=='OK' for line in replay.splitlines()):raise ValueError('owning v3 replay/check.py did not print OK')
    sys.path[:0]=[str(ROOT/'neural-interfaces-2026/experiments'),str(ROOT/'neural-interfaces-2026/2026-competition/tracks/bci_decoding')]
    import protocol
    d=protocol.load();score=np.empty((2,8,3));details=[]
    for i,a in enumerate(predictions):
        person=i//8;who='Dreyer2026Proteus/'+('P04','P10')[person];mask=(d['subject']==who)&(d['session']!='1');idx=np.flatnonzero(mask)
        if not np.array_equal(a['index'],idx) or len(idx)!=(377 if person==0 else 385):raise ValueError('full canonical held source rows differ')
        parts={}
        for k,arm in enumerate(('base','off','js')):
            m=protocol.score(d['y'][idx],a[arm+'_labels'],d,mask)
            if m['n_cells']!=6:raise ValueError('official six-cell coverage differs')
            score[person,i%8,k]=m['balanced_accuracy_cell_mean'];parts[arm]=m
        details.append(dict(fold=('P04','P10')[person],order=ORDERS[i%8],metrics=parts))
    delta=100*(score[:,:,2]-score[:,:,0]);weights=np.array([.2,.8]);point=float(weights@delta.mean(1));rng=np.random.default_rng(3211701);boot=[]
    for _ in range(2000):
        people=rng.integers(0,2,2);values=np.array([delta[p,rng.integers(0,8,8)].mean() for p in people]);w=weights[people];boot.append(float(w@values/w.sum()))
    report=dict(status='COMPLETE_CONDITIONAL_ENGINEERING_SENSITIVITY_NOT_FULLSYSTEMDEV',
                person_metrics_percent={p:{a:float(100*score[j,:,k].mean()) for k,a in enumerate(('base','off','js'))} for j,p in enumerate(('P04','P10'))},
                returning_weighted_JS_minus_base_pp=point,people_order_descriptive_interval_pp=np.percentile(boot,[2.5,97.5]).tolist(),
                equal_person_JS_minus_base_pp=float(delta.mean()),person_JS_minus_base_pp={p:float(delta[j].mean()) for j,p in enumerate(('P04','P10'))},
                order_returning_weighted_delta_pp={o:float(weights@delta[:,i]) for i,o in enumerate(ORDERS)},off_minus_base_exact_zero=bool(np.array_equal(score[:,:,0],score[:,:,1])),
                complete_cells=records,official_metrics=details,guard_sha256=sha(GUARD),source_sha256=sha('source.sha256'),replay_check_sha256=sha('replay_check.txt'),
                metric_owner='unchanged experiments/protocol.py:score official cell balanced_accuracy',metric_source_sha256=sha(ROOT/'neural-interfaces-2026/experiments/protocol.py'),
                scope='ONLYpairedfixed-production-priorengineeringconditional;primary+innerNNsourcefit;rest4_eyeauxNN/classical/teacherG/oldOOFexposureRETAINED',
                limitations='ONLYtwopeople/selecteddev;positiveCANNOTestablishindependentgeneralization/rank/sealedprojection/promotion;deterministicvalidationexecutorNOTstandardtiming',
                accuracy_computed=True,upload_or_promotion=False)
    Path('conditional_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
