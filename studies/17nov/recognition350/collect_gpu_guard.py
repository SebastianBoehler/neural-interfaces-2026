"""Complete-cohort schema adapter; unchanged protocol.score owns every metric."""
import json
import sys
from pathlib import Path

import numpy as np

from returning_assets import ROOT,sha
from returning_pair_metadata import held_metadata
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')

CELLS=ROOT/'ni26_recognition350_20261117_whole2'
OLD=ROOT/'ni26_js321_20261117_returning_pair_cpu1'
GUARD=ROOT/'ni26_recognition350_20261117_gpu_guard2/output/guard.json'
PRIMARY=ROOT/'ni26_js321_20261117_returning_pair_cpu_complete1/conditional_results.json'


def preflight():
    if sha(PRIMARY)!='d0532701c0b88a4acd8fd8bd3d6b7b98469ef1d6bae0758b34ce690d50313f04':raise ValueError('existing complete paired control changed')
    guard=json.loads(GUARD.read_text())
    if guard['status']!='POLICY350_GPU_SOURCE_FUNCTIONAL_PASS_NOT_ACCURACY':raise ValueError('owning actual guard absent')
    with np.load(ROOT/'neural-interfaces-2026/data/proteus_windows.npz',allow_pickle=True) as z:
        fields={k:z[k].copy() for k in ('subject','session','interface','split')}
    metadata=held_metadata(fields);arrays=[];proofs=[]
    for i in range(16):
        folder=CELLS/f'cell_{i}';proof=json.loads((folder/'proof.json').read_text());old=json.loads((OLD/f'cell_{i}/proof.json').read_text());fold=('P04_s23','P10_s23')[i//8]
        if (proof['status'],proof['index'],proof['fold'],proof['order'])!=('POLICY350_COMPLETE_WHOLE_CELL_NOT_SCORED',i,fold,ORDERS[i%8]):raise ValueError('incomplete source cohort')
        if proof['executor']!='cleanSIF_CPU_deterministic' or proof['guard_executor']!='cleanSIF_A100_deterministic' or proof['CPU_unknown215_proven']:raise ValueError('explicit stage executor/source scope differs')
        if proof['source_sha256']!=sha(CELLS/'source.sha256') or proof['owning_guard_sha256']!=sha(GUARD) or proof['query_labels_loaded'] or proof['predictions_sha256']!=sha(folder/'predictions.npz') or old['predictions_sha256']!=sha(OLD/f'cell_{i}/predictions.npz'):raise ValueError('whole source/prediction lock differs')
        if len(proof['records'])!=2 or any(r['recipe_maxabs']>1e-8 or r['rawview_maxabs']>1e-8 or not r['reset_copy_prefix_exact'] for r in proof['records']):raise ValueError('whole causal semantics failed')
        with np.load(folder/'predictions.npz') as z:a={k:z[k].copy() for k in z.files}
        with np.load(OLD/f'cell_{i}/predictions.npz') as z:
            for arm in ('base','js'):a[arm+'_labels']=z[arm+'_labels'].copy()
            if not np.array_equal(a['index'],z['index']) or not np.array_equal(z['base_labels'],z['off_labels']):raise ValueError('same complete control rows/zero differ')
        if not np.array_equal(a['index'],metadata[fold]['rows']) or proof['official_cell_keys']!=metadata[fold]['cell_keys']:raise ValueError('owning raw whole-fold metadata differs')
        for key in ('policy_labels','base_labels','js_labels'):
            if a[key].shape!=a['index'].shape or not np.isin(a[key],[0,1,2]).all():raise ValueError('whole class schema differs')
        if not np.isfinite(a['policy_scores']).all():raise ValueError('nonfinite complete scores')
        arrays.append(a);proofs.append(dict(index=i,fold=fold,order=ORDERS[i%8],proof_sha256=sha(folder/'proof.json'),predictions_sha256=sha(folder/'predictions.npz')))
    return arrays,proofs


def main(mode):
    arrays,proofs=preflight()
    if mode=='preflight':print('ALL16_POLICY_SOURCE_GUARDS_BEFORE_GOLD_OK',flush=True);return
    if mode!='collect' or not any(v.strip()=='OK' for v in Path('replay_check.txt').read_text().splitlines()):raise ValueError('explicit owning replay OK required')
    sys.path[:0]=[str(ROOT/'neural-interfaces-2026/experiments'),str(ROOT/'neural-interfaces-2026/2026-competition/tracks/bci_decoding')]
    import protocol
    d=protocol.load();scores=np.empty((2,8,3))
    for i,a in enumerate(arrays):
        person=('P04','P10')[i//8];mask=(d['subject']=='Dreyer2026Proteus/'+person)&(d['session']!='1');index=np.flatnonzero(mask)
        if not np.array_equal(index,a['index']):raise ValueError('official whole rows differ')
        for k,arm in enumerate(('base','js','policy')):scores[i//8,i%8,k]=protocol.score(d['y'][index],a[arm+'_labels'],d,mask)['balanced_accuracy_cell_mean']
    reports={};weights=np.array([.2,.8])
    for name,k in [('policy_minus_base',0),('policy_minus_JSall',1)]:
        delta=100*(scores[:,:,2]-scores[:,:,k]);rng=np.random.default_rng(3501701);boot=[]
        for _ in range(2000):
            people=rng.integers(0,2,2);value=np.array([delta[p,rng.integers(0,8,8)].mean() for p in people]);w=weights[people];boot.append(float(w@value/w.sum()))
        scheme=np.stack([delta[:,:4].mean(1),delta[:,4:6].mean(1),delta[:,6:8].mean(1)],1).mean(1)
        reports[name]=dict(primary_uniform8_weighted_pp=float(weights@delta.mean(1)),primary_95percent_descriptive_people_order=np.percentile(boot,[2.5,97.5]).tolist(),person_pp={p:float(delta[j].mean()) for j,p in enumerate(('P04','P10'))},equal_person_pp=float(delta.mean()),supplemental_equal_scheme_weighted_pp=float(weights@scheme),order_pp=dict(zip(ORDERS,(weights@delta).tolist())))
    Path('policy_results.json').write_text(json.dumps(dict(status='COMPLETE_POLICY350_CONDITIONAL_ENGINEERING_NOT_FULLSYSTEMDEV',contrasts=reports,cells=proofs,source_sha256=sha('source.sha256'),guard_sha256=sha(GUARD),metric_owner_sha256=sha(ROOT/'neural-interfaces-2026/experiments/protocol.py'),scope='Fixed direct-held-trained production classical/rest/eye+globalteacherG/oldOOF retained;notindependentgeneralization/rank/sealed/promotion',accuracy_computed=True,upload_or_promotion=False),indent=2))


if __name__=='__main__':main(sys.argv[1])
