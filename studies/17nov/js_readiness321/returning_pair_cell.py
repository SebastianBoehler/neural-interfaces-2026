"""Actual 3-arm, 2-session private conditional bridge cell; predictions only, never query labels."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from returning_pair_guard import ROOT,BUNDLE,models,zero,state
from benchmark_utils.streaming import fresh_model
from returning_pair_checks import Frames,compare
from package_guard import frame,owning_mix
from record import ft_clock_restart
from returning_assets import sha

ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
GUARD=ROOT/'ni26_js321_20261117_returning_pair_guard2/output/pair_guard.json'


def main(index):
    if index not in range(16):raise ValueError('fixed16cell index required')
    start=time.monotonic();fold=('P04_s23','P10_s23')[index//8];order=ORDERS[index%8]
    p=json.loads(GUARD.read_text())
    if p['status']!='actual_private_source_bridge_guards_passed_NOT_accuracy' or p['source_sha256']!=sha(GUARD.parent.parent/'source.sha256') or p['bundle_manifest_sha256']!=sha(BUNDLE/'bundle_complete.json'):
        raise ValueError('successful owning actual source guard required')
    bundle=json.loads((BUNDLE/'bundle_complete.json').read_text())
    for artifact in bundle['artifacts']:
        if artifact['fold']!=fold:continue
        for name,digest in artifact['files'].items():
            if sha(BUNDLE/artifact['path']/name)!=digest:raise ValueError('private source asset changed before fullcell')
    if torch.__version__!='2.14.1+cu130' or not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name() or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('explicit clean SIF/deterministic A100 required')
    if 'neural-interfaces-2026/.venv/' in Path('/proc/self/maps').read_text():raise ValueError('host numerical libraries loaded')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    data=ROOT/'neural-interfaces-2026/data/proteus_windows.npz';perm=ROOT/'neural-interfaces-2026/runs/shuffle_perms.npz'
    if sha(data)!='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327' or sha(perm)!='6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f':raise ValueError('raw/order data changed')
    with np.load(data,allow_pickle=True) as d:X=d['X'].copy();sub=d['subject'].copy();ses=d['session'].astype(str);channels=d['ch_names'].copy();interface=d['interface'].copy()
    who='Dreyer2026Proteus/'+fold[:3];canonical=np.flatnonzero((sub==who)&(ses!='1'));source=np.flatnonzero((sub==who)&(ses=='1'))[:17]
    with np.load(perm) as z:delivery=z[f'{order[:-1]}|{int(order[-1])}|seen_session'];delivery=delivery[sub[delivery]==who]
    if not np.array_equal(np.sort(delivery),canonical) or len(source)!=17 or X.shape[1:]!=(41,480):raise ValueError('actual fold/order/tensor coverage differs')
    if len(set(zip(ses[canonical],interface[canonical])))!=6:raise ValueError('complete six evaluation cells required before predictions')
    bank=ROOT/'ni26_js321_20261117_returning_view_bank1'/fold
    bankproof=json.loads((bank/'proof.json').read_text());file=bank/f'{order}.npz'
    if bankproof['status']!='full_native_raw_causal256_eightorder_bank_passed_NOT_accuracy' or sha(file)!=bankproof['files'][file.name]:raise ValueError('complete actual native raw view proof required')
    with np.load(file) as z:
        if not np.array_equal(z['index'],canonical):raise ValueError('raw bank canonical rows differ')
        views=z['view256'].copy()
    template=models(channels,fold,False,'cuda')
    for m in template.values():
        zero(m)
        for r in source:frame(m,X[r])
        m.reset_state();zero(m)
    values={k:np.full((len(canonical),3),np.nan) for k in template};labels={k:np.full(len(canonical),-1,dtype=np.int64) for k in template};records=[]
    with np.load(BUNDLE/fold/'js/weights.npz') as z:weights={k:z[k].copy(order='K') for k in z.files}
    reference=owning_mix()
    for session in dict.fromkeys(ses[delivery]):
        rows=delivery[ses[delivery]==session];cur={k:fresh_model(m) for k,m in template.items()};before={k:zero(m) for k,m in cur.items()}
        for m in cur.values():ft_clock_restart(m)
        frames=Frames(cur);mix_error=0.;view_error=0.;activity=0.
        for t,r in enumerate(rows):
            position=int(np.searchsorted(canonical,r));actual=frames(cur,X[r]);error,delta,ve=compare(cur,actual,t,weights,reference,views[position])
            mix_error=max(mix_error,error);view_error=max(view_error,ve);activity=max(activity,delta)
            for k,(label,score,trace) in actual.items():
                if not np.isfinite(score).all() or not np.allclose(score.sum(),1.):raise ValueError('invalid actual final posterior')
                values[k][position]=score;labels[k][position]=label
        for k,m in cur.items():
            if state(m)['immutable_sha256']!=before[k]['immutable_sha256']:raise ValueError('source immutable tensors changed')
        records.append(dict(session=str(session),windows=len(rows),owning_mix_maxabs=mix_error,raw_view_maxabs=view_error,JS_activity_max=activity,
                            activation={k:dict(rec=m.log6['rec'],ft_refits=m.ft.refits,ft_steps=m.ft.steps,ft_over=m.ft.over,cp_refits=m.cp.refits,cp_fused=m.cp.fused,cp_stopped=m.cp.stopped) for k,m in cur.items()},
                            off_base_final_bitwise=True,common_inner_gate_FT_bitwise=True,source_immutable=True))
        del cur;torch.cuda.empty_cache()
    if max(r['JS_activity_max'] for r in records)<=1e-12:raise ValueError('actual source JS final output inactive')
    if not all(np.isfinite(v).all() for v in values.values()) or any((v<0).any() for v in labels.values()):raise ValueError('partial actual package predictions')
    if not np.array_equal(values['base'],values['off']) or not np.array_equal(labels['base'],labels['off']):raise ValueError('disabled fullcase identity failed')
    np.savez_compressed('/out/predictions.npz',index=canonical,**{k+'_scores':v for k,v in values.items()},**{k+'_labels':v for k,v in labels.items()})
    proof=dict(status='complete_actual_conditional_pair_cell_NOT_scored',index=index,fold=fold,order=order,source_sha256=sha('source.sha256'),
               guard_sha256=sha(GUARD),bundle_manifest_sha256=sha(BUNDLE/'bundle_complete.json'),bank_sha256=sha(file),
               predictions_sha256=sha('/out/predictions.npz'),records=records,seconds=time.monotonic()-start,
               query_labels_read=False,accuracy_computed=False,executor='cleanSIF_A100_deterministic_validation_matched_independent_RNG',
               scope='CONDITIONAL_ENGINEERING;sharedrest_eye_auxNN/classical_production_priors/oldOOFexposure;NOTgenuineheldFULLSYSTEMDEV')
    Path('/out/proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)


if __name__=='__main__':main(int(sys.argv[1]))
