"""Real raw causal256 canary with original frozen TangentCore and original source reference."""
import ast
import copy
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter,hilbert,sosfilt,sosfilt_zi,sosfiltfilt

from returning_assets import ROOT,sha

FIT=ROOT/'ni26_js321_20261117_returning_view_fit1'
ARCHIVE=ROOT/'ni26_js321_20261117_active5/base_v66.zip'


def frozen_class():
    if sha(ARCHIVE)!='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed':raise ValueError('original ZIP changed')
    with zipfile.ZipFile(ARCHIVE) as z:text=z.read('submission.py').decode()
    wanted={'TangentCore','eig_apply_batch','log_floor','filtfilt_setup','filtfilt','softmax'}
    nodes=[n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    if {n.name for n in nodes}!=wanted:raise ValueError('actual frozen helper anchors differ')
    ns=dict(np=np,torch=torch,butter=butter,hilbert=hilbert,sosfilt=sosfilt,sosfilt_zi=sosfilt_zi,sosfiltfilt=sosfiltfilt,SFREQ=120.)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozenZIP/submission.py','exec'),ns)
    return ns['TangentCore']


def main(fold):
    if fold not in ("P04_s23","P10_s23"):raise ValueError("registered fold required")
    fit=FIT if fold=="P04_s23" else ROOT/"ni26_js321_20261117_returning_view_fit_p10_1"
    import sys
    torch.set_num_threads(1);start=time.monotonic();source=ROOT/'ni26_js321_20261117_returning2/source'
    sys.path[:0]=[str(source/'experiments'),str(ROOT/'neural-interfaces-2026/2026-competition/tracks/bci_decoding')]
    import ts_core as T
    meta=json.loads((fit/'fit.json').read_text());asset=fit/f'source_view_{fold}.npz'
    if meta['status']!='original_source_core_PCA256_fit_complete_NOT_raw_parity' or meta['artifact_sha256']!=sha(asset):raise ValueError('complete source fit required')
    with np.load(asset) as z:p={k[3:]:z[k].copy() for k in z.files if k.startswith('ts_')};m=z['tsem_m'].copy();V=np.concatenate((z['tsem_V'],z['route_V_hi']),1).astype(np.float64)
    with np.load(source/'data/proteus_windows.npz',allow_pickle=True) as z:
        rows=np.flatnonzero((z['subject']=='Dreyer2026Proteus/'+fold[:3])&(z['session'].astype(str)=='2'))[:31]
        X=z['X'][rows].copy();channels=z['ch_names'].copy()
    if len(rows)!=31 or X.shape!=(31,41,480) or V.shape!=(44239,256):raise ValueError('real canary shape differs')
    # Query labels are never read. Reference Feats uses the same real prefix and no fitted query statistics.
    fe=T.Feats(dict(X=X,y=np.full(31,-999),stream=np.array(['query']*31),ch_names=channels),T.parse('car_ov17_fs2_t0.1_le_k2'))
    ref_z=(fe.causal(np.arange(31),p['log_glob'],np.zeros(51))-p['mu'])/p['sd'];ref_u=(ref_z-m)@V
    logits=ref_z@p['W'].T+p['b'];logits-=logits.max(1,keepdims=True);ref_p=np.exp(logits);ref_p/=ref_p.sum(1,keepdims=True)
    cls=frozen_class();model=cls(p)
    def feed(mod,raw):
        out=[]
        for x in raw:
            q,z=mod.step(x);out.append(np.r_[q,(z-m)@V])
        return np.asarray(out)
    actual=feed(model,X);reference=np.c_[ref_p,ref_u]
    errors=dict(probability=float(np.abs(actual[:,:3]-reference[:,:3]).max()),view256=float(np.abs(actual[:,3:]-reference[:,3:]).max()))
    report=dict(status='raw_causal256_reference_guard_FAILED',errors=errors,reference_bound=1e-4,source_sha256=sha('source.sha256'),
                source_fit_sha256=sha(asset),fold=fold,query_rows=rows.tolist(),query_labels_read=False,accuracy_computed=False,
                raw_dtype=str(X.dtype),torch_version=torch.__version__,seconds=time.monotonic()-start)
    Path('raw_view_guard.json').write_text(json.dumps(report,indent=2))
    if max(errors.values())>1e-4:raise ValueError(f'original causal reference guard failed {errors}')
    before={k:v.copy() for k,v in p.items()};model.reset_state();reset=feed(model,X)
    fresh=cls(p);prefix=feed(fresh,X[:17]);clone=copy.deepcopy(fresh);a=feed(fresh,X[17:]);b=feed(clone,X[17:])
    same=max(float(np.abs(reset-actual).max()),float(np.abs(prefix-actual[:17]).max()),float(np.abs(a-b).max()),float(np.abs(a-actual[17:]).max()))
    if same>1e-8 or not all(np.array_equal(v,p[k]) for k,v in before.items()):raise ValueError('same-path/reset/prefix/deepcopy/weights guard failed')
    if np.abs(actual[:,3:]).max()<=1e-12 or model.n!=31:raise ValueError('full view or causal state inactive')
    report.update(status='raw_causal256_reference_and_samepath_passed_NOT_accuracy',samepath_maxabs=same,
                  high128_activity=float(np.abs(actual[:,131:]).max()),immutable_weights=True,seconds=time.monotonic()-start)
    Path('raw_view_guard.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    import sys
    main(sys.argv[1])
