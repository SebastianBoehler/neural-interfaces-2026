"""Original source-only tangent core/PCA256 fit; no query labels or accuracy."""
import ast
import hashlib
import io
import json
import resource
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from sklearn.utils.extmath import svd_flip
from threadpoolctl import threadpool_limits

from returning_assets import ROOT,sha


def digest(a):
    return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()


def main(fold):
    if fold not in ("P04_s23","P10_s23"):raise ValueError("registered source fold required")
    start=time.monotonic();source=ROOT/'ni26_js321_20261117_returning2/source'
    sys.path[:0]=[str(source/'experiments'),str(ROOT/'neural-interfaces-2026/2026-competition/tracks/bci_decoding')]
    import ts_core as T
    data=source/'data/proteus_windows.npz'
    if sha(data)!='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327':raise ValueError('data changed')
    with np.load(data,allow_pickle=True) as z:
        sub=z['subject'];session=z['session'].astype(str)
        held=(sub=='Dreyer2026Proteus/'+fold[:3])&(session!='1');rows=np.flatnonzero(~held);query=np.flatnonzero(held)
        d=dict(X=z['X'][rows].copy(),y=z['y'][rows].astype(np.int64),stream=sub[rows]+'|'+session[rows],ch_names=z['ch_names'].copy())
    expected=(2278,377) if fold=='P04_s23' else (2270,385)
    if (len(rows),len(query))!=expected or np.intersect1d(rows,query).size:raise ValueError('source fold differs')
    if d['X'].shape!=(expected[0],41,480):raise ValueError('model tensor changed')
    cfg=T.parse('car_ov17_fs2_t0.1_le_k2');t=time.monotonic();fe=T.Feats(d,cfg);feature_seconds=time.monotonic()-t
    if fe.y is not d['y'] or fe.F.shape!=(expected[0],44239):raise ValueError('source feature/label mapping differs')
    t=time.monotonic();core=T.Core(fe,['lr10'],{'lr10':np.full((len(rows),3),np.nan)})
    with threadpool_limits(limits=6):core.fit(np.arange(len(rows)))
    fit_seconds=time.monotonic()-t;sc=core.models[0].named_steps['standardscaler'];lr=core.models[0].named_steps['logisticregression']
    if not np.array_equal(lr.classes_,np.arange(3)):raise ValueError('class map differs')
    p=dict(bands=np.array(T.BANDSETS[cfg['bands']],dtype=np.float64),log_glob=np.stack(core.priors),k=np.array(2.),
           shrink=np.array(.1),mu=sc.mean_,sd=sc.scale_,W=lr.coef_,b=lr.intercept_,slow_cutoff=np.array(1.),
           slow_step=np.array(60),car=np.array(1),nsub=np.array(2))
    path=source/'scripts/export_tsem.py';tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pc_basis']
    if len(nodes)!=1:raise ValueError('original PCA exporter missing')
    ns=dict(np=np,svd=svd,svd_flip=svd_flip);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    t=time.monotonic();Z=(fe.F-sc.mean_)/sc.scale_
    with threadpool_limits(limits=6):m,V,S=ns['pc_basis'](Z,256)
    pca_seconds=time.monotonic()-t
    archive=ROOT/'ni26_js321_20261117_active5/base_v66.zip'
    if sha(archive)!='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed':raise ValueError('frozen ZIP changed')
    with zipfile.ZipFile(archive) as z,np.load(io.BytesIO(z.read('weights.npz'))) as w:
        low=V[:,:128].astype(w['tsem_V'].dtype);high=V[:,128:].astype(w['route_V_hi'].dtype)
    view=(Z-m)@np.concatenate((low,high),axis=1).astype(np.float64)
    arrays={**{'ts_'+k:v for k,v in p.items()},'tsem_m':m,'tsem_V':low,'route_V_hi':high,
            'idx_tr':rows,'pc_tr':view.astype(np.float32),'y_tr':d['y'],'singular_values':S}
    if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('nonfinite fit')
    artifact=f'source_view_{fold}.npz';np.savez(artifact,**arrays)
    report=dict(status='original_source_core_PCA256_fit_complete_NOT_raw_parity',fold=fold,source_rows=len(rows),held_rows=len(query),
                train_rows_sha256=digest(rows),held_rows_sha256=digest(query),data_sha256=sha(data),source_sha256=sha('source.sha256'),
                ts_core_sha256=sha(T.__file__),PCA_exporter_sha256=sha(path),artifact_sha256=sha(artifact),
                feature_seconds=feature_seconds,fit_seconds=fit_seconds,PCA_seconds=pca_seconds,seconds=time.monotonic()-start,
                maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,feature_threads=1,fit_PCA_threads=6,
                query_raw_used=False,query_labels_used=False,accuracy_computed=False,pc_storage='float32;basis_low_and_high_float32_like_frozen',
                scope='source-only new core/basis; raw causal canary and same-path/reset guards REQUIRED before use')
    Path('fit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main(sys.argv[1])
