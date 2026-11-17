"""Source galleries in new source PCA coordinates; frozen old-OOF calibration remains conditional."""
import ast
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from returning_assets import ROOT,sha


def functions(path,names,namespace):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('original source helper anchors differ')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def main():
    start=time.monotonic();main=ROOT/'neural-interfaces-2026';ns={'np':np,'CLIP':4.}
    functions(main/'replay/rlib2.py',('rstats','rz','contrast','onehot','basis','perp','gal_norm'),ns)
    functions(main/'replay/rlib.py',('weights',),ns)
    bio={'np':np};functions(main/'experiments/bio_gate.py',('Bio','fold_params'),bio)
    archive=ROOT/'ni26_js321_20261117_active5/base_v66.zip'
    if sha(archive)!='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed':raise ValueError('original ZIP changed')
    with zipfile.ZipFile(archive) as a,np.load(io.BytesIO(a.read('weights.npz'))) as w:
        names=[str(v) for v in w['route_names']];metric={k:w['bio_'+k].copy() for k in ('sd','V','B','W','S')};constants={k:w['bio_'+k].item() for k in ('tau','ramp','t0')}
    cache=main/'runs/replay_data/post_cache.npz'
    with np.load(cache) as z:oof=z['B2_pool|tsem|all_loso'].copy()
    with np.load(main/'data/proteus_windows.npz',allow_pickle=True) as z:X=z['X'].copy();sub=z['subject'].copy();ses=z['session'].astype(str)
    stream=sub+'|'+ses;person=np.array([s.split('/')[-1] for s in sub]);short=person+'|'+ses
    records=[]
    for fold,folder in (('P04_s23','returning_view_fit1'),('P10_s23','returning_view_fit_p10_1')):
        file=ROOT/f'ni26_js321_20261117_{folder}'/f'source_view_{fold}.npz'
        with np.load(file) as z:train=z['idx_tr'].copy();pc=z['pc_tr'].astype(np.float64);y=z['y_tr'].copy()
        held=np.flatnonzero((person==fold[:3])&(ses!='1'))
        if np.intersect1d(train,held).size or len(train)!=(2278 if fold[:3]=='P04' else 2270):raise ValueError('source gallery rows differ')
        per={};bias={}
        for s in dict.fromkeys(stream[train]):
            select=stream[train]==s;q=s.split('/')[1].split('|')[0];a,b=pc[select],y[select];po=oof[train[select]]
            per.setdefault(q,[]).append((a,b));bias.setdefault(q,[]).append(ns['contrast'](a,ns['weights'](po,len(b),'q70',b),256)-ns['contrast'](a,ns['onehot'](b),256))
        if set(per)!=set(names):raise ValueError('original gallery name coverage differs')
        gal=np.stack([np.mean([ns['contrast'](a,ns['onehot'](b),256) for a,b in per[q]],0) for q in names])
        pop=gal.mean(0);error=np.mean([np.mean(bias[q],0) for q in names],0);Q=ns['basis'](pop,error)
        prototype=bio['Bio'].__new__(bio['Bio']);prototype.win=np.hanning(120)
        features=np.stack([prototype.feat(X[r]) for r in train])
        arrays=bio['fold_params'](metric,features,short[train],names,**constants)
        arrays.update(route_gal=gal,route_pop=pop,route_b=error,route_Q=Q,route_names=np.array(names))
        if gal.shape!=(10,3,256) or ns['gal_norm'](gal,Q).shape!=(10,768):raise ValueError('actual package raw-gallery shape differs')
        if not all(np.isfinite(v).all() for k,v in arrays.items() if k!='route_names') or not np.allclose(Q.T@Q,np.eye(2),atol=1e-10):raise ValueError('nonfinite/degenerate source galleries')
        other='P10' if fold[:3]=='P04' else 'P04'
        expected_counts=np.array([3 if n==other else 1 for n in names])
        if not np.array_equal(arrays['bio_m'],expected_counts):raise ValueError('source-only biography session counts differ')
        out=Path(fold);out.mkdir();np.savez(out/'galleries.npz',**arrays)
        proof=dict(status='new_source_coordinate_galleries_complete_CONDITIONAL_oldOOF',fold=fold,source_rows=len(train),held_rows=len(held),
                   train_rows_sha256=hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(),source_basis_sha256=sha(file),arrays_sha256=sha(out/'galleries.npz'),
                   original_OOF_cache_sha256=sha(cache),OOF_key='B2_pool|tsem|all_loso',old_OOF_outer_exposure_RETAINED=True,
                   query_labels_read=False,accuracy_computed=False,source_sha256=sha('source.sha256'),
                   scope='sourcefit true galleries/newcoordinate Q using unchanged old OOF posterior calibration;not all-components-excluded generalization')
        (out/'proof.json').write_text(json.dumps(proof,indent=2));records.append(proof)
    Path('galleries_complete.json').write_text(json.dumps(dict(status='both_source_galleries_complete_CONDITIONAL',folds=records,seconds=time.monotonic()-start,accuracy_computed=False),indent=2))


if __name__=='__main__':main()
