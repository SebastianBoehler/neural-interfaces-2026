"""NEW genuine held-session Sinc weights; never historical checkpoint identity."""
import ast
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CFG='sincshallow_ea_ptm7r_rot05_sbn'
DATA='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(fold,seed):
    root=Path.cwd()/'source';sys.path[:0]=[str(root/'experiments'),str(root/'scripts')]
    sys.path.insert(0,'/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/2026-competition/tracks/bci_decoding')
    import protocol
    import session_bn as sb
    if fold not in ('P04_s23','P10_s23') or seed not in range(6):raise ValueError('registered fold/seed required')
    if sha(root/'data/proteus_windows.npz')!=DATA:raise ValueError('data changed')
    d=protocol.load();train,held={f:(a,b) for f,a,b in protocol.folds(d,'seen_session')}[fold]
    own=d['subject']==('Dreyer2026Proteus/'+fold[:3]);expected=own&(d['session']!='1')
    if not np.array_equal(held,expected) or not np.array_equal(train,~held) or set(d['session'][held])!={'2','3'}:
        raise ValueError('held-session exclusion differs')
    if not (own&train).any() or d['X'].shape[1:]!=(41,480):raise ValueError('own session1/tensor absent')
    # Training must never consume held labels or weights.
    d['y']=d['y'].copy();d['y'][held]=-999
    if 'w' in d:d['w']=d['w'].copy();d['w'][held]=np.nan
    model=sb.make(CFG,d,seed)
    recipe=dict(seed=model.seed,arch=model.arch,align=model.align,rot=model.rot,group=model.group,
                epochs=model.epochs,lr=model.lr,wd=model.wd,batch=model.batch,input_prior_k=model.k,
                causal=model.causal,k_train=model.k_train,aug=model.aug)
    expected=dict(seed=seed,arch='sincshallow',align=True,rot=.05,group=32,epochs=40,lr=.001,
                  wd=.001,batch=64,input_prior_k=5.,causal=False,k_train=20.,aug=False)
    if recipe!=expected:raise ValueError('documented recipe changed')
    trunk=Path(model.init.format(seed=seed));pins=json.loads((Path.cwd()/'inputs.json').read_text())
    if sha(trunk)!=pins['trunks'][str(seed)]:raise ValueError('pretrained trunk changed')
    return d,model,sb,np.flatnonzero(train),np.flatnonzero(held),recipe,trunk


def export(model,sb,inputs,path,latent_k=20.):
    model.net.cpu().eval();model.dev='cpu';model.set_k(latent_k)
    p=Path.cwd()/'source/scripts/export_external_teacher.py';tree=ast.parse(p.read_text())
    fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='device_free_trace']
    if len(fn)!=1:raise ValueError('owning exporter absent')
    ns={'torch':torch};exec(compile(ast.Module(body=fn,type_ignores=[]),str(p),'exec'),ns)
    trace=ns['device_free_trace'](sb.to_stream(model.net,latent_k),torch.from_numpy(inputs[:4]))
    sb.reset_stream(trace);trace.save(str(path));errors={}
    with torch.no_grad():
        for n in (5,17,len(inputs)):
            x=torch.from_numpy(inputs[:n]);ref=model.net(x).double().softmax(1).numpy()
            sb.reset_stream(trace);got=torch.cat([trace(x[i:i+1]) for i in range(n)]).double().softmax(1).numpy()
            errors[n]=float(np.abs(ref-got).max())
            if errors[n]>1e-4 or np.any((ref.argmax(1)!=got.argmax(1))&(np.sort(ref,axis=1)[:,-1]-np.sort(ref,axis=1)[:,-2]>2e-4)):
                raise ValueError('new own-weight eager/export parity failed')
        sb.reset_stream(trace);a,b=copy.deepcopy(trace),copy.deepcopy(trace)
        x=torch.from_numpy(inputs);left=[a(x[i:i+1]) for i in range(5)]
        for i in range(5):b(x[-1-i:len(x)-i])
        got=torch.cat(left+[a(x[i:i+1]) for i in range(5,len(x))]).double().softmax(1).numpy()
        if float(np.abs(got-ref).max())>1e-4:raise ValueError('deepcopy state interference')
    sb.reset_stream(trace);trace.save(str(path))
    if any(float(v.abs().sum()) for k,v in trace.named_buffers() if k.rsplit('.',1)[-1] in sb.STATE):raise ValueError('saved stream state not zero')
    return errors


def main(fold,seed,mode='train'):
    torch.set_num_threads(1);d,m,sb,train,held,recipe,trunk=prepare(fold,seed)
    meta=dict(fold=fold,seed=seed,recipe=recipe,data_sha256=DATA,trunk_sha256=sha(trunk),
              train_rows_sha256=hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(),
              held_rows_sha256=hashlib.sha256(held.astype('<i8').tobytes()).hexdigest(),train_count=len(train),held_count=len(held),
              historical_identity_claimed=False,scope='NEW held-session checkpoint for SAME-weight paired control/JS',
              source_sha256=sha('source.sha256'),job=os.environ.get('SLURM_JOB_ID'),accuracy_computed=False)
    if mode=='preflight':print(json.dumps(meta));return
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('explicit A100 required')
    out=Path('outputs')/f'{fold}_s{seed}';out.mkdir(parents=True,exist_ok=False)
    observed=[];aligned=m._aligned
    def training_input(rows,*args,**kwargs):
        observed.append(np.asarray(rows).copy())
        return aligned(rows,*args,**kwargs)
    m._aligned=training_input
    start=time.monotonic();m.fit(train);meta['fit_seconds']=time.monotonic()-start
    m._aligned=aligned
    actual=np.concatenate(observed)
    if not np.array_equal(np.sort(actual),train) or not np.array_equal(m.G,m.C[train].mean(0)):
        raise ValueError('actual fitting/normalization rows differ')
    if not np.isfinite(m.scale) or m.n_sbn<=0:raise ValueError('invalid fitted model')
    for v in m.net.state_dict().values():
        if not torch.isfinite(v).all():raise ValueError('nonfinite trained state')
    checkpoint=out/'checkpoint.pt'
    torch.save(dict(state={k:v.detach().cpu() for k,v in m.net.state_dict().items()},G=m.G,scale=m.scale,**meta),checkpoint)
    stream=d['stream'][held[0]];rows=held[d['stream'][held]==stream][:31]
    inputs=np.ascontiguousarray(m._aligned(rows,causal=True),dtype=np.float32)
    meta['export_errors']=export(m,sb,inputs,out/'trace_k20.pt')
    meta.update(checkpoint_sha256=sha(checkpoint),trace_sha256=sha(out/'trace_k20.pt'),seconds=time.monotonic()-start,
                status='new_genuine_held_session_weight_export_passed',device=torch.cuda.get_device_name(0),torch=torch.__version__)
    (out/'proof.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)


if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),*sys.argv[3:])
