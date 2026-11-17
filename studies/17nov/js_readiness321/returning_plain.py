"""Source-audited original normal-BN inner families, new matched weights for both arms."""
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

from returning_train import prepare,sha

CONFIG={'shallow_ea':('shallow',True,False),'shallow_raw':('shallow',False,False),
        'fbcnet_ea_tf':('fbcnet',True,True),'fbcnet_raw_tf':('fbcnet',False,True)}


def main(cfg,fold,seed,mode='fit'):
    if cfg not in CONFIG or seed not in range(5 if cfg.startswith('shallow') else 2):raise ValueError('unregistered family/seed')
    torch.set_num_threads(1);d,_,_,train,held,_,_=prepare(fold,seed)
    import run_cv
    run_cv.SEED[0]=seed;m=run_cv.CONFIGS[cfg](d);arch,align,fir=CONFIG[cfg]
    recipe=dict(cfg=cfg,arch=m.arch,align=m.align,train_fir=m.train_fir,epochs=m.epochs,lr=m.lr,wd=m.wd,batch=m.batch,
                k=m.k,rot=m.rot,aug=m.aug,init=m.init,adabn=m.adabn,seed=m.seed,ft=m.ft,net_kw=m.net_kw)
    expected=dict(cfg=cfg,arch=arch,align=align,train_fir=fir,epochs=40,lr=.001,wd=.001,batch=64,
                  k=5.,rot=0.,aug=False,init=None,adabn=None,seed=seed,ft=0,net_kw={})
    if recipe!=expected:raise ValueError(f'original documented recipe mismatch {recipe}')
    if mode=='preflight':print(json.dumps(recipe));return
    if mode=='recover':
        if (cfg,fold,seed)!=('shallow_ea','P04_s23',0):raise ValueError('only actual failed export checkpoint registered')
        import deep
        prior=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_returning_plain_pilot1/plain/shallow_ea_P04_s23_s0/checkpoint.pt')
        ck=torch.load(prior,map_location='cpu',weights_only=False)
        if ck['recipe']!=recipe or ck['train_count']!=len(train) or ck['held_count']!=len(held):raise ValueError('recovery source recipe differs')
        m.net=deep.build(arch,41,480,m.pretrained,**m.net_kw);m.net.load_state_dict(ck['state'],strict=True)
        m.G,m.scale=ck['G'],ck['scale'];meta={k:v for k,v in ck.items() if k not in ('state','G','scale')}
        if meta['train_rows_sha256']!=hashlib.sha256(train.astype('<i8').tobytes()).hexdigest():raise ValueError('recovery source mask differs')
        meta.update(recovered_checkpoint_sha256=sha(prior),recovery_source_sha256=sha('source.sha256'))
        start=time.monotonic();out=Path('plain')/f'{cfg}_{fold}_s{seed}';out.mkdir(parents=True,exist_ok=False)
    else:
        if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise ValueError('registered A100 required')
        seen=[];aligned=m._aligned
        def observed(rows,*args,**kw):seen.append(np.asarray(rows).copy());return aligned(rows,*args,**kw)
        m._aligned=observed;start=time.monotonic();m.fit(train);m._aligned=aligned;fit_seconds=time.monotonic()-start
        if not np.array_equal(np.sort(np.concatenate(seen)),train) or not np.array_equal(m.G,m.C[train].mean(0)):
            raise ValueError('actual source fit/EA rows differ')
        out=Path('plain')/f'{cfg}_{fold}_s{seed}';out.mkdir(parents=True,exist_ok=False)
        meta=dict(family=cfg,fold=fold,seed=seed,recipe=recipe,train_count=len(train),held_count=len(held),fit_seconds=fit_seconds,
                  train_rows_sha256=hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(),
                  held_rows_sha256=hashlib.sha256(held.astype('<i8').tobytes()).hexdigest(),source_sha256=sha('source.sha256'),
                  historical_identity_claimed=False,accuracy_computed=False,job=os.environ.get('SLURM_JOB_ID'))
    m.net.cpu().eval();m.dev='cpu';checkpoint=out/'checkpoint.pt'
    torch.save(dict(state=m.net.state_dict(),G=m.G,scale=m.scale,**meta),checkpoint)
    rows=held[d['stream'][held]==d['stream'][held[0]]][:31];x=torch.from_numpy(np.ascontiguousarray(m._aligned(rows,causal=True),dtype=np.float32))
    path=Path('source/scripts/export_external_teacher.py');nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='device_free_trace']
    if len(nodes)!=1:raise ValueError('original device-free exporter missing')
    ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    with torch.no_grad():eager_before=m.net(x[:5]).clone()
    state_before={k:v.clone() for k,v in m.net.state_dict().items()}
    # Exact metadata-only remedy from original export_external_teacher.py:65-68.
    m.net._chs_info=m.net._chs_info or [{'ch_name':f'ch{c}'} for c in range(41)]
    m.net._sfreq=m.net._sfreq or 120.;m.net._input_window_seconds=m.net._input_window_seconds or 4.
    with torch.no_grad():eager_after=m.net(x[:5])
    if not torch.equal(eager_before,eager_after) or not all(torch.equal(v,m.net.state_dict()[k]) for k,v in state_before.items()):
        raise ValueError('metadata-only export remedy changed eager/state')
    trace=ns['device_free_trace'](m.net,x[:4]);trace.eval();errors={}
    before={k:v.clone() for k,v in trace.state_dict().items()}
    with torch.no_grad():
        for n in (5,17,31):
            ref=m.net(x[:n]).double().softmax(1).numpy();got=torch.cat([trace(x[i:i+1]) for i in range(n)]).double().softmax(1).numpy()
            errors[n]=float(np.abs(ref-got).max())
            if errors[n]>1e-4 or np.any((ref.argmax(1)!=got.argmax(1))&(np.sort(ref,axis=1)[:,-1]-np.sort(ref,axis=1)[:,-2]>2e-4)):
                raise ValueError('own-weight CPU eager/B1 export guard failed')
        fresh,other=copy.deepcopy(trace),copy.deepcopy(trace);left=[]
        for i in range(31):left.append(fresh(x[i:i+1]));other(x[30-i:31-i])
        repeated=torch.cat(left).double().softmax(1).numpy()
        if not np.array_equal(repeated,got):raise ValueError('normalBN B1 interleaved copy/samepath differs')
    if not all(torch.equal(v,trace.state_dict()[k]) for k,v in before.items()):raise ValueError('normalBN buffers/weights mutated')
    trace.save(str(out/'trace.pt'))
    meta.update(status='new_source_plain_family_export_passed_NOT_accuracy',checkpoint_sha256=sha(checkpoint),trace_sha256=sha(out/'trace.pt'),
                export_errors=errors,immutable_buffers=True,samepath_interleaved_bitwise=True,seconds=time.monotonic()-start,
                scope='NEW same source weights in control/JS; fixed production prior conditional bridge only')
    (out/'proof.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)


if __name__=='__main__':main(sys.argv[1],sys.argv[2],int(sys.argv[3]),*sys.argv[4:])
