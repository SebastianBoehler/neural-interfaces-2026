"""NEW native-worker B1-only bank protocol; earlier batch guard failure remains failed."""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import returning_bank as bank
from returning_trace import STATE,reset_stream,immutable,zero


def forward(model,t):
    with torch.no_grad():return torch.cat([model(t[i:i+1]) for i in range(len(t))]).double().softmax(1).cpu().numpy()


def guarded(model,t):
    before=immutable(model)
    with torch.no_grad():
        for i in range(17):model(t[i:i+1])
    reset_stream(model);zero(model);got=forward(model,t)
    if any(float(v)!=len(t) for k,v in model.named_buffers() if k.rsplit('.',1)[-1]=='n'):raise ValueError('BN activity count differs')
    reset_stream(model);zero(model);repeat=forward(model,t);error=float(np.abs(got-repeat).max())
    if error>1e-8:raise ValueError(f'native full-session reset/self-repeat differs {error}')
    reset_stream(model);a,b=copy.deepcopy(model),copy.deepcopy(model);zero(a);zero(b)
    with torch.no_grad():
        prefix=[a(t[i:i+1]) for i in range(17)]
        for i in range(17):b(t[-1-i:len(t)-i])
        rest=[a(t[i:i+1]) for i in range(17,len(t))]
        replay=torch.cat(prefix+rest).double().softmax(1).cpu().numpy()
    interleaved=float(np.abs(got-replay).max())
    if interleaved>1e-8:raise ValueError(f'native interleaved deepcopy differs {interleaved}')
    if immutable(model)!=before or immutable(a)!=before or immutable(b)!=before:raise ValueError('immutable parameters/buffers changed')
    reset_stream(model);zero(model)
    return got,dict(self_repeat_max=error,interleaved_max=interleaved,immutable_unchanged=True,reset_zero=True)


def main(fold,seed):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.set_num_threads(1)
    if torch.__version__!='2.14.1+cu130' or not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('explicit qualified SIF/A100 required')
    if 'neural-interfaces-2026/.venv/' in Path('/proc/self/maps').read_text():raise ValueError('host numerical library loaded')
    start=time.monotonic()
    if bank.sha(bank.DATA)!='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327':raise ValueError('data changed')
    with np.load(bank.DATA,allow_pickle=True) as z:x=z['X'].copy();sub=z['subject'].copy();session=z['session'].astype(str)
    streams=sub+'|'+session;who='Dreyer2026Proteus/'+fold[:3];held=np.flatnonzero((sub==who)&(session!='1'))
    canonical=np.sort(held);Input=bank.input_class();models={};parameters={};records={}
    for kind in bank.KINDS:
        ckpt,trace,proof=bank.asset(kind,fold,seed);c=torch.load(ckpt,map_location='cpu',weights_only=False)
        if proof['held_rows_sha256']!=hashlib.sha256(held.astype('<i8').tobytes()).hexdigest():raise ValueError('held/source mismatch')
        parameters[kind]=dict(align=np.array(kind!='fbraw'),k=np.array(5.),G=c['G'],scale=np.array(c['scale']),shrink=np.array(.001))
        models[kind]=torch.jit.load(str(trace),map_location='cuda').eval();reset_stream(models[kind]);zero(models[kind])
        records[kind]=dict(checkpoint_sha256=bank.sha(ckpt),trace_sha256=bank.sha(trace))
    perm=bank.ROOT/'neural-interfaces-2026/runs/shuffle_perms.npz'
    if bank.sha(perm)!='6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f':raise ValueError('order file changed')
    out=Path('banks')/f'{fold}_s{seed}';out.mkdir(parents=True,exist_ok=False);checks=[]
    with np.load(perm) as perms:
        for order in bank.ORDERS:
            delivery=perms[f'{order[:-1]}|{int(order[-1])}|seen_session'];delivery=delivery[sub[delivery]==who]
            if not np.array_equal(np.sort(delivery),held):raise ValueError('order coverage differs')
            values={k:np.full((len(held),3),np.nan) for k in bank.KINDS}
            for name in dict.fromkeys(streams[delivery]):
                rows=delivery[streams[delivery]==name];positions=np.searchsorted(canonical,rows)
                for kind,model in models.items():
                    raw=Input(parameters[kind]);inputs=np.stack([raw(x[r].astype(np.float32)) for r in rows])
                    got,check=guarded(model,torch.from_numpy(inputs).cuda());checks.append(dict(order=order,session=name,kind=kind,**check))
                    if not np.isfinite(got).all() or not np.allclose(got.sum(1),1.):raise ValueError('invalid probabilities')
                    values[kind][positions]=got
            for kind,proba in values.items():
                np.savez_compressed(out/f'{order}_{kind}.npz',index=canonical,proba=proba,metadata=json.dumps(dict(fold=fold,seed=seed,kind=kind,order=order,asset=records[kind],accuracy_computed=False,executor='nativeSIF_B1_TF32off')))
            print('native',fold,seed,order,'complete',flush=True)
    proof=dict(status='native_returning_B1_bank_guard_passed',fold=fold,seed=seed,checks=checks,assets=records,
        files={p.name:bank.sha(p) for p in sorted(out.glob('*.npz'))},seconds=time.monotonic()-start,
        source_sha256=bank.sha('source.sha256'),accuracy_computed=False,old_batch_failure_preserved=True,torch=torch.__version__,
        device=torch.cuda.get_device_name(0),cudnn_tf32=False,matmul_tf32=False,permutation_sha256=bank.sha(perm),data_sha256=bank.sha(bank.DATA))
    (out/'proof.json').write_text(json.dumps(proof,indent=2))


if __name__=='__main__':
    import sys
    main(sys.argv[1],int(sys.argv[2]))
