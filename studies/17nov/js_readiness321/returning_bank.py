"""No-label raw-window production of genuine returning streams using frozen package input code."""
import ast
import copy
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from returning_assets import ROOT,KINDS,asset,sha

ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
ZIP=ROOT/'ni26_js321_20261117_active5/base_v66.zip'
DATA=ROOT/'neural-interfaces-2026/data/proteus_windows.npz'


def input_class():
    if sha(ZIP)!='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed':raise ValueError('frozen ZIP changed')
    with zipfile.ZipFile(ZIP) as z:source=z.read('submission.py').decode()
    tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('InputStream','eig_apply')]
    if len(nodes)!=2:raise ValueError('frozen input anchors differ')
    namespace={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_v66_inputs','exec'),namespace)
    return namespace['InputStream']


def main(fold,seed):
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('explicit A100 required')
    torch.set_num_threads(1);start=time.monotonic()
    if sha(DATA)!='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327':raise ValueError('data changed')
    with np.load(DATA,allow_pickle=True) as z:x=z['X'].copy();sub=z['subject'].copy();session=z['session'].astype(str)
    streams=sub+'|'+session;who='Dreyer2026Proteus/'+fold[:3]
    held=np.flatnonzero((sub==who)&(session!='1'));canonical=np.sort(held)
    if len(held) not in (377,385) or x.shape[1:]!=(41,480):raise ValueError('held tensor differs')
    InputStream=input_class();models={};parameters={};records={}
    sbpath=ROOT/'ni26_js321_20261117_returning2/source/experiments';sys.path.insert(0,str(sbpath))
    from session_bn import STATE,reset_stream
    for kind in KINDS:
        ckpt,trace,proof=asset(kind,fold,seed);c=torch.load(ckpt,map_location='cpu',weights_only=False)
        if proof['held_rows_sha256']!=__import__('hashlib').sha256(held.astype('<i8').tobytes()).hexdigest():raise ValueError('held/source identity differs')
        parameters[kind]=dict(align=np.array(kind!='fbraw'),k=np.array(5.),G=c['G'],scale=np.array(c['scale']),shrink=np.array(.001))
        models[kind]=torch.jit.load(str(trace),map_location='cuda').eval();reset_stream(models[kind])
        if any(float(v.abs().sum()) for n,v in models[kind].named_buffers() if n.rsplit('.',1)[-1] in STATE):raise ValueError('asset starts with session state')
        records[kind]=dict(checkpoint_sha256=sha(ckpt),trace_sha256=sha(trace),source=proof.get('source_sha256',proof.get('source_manifest')))
    out=Path('banks')/f'{fold}_s{seed}';out.mkdir(parents=True,exist_ok=False);errors={};calls={k:0 for k in KINDS}
    perm=ROOT/'neural-interfaces-2026/runs/shuffle_perms.npz'
    if sha(perm)!='6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f':raise ValueError('pinned permutation changed')
    with np.load(perm) as perms:
        for order in ORDERS:
            scheme,number=order[:-1],int(order[-1]);delivery=perms[f'{scheme}|{number}|seen_session']
            delivery=delivery[sub[delivery]==who]
            if not np.array_equal(np.sort(delivery),held):raise ValueError('order coverage differs')
            values={k:np.full((len(held),3),np.nan) for k in KINDS}
            for name in dict.fromkeys(streams[delivery]):
                rows=delivery[streams[delivery]==name];positions=np.searchsorted(canonical,rows)
                for kind,model in models.items():
                    raw=InputStream(parameters[kind]);inputs=np.stack([raw(x[r].astype(np.float32)) for r in rows]);reset_stream(model)
                    with torch.no_grad():
                        t=torch.from_numpy(inputs).cuda()
                        for i in range(17):model(t[i:i+1])
                        reset_stream(model)
                        got=torch.cat([model(t[i:i+1]) for i in range(len(t))]).double().softmax(1).cpu().numpy()
                        calls[kind]+=len(t)
                        reset_stream(model);prefix=torch.cat([model(t[i:i+1]) for i in range(17)]).double().softmax(1).cpu().numpy()
                        if float(np.abs(prefix-got[:17]).max())>1e-8:raise ValueError('phase-matched reset/prefix differs')
                        reset_stream(model);batch=model(t).double().softmax(1).cpu().numpy()
                        delta=float(np.abs(got-batch).max());errors[kind]=max(errors.get(kind,0.),delta)
                        if delta>1e-4:raise ValueError('production window vs batch prefix differs')
                    if not np.isfinite(got).all() or not np.allclose(got.sum(1),1.):raise ValueError('invalid probabilities')
                    values[kind][positions]=got;reset_stream(model)
                    if any(float(v.abs().sum()) for n,v in model.named_buffers() if n.rsplit('.',1)[-1] in STATE):raise ValueError('reset state not zero')
            for kind,proba in values.items():
                np.savez_compressed(out/f'{order}_{kind}.npz',index=canonical,proba=proba,
                    metadata=json.dumps(dict(fold=fold,seed=seed,kind=kind,order=order,asset=records[kind],accuracy_computed=False)))
    if any(n!=8*len(held) for n in calls.values()):raise ValueError('inactive/incomplete bank')
    proof=dict(status='returning_raw_production_bank_passed',fold=fold,seed=seed,assets=records,window_calls=calls,
               zip_sha256=sha(ZIP),data_sha256=sha(DATA),permutation_sha256=sha(perm),batch_window_errors=errors,seconds=time.monotonic()-start,source_sha256=sha('source.sha256'),accuracy_computed=False,
               files={p.name:sha(p) for p in sorted(out.glob('*.npz'))},torch=torch.__version__,device=torch.cuda.get_device_name(0))
    (out/'proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)


if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
