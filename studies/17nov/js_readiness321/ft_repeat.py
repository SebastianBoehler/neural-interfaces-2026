"""OriginaloldJS two-fresh-model215FT/CP control, before optimized FT interpretation."""
import json
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from guard_shared import OLD,DATA,load,frame
from phase_state import neural_state,rng
from reference import sha


def main():
    started=time.monotonic();torch.set_num_threads(1)
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('explicit A100 required')
    with np.load(DATA,allow_pickle=False) as z:x=z['X'][:215].copy();channels=z['ch_names']
    a=load(OLD,'old215_controlA',channels,True);b=load(OLD,'old215_controlB',channels,True)
    initial=(neural_state(a),neural_state(b));before=rng();maximum={k:0. for k in ('out','core1','mix','u','shadow','out_v5','cc','p_ft')};labels=0;first={};refits=[]
    for i,value in enumerate(x):
        la,pa,ta=frame(a,value);lb,pb,tb=frame(b,value);labels+=int(la!=lb)
        pairs={'out':(pa,pb),'cc':(ta['js']['cc'],tb['js']['cc']),**{k:(ta[k],tb[k]) for k in ('core1','mix','u','shadow','out_v5')}}
        for name,(v,w) in pairs.items():
            e=float(abs(np.asarray(v)-np.asarray(w)).max());maximum[name]=max(maximum[name],e)
            if e and name not in first:first[name]=i
        if set(ta['v6']['p_ft'])!=set(tb['v6']['p_ft']):raise ValueError('original FT activity differs')
        for slot,v in ta['v6']['p_ft'].items():maximum['p_ft']=max(maximum['p_ft'],float(abs(v-tb['v6']['p_ft'][slot]).max()))
        if a.ft.refits and (not refits or refits[-1]['refits']!=a.ft.refits):
            error=0.
            for slot,m in a.ft.nets.items():
                other=dict(b.ft.nets[slot].named_parameters())
                for name,p in m.named_parameters():error=max(error,float((p-other[name]).abs().max()))
            generators=[{s:hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest() for s,g in model.ft.gens.items()} for model in (a,b)]
            if generators[0]!=generators[1]:raise ValueError('original augmentation generators diverged')
            refits.append(dict(window=i,refits=a.ft.refits,parameter_max_error=error,generator_sha256=generators))
    final=(neural_state(a),neural_state(b));after=rng()
    for start,end in zip(initial,final):
        if start['immutable_sha256']!=end['immutable_sha256']:raise ValueError('original frozen neural parameter/non-stream buffer changed')
    report=dict(status='original_JS215_FT_selfrepeat_characterized',max_errors=maximum,first_nonzero=first,label_changes=labels,
                ft_parameters=refits,ft_refits=[a.ft.refits,b.ft.refits],cp_refits=[a.cp.refits,b.cp.refits],cp_fused=[a.cp.fused,b.cp.fused],
                rng_before=before,rng_after=after,initial=initial,final=final,source_sha256=sha(Path('source.sha256')),
                old_bundle_sha256=sha(OLD/'submission.py'),old_weights_sha256=sha(OLD/'weights.npz'),seconds=time.monotonic()-started,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                accuracy_computed=False,scope='original self-control only;no tolerance or optimization guard pass')
    Path('ft_repeat.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
