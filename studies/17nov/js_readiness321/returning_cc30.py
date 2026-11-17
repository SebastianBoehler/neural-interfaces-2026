"""Exact §304 CC30 recipe on NEW returning held-session source folds."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from reference import export,original,sha


def main(fold,seed):
    if fold not in ('P04_s23','P10_s23') or seed not in range(6):raise ValueError('registered fold/seed required')
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('explicit A100 required')
    torch.set_num_threads(1);module=original()
    if sha(module.protocol.DATA)!='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327':raise ValueError('public data changed')
    d=module.protocol.load()
    train,held={f:(a,b) for f,a,b in module.protocol.folds(d,'seen_session')}[fold]
    who=d['subject']==('Dreyer2026Proteus/'+fold[:3]);expected=who&(d['session']!='1')
    if not np.array_equal(held,expected) or not np.array_equal(train,~held) or not (who&train).any():raise ValueError('wrong exclusion')
    rows=np.flatnonzero(train);query=np.flatnonzero(held)
    d['y']=d['y'].copy();d['y'][held]=-999
    if 'w' in d:d['w']=d['w'].copy();d['w'][held]=np.nan
    model=module.make(d,'conditional30',seed)
    start=time.monotonic();model.fit(rows);model.set_k(2.)
    if not np.array_equal(np.sort(model._order),rows) or not np.array_equal(model.G,model.C[rows].mean(0)):
        raise ValueError('actual training/normalization rows differ')
    if model.k!=5. or model.repair_stats['gradient_norm']<=0:raise ValueError('inactive recipe/input prior differs')
    out=Path('cc30')/f'{fold}_s{seed}';out.mkdir(parents=True,exist_ok=False)
    meta=dict(fold=fold,seed=seed,conditional_coral=30.,input_prior_k=5.,latent_bn_k=2.,
              train_rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
              held_rows_sha256=hashlib.sha256(query.astype('<i8').tobytes()).hexdigest(),train_count=len(rows),held_count=len(query),
              source_manifest=sha('source.sha256'),fit_seconds=time.monotonic()-start,job=os.environ.get('SLURM_JOB_ID'),
              historical_identity_claimed=False,accuracy_computed=False,scope='NEW returning held-session CC30')
    ckpt=out/'checkpoint.pt';trace=out/'trace_k2.pt'
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.net.state_dict().items()},G=model.G,scale=model.scale,**meta),ckpt)
    stream=d['stream'][query[0]];part=query[d['stream'][query]==stream][:31]
    inputs=np.ascontiguousarray(model._aligned(part,causal=True),dtype=np.float32)
    meta['export']=export(model,module,trace,inputs)
    meta.update(checkpoint_sha256=sha(ckpt),trace_sha256=sha(trace),seconds=time.monotonic()-start,
                status='new_returning_CC30_source_export_passed')
    (out/'proof.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)


if __name__=='__main__':
    import sys
    main(sys.argv[1],int(sys.argv[2]))
