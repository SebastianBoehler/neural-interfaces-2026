"""One independent seed: real entry smoke, unchanged40epoch fit and complete banks."""
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from shared339 import src,inputs,logits,fingerprint,OUTER,CAL,VIEWS,BASE_SHA,sha,digest


def main():
    root=Path(os.environ['CAL_ROOT']);seed=int(sys.argv[1]);start=time.monotonic();torch.set_num_threads(6)
    if seed not in (0,1) or not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise ValueError('registeredseed/A100route')
    torch.cuda.reset_peak_memory_stats();d,rows,cal,people,perms=inputs();m=src.model322;m.SEED=seed
    target=root/f'seed{seed}'
    if target.exists():raise FileExistsError(target)
    # Same per-seed one-epoch entry point checks actual gradient/update and cost BEFORE40epochs.
    x,y,scale,mu,sd,cw,groups=m.setup(d,rows,'cuda',epochs=1);probe=m.Factor('population',False,'cuda');before=fingerprint(probe)
    tick=time.monotonic();smoke=m.train(probe,x,y,mu,sd,cw,groups);torch.cuda.synchronize();seconds=time.monotonic()-tick
    if before==fingerprint(probe) or smoke['max_gradient']<=0:raise ValueError('realtrainstepno-op')
    projection=1.5*seconds*40+120
    cost=dict(seed=seed,smoke=smoke,seconds=seconds,projection_seconds=projection,ceiling=480,
              device=torch.cuda.get_device_name(),source_sha256=sha(root/'source.sha256'),calibration_labels_opened=False)
    (root/f'cost_s{seed}.json').write_text(json.dumps(cost,indent=2))
    if projection>480:raise ValueError('independentproducerbudgetstop')
    del probe,x,y;gc.collect();torch.cuda.empty_cache()
    # Factor/trainer each explicitly reset to the SAME registered seed; no probe RNG continuation.
    m.SEED=seed;x,y,scale,mu,sd,cw,groups=m.setup(d,rows,'cuda');model=m.Factor('population',False,'cuda')
    initial=m.param_hash(model)
    with np.load(src.PROD/'output'/f'{OUTER}_s{seed}'/'result.npz') as z:original_meta=json.loads(str(z['metadata']))
    if initial!=original_meta['initial_sha256']['P0']:raise ValueError('original323seedinitializationdiffers')
    tick=time.monotonic();training=m.train(model,x,y,mu,sd,cw,groups);torch.cuda.synchronize();train_seconds=time.monotonic()-tick
    if len(groups)!=40 or training['steps']!=sum(map(len,groups)):raise ValueError('incompletefixed40epochfit')
    state=fingerprint(model);tmp=Path(tempfile.mkdtemp(prefix=f'.seed{seed}-',dir=root))
    torch.save(dict(state=model.state_dict(),scale=scale,seed=seed,outer=OUTER,cal=CAL),tmp/'model.pt')
    bank,checks=logits(model,scale,d,cal,perms,'cuda')
    if fingerprint(model)!=state:raise ValueError('newfittedmodelmutated')
    original=src.load(OUTER,seed,'cuda');oldnet,oldscale=original[5]['P'];oldbank,oldchecks=logits(oldnet,oldscale,d,cal,perms,'cuda')
    expected={v+'|'+o for v in VIEWS for o in perms}
    if set(bank)!=expected or set(oldbank)!=expected:raise ValueError('completebankfirst')
    with (tmp/'logits.npz').open('xb') as f:np.savez_compressed(f,index=cal,**{kind+'|'+k:v for kind,b in (('held8',bank),('insample9',oldbank)) for k,v in b.items()})
    record=dict(seed=seed,source_sha256=sha(root/'source.sha256'),scientific_base=BASE_SHA,initial_param_sha256=initial,
        checkpoint_sha256=sha(tmp/'model.pt'),logits_sha256=sha(tmp/'logits.npz'),trained=fingerprint(model),training=training,
        fit_people=sorted(set(people[rows])),outer=OUTER,cal=CAL,fit_rows_sha256=digest(rows),cal_rows_sha256=digest(cal),
        fit_input_sha256=digest(d['X'][rows]),scale=scale,checks=checks,source9_checks=oldchecks,
        original_source9_checkpoint=original[7]['checkpoints']['P'],
        batches_sha256=hashlib.sha256(b''.join(b.tobytes() for ep in groups for b in ep)).hexdigest(),
        cost=cost,train_seconds=train_seconds,seconds=time.monotonic()-start,calibration_labels_opened=False,outer_labels_opened=False,
        device=torch.cuda.get_device_name(),node=os.uname().nodename,peak_bytes=torch.cuda.max_memory_allocated(),
        kernel=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            deterministic=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark))
    (tmp/'record.json').write_text(json.dumps(record,indent=2));os.rename(tmp,target);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
