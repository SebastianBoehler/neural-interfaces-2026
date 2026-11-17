"""One complete person/seed saved-weight inference cell; no label reads."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import source332 as src
from predict332 import run


def produce(held,seed):
    root=Path(os.environ['MOMENT_ROOT']);start=time.monotonic();torch.set_num_threads(6)
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise ValueError('registeredA100required')
    guard=json.loads((root/'cpu_guard.json').read_text())
    if not guard['passed'] or guard['labels_opened'] or guard['source_sha256']!=src.sha(root/'source.sha256'):raise ValueError('passedCPUguard/sourcefirst')
    d,fit,test,donor,perms,nets,original,meta=src.load(held,seed,'cuda')
    output,checks,donors=run(d,test,donor,perms,nets,original,'cuda')
    if set(output)!={f'{n}{v}|{o}' for n in 'PS' for v in ('F','M','V','B','D') for o in src.ORDERS}:raise ValueError('incomplete332cell')
    report=dict(held=held,seed=seed,source=meta,checks=checks,donor_order_rows=donors,
        donor_cyclic_repeats={s:max(0,int(np.sum(d['stream'][test]==s))-len(donor)) for s in np.unique(d['stream'][test])},
        source_sha256=src.sha(root/'source.sha256'),held_labels_opened=False,seconds=time.monotonic()-start,
        device=torch.cuda.get_device_name(),peak_gpu_bytes=torch.cuda.max_memory_allocated())
    path=root/'output'/f'{held}_s{seed}.npz';path.parent.mkdir(exist_ok=True)
    with path.open('xb') as out:np.savez_compressed(out,index=test,metadata=json.dumps(report),**output)
    print(json.dumps(dict(output=str(path),seconds=report['seconds'],peak_gpu_bytes=report['peak_gpu_bytes'])),flush=True)


if __name__=='__main__':produce(sys.argv[1],int(sys.argv[2]))
