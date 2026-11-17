"""One fixed whole-source arm; nativehead/scaler/loss/batches exceptONEfactor."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from recipe336 import source,native,observe,inference,state_sha,sha,digest

ARMS=('A','H','U','W','B')


def main():
    root=Path(os.environ['BRIDGE_ROOT']);arm=ARMS[int(sys.argv[1])];torch.set_num_threads(6);start=time.monotonic()
    guard=json.loads((root/'guard.json').read_text())
    if not guard['passed'] or not guard['budget_passed'] or guard['source_sha256']!=sha(root/'source.sha256'):raise ValueError('nativeanchorguardfirst')
    d,fit,query,people,y=source();model=native(d);trace=observe(model,arm,fit);model.fit(fit);model.set_k(20.)
    if len(trace['batches'])!=40 or trace['steps']<1:raise ValueError('incomplete40epochfit')
    output,checks=inference(model,d,fit,query,y)
    directory=root/'output'/arm;directory.mkdir(parents=True,exist_ok=False)
    checkpoint=directory/'model.pt';torch.save(dict(state=model.net.state_dict(),G=model.G,scale=model.scale,arm=arm,trace=trace),checkpoint)
    result=dict(arm=arm,source_sha256=sha(root/'source.sha256'),checkpoint_sha256=sha(checkpoint),
        final_state_sha256=state_sha(model.net),source_rows_sha256=digest(fit),query_rows_sha256=digest(query),
        source_people=sorted(set(people[fit])),trace=trace,inference=checks,seconds=time.monotonic()-start,
        device=model.dev,query_labels_opened=False,node=os.uname().nodename)
    with (directory/'record.json').open('x') as out:json.dump(result,out,indent=2)
    with (directory/'predictions.npz').open('xb') as out:np.savez_compressed(out,index=query,**output)
    print(json.dumps(dict(arm=arm,seconds=result['seconds'],state=result['final_state_sha256'])),flush=True)


if __name__=='__main__':main()
