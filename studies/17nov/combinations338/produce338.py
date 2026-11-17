"""One prospectively fixed source instrument/combination; no query labels."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from recipe338 import old,source,observe,CELLS


def run(scope,arm,epochs=40,smoke=False):
    d,fit,query,people,y=source(scope);model=old.native(d,epochs);trace=observe(model,arm,fit)
    model.fit(fit);model.set_k(20.)
    if len(trace['batches'])!=epochs or trace['steps']<=0:raise ValueError('incompletefit')
    if ('H' in arm)==trace['head_is_parametrized']:raise ValueError('headinteractionno-op')
    if ('W' in arm)!=(trace['criterion_weights'] is not None):raise ValueError('lossinteractionno-op')
    return d,fit,query,people,y,model,trace


def main():
    root=Path(os.environ['COMBO_ROOT']);torch.set_num_threads(6);start=time.monotonic()
    scope,arm=CELLS[int(sys.argv[1])];guard=json.loads((root/'guard.json').read_text())
    if not guard['passed'] or guard['source_sha256']!=old.sha(root/'source.sha256'):raise ValueError('guardfirst')
    d,fit,query,people,y,model,trace=run(scope,arm)
    output,checks=old.inference(model,d,fit,query,y)
    folder=root/'output'/scope/arm;folder.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state=model.net.state_dict(),G=model.G,scale=model.scale,trace=trace),folder/'model.pt')
    with (folder/'predictions.npz').open('xb') as f:np.savez_compressed(f,index=query,**output)
    record=dict(scope=scope,arm=arm,source_sha256=old.sha(root/'source.sha256'),trace=trace,
                source_rows_sha256=old.digest(fit),query_rows_sha256=old.digest(query),
                checkpoint_sha256=old.sha(folder/'model.pt'),prediction_sha256=old.sha(folder/'predictions.npz'),
                inference=checks,device=model.dev,node=os.uname().nodename,query_labels_opened=False,seconds=time.monotonic()-start)
    (folder/'record.json').write_text(json.dumps(record,indent=2));print(scope,arm,record['seconds'],flush=True)


if __name__=='__main__':main()
