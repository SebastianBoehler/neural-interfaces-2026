"""Actual frozen external-head source cost and decomposition; no task map fit/score."""
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from returning_bank import ROOT,input_class,sha


def main():
    torch.set_num_threads(1);start=time.monotonic();package=ROOT/'ni26_js321_20261117_orchestration4/prepared_jsall'
    with np.load(package/'weights.npz') as z:p={k:z['net34_'+k].copy() for k in ('align','k','G','scale','shrink')};name=str(z['net34_jit_file'])
    if not bool(p['align']) or float(p['k'])!=5. or name!='teacher.pt':raise ValueError('actual external teacher input differs')
    file=package/name;model=torch.jit.load(str(file),map_location='cpu').eval()
    if tuple(model.lin.weight.shape)!=(3,38):raise ValueError('actual external head/map dimensions differ')
    data=ROOT/'neural-interfaces-2026/data/proteus_windows.npz'
    if sha(data)!='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327':raise ValueError('data changed')
    with np.load(data,allow_pickle=True) as z:
        rows=np.flatnonzero((z['subject']=='Dreyer2026Proteus/P04')&(z['session'].astype(str)=='1'))[:12];raw=z['X'][rows].copy()
    stream=input_class()(p);x=torch.from_numpy(np.stack([stream(v) for v in raw]));before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        t=time.monotonic();features=model.heads(x);seconds=time.monotonic()-t
        logits=model(x);reconstructed=model.lin(features);delta=float((logits-reconstructed).abs().max())
    if features.shape!=(12,38) or not torch.isfinite(features).all() or delta>1e-8:raise ValueError('actual head extraction/composition failed')
    if not all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('external weights/buffers changed')
    proof=dict(status='actual_external_teacher_source_cost_and_decomposition_passed_NOT_mapfit',source_rows=rows.tolist(),feature_shape=list(features.shape),
               feature_seconds=seconds,composition_maxabs=delta,immutable_weights_buffers=True,teacher_sha256=sha(file),weights_sha256=sha(package/'weights.npz'),
               source_sha256=sha('source.sha256'),maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.monotonic()-start,
               query_labels_read=False,accuracy_computed=False,scope='frozen external heads/input productionG;sourceonly12windows;fullmap cost unmeasured')
    Path('teacher_cost.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
