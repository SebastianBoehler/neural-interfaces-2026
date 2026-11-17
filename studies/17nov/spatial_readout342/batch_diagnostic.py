"""Same-weight source-only B1/B3/B8 and Torch/NumPy pooling diagnostic."""
import json
from pathlib import Path

import numpy as np
import torch

from common import OLD, encoder, gpu, inputs, layouts, load, state_sha


def main():
    torch.set_num_threads(2);gpu();d,people,source,_=load()
    probe=np.load(OLD/'cpu_a100_embeddings.npz');rows=probe['rows']
    if not set(people[rows])<={'P04','P05','P06','P07','P08','P10'}:
        raise ValueError('source-only rows required')
    # Original3 sourceproberows are consecutive P04 rows inside original8-row batch.
    old=np.load(OLD/'source_banks.npz');oldrows=old['source_rows']
    positions=np.array([np.flatnonzero(oldrows==r)[0] for r in rows])
    start=(positions[0]//8)*8;batchrows=oldrows[start:start+8]
    if len(batchrows)!=8 or not set(people[batchrows])<={'P04','P05','P06','P07','P08','P10'}:
        raise ValueError('source-only original fullbatch unavailable')
    model,upstream=encoder('pretrained');before=state_sha(model);model.cuda()
    chans=upstream.get_input_chans(d['ch_names'].astype(str).tolist());outputs={};pooling={}
    with torch.no_grad():
        cpu=model.cpu().forward_features(inputs(d['X'][rows]),input_chans=chans,return_patch_tokens=True).numpy()
        model.cuda()
        for name,index in [('B1',rows[:1]),('B3',rows),('B8',batchrows)]:
            value=model.forward_features(inputs(d['X'][index]).cuda(),input_chans=chans,return_patch_tokens=True)
            patch=value.cpu().numpy();outputs[name]=patch
            own=layouts(patch)
            torchpool=value.mean(1).cpu().numpy()
            torchtime=value.reshape(-1,41,4,400).mean(1).flatten(1).cpu().numpy()
            pooling[name]=dict(pooled_max_abs=float(np.max(np.abs(own['pooled400']-torchpool))),
                               temporal_max_abs=float(np.max(np.abs(own['temporal1600']-torchtime))))
    if before!=state_sha(model):
        raise ValueError('same-weight frozen state changed')
    selected=positions-start;b8=outputs['B8'][selected];b3=outputs['B3'];b1=outputs['B1']
    allcomp={}
    for name,a,b in [('sameweightCPU_B3_GPU_B3',cpu,b3),('GPU_B1_B3',b1,b3[:1]),
                     ('GPU_B3_B8',b3,b8),('oldGPU_B3_newGPU_B3',probe['pretrained_A100'],b3)]:
        allcomp[name]=dict(full_patch_max_abs=float(np.max(np.abs(a-b))),
                          original_CPU_GPU_bound_pass=bool(np.allclose(a,b,atol=2e-3,rtol=1e-3)),
                          layouts={k:float(np.max(np.abs(v-layouts(b)[k]))) for k,v in layouts(a).items()})
    expected=old['pretrained_retained'][start:start+8]
    err=np.abs(layouts(outputs['B8'])['temporal1600']-expected)
    result=dict(scope='source-only same-weight numerical diagnostic; no labels/fitting/taskmetrics',
                device=torch.cuda.get_device_name(0),state_sha256=before,source_rows=rows.tolist(),batch_rows=batchrows.tolist(),
                pooling_reduction=pooling,comparisons=allcomp,old_temporal_sameB8_max_abs=float(err.max()),
                old_temporal_sameB8_registered_bound_pass=bool(np.allclose(layouts(outputs['B8'])['temporal1600'],expected,atol=2e-4,rtol=1e-4)))
    Path('batch_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
