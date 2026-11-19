"""Check current-window TorchScript against native bounded CPU inference."""
import copy
from types import MethodType

import numpy as np
import torch

from bounded_bn import normalize
from encoder312 import fingerprint
from held_distill import raw_probability
from source_targets import raw_logits


TOL=2e-5
HISTORY={"hist_n","hist_s1","hist_s2","hist_mass","hist_w1","hist_w2","hist_clipped"}


def reset(module):
    if module.reset().item()!=0:
        raise ValueError("native actor reset did not return zero")
    state={name.rsplit(".",1)[-1]:v for name,v in module.named_buffers() if ".hist_" in name}
    if set(state)!=HISTORY or any(torch.count_nonzero(v).item() for v in state.values()):
        raise ValueError("native actor reset left history")


def constants(module):
    return {name:v.detach().cpu().clone() for name,v in module.state_dict().items()
            if name.rsplit(".",1)[-1] not in HISTORY}


@torch.no_grad()
def reference(model,layer,sample,scale):
    model.skip=False
    prior=layer.forward
    before=fingerprint(model)
    probes=[]
    results=[]
    for data in sample:
        voltage=data/scale
        tensor=torch.from_numpy(voltage[:10])
        if not torch.equal(raw_logits(model,tensor),model(tensor)):
            raise ValueError("ordinary native head differs from logical zero features")
        original=raw_probability(model,voltage)
        try:
            layer.forward=MethodType(lambda self,x:normalize(self,x,probes),layer)
            bounded=raw_probability(model,voltage)
        finally:
            layer.forward=prior
        if not np.array_equal(original[0],bounded[0]) or not np.isfinite(bounded).all():
            raise ValueError("native bounded batch oracle changed the first window")
        results.append(bounded)
    if fingerprint(model)!=before:
        raise ValueError("native oracle changed learned parameters")
    return results,sum(map(int,probes))


@torch.no_grad()
def check(module,sample,targets):
    frames=[torch.from_numpy(data).unsqueeze(1).contiguous() for data in sample]
    with torch.jit.optimized_execution(False):
        reset(module)
        frozen=constants(module)
        worst=0.
        for i,(batch,target) in enumerate(zip(frames,targets,strict=True)):
            reset(module)
            fresh=copy.deepcopy(module)
            reset(fresh)
            if not torch.equal(fresh(batch[0]),module(batch[0])):
                raise ValueError("fresh and reused actor first windows differ")
            reset(module)
            native=torch.stack([module(frame) for frame in batch])
            if native.dtype!=torch.float64 or native.shape!=(len(batch),1,3):
                raise ValueError("export returned a noncanonical posterior")
            output=native[:,0].cpu().numpy()
            difference=float(np.max(np.abs(output-target)))
            if (not np.isfinite(output).all() or np.min(output)<0 or difference>TOL or
                    not np.array_equal(output.argmax(1),target.argmax(1))):
                raise ValueError(f"incremental/batch parity failed {i}: {difference:.6g}, "
                                 f"{np.count_nonzero(output.argmax(1)!=target.argmax(1))} decisions")
            np.testing.assert_allclose(output.sum(1),1.,rtol=0,atol=1e-12)
            worst=max(worst,difference)
            reset(module)
            prefix=torch.stack([module(frame) for frame in batch[:17]])
            if not torch.equal(prefix,native[:17]):
                raise ValueError("future source windows changed the prefix")
            for frame in frames[1-i][:17]:
                module(frame)
            reset(module)
            if not torch.equal(prefix,torch.stack([module(frame) for frame in batch[:17]])):
                raise ValueError("foreign source session survived reset")
        reset(module)
        after=constants(module)
        if frozen.keys()!=after.keys() or any(not torch.equal(val,after[k]) for k,val in frozen.items()):
            raise ValueError("native weights changed during source streaming")
    return dict(windows=sum(map(len,sample)),max_abs=worst,changed_decisions=0,first_windows=2,prefix=2,reset=2)
