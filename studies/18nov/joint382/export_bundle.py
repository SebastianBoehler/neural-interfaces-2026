"""Certify fixed-shape scripted source training against actual Galvani A6 operators."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from torch.nn import functional as F

PRIMARY=Path('/mnt/lustre/work/gehler/gfh527/ni26_joint382_v1_20261118')
sys.path.insert(0,str(PRIMARY))
from restore import joint,prior,init,source_arrays
from kernel import configure,schedule


class TraceBody(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        for name, module in model.named_children():
            self.add_module(name,module)
        self.stride_factor=model.stride_factor
        self.n_times_padded=model.n_times_padded
        self.forward_impl=type(model).forward

    def forward(self,x):
        return self.forward_impl(self,x)


def main():
    assert 'A100' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    primary_sha='67829e9fe023605f5426d1bb035f9b51f00e8fc026c9b6824fdae3035b92c5a9'
    joint.check_manifest(PRIMARY,primary_sha)
    d,people,rows,runtime=prior.source()
    prior.owners(rows)
    arrays=source_arrays(d,rows,runtime)
    x,y,w=[torch.from_numpy(arrays[k]).cuda() for k in ('x','y','weights')]
    b=schedule(arrays['streams'],382,1)[0]
    dest=Path('bundle')
    dest.mkdir()
    np.savez_compressed(dest/'source.npz',**arrays,source_rows=rows)
    rec={}
    probes={}
    for arm,name in [('J','learned'),('K','native')]:
        model=init(runtime.net,arm).cuda().eval()
        configure(model,arm)
        traced=torch.jit.trace(TraceBody(model).eval(),x[b],check_trace=False)
        path=dest/(name+'.ts')
        torch.jit.save(traced,str(path))
        traced=torch.jit.load(str(path),map_location='cuda').eval()
        assert set(model.state_dict())==set(traced.state_dict())
        assert all(torch.equal(v,traced.state_dict()[k]) for k,v in model.state_dict().items())
        configure(traced,arm)
        prob_error=grad_error=0.
        for i,episode in enumerate([b,schedule(arrays['streams'],1382,1)[0]]):
            model.zero_grad()
            traced.zero_grad()
            real=model(x[episode])
            trial=traced(x[episode])
            prob_error=max(prob_error,float((real.detach()-trial.detach()).abs().max()))
            assert torch.allclose(real,trial,atol=2e-5,rtol=1e-5) and torch.equal(real.argmax(1),trial.argmax(1))
            if i==0:
                probes[name]=real.detach().cpu().numpy()
            for output in (real,trial):
                (F.cross_entropy(output,y[episode],reduction='none')*w[episode]).mean().backward()
            other=dict(traced.named_parameters())
            for key,p in model.named_parameters():
                if p.requires_grad:
                    q=other[key]
                    assert p.grad is not None and q.grad is not None
                    grad_error=max(grad_error,float((q.grad-p.grad).abs().max()))
                    assert torch.allclose(p.grad,q.grad,atol=2e-5,rtol=1e-4),key
        rec[name]=dict(source_loss_gradient_max=grad_error,logit_max=prob_error)
    np.savez_compressed(dest/'probes.npz',**probes)
    shutil.copyfile(PRIMARY/'kernel.py',dest/'kernel.py')
    shutil.copyfile('fit_bundle.py',dest/'fit_bundle.py')
    hashes={p.name:joint.sha(p) for p in dest.iterdir()}
    doc=dict(primary_source=primary_sha,source_only=True,people=joint.FIT_PEOPLE,
             fixed_training_batch_length=48,source_rows_sha256=joint.digest(rows),files=hashes,
             source_guards=rec,export_source=joint.sha(Path('source.sha256')),
             torch=str(torch.__version__))
    (dest/'bundle.json').write_text(json.dumps(doc,indent=2))
    print(doc,flush=True)


if __name__=='__main__':
    main()
