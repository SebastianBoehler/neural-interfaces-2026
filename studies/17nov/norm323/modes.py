"""Create fixed/causal inference views without changing trained parameters."""
import copy
import numpy as np
import torch
from session_bn import SessionBN, swap_bn


def state_equal(a,b):
 pa=dict(a.named_parameters());pb=dict(b.named_parameters())
 if set(pa)!=set(pb) or any(not torch.equal(pa[k],pb[k]) for k in pa):raise ValueError('parameter conversion changed weights')
 ba={k:v for k,v in a.named_buffers() if not k.endswith('num_batches_tracked')}
 bb={k:v for k,v in b.named_buffers() if not k.endswith('num_batches_tracked')}
 if set(ba)!=set(bb) or any(not torch.equal(ba[k],bb[k]) for k in ba):raise ValueError('BN conversion changed buffers')


def views(model,probe):
 fixed=copy.deepcopy(model).eval();causal=copy.deepcopy(model).eval()
 existing=[m for m in fixed.body.modules() if isinstance(m,SessionBN)]
 if not existing:
  if swap_bn(causal.body,False,20.)<1:raise ValueError('no BN available for causal inference')
  causal.eval();state_equal(fixed,causal)
  for m in causal.body.modules():
   if isinstance(m,SessionBN):m.k=None
  with torch.no_grad():a=fixed(probe);b=causal(probe)
  if not torch.allclose(a,b,atol=2e-5,rtol=2e-5):raise ValueError('fixed converted BN differs')
 else:
  state_equal(fixed,causal)
  for m in fixed.body.modules():
   if isinstance(m,SessionBN):m.k=None
 for m in causal.body.modules():
  if isinstance(m,SessionBN):m.k=20.
 state_equal(fixed,causal)
 return fixed,causal


def probabilities(model,x,device):
 with torch.no_grad():return torch.softmax(model(torch.as_tensor(x,device=device).float()).double(),1).cpu().numpy()
