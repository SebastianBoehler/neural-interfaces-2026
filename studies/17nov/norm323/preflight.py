"""Real-source no-score fit/mode smoke before the worst-session GPU cell."""
import copy
import sys
from pathlib import Path
import numpy as np
import torch
import model322, data323, encoder312, session_bn, source_labels, modes
from data323 import ROOT, load
from model322 import Factor, setup, param_hash
from modes import probabilities, views

for mod,p in ((model322,ROOT/'model322.py'),(data323,ROOT/'data323.py'),(encoder312,ROOT/'encoder312.py'),
              (session_bn,ROOT/'experiments/session_bn.py'),(source_labels,ROOT/'source_labels.py'),(modes,ROOT/'modes.py')):
 if Path(mod.__file__).resolve()!=p.resolve():raise ValueError(f'wrong module path {mod.__name__}')
torch.set_num_threads(2);d,fit,test=load('P10');model322.SEED=0
if not (d['y'][test]==-1).all():raise ValueError('held labels loaded')
x,y,scale,mu,sd,weights,groups=setup(d,fit,'cpu',1)
rows=groups[0][0];initial={}
for norm in ('population','session'):
 for skip in (False,True):
  m=Factor(norm,skip,'cpu');initial[norm+str(skip)]=param_hash(m)
  m.moment_mean.copy_(mu);m.moment_sd.copy_(sd);m.train();torch.manual_seed(0)
  loss=torch.nn.functional.cross_entropy(m(x[rows]),y[rows],weight=weights);loss.backward()
  g=m.task.weight.grad[:,-287:]
  if not torch.isfinite(g).all() or (skip and float(g.abs().max())<=0) or (not skip and float(g.abs().max())!=0):
   raise ValueError('skip gradient inactive or leaked')
  fixed,causal=views(m.eval(),x[:16]);q=probabilities(causal,x[:20].numpy(),'cpu')
  prefix=probabilities(causal,x[:10].numpy(),'cpu')
  if not np.allclose(q[:10],prefix,rtol=2e-5,atol=2e-5):raise ValueError('prefix guard')
  if np.array_equal(probabilities(fixed,x[:20].numpy(),'cpu'),q):raise ValueError('causal inference inactive')
if len(set(initial.values()))!=1:raise ValueError('initial model parameters differ')
print('OK: source-only fit, held-label exclusion, module paths, active skip and same-weight causal conversion/prefix',flush=True)
