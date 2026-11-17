"""Real-source one-step import, mechanism, leakage and prefix smoke; no accuracy."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data322 import DATA_SHA, FIT, ROOT, load, sha
from model322 import Factor, setup, SEED, param_hash
import deep, protocol, session_bn, encoder312, source_labels

for mod,path in ((deep,ROOT/'experiments/deep.py'),(protocol,ROOT/'experiments/protocol.py'),
                 (session_bn,ROOT/'experiments/session_bn.py'),(encoder312,ROOT/'encoder312.py'),
                 (source_labels,ROOT/'source_labels.py')):
 if Path(mod.__file__).resolve()!=path.resolve():raise ValueError(f'module path differs: {mod.__name__}')
torch.set_num_threads(2)
d,global_rows,fit,val=load(False)
if set(np.char.split(d['subject'][fit],sep='/').tolist()[i][-1] for i in range(len(fit)))!=set(FIT):
 raise ValueError('fit people differ')
x,y,scale,mu,sd,weights,groups=setup(d,fit,'cpu',1)
rows=next((r for r in groups[0] if len(r)==64 and d['stream'][fit][r[0]]!=d['stream'][fit][r[32]]),None)
if rows is None:raise ValueError('no two-session fit minibatch')
models={};initial={};outputs={};grad={}
for norm in ('population','session'):
 for skip in (False,True):
  name=f'{norm}_{int(skip)}';m=Factor(norm,skip,'cpu');initial[name]=param_hash(m)
  m.moment_mean.copy_(mu);m.moment_sd.copy_(sd);m.train();torch.manual_seed(SEED)
  outputs[name]=m(x[rows]).detach().clone()
  torch.manual_seed(SEED);m.zero_grad(set_to_none=True)
  loss=torch.nn.functional.cross_entropy(m(x[rows]),y[rows],weight=weights);loss.backward()
  g=m.task.weight.grad[:,-287:]
  if not torch.isfinite(loss) or not torch.isfinite(g).all():raise ValueError('invalid gradient')
  grad[name]=float(g.abs().max())
  if (skip and grad[name]<=0) or (not skip and grad[name]!=0):raise ValueError('skip gradient inactive or leaked')
  models[name]=m.eval()
if len(set(initial.values()))!=1:raise ValueError('factor arms did not share initialization')
if torch.equal(outputs['population_0'],outputs['session_0']):raise ValueError('normalization is inactive')
if torch.equal(outputs['population_0'],outputs['population_1']) or torch.equal(outputs['session_0'],outputs['session_1']):
 raise ValueError('skip knob is inactive')
first=np.flatnonzero(d['stream'][fit]==d['stream'][fit][0])[:20]
second=np.flatnonzero(d['stream'][fit]!=d['stream'][fit][0])[:10]
if len(first)<10 or len(second)<5:raise ValueError('insufficient session rows')
for name,m in models.items():
 before=encoder312.fingerprint(m)
 with torch.no_grad():
  allp=m(x[first]);pref=m(x[first[:10]]);m(x[second]);again=m(x[first])
 if not torch.allclose(allp[:10],pref,atol=1e-5,rtol=1e-5) or not torch.equal(allp,again):
  raise ValueError(f'{name}: prefix/session reset failed')
 if before!=encoder312.fingerprint(m):raise ValueError('model weights changed during evaluation')
ref=session_bn.make('fbcnet_raw_sbn',d,SEED)
if ref.align is not False or ref.epochs!=40:raise ValueError('raw reference settings differ')
result=dict(passed=True,fit_people=FIT,validation_labels_opened=False,outer_P01_labels_opened=False,
            data_sha256=DATA_SHA,source_manifest_sha256=sha(ROOT/'source322.sha256'),initial_sha256=initial,
            skip_gradient_max=grad,fit_rows_sha256=hashlib.sha256(global_rows[fit].tobytes()).hexdigest(),
            validation_rows_sha256=hashlib.sha256(global_rows[val].tobytes()).hexdigest(),scale=scale,
            cpu_only=True,reference_align=False,reference_epochs=40)
with (ROOT/'smoke.json').open('x') as f:json.dump(result,f,indent=2)
print('OK: §322 no-score source/split, module, matched initialization, active BN/skip, gradient, causal/reset guards',flush=True)
