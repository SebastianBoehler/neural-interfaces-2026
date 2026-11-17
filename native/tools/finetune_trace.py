"""Observe the owning Python RAW first refit; compare every stage to native trace."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import textwrap
import types

import numpy as np
import torch
from torch.jit._pickle import restore_type_tag


def strings(names):
    out=np.zeros((len(names),max(map(len,names))+1),np.uint8)
    for i,name in enumerate(names):out[i,:len(name)]=list(name.encode())
    return torch.from_numpy(out)


def difference(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('trace shape/dtype differs')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('nonfinite trace')
    delta=(a.to(torch.float64)-b.to(torch.float64)).abs()
    return dict(max_abs=float(delta.max()),different=int(torch.count_nonzero(a!=b)),rms=float(delta.square().mean().sqrt()))


def main():
    p=argparse.ArgumentParser()
    for key in ('package','fixture','native','production','project','out'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);sys.path.insert(0,str(args.project/'2026-competition/tracks/bci_decoding'))
    spec=importlib.util.spec_from_file_location('actual_frozen_FT_trace',args.package/'submission.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with np.load(args.package/'weights.npz',allow_pickle=False) as f:z={key:f[key] for key in f.files}
    f=torch.load(args.fixture,weights_only=True,map_location='cpu')
    with torch.serialization.safe_globals([restore_type_tag]):
        native=torch.load(args.native,weights_only=True,map_location='cpu');production=torch.load(args.production,weights_only=True,map_location='cpu')
    w=z['member_weights'].astype(np.float64);w/=w.sum();ft=module.FineTune(z,w,'cpu',args.package);ft.load()
    member=ft.member['fbraw'];column=f['input_ids'].tolist().index(member);model=ft.nets['fbraw'];ft.nets={'fbraw':model}
    ft.X['fbraw']=[row.numpy() for row in f['unknown_input'][:,column]]
    P=f['unknown_relabel_20'].numpy();labels=P.argmax(1);weights=P.max(1);out={'native_format':torch.tensor(1)}
    names=[name for name,q in model.named_parameters() if q.requires_grad]+[name for name,_ in model.named_buffers() if name.rsplit('.',1)[-1] in ('w','b')]
    buffers=[name for name,_ in model.named_buffers()];out['parameter_names']=strings(names);out['buffer_names']=strings(buffers)
    def save(key,value):out[key]=value.detach().cpu().clone()
    def capture(key,value,step):save(('step'+str(step)+'_' if step is not None else '')+key,value)
    def capture_buffers(prefix,m,step):
        for i,(_,value) in enumerate(m.named_buffers()):capture(prefix+str(i),value,step)
    def capture_parameters(prefix,parameters,step):
        for i,value in enumerate(parameters):capture(prefix+str(i),value,step)
    def capture_optimizer(opt,ps,step):
        for i,value in enumerate(ps):
            capture('after_'+str(i),value,step);capture('m_'+str(i),opt.state[value]['exp_avg'],step);capture('v_'+str(i),opt.state[value]['exp_avg_sq'],step)
            capture('adam_step_'+str(i),torch.tensor(int(opt.state[value]['step']),dtype=torch.int64),step)
    original=textwrap.dedent(inspect.getsource(module.FineTune._refit));changed=original
    replacements={
      'X = torch.from_numpy(np.stack(self.X[s][:t])).to(self.dev)':'X = torch.from_numpy(np.stack(self.X[s][:t])).to(self.dev)\n        capture("X", X, None)',
      'opt.zero_grad()':'opt.zero_grad()\n            capture_parameters("before_", ps, _)',
      'zc = m(X)[:, :3]':'zc = m(X)[:, :3]\n                capture("clean", zc, _)\n                capture_buffers("clean_buffer_", m, _)',
      'za = m(strong(X, self.gens[s]))[:, :3]':'aug = strong(X, self.gens[s])\n            capture("augmented", aug, _)\n            capture("generator_after", self.gens[s].get_state(), _)\n            za = m(aug)[:, :3]\n            capture("logits", za, _)\n            capture_buffers("aug_buffer_", m, _)',
      'loss.backward()':'capture("ce", ce, _)\n            capture("kl", F.kl_div(F.log_softmax(za, 1), F.softmax(zc, 1), reduction="batchmean"), _)\n            capture("loss", loss, _)\n            loss.backward()\n            capture_parameters("gradient_", [q.grad for q in ps], _)',
      'opt.step()':'opt.step()\n            capture_optimizer(opt, ps, _)',
      'm(X)  # StreamBN state over windows 0..t-1':'m(X)  # StreamBN state over windows 0..t-1\n            capture_buffers("rebuilt_buffer_", m, None)'}
    for before,after in replacements.items():
        if changed.count(before)!=1:raise ValueError('owning FT trace anchor changed:'+before)
        changed=changed.replace(before,after)
    environment=dict(module.__dict__,capture=capture,capture_buffers=capture_buffers,capture_parameters=capture_parameters,capture_optimizer=capture_optimizer)
    exec(compile(changed,'owning_FT_observations_only','exec'),environment)
    ft._refit=types.MethodType(environment['_refit'],ft)
    save('labels',torch.from_numpy(labels));save('weights',torch.from_numpy(weights.astype(np.float32)));save('generator_before',ft.gens['fbraw'].get_state())
    with torch.jit.optimized_execution(False):ft._refit(20,labels,weights)
    with torch.no_grad():
        logits=model(f['unknown_input'][20,column][None])[:,:3];save('current_logits',logits);save('current_p',torch.softmax(logits.double(),1)[0]);capture_buffers('current_buffer_',model,None)
    if set(native)!=set(out):raise ValueError('trace key sets differ:'+str(set(native)^set(out)))
    comparisons={key:difference(native[key],out[key]) for key in sorted(out)}
    original_vs_SIF=difference(out['current_p'],f['unknown_ft'][20,0]);native_vs_production=difference(native['current_p'],production['unknown_ft_fbraw_20'])
    torch.save(out,args.out.with_suffix('.pt'))
    result=dict(original_method_sha256=hashlib.sha256(original.encode()).hexdigest(),observed_method_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        original_vs_SIF=original_vs_SIF,native_vs_production=native_vs_production,comparisons=comparisons,source_prefix=20,current_window=20,refit=1,steps=5,
        raw_only=True,task_labels_loaded=False,accuracy_computed=False,optimizer_repaired=False,tolerance_changed=False)
    args.out.write_text(json.dumps(result,indent=2))
    if original_vs_SIF['max_abs']>1e-6 or native_vs_production['max_abs']!=0:raise ValueError('trace does not reproduce owning consumers; no attribution allowed')


if __name__=='__main__':main()
