"""Observe original first SOURCE batch forward/backward; never construct an optimizer."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import types

import torch
from torch.jit._pickle import restore_type_tag


def main():
    p=argparse.ArgumentParser()
    for key in ('fixture','native','project','out'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(args.project/'experiments'))
    import session_bn
    from deep import build,param_groups
    f=torch.load(args.fixture,weights_only=True,map_location='cpu',mmap=True)
    with torch.serialization.safe_globals([restore_type_tag]):native=torch.load(args.native,weights_only=True,map_location='cpu')
    initial={k[len('initial_'):]:v for k,v in f.items() if k.startswith('initial_')}
    torch.manual_seed(int(f['initialization_seed']));net=build('fbcnet',41,480,pretrained=False)
    if session_bn.swap_bn(net,False,20.)!=1:raise ValueError('original ONEBN graph changed')
    net.spatial_conv[1].m=32;net.train();param_groups(net,.001,1)
    if set(net.state_dict())!=set(initial) or any(not torch.equal(v,initial[k]) for k,v in net.state_dict().items()):raise ValueError('original trace initialization does not reproduce source fixture')
    out={'native_format':torch.tensor(1,dtype=torch.int64)}
    def save(name,value):
        if not torch.isfinite(value).all():raise ValueError('nonfinite original stage:'+name)
        out[name]=value.detach().clone();out[name+'_stride']=torch.tensor(value.stride(),dtype=torch.int64)
        if value.requires_grad:
            def gradient(g):
                if not torch.isfinite(g).all():raise ValueError('nonfinite original gradient:'+name)
                out[name+'_grad']=g.detach().clone();out[name+'_grad_stride']=torch.tensor(g.stride(),dtype=torch.int64)
            value.register_hook(gradient)
    def hook(name):return lambda module,inputs,output:save(name,output)
    handles=[]
    for name,module in [('fir',net.spectral_filtering),('conv',net.spatial_conv[0]),('bn',net.spatial_conv[1]),('activation',net.spatial_conv[2]),('logvar',net.temporal_layer),('features',net.flatten_layer),('logits',net.final_layer),('spatial_weight',net.spatial_conv[0].parametrizations.weight),('head_weight',net.final_layer.parametrizations.weight)]:handles.append(module.register_forward_hook(hook(name)))
    handles.append(net.temporal_layer.register_forward_pre_hook(lambda module,inputs:save('segments',inputs[0])))
    original=textwrap.dedent(inspect.getsource(session_bn.SessionBN.forward));changed=original
    replacements={
      'r1 = z.mean(zr) if red else z':'r1 = z.mean(zr) if red else z\n        save("bn_r1", r1)',
      'r2 = (z * z).mean(zr) if red else z * z':'r2 = (z * z).mean(zr) if red else z * z\n        save("bn_r2", r2)',
      'mean = r1.mean(1, keepdim=True)':'mean = r1.mean(1, keepdim=True)\n            save("bn_mean", mean)',
      'var = (r2.mean(1, keepdim=True) - mean ** 2).clamp_min(0)':'var = (r2.mean(1, keepdim=True) - mean ** 2).clamp_min(0)\n            save("bn_var", var)'}
    for before,after in replacements.items():
        if changed.count(before)!=1:raise ValueError('original BN observation anchor changed:'+before)
        changed=changed.replace(before,after)
    environment=dict(session_bn.__dict__,save=save);exec(compile(changed,'original_SessionBN_observations_only','exec'),environment)
    net.spatial_conv[1].forward=types.MethodType(environment['forward'],net.spatial_conv[1])
    count=int(f['batch_sizes'][0]);ids=f['batch_indices'][0,:count];x=f['source_X'][ids];y=f['source_y'][ids]
    actual_rfft=torch.fft.rfft;counts={'fir':0,'input':0}
    def observed_rfft(value,*positional,**keywords):
        result=actual_rfft(value,*positional,**keywords)
        if value.ndim==3 and value.shape[:2]==(1,41):
            save('fir_spectrum_'+str(counts['fir']),result);counts['fir']+=1
        elif value.shape==(count,41,480):
            save('input_spectrum_'+str(counts['input']),result);counts['input']+=1
        return result
    save('input',x);torch.fft.rfft=observed_rfft
    try:logits=net(x)
    finally:torch.fft.rfft=actual_rfft
    if counts!={'fir':9,'input':9}:raise ValueError('actual original FIR FFT consumers changed:'+str(counts))
    loss=torch.nn.functional.cross_entropy(logits,y)
    if not torch.equal(logits,f['first_logits']) or not torch.equal(loss,f['first_loss']):raise ValueError('original stage trace does not reproduce actual source forward/loss BYTE_EXACT')
    loss.backward();out['loss']=loss.detach().clone()
    for name,value in net.named_parameters():
        if value.grad is not None:
            if not torch.equal(value.grad,f['first_gradient_'+name]):raise ValueError('original stage trace does not reproduce actual source gradient BYTE_EXACT:'+name)
            save('parameter_gradient_'+name,value.grad)
    for name,value in net.state_dict().items():
        if 'running_' in name:
            if not torch.equal(value,f['first_state_'+name]):raise ValueError('original population state reproduction differs:'+name)
            out['state_'+name]=value.detach().clone()
        elif not torch.equal(value,initial[name]):raise ValueError('original diagnostic mutated a parameter:'+name)
    if set(native)!=set(out):raise ValueError('stage keysets differ:'+str(set(native)^set(out)))
    comparisons={};layouts={}
    for key in sorted(out):
        a,b=native[key],out[key]
        if key.endswith('_stride'):
            layouts[key]=dict(native=a.tolist(),original=b.tolist(),equal=a.shape==b.shape and torch.equal(a,b));continue
        if a.numel()!=b.numel() or a.dtype!=b.dtype or not torch.isfinite(a).all():raise ValueError('stage element-count/dtype/finite differs:'+key)
        dtype=torch.complex128 if a.is_complex() else torch.float64
        delta=(a.reshape(-1).to(dtype)-b.reshape(-1).to(dtype)).abs()
        comparisons[key]=dict(native_shape=list(a.shape),original_shape=list(b.shape),max_abs=float(delta.max()),rms=float(delta.square().mean().sqrt()),different=int(torch.count_nonzero(a.reshape(-1)!=b.reshape(-1))))
    torch.save(out,args.out.with_suffix('.pt'))
    result=dict(comparisons=comparisons,layouts=layouts,original_actual_production_reproduction='BYTE_EXACT logits/loss/all6gradients/population state',native_reproduction_required_by_native_test='BYTE_EXACT logits/loss/all6gradients/population state',optimizer_updates=0,fit_calls=0,task_accuracy_computed=False,only_existing_source_batch_read=True,
        original_method_sha256=hashlib.sha256(original.encode()).hexdigest(),observed_method_sha256=hashlib.sha256(changed.encode()).hexdigest(),forward_order=['input','fir','spatial_weight','conv','bn_r1','bn_r2','bn_mean','bn_var','bn','activation','segments','logvar','features','head_weight','logits'])
    args.out.write_text(json.dumps(result,indent=2));print('ORIGINAL_SOURCE_STAGE_TRACE_REPRODUCES_PRODUCTION_BYTE_EXACT optimizer_updates=0')


if __name__=='__main__':main()
