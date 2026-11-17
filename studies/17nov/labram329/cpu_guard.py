"""Real cached-asset CPU import/strict-load/input/gradient guard before329repair release."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parent
ASSETS=Path('/mnt/lustre/work/gehler/gfh527/ni26_labram329_2080_v1_20261117/assets')
DATA=Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/data/proteus_windows.npz')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(6)
    if sha(ASSETS/'atom-large.safetensors')!='a28e97f7bd73bd7ff55f3b5d680f48807c335cf3134d1281228a061056aaf52b':raise ValueError('actualweightchanged')
    sys.path.insert(0,str(ASSETS/'runtime'))
    name='ni26_atom329_cpu_guard_upstream';spec=importlib.util.spec_from_file_location(name,ASSETS/'labram.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    torch.manual_seed(329);model=module.labram_large_patch200_200(num_classes=0,init_values=.1,use_mean_pooling=False).eval();del model.logit_scale
    own=model.state_dict();mapped={};removed=[]
    for name,value in load_file(str(ASSETS/'atom-large.safetensors')).items():
        k=name[8:] if name.startswith('student.') else name
        if k in own:mapped[k]=value
        elif name.startswith(('lm_head.','projection_head.','student.lm_head.','student.mask_token')):removed.append(name)
        else:raise ValueError('unexpectedkey:'+name)
    if set(mapped)!=set(own):raise ValueError('missingencoderkeys:'+str(sorted(set(own)-set(mapped))))
    model.load_state_dict(mapped,strict=True)
    with np.load(DATA,allow_pickle=False) as z:
        people=np.array([str(s).split('/')[-1] for s in z['subject']]);rows=np.flatnonzero(people=='P04')[:3]
        x=resample_poly(z['X'][rows].astype(np.float64),5,3,axis=-1).astype(np.float32)/100;names=z['ch_names'].astype(str).tolist()
    chans=module.get_input_chans(names)
    if x.shape!=(3,41,800) or len(set(chans))!=42:raise ValueError('input/channelmapfailed')
    value=torch.from_numpy(x.reshape(3,41,4,200)).requires_grad_(True)
    output=model.forward_features(value,input_chans=chans,return_patch_tokens=True)
    if output.shape!=(3,164,400) or not torch.isfinite(output).all():raise ValueError('realforwardfailed')
    torch.manual_seed(329);head=torch.randn(400,3);loss=(output.mean(1)@head).square().mean();loss.backward()
    if not torch.isfinite(value.grad).all() or not torch.any(value.grad[..., -1]!=0):raise ValueError('realgradientfailed')
    result=dict(passed=True,labels_opened=False,source_rows=rows.tolist(),output_shape=list(output.shape),
        removed_pretraining_keys=removed,loaded_encoder_keys=len(mapped),input_gradient_max=float(value.grad.abs().max()),
        module_sha256=sha(ASSETS/'labram.py'),weight_sha256=sha(ASSETS/'atom-large.safetensors'),
        fixed_import='sys.modules[spec.name]=module before exec_module;actualtimmregistrationworks')
    with (ROOT/'cpu_guard.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
