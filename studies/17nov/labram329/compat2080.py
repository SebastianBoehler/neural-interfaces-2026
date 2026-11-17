"""Separate CPU/2080Ti numerical readiness experiment; never an A100 substitute (§329)."""
import copy
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parent
ORIGINAL=Path('/mnt/lustre/work/gehler/gfh527/ni26_labram329_v3_20261117')
ORIGINAL_SHA='936aecfa1040d3c75c767feef448da3f17662b5f908587b59d8a2405ec21118f'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    if Path(value.__file__).resolve()!=path.resolve():raise ValueError('modulepath differs')
    return value


def main():
    started=time.monotonic();torch.set_num_threads(6)
    original=module('ni26_atom329_original',ORIGINAL/'preflight.py')
    sha,fetch=original.sha,original.fetch
    if sha(ORIGINAL/'source.sha256')!=ORIGINAL_SHA:raise ValueError('A100source identity differs')
    for directory in (ROOT,ORIGINAL):
        for line in (directory/'source.sha256').read_text().splitlines():
            expected,name=line.split(maxsplit=1)
            if sha(directory/name)!=expected:raise ValueError('immutable source changed')
    if sha(original.DATA)!=original.DATA_SHA:raise ValueError('official tensor changed')
    assets=ROOT/'assets';assets.mkdir(exist_ok=False)
    hf=f'https://huggingface.co/Pan-Junyu/LaBraM-at-Scale/resolve/{original.WEIGHT_REV}/'
    code=f'https://raw.githubusercontent.com/AlbertMentat/LaBraM-at-Scale/{original.CODE_REV}/'
    for base,name,target in ((hf,'metadata/atom-large.weights.json','weights.json'),(hf,'LICENSE','LICENSE-weights'),
        (code,'LICENSE','LICENSE-code'),(code,'THIRD_PARTY_NOTICES.md','THIRD_PARTY_NOTICES.md'),
        (code,'third_party/LaBraM-LICENSE.txt','LaBraM-LICENSE.txt'),(code,'models/labram.py','labram.py'),
        (hf,'atom/atom-large.safetensors','atom-large.safetensors')):fetch(base+name,assets/target)
    meta=json.loads((assets/'weights.json').read_text())
    if meta['license']!='MIT' or meta['weights_scope']!='Pretrained encoder and pretraining heads.':raise ValueError('license/lineage differs')
    if sha(assets/'atom-large.safetensors')!=original.WEIGHT_SHA or (assets/'atom-large.safetensors').stat().st_size!=212001576:
        raise ValueError('actual weight hash/bytes differ')
    for name in ('LICENSE-code','LICENSE-weights'):
        if not (assets/name).read_text().startswith('MIT License'):raise ValueError('actual MITtext missing')
    wheel='timm-0.9.16-py3-none-any.whl'
    fetch('https://files.pythonhosted.org/packages/68/99/2018622d268f6017ddfa5ee71f070bad5d07590374793166baa102849d17/'+wheel,assets/wheel)
    if sha(assets/wheel)!='bf5704014476ab011589d3c14172ee4c901fd18f9110a928019cac5be2945914':raise ValueError('timmwheel changed')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(assets/'runtime'),str(assets/wheel)],check=True)
    sys.path.insert(0,str(assets/'runtime'));upstream=module('ni26_atom329_compat_upstream',assets/'labram.py')
    with np.load(original.DATA,allow_pickle=False) as z:
        people=np.array([str(s).split('/')[-1] for s in z['subject']]);rows=np.flatnonzero(people=='P04')[:3]
        raw=z['X'][rows];names=z['ch_names'].astype(str).tolist()
    normalized=[n.upper().replace('EEG-','').replace('EEG ','').replace('-REF','').replace('-LE','').strip() for n in names]
    if any(n not in upstream._STD1020_UPPER2IDX for n in normalized):raise ValueError('unknown channel')
    chans=upstream.get_input_chans(names)
    if len(chans)!=42 or len(set(chans))!=42:raise ValueError('collapsed channelmapping')
    x=resample_poly(raw.astype(np.float64),5,3,axis=-1).astype(np.float32)/100
    if x.shape!=(3,41,800):raise ValueError('inputshape differs')
    tensor=torch.from_numpy(x.reshape(3,41,4,200))
    weight=load_file(str(assets/'atom-large.safetensors'));banks={};comparisons={};removed=[]
    if not torch.cuda.is_available() or '2080 Ti' not in torch.cuda.get_device_name(0):raise ValueError('explicit2080Ti route required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.enable_cudnn_sdp(False)
    for prior in ('pretrained','random'):
        torch.manual_seed(329);model=upstream.labram_large_patch200_200(num_classes=0,init_values=.1,use_mean_pooling=False).eval();del model.logit_scale
        if prior=='pretrained':
            own=model.state_dict();mapped={}
            for name,value in weight.items():
                k=name[8:] if name.startswith('student.') else name
                if k in own:mapped[k]=value
                elif name.startswith(('lm_head.','projection_head.','student.lm_head.','student.mask_token')):removed.append(name)
                else:raise ValueError('unexpected checkpointkey:'+name)
            if set(mapped)!=set(own):raise ValueError('missingencoderkeys')
            model.load_state_dict(mapped,strict=True)
        before=original.state_sha(model)
        with torch.no_grad():cpu=model.forward_features(tensor,input_chans=chans,return_patch_tokens=True).numpy()
        model.cuda();xt=tensor.cuda()
        with torch.no_grad():
            gpu=model.forward_features(xt,input_chans=chans,return_patch_tokens=True).cpu().numpy()
            if not np.allclose(cpu,gpu,atol=2e-3,rtol=1e-3):raise ValueError('CPU2080numericalparity failed')
            rng=np.random.default_rng(329);head=rng.normal(size=(400,3));cp=cpu.mean(1)@head;gp=gpu.mean(1)@head
            margin=np.sort(cp,axis=1)[:,-1]-np.sort(cp,axis=1)[:,-2]
            if np.any((cp.argmax(1)!=gp.argmax(1))&(margin>1e-6)):raise ValueError('nontiedheadargmax differs')
            a=model.forward_features(xt[:1],input_chans=chans,return_patch_tokens=True)
            model.forward_features(xt[1:2],input_chans=chans,return_patch_tokens=True)
            if not torch.equal(a,model.forward_features(xt[:1],input_chans=chans,return_patch_tokens=True)):raise ValueError('statelessinterleavingfailed')
            replica=copy.deepcopy(model)
            if not torch.equal(a,replica.forward_features(xt[:1],input_chans=chans,return_patch_tokens=True)):raise ValueError('deepcopyfailed')
            del replica
            amp=model.forward_features(2*xt[:1],input_chans=chans,return_patch_tokens=True)
            if torch.equal(a,amp):raise ValueError('amplitudepath inactive')
            times=[]
            for _ in range(35):
                torch.cuda.synchronize();t=time.perf_counter();model.forward_features(xt[:1],input_chans=chans,return_patch_tokens=True);torch.cuda.synchronize();times.append(time.perf_counter()-t)
        g=xt[:1].clone().requires_grad_(True);model.forward_features(g,input_chans=chans,return_patch_tokens=True).square().sum().backward()
        if not torch.isfinite(g.grad).all() or not torch.any(g.grad[..., -1]!=0):raise ValueError('lastsamplegradientfailed')
        if original.state_sha(model)!=before:raise ValueError('frozenstatechanged')
        banks[prior+'_cpu']=cpu;banks[prior+'_2080']=gpu
        comparisons[prior]=dict(max_abs=float(np.max(np.abs(cpu-gpu))),nontied_head_changes=int(np.sum((cp.argmax(1)!=gp.argmax(1))&(margin>1e-6))),
            B1_median_ms=float(1000*np.median(times[5:])),B1_p90_ms=float(1000*np.quantile(times[5:],.9)),
            amplitude_max_abs=float((a-amp).abs().max()),encoder_state_sha256=before)
        del model
    if comparisons['pretrained']['B1_median_ms']>100:raise ValueError('2080sourcepilot budgetnotcredible above100msB1')
    np.savez(ROOT/'cpu2080_embeddings.npz',rows=rows,input=x,**banks)
    result=dict(passed=True,labels_opened=False,device=torch.cuda.get_device_name(0),compared_devices=['CPU','2080Ti'],A100_verified=False,
        comparisons=comparisons,channel_indices=chans,source_rows=rows.tolist(),removed_pretraining_keys=removed,
        source_sha256=sha(ROOT/'source.sha256'),weight_sha256=original.WEIGHT_SHA,actual_weight_bytes=212001576,
        code_revision=original.CODE_REV,weight_revision=original.WEIGHT_REV,module_sha256=sha(assets/'labram.py'),
        embedding_artifact_sha256=sha(ROOT/'cpu2080_embeddings.npz'),original_A100_source_sha256=ORIGINAL_SHA,
        attention_backend='explicit PyTorch SDPA MATH only;Flash/Efficient/cuDNNdisabled',TF32=False,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
        limits='CPU2080readinessonly;A100verificationpending;nosource/outer/workeraccuracy')
    with (ROOT/'compatibility.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    import source2080
    source2080.main()


if __name__=='__main__':main()
