"""Pinned public Atom-L assets and label-free real-window A100 readiness (§329)."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
CODE_REV='59e4e7ea2d855cda5573c312b380b109f1856764'
WEIGHT_REV='72189644cef9a91b7b436c5f37f083c1366e2f21'
WEIGHT_SHA='a28e97f7bd73bd7ff55f3b5d680f48807c335cf3134d1281228a061056aaf52b'
DATA=Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/data/proteus_windows.npz')
DATA_SHA='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def fetch(url,path):
    if path.exists():raise ValueError('fresh preflight assets already exist')
    with urllib.request.urlopen(url,timeout=60) as src,path.open('xb') as out:
        while True:
            block=src.read(1<<20)
            if not block:break
            out.write(block)


def state_sha(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    started=time.monotonic();torch.set_num_threads(6)
    cpu_path=Path('/mnt/lustre/work/gehler/gfh527/ni26_atom329_cpuguard_v1_20261117/cpu_guard.json')
    cpu=json.loads(cpu_path.read_text())
    if not cpu['passed'] or cpu['labels_opened'] or cpu['loaded_encoder_keys']!=425:raise ValueError('actualCPUstrictloadguardrequired')
    if sha(DATA)!=DATA_SHA:raise ValueError('official tensor changed')
    for line in (ROOT/'source.sha256').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        if sha(ROOT/name)!=expected:raise ValueError('source changed')
    ASSETS.mkdir(exist_ok=False)
    base=f'https://huggingface.co/Pan-Junyu/LaBraM-at-Scale/resolve/{WEIGHT_REV}/'
    fetch(base+'metadata/atom-large.weights.json',ASSETS/'weights.json')
    fetch(base+'LICENSE',ASSETS/'LICENSE-weights')
    meta=json.loads((ASSETS/'weights.json').read_text())
    if meta['license']!='MIT' or meta['weights_scope']!='Pretrained encoder and pretraining heads.':raise ValueError('weight license/lineage differs')
    code=f'https://raw.githubusercontent.com/AlbertMentat/LaBraM-at-Scale/{CODE_REV}/'
    fetch(code+'LICENSE',ASSETS/'LICENSE-code')
    fetch(code+'THIRD_PARTY_NOTICES.md',ASSETS/'THIRD_PARTY_NOTICES.md')
    fetch(code+'third_party/LaBraM-LICENSE.txt',ASSETS/'LaBraM-LICENSE.txt')
    fetch(code+'models/labram.py',ASSETS/'labram.py')
    for name in ('LICENSE-code','LICENSE-weights'):
        if not (ASSETS/name).read_text().startswith('MIT License'):raise ValueError('actual MIT license missing')
    fetch(base+'atom/atom-large.safetensors',ASSETS/'atom-large.safetensors')
    if (ASSETS/'atom-large.safetensors').stat().st_size!=212001576 or sha(ASSETS/'atom-large.safetensors')!=WEIGHT_SHA:
        raise ValueError('actual checkpoint bytes/hash differ')
    fetch('https://files.pythonhosted.org/packages/68/99/2018622d268f6017ddfa5ee71f070bad5d07590374793166baa102849d17/timm-0.9.16-py3-none-any.whl',ASSETS/'timm-0.9.16-py3-none-any.whl')
    if sha(ASSETS/'timm-0.9.16-py3-none-any.whl')!='bf5704014476ab011589d3c14172ee4c901fd18f9110a928019cac5be2945914':raise ValueError('pinned timm wheel changed')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(ASSETS/'runtime'),str(ASSETS/'timm-0.9.16-py3-none-any.whl')],check=True)
    sys.path.insert(0,str(ASSETS/'runtime'))
    spec=importlib.util.spec_from_file_location('ni26_atom329_upstream',ASSETS/'labram.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=ASSETS/'labram.py':raise ValueError('upstream module path differs')
    with np.load(DATA,allow_pickle=False) as z:
        names=z['ch_names'].astype(str).tolist();people=np.array([str(x).split('/')[-1] for x in z['subject']])
        rows=np.flatnonzero(people=='P04')[:3];raw=z['X'][rows]
    normalized=[n.upper().replace('EEG-','').replace('EEG ','').replace('-REF','').replace('-LE','').strip() for n in names]
    if any(n not in module._STD1020_UPPER2IDX for n in normalized):raise ValueError('unknown channel would silently map to FP1')
    chans=module.get_input_chans(names)
    if len(set(chans))!=42 or len(chans)!=42:raise ValueError('channel map collapses electrodes')
    x=resample_poly(raw.astype(np.float64),5,3,axis=-1).astype(np.float32)/100
    if x.shape!=(3,41,800) or not np.allclose(resample_poly((raw*2).astype(np.float64),5,3,axis=-1)/100,2*x,atol=1e-6):raise ValueError('amplitude/input contract failed')
    sd=load_file(str(ASSETS/'atom-large.safetensors'),device='cpu')
    torch.manual_seed(329);model=module.labram_large_patch200_200(num_classes=0,init_values=.1,use_mean_pooling=False).eval()
    # The release is an encoder exported under student.* plus pretraining decoder/heads.
    keys=list(sd);(ROOT/'checkpoint_keys.json').write_text(json.dumps(keys,indent=2))
    del model.logit_scale  # unused image/text readout scalar, absent from pretrained EEG encoder
    own=model.state_dict();mapped={};removed=[]
    for key,value in sd.items():
        name=key[len('student.'):] if key.startswith('student.') else key
        if name in own:
            if value.shape!=own[name].shape:raise ValueError(f'encoder shape mismatch:{key}')
            mapped[name]=value
        else:removed.append(key)
    if set(mapped)!=set(own):raise ValueError(f'missing encoder keys:{sorted(set(own)-set(mapped))}')
    allowed=('decoder.','decode_task_layer.','lm_head.','projection_head.','mask_token','student.head.','student.lm_head.','student.mask_token','teacher.','head.')
    if any(not key.startswith(allowed) for key in removed):raise ValueError(f'unexpected nonencoder keys:{removed}')
    model.load_state_dict(mapped,strict=True)
    torch.manual_seed(329);random_model=module.labram_large_patch200_200(num_classes=0,init_values=.1,use_mean_pooling=False).eval()
    del random_model.logit_scale
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('registered A100 required;no fallback')
    model.cuda();random_model.cuda();tensor=torch.from_numpy(x.reshape(3,41,4,200)).cuda();before=state_sha(model)
    def forward(value):return model.forward_features(value,input_chans=chans,return_patch_tokens=True)
    with torch.no_grad():
        a=forward(tensor[:1]);b=forward(tensor[1:2]);c=forward(tensor[:1]);amplitude=forward(2*tensor[:1])
        random_out=random_model.forward_features(tensor[:1],input_chans=chans,return_patch_tokens=True)
        if torch.equal(a,random_out) or not torch.isfinite(random_out).all():raise ValueError('pretrained/random path inactive')
        z=(tensor[:1]-tensor[:1].mean((2,3),keepdim=True))/tensor[:1].std((2,3),keepdim=True)
        zscore_out=forward(z)
        if torch.equal(a,zscore_out):raise ValueError('normalization contrast inactive')
        if not torch.equal(a,c) or not torch.isfinite(a).all() or torch.equal(a,amplitude):raise ValueError('state/amplitude path failed')
        replica=copy.deepcopy(model)
        if not torch.equal(a,replica.forward_features(tensor[:1],input_chans=chans,return_patch_tokens=True)):raise ValueError('deepcopy failed')
        del replica
        times=[]
        for _ in range(35):
            torch.cuda.synchronize();t=time.perf_counter();forward(tensor[:1]);torch.cuda.synchronize();times.append(time.perf_counter()-t)
    grad=tensor[:1].clone().requires_grad_(True);forward(grad).square().sum().backward()
    if not torch.isfinite(grad.grad).all() or not torch.any(grad.grad[..., -1]!=0):raise ValueError('active finalsample/input gradient failed')
    if state_sha(model)!=before:raise ValueError('frozen model changed')
    result=dict(passed=True,labels_opened=False,data_sha256=DATA_SHA,source_sha256=sha(ROOT/'source.sha256'),
        code_revision=CODE_REV,weight_revision=WEIGHT_REV,weight_sha256=WEIGHT_SHA,actual_weight_bytes=212001576,
        module_sha256=sha(ASSETS/'labram.py'),source_rows=rows.tolist(),channel_indices=chans,
        tensor_shape=list(tensor.shape),embedding_shape=list(a.shape),removed_pretraining_keys=removed,
        encoder_parameters=sum(p.numel() for p in model.parameters()),B1_median_ms=float(1000*np.median(times[5:])),
        B1_p90_ms=float(1000*np.quantile(times[5:],.9)),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        normalization_embedding_max_abs=float((a-zscore_out).abs().max()),random_embedding_max_abs=float((a-random_out).abs().max()),
        output_contract='pretrained postnorm164patchtokens×400;no learned taskhead;unused image/text logit_scale removed',
        amplitude_embedding_max_abs=float((a-amplitude).abs().max()),total_seconds=time.monotonic()-started,
        limits='no source/outer/current-system accuracy;pretraining participant overlap unknown')
    with (ROOT/'preflight.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
