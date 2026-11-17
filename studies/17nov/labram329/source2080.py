"""Frozen Atom-L prior/readout/raw-information factorial after passed §329 engineering."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
from safetensors.torch import load_file
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
PREFLIGHT=ROOT
VISUAL=Path('/mnt/lustre/work/gehler/gfh527/ni26_visual324_v2_20261117')
MICRO=Path('/mnt/lustre/work/gehler/gfh527/ni26_microstate328_v2_20261117')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    if Path(result.__file__).resolve()!=path.resolve():raise ValueError('modulepath differs')
    return result


def main():
    started=time.monotonic();torch.set_num_threads(6)
    guard=json.loads((PREFLIGHT/'compatibility.json').read_text())
    if not guard['passed'] or guard['labels_opened'] or guard['source_sha256']!=sha(PREFLIGHT/'source.sha256'):
        raise ValueError('passed exact preflight required')
    for directory in (ROOT,PREFLIGHT,MICRO):
        for line in (directory/'source.sha256').read_text().splitlines():
            expected,name=line.split(maxsplit=1)
            if sha(directory/name)!=expected:raise ValueError('immutable code changed')
    if sha(VISUAL/'baseline.py')!='f01cac4339a42e6c8c10fab6c6f56a71139127f056ad8246aa1554345edcdbc7' or sha(VISUAL/'shared.py')!='ae8d03f12d5a564eb88d79fd2b0cc5d146849b34d9b83b6932049a58ea25589a':
        raise ValueError('unchanged raw reference wrapper differs')
    assets=PREFLIGHT/'assets'
    if sha(assets/'atom-large.safetensors')!=guard['weight_sha256'] or sha(assets/'labram.py')!=guard['module_sha256']:
        raise ValueError('passed assets changed')
    sys.path[:0]=[str(assets/'runtime'),str(VISUAL)]
    import shared
    if sha(shared.BASE/'source.sha256')!=shared.BASE_SHA:raise ValueError('raw reference dependency source differs')
    for line in (shared.BASE/'source.sha256').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        if sha(shared.BASE/name)!=expected:raise ValueError('raw reference dependency changed')
    upstream=module('ni26_atom329_frozen_upstream',assets/'labram.py')
    representation=module('ni26_micro328_raw',MICRO/'representation.py')
    d,fit,query,people=shared.load_data();rows=np.r_[fit,query];n=len(fit)
    chans=guard['channel_indices']
    raw=d['X'][rows];scale=float(np.sqrt(np.mean(raw[:n].astype(np.float64)**2)))
    rawblock=representation.raw287(raw,scale)
    if np.array_equal(rawblock[:1],representation.raw287(raw[:1]*2,scale)):raise ValueError('raw amplitude sidechannel inactive')
    if not torch.cuda.is_available() or '2080 Ti' not in torch.cuda.get_device_name(0):raise ValueError('explicit2080route required')
    if torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled() or torch.backends.cuda.cudnn_sdp_enabled() or not torch.backends.cuda.math_sdp_enabled():raise ValueError('SDPA MATH backendguard failed')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:raise ValueError('TF32mustbeoff')
    banks={'raw287':rawblock};weights=load_file(str(assets/'atom-large.safetensors'),device='cpu')
    for prior in ('pretrained','random'):
        torch.manual_seed(329)
        model=upstream.labram_large_patch200_200(num_classes=0,init_values=.1,use_mean_pooling=False).eval()
        del model.logit_scale
        if prior=='pretrained':
            own=model.state_dict();mapped={}
            for name,value in weights.items():
                k=name[8:] if name.startswith('student.') else name
                if k in own:mapped[k]=value
                elif name not in guard['removed_pretraining_keys']:raise ValueError('unexpected nonencoder key')
            if set(mapped)!=set(own):raise ValueError('encoderload incomplete')
            model.load_state_dict(mapped,strict=True)
        model.cuda();before=shared.state_sha(model);pooled=[];retained=[]
        with torch.no_grad():
            for start in range(0,len(raw),8):
                x=resample_poly(raw[start:start+8].astype(np.float64),5,3,axis=-1).astype(np.float32)/100
                tensor=torch.from_numpy(x.reshape(-1,41,4,200)).cuda()
                output=model.forward_features(tensor,input_chans=chans,return_patch_tokens=True)
                if output.shape!=(len(tensor),164,400):raise ValueError('wrong patchstructure')
                pooled.append(output.mean(1).cpu().numpy())
                retained.append(output.reshape(-1,41,4,400).mean(1).flatten(1).cpu().numpy())
            # Prefix/batch parity on real source rows, prior-specific frozen state.
            tensor=torch.from_numpy((resample_poly(raw[:1].astype(np.float64),5,3,axis=-1).astype(np.float32)/100).reshape(1,41,4,200)).cuda()
            single=model.forward_features(tensor,input_chans=chans,return_patch_tokens=True).mean(1).cpu().numpy()
            if not np.allclose(single,pooled[0][:1],atol=2e-4,rtol=1e-4):raise ValueError('B1/batched feature parity failed')
        if shared.state_sha(model)!=before:raise ValueError('frozen state changed')
        for kind,parts in (('pooled',pooled),('retained',retained)):
            bank=np.concatenate(parts);banks[f'{prior}_{kind}']=bank
            banks[f'{prior}_{kind}_raw']=np.c_[bank,rawblock]
            if not np.array_equal(banks[f'{prior}_{kind}_raw'][:,-287:],rawblock):raise ValueError('rawfactor differs')
        del model
    if any(not np.isfinite(b).all() for b in banks.values()):raise ValueError('nonfinitebank')
    yfit=shared.labels(fit,len(d['X']));models={};fit_quality={}
    np.savez(ROOT/'source_banks.npz',source_rows=fit,query_rows=query,**banks)
    for name,bank in banks.items():
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=2000))
        m.fit(bank[:n],yfit)
        if int(m[-1].n_iter_.max())>=2000:raise ValueError('fixedhead didnotconverge')
        models[name]=m
        pfit=m.predict_proba(bank[:n]);pred=pfit.argmax(1)
        fit_quality[name]=dict(iterations=int(m[-1].n_iter_.max()),source_train_logloss=float(log_loss(yfit,pfit,labels=[0,1,2])),
            source_train_accuracy=float(np.mean(pred==yfit)),source_class_recalls=[float(np.mean(pred[yfit==c]==c)) for c in range(3)],
            source_probability_sd=np.std(pfit,axis=0).tolist())
    cpu_reference=np.load(ROOT/'cpu2080_embeddings.npz');reference_rows=cpu_reference['rows']
    positions=np.array([int(np.flatnonzero(rows==r)[0]) for r in reference_rows])
    if np.any(positions>=n):raise ValueError('CPUreadout reference must be fitrows')
    readout_errors={}
    for name,m in models.items():
        if name=='raw287':cpu_features=rawblock[positions]
        else:
            prior='pretrained' if name.startswith('pretrained') else 'random'
            patch=cpu_reference[prior+'_cpu']
            cpu_features=patch.mean(1) if '_pooled' in name else patch.reshape(-1,41,4,400).mean(1).reshape(-1,1600)
            if name.endswith('_raw'):cpu_features=np.c_[cpu_features,rawblock[positions]]
        pcpu=m.predict_proba(cpu_features);pgpu=m.predict_proba(banks[name][positions])
        margin=np.sort(pcpu,axis=1)[:,-1]-np.sort(pcpu,axis=1)[:,-2]
        err=float(np.max(np.abs(pcpu-pgpu)));changes=int(np.sum((pcpu.argmax(1)!=pgpu.argmax(1))&(margin>1e-6)))
        if err>1e-3 or changes:raise ValueError('actualfittedCPU2080readoutguard failed:'+name)
        readout_errors[name]=dict(probability_max_abs=err,nontied_argmax_changes=changes)
    (ROOT/'fitted_readout_guard.json').write_text(json.dumps(readout_errors,indent=2))
    from baseline import predict
    raw_reference,reference_meta=predict(d,fit,query,yfit)
    # All representations, sourcefits and rawreference must complete before either query labels open.
    yquery=shared.labels(query,len(d['X']))
    reader=module('ni26_atom329_labels',shared.LABELS)
    interface=reader.read_selected(shared.DATA,'interface',query,len(d['X']))
    sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'2026-competition')]
    import protocol
    if Path(protocol.__file__).resolve()!=ROOT/'experiments/protocol.py':raise ValueError('metricprotocol differs')
    metadata=dict(stream=d['stream'][query],interface=interface)
    def score(pred):
        return {person:protocol.score(yquery[people[query]==person],pred[people[query]==person],metadata,people[query]==person) for person in ('P02','P03')}
    results={name:dict(people=score(m.predict(banks[name][n:])),dimensions=banks[name].shape[1],feature_sha256=shared.digest(banks[name])) for name,m in models.items()}
    reference={order:score(p.argmax(1)) for order,p in raw_reference.items()}
    result=dict(source_only=True,device=torch.cuda.get_device_name(0),attention_backend='SDPA MATH only',TF32=False,A100_verified=False,CPU2080_fitted_readout=readout_errors,P01_labels_opened=False,fit_people=sorted(set(people[fit])),previously_exposed_queries=['P02','P03'],
        source_sha256=sha(ROOT/'source.sha256'),preflight_source_sha256=guard['source_sha256'],raw_reference_metadata=reference_meta,
        results=results,source_fit_quality=fit_quality,source_banks_sha256=sha(ROOT/'source_banks.npz'),raw_reference_by_order=reference,seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limits='no outer/current-system/sealedgain;upstream participant exposure unknown;all heads prospectively fixedC')
    with (ROOT/'source_result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
