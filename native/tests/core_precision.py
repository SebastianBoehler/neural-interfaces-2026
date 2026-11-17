"""Host-Python source-only precision diagnosis, preserving all declared guards."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.jit._pickle import restore_type_tag

BASE=Path('/mnt/lustre/work/gehler/gfh527')
PACKAGE=BASE/'ni26_native353_qualify_v1_20261117/package'
NATIVE=BASE/'ni26_native353_parity_v3_20261117/component_output.pt'
SOURCE=PACKAGE/'source_fixture.pt'
STAGES=('car_mean','car','filtered64','filtered32','analytic','time_mean','covariance',
        'log_covariance','reference_log','whitening','tangent','slow64','slow','feature',
        'standardized','logits','probabilities')


def load(path):
    with torch.serialization.safe_globals([restore_type_tag]):return torch.load(path,weights_only=True)


def save(path,values):
    torch.save({**{k:torch.from_numpy(np.array(v,copy=True)) for k,v in values.items()},'native_format':torch.tensor(1)},path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['reference','prepare_means','report']);args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode=='reference':
        reference,native=load(SOURCE),load(NATIVE);overall={}
        for key in ('p1','p2','features','tail','reapply'):
            a,b=native[key],reference[key]
            if a.shape!=b.shape or not torch.isfinite(a).all():raise ValueError(f'invalid output {key}')
            error=float(abs(a-b).max());limit=1e-4 if key=='features' else 1e-7
            changes=None if key=='features' else int((a.argmax(-1)!=b.argmax(-1)).sum())
            overall[key]=dict(max_abs=error,limit=limit,changed_labels=changes,passed=error<=limit and not changes)
            print(key,json.dumps(overall[key]),flush=True)
        Path('overall.json').write_text(json.dumps(overall,indent=2))
        official=BASE/'neural-interfaces-2026/2026-competition/tracks/bci_decoding';sys.path.insert(0,str(official))
        spec=importlib.util.spec_from_file_location('core_precision_frozen',PACKAGE/'submission.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with np.load(PACKAGE/'weights.npz',allow_pickle=False) as z:parameters={k:z[k] for k in z.files}
        raw=reference['x'][0].numpy();values={}
        for stem,output in [('ts_','p1'),('ts2_','p2')]:
            z=module.with_prefix(parameters,stem);core=module.TangentCore(z)
            mean=raw.mean(0,keepdims=True);car=raw-mean
            b64=np.stack([module.filtfilt(*f,car) for f in core.bank]);b32=b64.astype(np.float32)
            band=module.hilbert(b32,axis=-1).astype(np.complex64) if core.hilbert else b32
            means,covariances=[],[]
            for segment in [band]+np.array_split(band,core.nsub,axis=-1):
                center=segment.mean(-1,keepdims=True);means.append(center)
                centered=(segment-center).astype(np.complex128 if core.hilbert else np.float64)
                covariance=centered@(centered.conj() if core.hilbert else centered).transpose(0,2,1)/segment.shape[-1]
                trace=np.trace(covariance,axis1=1,axis2=2)[:,None,None].real/41
                covariances.append((1-core.shrink)*covariance+core.shrink*trace*np.eye(41))
            cov=np.stack(covariances,1).reshape(-1,41,41)
            log=module.eig_apply_batch(cov,module.log_floor);ref=(core.k*core.prior+log)/(core.k+1)
            whitening=module.eig_apply_batch(ref,lambda w:np.exp(-.5*w))
            tangent=module.eig_apply_batch(whitening@cov@whitening,module.log_floor)
            iu=np.triu_indices(41);v=tangent.real[:,iu[0],iu[1]]*np.where(iu[0]==iu[1],1.,np.sqrt(2.))
            if core.hilbert:
                ju=np.triu_indices(41,1);v=np.concatenate([v,np.sqrt(2.)*tangent.imag[:,ju[0],ju[1]]],1)
            slow64=module.sosfiltfilt(core.slow_sos,car,axis=-1)[...,core.slow_step//2::core.slow_step].reshape(-1)
            slow=slow64.astype(np.float32).astype(np.float64);feature=np.concatenate([v.reshape(-1),slow-slow/(core.k+1)])
            standardized=(feature-core.mu)/core.sd;logits=core.W@standardized+core.b;probability=module.softmax(logits)
            actual_p,actual_f=core.step(raw)
            if not np.array_equal(probability,actual_p) or np.max(np.abs(actual_p-reference[output][0].numpy()))>1e-12:raise ValueError('diagnostic reference does not reproduce original first window')
            if stem=='ts_' and not np.array_equal(standardized,actual_f):raise ValueError('diagnostic source features changed')
            arrays=[mean,car,b64,b32,band,np.stack(means,1),cov,log,ref,whitening,tangent,slow64,slow,feature,standardized,logits,probability]
            values.update({stem+k:v for k,v in zip(STAGES,arrays)})
        save('reference_stages.pt',values)
    elif args.mode=='prepare_means':
        reference=load('reference_stages.pt');native=load('native_stages.pt')
        values={k:v.numpy() for k,v in reference.items() if k!='native_format'}
        for stem in ('ts_','ts2_'):
            for case in ('native','refcar'):
                b=native[stem+case+'_analytic'].numpy()
                values[stem+case+'_np_time_mean']=np.stack([s.mean(-1,keepdims=True) for s in [b]+np.array_split(b,2,axis=-1)],1)
        save('mean_oracles.pt',values)
    else:
        reference,native,counter=load('reference_stages.pt'),load('native_stages.pt'),load('mean_stages.pt')
        fixture,component=load(SOURCE),load(NATIVE);results={}
        for stem,output in [('ts_','p1'),('ts2_','p2')]:
            native_replay_error=float(abs(native[stem+'native_probabilities']-component[output][0]).max())
            if native_replay_error>1e-12:raise ValueError('diagnostic C++ does not reproduce captured first window')
            stages={}
            for key in STAGES:
                a,b=native[stem+'native_'+key],reference[stem+key]
                stages[key]=dict(max_abs=float(abs(a-b).max()),exact=bool(torch.equal(a,b)))
            interventions={}
            for name,bank in [('native',native),('refcar',native),('npmean',counter),('both',counter)]:
                a=bank[stem+name+'_probabilities'];b=reference[stem+'probabilities']
                interventions[name]=dict(max_probability_error=float(abs(a-b).max()),changed_labels=int(a.argmax()!=b.argmax()))
            results[stem]=dict(stages=stages,interventions=interventions,native_replay_error=native_replay_error)
            print(stem,json.dumps(results[stem]),flush=True)
        source_sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        report=dict(overall=json.loads(Path('overall.json').read_text()),first_window=results,
                    host_python_fixture=True,clean_SIF_worker_claim=False,no_labels_accuracy_or_fit=True,
                    fixture_sha256=source_sha(SOURCE),native_output_sha256=source_sha(NATIVE))
        Path('result.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
