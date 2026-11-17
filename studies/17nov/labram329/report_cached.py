"""Report-only continuation of completed329features/heads after guarded metric-import failure."""
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/lustre/work/gehler/gfh527/ni26_labram329_A100source_v1_20261117')
SOURCE_SHA='f5dcaf520fb65d5587cf5a4ad3373c22e35bd4761e1ee44f64c9eeb9792a84be'
BANK_SHA='524610d6f21633a2e93cda04007dc82996215ef60a0f14a0bbd28edd9ca2541a'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


def main():
    started=time.monotonic();torch.set_num_threads(6)
    sys.path[:0]=[str(SOURCE/'assets/runtime'),'/mnt/lustre/work/gehler/gfh527/ni26_visual324_v2_20261117']
    import shared
    if shared.sha(SOURCE/'source.sha256')!=SOURCE_SHA or shared.sha(SOURCE/'source_banks.npz')!=BANK_SHA:raise ValueError('completedsource/bankidentityfailed')
    for folder in (ROOT,SOURCE,shared.BASE):
        for line in (folder/'source.sha256').read_text().splitlines():
            expected,name=line.split(maxsplit=1)
            if shared.sha(folder/name)!=expected:raise ValueError('immutabledependencychanged')
    guard=json.loads((SOURCE/'compatibility.json').read_text());readout=json.loads((SOURCE/'fitted_readout_guard.json').read_text())
    if not guard['passed'] or guard['labels_opened'] or len(readout)!=9 or any(v['probability_max_abs']>1e-3 or v['nontied_argmax_changes'] for v in readout.values()):raise ValueError('numericalguardfailed')
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('sameA100route required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.enable_cudnn_sdp(False)
    d,fit,query,people=shared.load_data();n=len(fit);yfit=shared.labels(fit,len(d['X']))
    with np.load(SOURCE/'source_banks.npz') as z:
        if not np.array_equal(z['source_rows'],fit) or not np.array_equal(z['query_rows'],query):raise ValueError('sourcebankrowschanged')
        banks={name:z[name] for name in z.files if name not in ('source_rows','query_rows')}
    if len(banks)!=9:raise ValueError('complete9banksrequired')
    models={};fit_quality={}
    for name,bank in banks.items():
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=2000));m.fit(bank[:n],yfit)
        if int(m[-1].n_iter_.max())>=2000:raise ValueError('fixedsourceheadconvergencefailed')
        p=m.predict_proba(bank[:n]);pred=p.argmax(1);models[name]=m
        np.savez(ROOT/(name+'_fitted_head.npz'),coef=m[-1].coef_,intercept=m[-1].intercept_,classes=m[-1].classes_,
            scaler_mean=m[0].mean_,scaler_scale=m[0].scale_,iterations=m[-1].n_iter_,fit_rows=fit)
        fit_quality[name]=dict(iterations=int(m[-1].n_iter_.max()),source_train_logloss=float(log_loss(yfit,p,labels=[0,1,2])),
            source_class_recalls=[float(np.mean(pred[yfit==c]==c)) for c in range(3)],source_train_accuracy=float(np.mean(pred==yfit)))
    cpu_reference=np.load(SOURCE/'cpu_a100_embeddings.npz');positions=np.array([int(np.flatnonzero(fit==r)[0]) for r in cpu_reference['rows']]);actual_readouts={}
    for name,m in models.items():
        if name=='raw287':cpu_features=banks['raw287'][positions]
        else:
            prior='pretrained' if name.startswith('pretrained') else 'random';patch=cpu_reference[prior+'_cpu']
            cpu_features=patch.mean(1) if '_pooled' in name else patch.reshape(-1,41,4,400).mean(1).reshape(-1,1600)
            if name.endswith('_raw'):cpu_features=np.c_[cpu_features,banks['raw287'][positions]]
        a=m.predict_proba(cpu_features);b=m.predict_proba(banks[name][positions]);error=float(np.max(np.abs(a-b)))
        margin=np.sort(a,axis=1)[:,-1]-np.sort(a,axis=1)[:,-2];changes=int(np.sum((a.argmax(1)!=b.argmax(1))&(margin>1e-6)))
        if error>1e-3 or changes:raise ValueError('canonicalrefittedreadoutCPU_A100guardfailed')
        actual_readouts[name]=dict(probability_max_abs=error,nontied_argmax_changes=changes)
    (ROOT/'canonical_fitted_readout_guard.json').write_text(json.dumps(actual_readouts,indent=2))
    from baseline_saved import predict
    baseline,baseline_meta=predict(d,fit,query,yfit,ROOT/'raw_reference_checkpoint.pt')
    np.savez(ROOT/'raw_reference.npz',index=query,**baseline)
    query_probabilities={name:m.predict_proba(banks[name][n:]) for name,m in models.items()}
    np.savez(ROOT/'query_probabilities.npz',index=query,**query_probabilities)
    (ROOT/'source_quality.json').write_text(json.dumps(fit_quality,indent=2))
    (ROOT/'recovered_artifacts.json').write_text(json.dumps({p.name:shared.sha(p) for p in ROOT.glob('*') if p.suffix in ('.npz','.pt')},indent=2))
    sys.path.insert(0,str(ROOT/'2026-competition'))
    protocol=module('ni26_atom329_report_protocol',ROOT/'experiments/protocol.py')
    if Path(protocol.__file__).resolve()!=ROOT/'experiments/protocol.py':raise ValueError('namedprotocolpathdiffers')
    metric_path=Path(sys.modules[protocol.balanced_accuracy.__module__].__file__)
    if shared.sha(metric_path)!=shared.sha(ROOT/'2026-competition/benchmark_utils/metrics.py'):raise ValueError('actualmetriccodeidentityfailed')
    yquery=shared.labels(query,len(d['X']));reader=module('ni26_atom329_report_labels',shared.LABELS)
    interface=reader.read_selected(shared.DATA,'interface',query,len(d['X']))
    metadata=dict(stream=d['stream'][query],interface=interface)
    def score(pred):return {person:protocol.score(yquery[people[query]==person],pred[people[query]==person],metadata,people[query]==person) for person in ('P02','P03')}
    results={name:dict(people=score(query_probabilities[name].argmax(1)),dimensions=banks[name].shape[1],feature_sha256=shared.digest(banks[name])) for name,m in models.items()}
    result=dict(source_only=True,P01_labels_opened=False,device=torch.cuda.get_device_name(0),attention_backend='SDPA MATH only',TF32=False,
        source_sha256=shared.sha(ROOT/'source.sha256'),producer_source_sha256=SOURCE_SHA,source_banks_sha256=BANK_SHA,
        results=results,source_fit_quality=fit_quality,raw_reference_by_order={order:score(p.argmax(1)) for order,p in baseline.items()},
        raw_reference_metadata=baseline_meta,CPU_A100_fitted_readout=actual_readouts,seconds=time.monotonic()-started,
        limits='canonicalrecoveredfixedhead/rawreferenceinstrument;identitytolostprocessoutputsnotclaimed;twoexposedpeople;noouter/current-systemgain')
    with (ROOT/'source_result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
