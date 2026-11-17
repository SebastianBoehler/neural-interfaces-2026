"""Complete eight-arm source-only selection/refit before held P09 labels (§342)."""
import json
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from common import DATA, LABEL, OLD, encoder, gpu, inputs, labels, layouts, load, module, sha, state_sha
from head import Head, predict

ROOT=Path(__file__).resolve().parent
GUARD=Path('/mnt/lustre/work/gehler/gfh527/ni26_spatial342_v1_20261117')


def cell_nll(y, p, stream, interface):
    cells=np.char.add(np.char.add(stream.astype(str),'|'),interface.astype(str))
    losses=-np.log(np.maximum(p[np.arange(len(y)),y],1e-300))
    return float(np.mean([losses[cells==cell].mean() for cell in np.unique(cells)]))


def main():
    started=time.monotonic();torch.set_num_threads(2);gpu()
    for directory in (ROOT,GUARD):
        for line in (directory/'source.sha256').read_text().splitlines():
            expected,name=line.split(maxsplit=1)
            if sha(directory/name)!=expected:
                raise ValueError('immutable source changed')
    smoke=json.loads((GUARD/'gpu_guard.json').read_text())
    if not smoke['passed'] or smoke['P09_labels_opened']:
        raise ValueError('passed engineering smoke required')
    d,people,source,query=load();rows=np.r_[source,query];n=len(source)
    y=labels(source,people)
    inner=people[source]!='P10';cal=~inner
    reader=module('ni26_s342_source_metadata',LABEL)
    interface=reader.read_selected(DATA,'interface',source,len(people))
    streams=np.char.add(np.char.add(d['subject'][source],'|'),d['session'][source])
    head=Head();quality={};params={};posteriors={};parity={};fit_count=0
    if sha(OLD/'source_banks.npz')!='524610d6f21633a2e93cda04007dc82996215ef60a0f14a0bbd28edd9ca2541a':
        raise ValueError('old feature bank identity differs')
    old=np.load(OLD/'source_banks.npz');oldrows=np.r_[old['source_rows'],old['query_rows']]
    oldpositions=np.array([np.flatnonzero(oldrows==row)[0] for row in rows])
    cpu=np.load(OLD/'cpu_a100_embeddings.npz')
    probe_positions=np.array([np.flatnonzero(source==row)[0] for row in cpu['rows']])
    for prior in ('pretrained','random'):
        model,upstream=encoder(prior);chans=upstream.get_input_chans(d['ch_names'].astype(str).tolist())
        model.cuda();before=state_sha(model)
        patch=np.lib.format.open_memmap(ROOT/f'{prior}_patch.npy',mode='w+',dtype=np.float32,shape=(len(rows),164,400))
        with torch.no_grad():
            for start in range(0,len(rows),8):
                patch[start:start+8]=model.forward_features(inputs(d['X'][rows[start:start+8]]).cuda(),
                    input_chans=chans,return_patch_tokens=True).cpu().numpy()
        if state_sha(model)!=before or not np.isfinite(patch).all():
            raise ValueError('frozen encoder/bank guard failed')
        del model;patch.flush()
        banks=layouts(patch);cpu_banks=layouts(cpu[prior+'_cpu'])
        for layout,oldname in (('pooled400','pooled'),('temporal1600','retained')):
            expected=old[f'{prior}_{oldname}'][oldpositions]
            if not np.allclose(banks[layout],expected,atol=2e-4,rtol=1e-4):
                raise ValueError('exact-row old feature parity failed')
        for layout,bank in banks.items():
            name=f'{prior}_{layout}';factor=np.sqrt(bank.shape[1]/400)
            scaler=StandardScaler().fit(bank[:n][inner])
            x=np.ascontiguousarray(scaler.transform(bank[:n][inner]).astype(np.float64)/factor)
            z=np.ascontiguousarray(scaler.transform(bank[:n][cal]).astype(np.float64)/factor)
            tuning=[]
            for c in (.001,.01,.1):
                theta,info=head.fit(x,np.ascontiguousarray(y[inner],dtype=np.int64),c)
                value=cell_nll(y[cal],predict(z,theta),streams[cal],interface[cal])
                tuning.append(dict(C=c,source_held_P10_cell_NLL=value,fit=info));fit_count+=1
                params[f'{name}_inner_{c}_theta']=theta
            chosen=min(tuning,key=lambda record:(record['source_held_P10_cell_NLL'],record['C']))['C']
            params[name+'_inner_mean']=scaler.mean_;params[name+'_inner_scale']=scaler.scale_
            scaler=StandardScaler().fit(bank[:n])
            x=np.ascontiguousarray(scaler.transform(bank[:n]).astype(np.float64)/factor)
            theta,info=head.fit(x,np.ascontiguousarray(y,dtype=np.int64),chosen);fit_count+=1
            psource=predict(x,theta)
            pcpu=predict(np.ascontiguousarray(scaler.transform(cpu_banks[layout]).astype(np.float64)/factor),theta)
            pgpu=psource[probe_positions]
            margin=np.sort(pcpu,axis=1)[:,-1]-np.sort(pcpu,axis=1)[:,-2]
            error=float(np.abs(pcpu-pgpu).max());changes=int(np.sum((pcpu.argmax(1)!=pgpu.argmax(1))&(margin>1e-6)))
            if error>1e-3 or changes:
                raise ValueError('fitted CPU/A100 readout guard failed')
            parity[name]=dict(max_probability_error=error,nontied_changes=changes)
            posteriors[name]=predict(np.ascontiguousarray(scaler.transform(bank[n:]).astype(np.float64)/factor),theta)
            params[name+'_theta']=theta;params[name+'_mean']=scaler.mean_;params[name+'_scale']=scaler.scale_
            quality[name]=dict(tuning=tuning,selected_C=chosen,final_fit=info,source_accuracy=float(np.mean(psource.argmax(1)==y)),
                               source_NLL=float(-np.log(psource[np.arange(n),y]).mean()),encoder_state_sha256=before)
            print(json.dumps(dict(completed=name,fit_count=fit_count,seconds=time.monotonic()-started)),flush=True)
    if fit_count!=32 or len(posteriors)!=8:
        raise ValueError('complete matrix required before query labels')
    np.savez(ROOT/'all_fits.npz',source_rows=source,query_rows=query,**params)
    np.savez(ROOT/'query_posteriors.npz',query_rows=query,**posteriors)
    premetric=dict(passed=True,fit_count=fit_count,source_quality=quality,fitted_readout_parity=parity,
                   all_fits_sha256=sha(ROOT/'all_fits.npz'),posterior_sha256=sha(ROOT/'query_posteriors.npz'),
                   source_sha256=sha(ROOT/'source.sha256'),P09_labels_opened=False,seconds=time.monotonic()-started)
    (ROOT/'complete_before_labels.json').write_text(json.dumps(premetric,indent=2))
    # First access to held P09 labels, after persisted complete fitted artifacts and guards.
    yquery=reader.read_training_labels(DATA,query,len(people))
    qi=reader.read_selected(DATA,'interface',query,len(people))
    sys.path[:0]=[str(OLD/'2026-competition'),str(OLD/'experiments')]
    protocol=module('ni26_s342_official_protocol',OLD/'experiments/protocol.py')
    metric_path=Path(inspect.getfile(protocol.balanced_accuracy)).resolve()
    if metric_path.name!='metrics.py' or sha(metric_path)!='fcc739400806a74259f248618501376a4fe53f2b5a40ccdc20db6a9e2369787d':
        raise ValueError('official metric ownership/hash differs')
    metadata=dict(stream=np.char.add(np.char.add(d['subject'][query],'|'),d['session'][query]),interface=qi)
    results={name:protocol.score(yquery,p.argmax(1),metadata,np.ones(len(query),bool)) for name,p in posteriors.items()}
    result=dict(metric_module_path=str(metric_path),metric_module_sha256=sha(metric_path),scope='INDIVIDUAL-MODEL SOURCE6 held P09; previously exposed, not blind',results=results,
                complete_before_labels_sha256=sha(ROOT/'complete_before_labels.json'),seconds=time.monotonic()-started,
                limits='No FULL-SYSTEM DEV/SEALED/outer inference. One held person, one seed. No raw-factor branch.')
    (ROOT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
