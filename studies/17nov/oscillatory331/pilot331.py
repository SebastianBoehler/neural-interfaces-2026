"""Fixed source7 pilot; allmodels/checkpoints/querybanks precede querylabels."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from data331 import load,labels,official,sha,digest
from train331 import fit,predict
from activity331 import inference,fingerprint
from reference331 import predict as reference_predict,ORDERS


def main():
    root=Path(os.environ['OSC_ROOT']);torch.set_num_threads(6);started=time.monotonic()
    if os.environ.get('NI26_DEVICE')!='cpu' or torch.cuda.is_available():raise ValueError('explicitCPUonlyroute required')
    guard=json.loads((root/'guard.json').read_text())
    if not guard['passed'] or not guard['budget_passed'] or guard['source_sha256']!=sha(root/'source.sha256') or guard['scaler_sha256']!=sha(root/'scaler.npz'):
        raise ValueError('passed source/scaler/costguard required')
    d,source,query,people=load()
    with np.load(root/'scaler.npz') as z:
        rms=float(z['rms']);mean=torch.from_numpy(z['mean']);sd=torch.from_numpy(z['sd']);stats=json.loads(str(z['metadata']))
    if stats['fit_rows_sha256']!=digest(source) or stats['fit_input_sha256']!=digest(d['X'][source]):raise ValueError('scaler sourcechanged')
    yfit=labels(source,len(d['X']));x=(d['X'][source]/rms).astype(np.float32);q=(d['X'][query]/rms).astype(np.float32)
    probe=np.array([np.flatnonzero(people==f'P{i:02d}')[0] for i in (4,5,6,7)])
    xx=torch.from_numpy((d['X'][probe]/rms).astype(np.float32));training={};checks={};banks={};checkpoints={}
    for arm in ('L','F','S'):
        model,meta=fit(arm,x,yfit,people[source],mean,sd);checks[arm]=inference(model,xx)
        before=fingerprint(model);banks[arm]=predict(model,q)
        if fingerprint(model)!=before or not np.isfinite(banks[arm]).all() or not np.allclose(banks[arm].sum(1),1,atol=1e-6):raise ValueError('querybank/statefailed')
        path=root/f'{arm}.pt'
        with path.open('xb') as out:torch.save(dict(state=model.state_dict(),source=stats,training=meta),out)
        checkpoints[arm]=sha(path);training[arm]=meta
    reference_started=time.monotonic();reference,reference_meta=reference_predict(d,source,query,yfit)
    reference_meta['seconds']=time.monotonic()-reference_started
    # Every prescribedfit/predictionguard is completebefore these two exposed queries.
    metric=official();yquery=labels(query,len(d['X']));results={}
    for arm,bank in list(banks.items())+[(f'R_{o}',p) for o,p in reference.items()]:
        per={}
        for person in ('P02','P03'):
            choose=people[query]==person;mask=people==person
            value=metric.score(yquery[choose],bank[choose].argmax(1),d,mask)
            per[person]=value
        results[arm]=dict(people=per,person_mean_official_BA=float(np.mean([v['balanced_accuracy_cell_mean'] for v in per.values()])))
    reference_summary={}
    for person in ('P02','P03'):
        per_order={o:results[f'R_{o}']['people'][person]['balanced_accuracy_cell_mean'] for o in ORDERS}
        reference_summary[person]=dict(by_order=per_order,scheme_equal=float(np.mean([np.mean([v for o,v in per_order.items() if o.startswith(s)]) for s in ('within','full','runs')])))
    report=dict(scope='selected source-only whole decoder;two exposed people;not outer/current-system gain',
        P01_used=False,source_sha256=sha(root/'source.sha256'),scaler_sha256=sha(root/'scaler.npz'),source=stats,
        source_rows_sha256=digest(source),query_rows_sha256=digest(query),checkpoints=checkpoints,
        training=training,final_checks=checks,reference=reference_meta,reference_scheme_equal=reference_summary,
        results=results,prediction_sha256={a:digest(v) for a,v in banks.items()},torch_version=torch.__version__,
        total_seconds=time.monotonic()-started,official_metric_module=str(metric.__file__),official_metric_sha256=sha(metric.__file__))
    with (root/'results.json').open('x') as out:json.dump(report,out,indent=2)
    with (root/'predictions.npz').open('xb') as out:np.savez(out,index=query,**banks,**{f'R_{k}':v for k,v in reference.items()})
    print('331completefixedsourcepilot;allfits/guards/banksbeforequerylabels',flush=True)


if __name__=='__main__':main()
