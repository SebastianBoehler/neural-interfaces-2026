"""Actual-source baseline parity, active Gaussian/loss/control and timing smoke."""
import json
from pathlib import Path
import time
import torch
from data354 import source,original,ARMS
from train354 import attach


def main():
    root=Path.cwd();start=time.monotonic();torch.set_num_threads(2)
    data,fit,held,people,y=source('P02')
    reference=original.native(data,1);reference.fit(fit)
    state=original.state_sha(reference.net);records={};initial=[]
    for arm in ARMS:
        model=original.native(data,1);trace=attach(model,arm);tick=time.monotonic();model.fit(fit)
        trace['seconds']=time.monotonic()-tick;records[arm]=trace;initial.append(trace['initial_backbone_sha256'])
        if arm=='O' and original.state_sha(model.net)!=state:raise ValueError('custom training differs from ORIGINAL one-epoch raw reference')
        if arm in ('M','L','G'):
            expected={'M':47026,'L':47024,'G':47075}[arm]
            if trace['adapter_parameters']!=expected:raise ValueError('capacity count mismatch')
            if model.net.temporal_layer.adapter.cache is not None:raise ValueError('training graph cache not cleared')
        model.net.eval();before=original.state_sha(model.net)
        rows=fit[:7];p=model.predict_proba_stream(rows)
        if p.shape!=(7,3) or before!=original.state_sha(model.net):raise ValueError('source inference/state guard')
    if len(set(initial))!=1:raise ValueError('different backbone initialization')
    if len({tuple(records[a]['plans']) for a in ('A','M','L','G')})!=1:raise ValueError('matched sampler differs')
    record=dict(passed=True,original_one_epoch_BITWISE=True,source_sha256=original.sha(root/'source.sha256'),
                records=records,seconds=time.monotonic()-start,query_labels_opened=False,query_forward=False,
                scope='real SOURCE-only smoke; not accuracy or full-system proof')
    (root/'smoke.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
