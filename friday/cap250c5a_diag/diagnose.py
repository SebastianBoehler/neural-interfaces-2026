"""Separate personalizer equivalence from independently replayed upstream drift."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from qualify import empty, fresh_model, instrument, replay, stream, trials


def bits(x):
    return hashlib.sha256(x.dtype.str.encode()+x.tobytes()).hexdigest()


def run(out):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    names, series = trials(5, True)
    template = replay.load(Path('/package'), names)
    assert stream(template, series['unrecognized'], False, 1)['labels']
    warm = {r: stream(template, series[r], False, 17) for r in ('unrecognized','recognized')}
    unknown = series['unrecognized']
    model = fresh_model(template)
    empty(model)
    eager = model.cp
    lazy = deepcopy(eager)
    e, d = instrument(eager, False), instrument(lazy, True)
    emit = eager.step
    rows = []
    def track(x, posterior, filtered, recognized):
        # Capture the same live input for each stage. Only stock's answer reaches the model.
        incoming = {'base':bits(posterior),'filtered':bits(filtered),'window':bits(x)}
        incoming['ft_count'] = model.ft.refits
        incoming['ft_tail'] = {name:bits(stack[-1]) for name,stack in model.ft.X.items() if stack}
        incoming['ft_weight'] = [int(model.ft.lab[-1]),float(model.ft.wt[-1])]
        out = emit(x, posterior, filtered, recognized)
        shadow = lazy.step(x, posterior, filtered, recognized)
        incoming.update(stock=bits(out),deferred=bits(shadow),values=out.tolist())
        rows.append(incoming)
        assert incoming['stock'] == incoming['deferred'], len(rows)-1
        return out
    eager.step = track
    labels = []
    for r in unknown.rows.tolist():
        z = model.predict(unknown.eeg[r:r+1].to('cuda:0'))
        assert z.shape==(1,) and z.dtype==torch.int64 and z.device.type=='cpu'
        labels.append(int(z[0]))
    torch.cuda.synchronize()
    assert labels == unknown.anchor['labels'] and replay.activity(model)==unknown.anchor['activity']
    assert labels[:17]==warm['unrecognized']['labels']
    assert [r['stock'] for r in rows[:17]]==warm['unrecognized']['terminal']
    assert [r['ft_count'] for r in rows[:22]]==[0]*20+[1,1]
    attrs = ('t','stopped','refits','fused','nfail','dead','log')
    assert all(getattr(eager,k)==getattr(lazy,k) for k in attrs) and not lazy.dead
    assert all(e[k] == d[k] for k in ('selected','consumes','queries'))
    assert len(e['cov'])==len(rows)==len(set(e['cov'])) and len(d['cov'])==len(set(d['cov']))
    assert set(d['cov'])==set(d['consumes'])|set(d['queries'])
    assert len(d['cov']) <= .85*len(e['cov'])
    result = dict(rows=unknown.rows.tolist(),labels=labels,activity=replay.activity(model),
                  stock_cov=len(e['cov']),deferred_cov=len(d['cov']),trace=rows,
                  source=hashlib.sha256(Path('source.sha256').read_bytes()).hexdigest())
    dest=Path(out)
    dest.write_text(json.dumps(result)+'\n')
    dest.with_suffix('.sha256').write_text(hashlib.sha256(dest.read_bytes()).hexdigest()+'\n')
    print('shared-input stock/deferred scores',len(rows),'bitwise; covariance',
          len(e['cov']),len(d['cov']),flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
