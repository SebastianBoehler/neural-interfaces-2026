"""Run causal F-C9 and fixed scale in the exact staged cap250 worker."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from data import permutation, sha
from reliability import Reliability
import replay
from benchmark_utils.streaming import fresh_model

SEEDS = (411, 1411)
ARMS = ('stock','fixed','adaptive')


def predict(template, eeg, indices, name):
    model = fresh_model(template)
    assert not model.log6['n'] and not model.v5hist and model.ft.refits == model.cp.t == 0
    assert 'mix6' not in vars(model)
    policy = None if name == 'stock' else Reliability(name)
    if policy is not None:
        policy.install(model)
    labels = []
    torch.cuda.synchronize()
    start = perf_counter()
    for r in indices.tolist():
        label = model.predict(eeg[r:r+1].to('cuda:0'))
        assert label.shape == (1,) and label.device.type == 'cpu' and label.dtype == torch.int64
        labels.append(int(label[0]))
    torch.cuda.synchronize()
    wall = perf_counter()-start
    work = replay.activity(model)
    assert work['errors']==0 and work['windows']==len(indices)
    assert not any(x.dead for x in (model,model.cp,model.cf))
    assert len(model.v5hist) == len(labels) and np.isin(labels,(0,1,2)).all()
    inner = [int(x.argmax()) for x in model.v5hist]
    trace = {} if policy is None else policy.reads
    changed = 0 if policy is None else policy.n_changed
    if work['recognized'] != -1:
        assert not trace
    if trace:
        assert all(t[0][0]==20 and t[0][1]== (1. if name=='neutral' else .75)
                   for t in trace.values())
        assert all(len(t)==len(indices)-20 for t in trace.values())
        assert changed==0 if name=='neutral' else changed>0
    if policy is not None:
        del model.mix6
    return dict(labels=labels,inner=inner,cp_fail=model.cp.nfail,activity=work,
                coefficient=trace,changed=changed,wall_s=wall)


def match(result, stock, old):
    base=result['stock']
    assert base['labels']==stock['labels'] and base['activity']==stock['activity']
    recognized = base['activity']['recognized']
    assert recognized == -1 and len(stock['labels']) >= 100
    for name,value in result.items():
        assert value['inner']==old['labels'] and value['activity']==stock['activity'], name
        if name in ARMS[1:]:
            assert value['coefficient'] and value['changed'] > 0
    if 'neutral' in result:
        assert result['neutral']['labels']==base['labels']
    assert any(len({v for _,v in trace})>1
               for trace in result['adaptive']['coefficient'].values())


def smoke_last(template, x, rows, result):
    prefix=predict(template,x,rows[:36],'adaptive')
    chosen=result['adaptive']
    assert prefix['labels']==chosen['labels'][:36] and prefix['inner']==chosen['inner'][:36]
    assert prefix['coefficient'].keys()==chosen['coefficient'].keys()
    for name,points in prefix['coefficient'].items():
        np.testing.assert_allclose(points,chosen['coefficient'][name][:len(points)],atol=2e-5,rtol=0)
    px, ps, channels = replay.read_source()
    anchor=json.loads(Path('/normal.json').read_text())['streams']['first']['P10-3']
    query=dict(ps)['P10-3'][:31]
    ranked=predict(template,px,query,'adaptive')
    assert channels and ranked['activity']['recognized']==10 and not ranked['coefficient']
    assert ranked['labels']==anchor['labels'][:31]


def main(index, smoke):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    with np.load('/external.npz',allow_pickle=False) as z:
        x=torch.from_numpy(z['X'])
        people,days=(z[k] for k in ('subject','session'))
        channels=z['ch_names'].tolist()
    assert x.shape==(2130,41,480) and x.dtype==torch.float32
    person=sorted(set(people))[index]
    template=replay.load(Path('/package'),channels)
    groups=np.char.add(np.char.add(people,'|'),days)
    cap,old=[json.loads(Path(f'/prior/{arm}_{index}.json').read_text()) for arm in ('cap','old')]
    result={}
    for order in SEEDS:
        perm=permutation(groups,None,'full',order)
        for day in sorted(set(days[people==person])):
            name=f'{person}|{day}|{order}'
            rows=perm[(people==person)&(days==day)]
            assert rows.tolist()==cap[name]['rows']==old[name]['rows']
            arms=(*ARMS,'neutral') if smoke else ARMS
            if not smoke:
                rotate=(index+int(day)+SEEDS.index(order))%len(arms)
                arms=arms[rotate:]+arms[:rotate]
            observed={arm:predict(template,x,rows,arm) for arm in arms}
            match(observed,cap[name],old[name])
            result[name]=dict(rows=rows.tolist(),arms=observed)
            if smoke:
                smoke_last(template,x,rows,observed)
                break
        if smoke:
            break
    assert len(result)==(1 if smoke else 4)
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    path=Path('smoke.json' if smoke else f'held_{index}.json')
    path.write_text(json.dumps(dict(source=sha(Path('source.sha256')),sessions=result))+'\n')
    path.with_suffix('.sha256').write_text(sha(path)+'\n')
    print(person,len(result),'exact stock and inner rows, causal scale active',flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]),sys.argv[2:]==['smoke'])
