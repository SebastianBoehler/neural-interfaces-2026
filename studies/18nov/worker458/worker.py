"""Compare the same worker score stream with and without the all-source native graph."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import replay
from benchmark_utils.streaming import fresh_model
from data import permutation
from graph import SEEDS,SessionGraph


def measure(person,arm,rep,smoke):
    assert torch.__version__=='2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    with np.load('/external.npz') as z:
        x,people,days,channels=torch.from_numpy(z['X']),z['subject'],z['session'],z['ch_names'].tolist()
    who=sorted(set(people))[person]
    streams=np.char.add(np.char.add(people,'|'),days)
    day=sorted(set(days[people==who]))[0]
    order=permutation(streams,None,'full',0)
    rows=order[(people[order]==who)&(days[order]==day)]
    name=f'{who}|{day}|0'
    archive=json.loads(Path(f'/archive/cap_{person}.json').read_text())[name]
    assert archive['rows']==rows.tolist()
    if smoke:
        rows=rows[:17]
    template=replay.load(Path('/package'),channels)
    torch.cuda.synchronize()
    started=time.perf_counter()
    model=fresh_model(template)
    original=model.cp.step
    observed=[]
    def measure_scores(*args):
        scores=original(*args)
        observed.append(np.array(scores,copy=True))
        return scores
    model.cp.step=measure_scores
    graph=None
    if arm=='graph':
        with torch.inference_mode():
            graph=SessionGraph()
    torch.cuda.synchronize()
    setup=time.perf_counter()-started
    labels={arm:[] for arm in ('stock','454','1454','pair')}
    durations=[]
    for t,row in enumerate(rows):
        window=x[row:row+1].cuda()
        label=model.predict(window)
        assert label.shape==(1,) and label.dtype==torch.int64 and label.device.type=='cpu'
        assert len(observed)==1
        raw=observed.pop()
        assert raw.argmax()==int(label[0])
        labels['stock'].append(int(label[0]))
        if graph:
            tick=time.perf_counter()
            with torch.inference_mode():
                offset=graph.step(window)
            if t==0:
                assert not offset.any()
            if t<90:
                with np.errstate(divide='ignore'):
                    raw=np.log(raw.astype(np.float64))
            else:
                assert np.isfinite(raw).all() and (raw<=0).all()
            for key,d in zip(('454','1454','pair'),(*offset,offset.mean(0))):
                labels[key].append(int(np.argmax(raw+d)))
            durations.append(time.perf_counter()-tick)
    torch.cuda.synchronize()
    total=time.perf_counter()-started
    assert labels['stock']==archive['labels'][:len(rows)]
    counters=replay.activity(model)
    if not smoke:
        assert counters==archive['activity']
    if graph:
        reference=json.loads(Path('/reference/'+('external_0_smoke.json' if smoke else f'external_0_{person}.json')).read_text())
        check=reference['sessions'][name]
        assert check['rows'][:len(rows)]==rows.tolist()
        for seed in SEEDS:
            assert check['arms'][f'{seed}_memory'][:len(rows)]==labels[str(seed)]
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    result=dict(session=name,device=torch.cuda.get_device_name(),windows=len(rows),setup=setup,seconds=total,
                graph_increment_median_ms=1000*np.median(durations) if durations else None,
                labels={k:hashlib.sha256(bytes(v)).hexdigest() for k,v in labels.items() if v},counters=counters)
    Path(f'{person}_{rep}_{arm}.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    measure(int(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[3]=='smoke')
