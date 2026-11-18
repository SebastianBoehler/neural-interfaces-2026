"""Predict blind on each original Scherer order using a clean, unmodified cap250."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import replay
from benchmark_utils.streaming import fresh_model
from data import permutation
from external_actor import Actor, features

SEEDS = (454,1454)
ORDERS = ((0,1),(411,1411))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream(template, actors, eeg, rows):
    model = fresh_model(template)
    predict = model.cp.step
    observed = []

    def capture(*args):
        result = predict(*args)
        observed.append(np.array(result,copy=True))
        return result

    model.cp.step = capture
    output = {'stock':[]}
    states = {s:{} for s in SEEDS}
    activity = {s:0. for s in SEEDS}
    for t,row in enumerate(rows):
        window = eeg[row:row+1].cuda()
        label = model.predict(window)
        assert label.device.type == 'cpu' and label.dtype == torch.int64 and label.shape == (1,)
        assert len(observed) == 1
        raw = observed.pop()
        assert raw.shape == (3,) and raw.argmax() == int(label[0])
        if t < 90:
            assert (raw>=0).all() and np.isclose(raw.sum(),1,atol=1e-5)
            with np.errstate(divide='ignore'):
                base = np.log(raw.astype(np.float64))
        else:
            assert (raw<=0).all() and np.isfinite(raw).all()
            base = raw.astype(np.float64)
        output['stock'].append(int(label[0]))
        spectral = features(window)
        for seed,actor in actors.items():
            for arm,d in actor.step(spectral,states[seed],t).items():
                corrected = int(np.argmax(base+d))
                if arm == 'source':
                    assert corrected == int(label[0])
                else:
                    output.setdefault(f'{seed}_{arm}',[]).append(corrected)
                    activity[seed] = max(activity[seed],float(abs(d).max()))
    assert all(value>0 for value in activity.values())
    counters = replay.activity(model)
    assert counters['recognized']==-1 and counters['windows']==len(rows) and counters['errors']==0
    assert not model.dead and not model.cp.dead and not model.cf.dead and not counters['ft_over']
    return dict(rows=rows.tolist(), arms=output, counters=counters, max_correction=activity)


def main(person,pair,smoke):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    with np.load('/external.npz') as z:
        x,people,days,channels = torch.from_numpy(z['X']),z['subject'],z['session'],z['ch_names'].tolist()
    assert x.shape == (2130,41,480) and x.dtype == torch.float32
    streams = np.char.add(np.char.add(people,'|'),days)
    who = sorted(set(people))[person]
    archive = Path('/archive')/f'cap_{person}.json'
    control = json.loads(archive.read_text())
    template = replay.load(Path('/package'),channels)
    actors = {s:Actor(s) for s in SEEDS}
    sessions = {}
    for order in ORDERS[pair]:
        perm = permutation(streams,None,'full',order)
        for day in sorted(set(days[people==who])):
            name = f'{who}|{day}|{order}'
            rows = perm[(people==who)&(days==day)]
            assert rows.tolist() == control[name]['rows']
            entry = stream(template,actors,x,rows)
            assert entry['arms']['stock'] == control[name]['labels'], name
            assert entry['counters'] == control[name]['activity'], name
            sessions[name] = entry
            if smoke:
                prefix = stream(template,actors,x,rows[:17])
                assert all(v==entry['arms'][k][:17] for k,v in prefix['arms'].items())
                break
        if smoke:
            break
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    suffix = 'smoke' if smoke else str(person)
    path = Path(f'external_{pair}_{suffix}.json')
    path.write_text(json.dumps(dict(archive=digest(archive), source=digest(Path('source.sha256')),
        models={s:digest(Path('/weights')/f'all_{s}.pt') for s in SEEDS}, sessions=sessions)))


if __name__ == '__main__':
    main(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3:] == ['smoke'])
