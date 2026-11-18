"""Compare frozen member outputs before any same-copy learning, with JIT graph optimization on/off."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model


def probe(template, x, rows):
    model = fresh_model(template)
    for r in rows:
        label = model.predict(x[r:r+1].to('cuda:0'))
        if r == rows[0]:
            assert model.ft.steps == 0
            metadata = {i: type(net).__name__ for _, nets in model.nets.groups for i, net in nets}
            member = {i: dict(net=metadata[i], p=p.tolist(),
                              input_sha256=hashlib.sha256(model.nets.last_inputs[i].tobytes()).hexdigest())
                      for i, p in model.trace['v6']['members'].items()}
            first = int(label[0])
    return first, member


def main(mode):
    with np.load('/external.npz') as archive:
        x = torch.from_numpy(archive['X'])
        names = archive['ch_names'].tolist()
    template = replay.load(Path('/package'), names)
    ref = next(iter(json.loads(Path('/prior/cap_5.json').read_text()).values()))
    rows = np.array(ref['rows'])
    ctx = torch.jit.optimized_execution(False) if mode == 'jitoff' else nullcontext()
    with ctx:
        before, cold = probe(template, x, rows)
        after, warm = probe(template, x, rows[:1])
    assert before == after
    assert cold.keys() == warm.keys()
    out = {}
    for key, a in cold.items():
        b = warm[key]
        assert a['input_sha256'] == b['input_sha256'] and a['net'] == b['net']
        delta = float(np.abs(np.array(a['p'])-b['p']).max())
        if delta:
            out[int(key)] = dict(net=a['net'], max_abs=delta)
    Path(f'graph_{mode}.json').write_text(json.dumps(dict(n_members=len(cold), changed=out)))


if __name__ == '__main__':
    main(sys.argv[1])
