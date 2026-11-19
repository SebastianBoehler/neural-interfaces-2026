"""Pass the frozen F-C6 rule through the exact C7 worker on §411b orders."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from actor import stream
from data import permutation
import replay

ORDERS = (411, 1411)


def main(index, smoke):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    with np.load('/external.npz', allow_pickle=False) as z:
        x = torch.from_numpy(z['X'])
        people, days, channels = (z[k] for k in ('subject','session','ch_names'))
    assert x.shape == (2130,41,480) and x.dtype == torch.float32
    person = sorted(set(people))[index]
    groups = np.char.add(np.char.add(people, '|'), days)
    cap, old = (json.loads(Path(f'/prior/{a}_{index}.json').read_text()) for a in ('cap','old'))
    template = replay.load(Path('/package'),channels.tolist())
    records = {}
    for order in ORDERS:
        perm = permutation(groups, None, 'full', order)
        for day in sorted(set(days[people==person])):
            name = f'{person}|{day}|{order}'
            rows = perm[(people==person)&(days==day)]
            assert rows.tolist() == cap[name]['rows'] == old[name]['rows']
            labels, activity, features = stream(template, {}, x, rows)
            assert set(labels) == {'stock','inner','causal'}
            assert labels['stock'] == cap[name]['labels'] and activity == cap[name]['activity']
            assert labels['inner'] == old[name]['labels']
            records[name] = dict(rows=rows.tolist(), predictions=labels, activity=activity)
            if smoke:
                shorter, _, probe = stream(template, {}, x, rows[:17])
                assert all(shorter[a] == v[:17] for a,v in labels.items())
                np.testing.assert_allclose(probe,features[:17],atol=2e-4,rtol=1e-5)
                break
        if smoke:
            break
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    path = Path('smoke.json' if smoke else f'held_{index}.json')
    source = hashlib.sha256(Path('source.sha256').read_bytes()).hexdigest()
    path.write_text(json.dumps(dict(source=source,sessions=records))+'\n')
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    n = sum(np.count_nonzero(np.asarray(v['predictions']['causal']) != v['predictions']['stock'])
            for v in records.values())
    print(person,'sessions',len(records),'causal switches',n,flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]),sys.argv[2:]==['smoke'])
