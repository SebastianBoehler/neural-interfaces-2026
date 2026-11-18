"""Run the two cognitive instruments separately through the frozen native worker."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import worker as frozen


def source(kind, person):
    paths = sorted(Path('/datasets/shin2017b').glob('windows_*.npz')) if kind == 'A' else [Path('/datasets/shin2018/windows.npz')]
    subject = ('Shin2017B/' if kind == 'A' else 'Shin2018/') + str(person)
    for path in paths:
        with np.load(path) as z:
            which = np.flatnonzero(z['subject'] == subject)
            if not len(which):
                continue
            assert z['classes'].tolist() == (['rest', 'subtraction'] if kind == 'A' else ['baseline', 'WG'])
            session, run, onset = (z[k][which] for k in ('session', 'run', 'onset'))
            tags = np.char.add(np.char.add(session, '|'), run)
            selected = []
            for tag in sorted(set(tags)):
                local = np.flatnonzero(tags == tag)
                early, late = np.split(local, 2) if kind == 'A' else (local[::2], local[1::2])
                assert len(early) == 20 and np.allclose(onset[late] - onset[early], 4)
                if kind == 'W':
                    assert np.all(z['win'][which[early]] == 0)
                selected.extend(early)
            selected = np.array(sorted(selected))
            assert len(selected) == 60 and len(set(tags)) == 3
            real_rows = which[selected]
            x = z['X'][real_rows]
            assert x.shape == (60, 41, 480) and x.dtype == np.float32
            return torch.from_numpy(x), session[selected], tags[selected], real_rows, z['ch_names'].tolist(), path.name
    raise ValueError(f'Unknown subject {subject}')


def shuffled(session, person, seed):
    for index, name in enumerate(sorted(set(session))):
        rows = np.flatnonzero(session == name)
        yield name, np.random.default_rng([415, person, seed, index]).permutation(rows)


def main(arm, task):
    assert 'A100' in torch.cuda.get_device_name() and torch.__version__ == '2.14.1+cu130'
    smoke = task == -1
    shard = max(0, task)
    kind = 'A' if shard < 6 else 'W'
    first = (shard % 6) * 5 + 1
    last = min(first + 4, 29 if kind == 'A' else 26)
    model = None
    out = []
    cases = [('A', 1), ('W', 1)] if smoke else [(kind, i) for i in range(first, last+1)]
    for kind, person in cases:
        x, sessions, tags, raw, names, filename = source(kind, person)
        if model is None:
            model = frozen.replay.load(Path('/package'), names)
            px, ps, pn = frozen.replay.read_source()
            assert pn == names
        assert pn == names
        for seed in (415, 1415):
            for session, rows in shuffled(sessions, person, seed):
                rec = frozen.run(model, x, rows, arm)
                if smoke:
                    assert frozen.run(model, x, rows[:7], arm)['labels'] == rec['labels'][:7]
                out.append(dict(kind=kind, person=person, session=str(session), seed=seed,
                                filename=filename, raw=raw[rows].tolist(), groups=tags[rows].tolist(), **rec))
            if smoke:
                break
    if smoke and arm == 'cap':
        archived = json.loads(Path('/prior.json').read_text())['streams']['first']['P01-1']
        assert frozen.run(model, px, ps[0][1][:31], arm)['labels'] == archived['labels'][:31]
    Path(f'{arm}_{task}.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
