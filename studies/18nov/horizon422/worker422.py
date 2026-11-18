"""Intervene on native FT settings while preserving the rest of cap250."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import worker as native
from worker415 import shuffled, source


def main(shard):
    assert 'A100' in torch.cuda.get_device_name() and torch.__version__ == '2.14.1+cu130'
    people = [1] if shard == -1 else [26] if shard == -2 else range(1+5*shard, min(26, 5+5*shard)+1)
    result = []
    for person in people:
        x, sessions, tags, raw, channels, _ = source('W', person)
        archives = []
        for folder, offset in (('/long', 6), ('/short', 0)):
            path = Path(folder) / f'cap_{(person-1)//5+offset}.json'
            archives.append({r['seed']: r for r in json.loads(path.read_text()) if r['person'] == person})
        for arm in ('stock', 'zero', 'off'):
            model = native.replay.load(Path('/package'), channels)
            assert model.ft.on and model.ft.n_steps > 0
            if arm == 'zero':
                model.ft.n_steps = 0
            if arm == 'off':
                model.ft.on = False
            for seed in (415, 1415):
                [(session, order)] = list(shuffled(sessions, person, seed))
                for horizon, ref in zip((60, 20), archives):
                    reference = ref[seed]
                    assert raw[order].tolist() == reference['raw'] and tags[order].tolist() == reference['groups']
                    pieces = [native.run(model, x, b, 'cap') for b in np.split(order, 60//horizon)]
                    labels = sum((p['labels'] for p in pieces), [])
                    activity = [p['activity'] for p in pieces]
                    for a in activity:
                        count = 4 if horizon == 60 and arm != 'off' else 0
                        assert a['ft_refits'] == a['ft_relabels'] == count
                        assert a['ft_steps'] == (20*count if arm == 'stock' else 0)
                        assert a['ft_capped'] == a['ft_over'] == a['errors'] == 0 and a['recognized'] == -1
                    if arm == 'stock':
                        baseline = reference['activity']
                        assert activity == (baseline if isinstance(baseline, list) else [baseline])
                    length = 60 if arm == 'stock' or horizon == 20 else 20
                    assert labels[:length] == reference['labels'][:length]
                    result.append(dict(person=person, seed=seed, horizon=horizon, arm=arm,
                                       raw=reference['raw'], labels=labels, activity=activity))
            del model
    Path(f'run_{shard}.json').write_text(json.dumps(result))


if __name__ == '__main__':
    main(int(sys.argv[1]))
