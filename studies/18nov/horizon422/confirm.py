"""Prospective §422b replication on two previously unexamined permutations."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import worker as native
from worker415 import shuffled, source


def main(arm, shard):
    assert 'A100' in torch.cuda.get_device_name() and torch.__version__ == '2.14.1+cu130'
    people = [1] if shard == -1 else range(1+5*shard, min(26, 5+5*shard)+1)
    result = []
    for person in people:
        x, sessions, tags, raw, names, _ = source('W', person)
        model = native.replay.load(Path('/package'), names)
        if arm != 'old':
            assert model.ft.on and model.ft.n_steps > 0
            if arm == 'zero':
                model.ft.n_steps = 0
            if arm == 'off':
                model.ft.on = False
        for seed in (2422,3422):
            [(_, order)] = list(shuffled(sessions, person, seed))
            labels = {}
            for h in (60,20):
                blocks = [native.run(model,x,b,'old' if arm == 'old' else 'cap') for b in np.split(order,60//h)]
                activity = [b['activity'] for b in blocks]
                if arm != 'old':
                    for a in activity:
                        fits = 4 if arm != 'off' and h == 60 else 0
                        assert a['ft_refits'] == a['ft_relabels'] == fits
                        assert a['ft_steps'] == (20*fits if arm == 'stock' else 0)
                        assert a['ft_capped'] == a['ft_over'] == a['errors'] == 0 and a['recognized'] == -1
                labels[h] = sum((b['labels'] for b in blocks), [])
                result.append(dict(person=person,seed=seed,horizon=h,raw=raw[order].tolist(),
                                   labels=labels[h],activity=activity))
            assert labels[20][:20] == labels[60][:20]
        del model
    Path(f'{arm}_{shard}.json').write_text(json.dumps(result))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
