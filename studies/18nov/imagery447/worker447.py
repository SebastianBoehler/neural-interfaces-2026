"""Compare frozen package outputs within unbroken Chanɡ recordings."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import worker as frozen

PEOPLE = (211, 216, 221, 222, 226, 227)


def main(arm, task):
    assert 'A100' in torch.cuda.get_device_name() and torch.__version__ == '2.14.1+cu130'
    person = PEOPLE[max(task, 0)]
    results = {}
    template = None
    for repeat in (1, 2):
        for condition in ('MI', 'ME'):
            key = f'S{person}_{condition}{repeat}'
            with np.load(key + '.npz') as src:
                x = torch.from_numpy(src['X'])
                names = src['names'].tolist()
            if template is None:
                template = frozen.replay.load(Path('/package'), names)
            assert x.shape == (30, 41, 480) and x.dtype == torch.float32
            for seed in (447, 1447):
                rows = np.random.default_rng([447, person, repeat, seed]).permutation(30)
                result = frozen.run(template, x, rows, arm)
                results[f'{key}-{seed}'] = result
                if task < 0:
                    assert frozen.run(template, x, rows[:7], arm)['labels'] == result['labels'][:7]
                    break
        if task < 0:
            break
    if task < 0 and arm == 'cap':
        px, ps, pn = frozen.replay.read_source()
        assert pn == names
        prior = json.loads(Path('/prior.json').read_text())['streams']['first']['P01-1']
        assert frozen.run(template, px, ps[0][1][:31], arm)['labels'] == prior['labels'][:31]
    Path(f'{arm}_{task}.json').write_text(json.dumps(results))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
