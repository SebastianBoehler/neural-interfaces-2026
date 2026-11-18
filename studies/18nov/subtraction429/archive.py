"""Exact historical archive through the official fresh-session worker."""
import json
from pathlib import Path
import sys

import torch
import worker as frozen

from wang_inputs import load, orders


def main(arm, person, smoke):
    assert torch.__version__ == '2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    bank, names = load(person, smoke)
    template = frozen.replay.load(Path('/package'), names)
    result = []
    for visit, wave in enumerate(bank, 1):
        x = torch.from_numpy(wave)
        for seed, rows in orders(person, visit, smoke):
            rec = frozen.run(template, x, rows, arm)
            if smoke:
                assert frozen.run(template, x, rows[:6], arm)['labels'] == rec['labels'][:6]
            result.append(dict(visit=visit, seed=seed, arm=arm, **rec))
    if smoke and arm == 'cap':
        px, streams, pn = frozen.replay.read_source()
        assert names == pn
        ref = json.loads(Path('/prior.json').read_text())['streams']['first']['P01-1']
        assert frozen.run(template, px, streams[0][1][:31], arm)['labels'] == ref['labels'][:31]
    name = 'smoke' if smoke else str(person)
    Path(f'{arm}_{name}.json').write_text(json.dumps(result))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), len(sys.argv)>3)
