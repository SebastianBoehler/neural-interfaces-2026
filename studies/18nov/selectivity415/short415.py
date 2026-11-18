"""Apply two additional resets to the exact same 60-window Shin2018 delivery."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import worker as frozen
from worker415 import shuffled, source


def main(arm, shard):
    assert 'A100' in torch.cuda.get_device_name() and torch.__version__ == '2.14.1+cu130'
    smoke = shard == -1
    people = [1] if smoke else range(1 + 5*shard, min(5 + 5*shard, 26) + 1)
    model = None
    records = []
    for person in people:
        x, sessions, tags, raw, channels, filename = source('W', person)
        assert len(set(sessions)) == 1
        if model is None:
            model = frozen.replay.load(Path('/package'), channels)
            names = channels
        assert names == channels
        for seed in (415, 1415):
            [(session, order)] = list(shuffled(sessions, person, seed))
            pieces = [frozen.run(model, x, block, arm) for block in np.split(order, 3)]
            if smoke:
                assert frozen.run(model, x, order[:20], arm)['labels'] == pieces[0]['labels']
            if arm == 'cap':
                assert all(r['activity']['ft_refits'] == 0 for r in pieces)
            records.append(dict(person=person, seed=seed, session=str(session), raw=raw[order].tolist(),
                                groups=tags[order].tolist(), filename=filename,
                                labels=sum((r['labels'] for r in pieces), []),
                                activity=[r['activity'] for r in pieces]))
            if smoke:
                break
    Path(f'{arm}_{shard}.json').write_text(json.dumps(records))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
