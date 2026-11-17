"""Repeat all frozen §380 decisions with the independent §379 CPU step implementation."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from pilot import ARMS, DualScale, OWNER, train, verify
from score379 import PEOPLE, ORDERS, sha

READER = Path(sys.argv[1]).resolve()
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location('audit379_pinned',READER)
audit = module_from_spec(spec)
spec.loader.exec_module(audit)


@torch.no_grad()
def main(root):
    verify()
    torch.set_num_threads(2)
    with np.load(train.DATA) as z:
        raw = torch.from_numpy(z['X']).float()
        people = np.array([s.rsplit('/',1)[-1] for s in z['subject']])
        streams = np.char.add(np.char.add(people,'|'), z['session'])
    with np.load(train.PERM) as z:
        orders = {o:z[f'{o[:-1]}|{o[-1]}|all_loso'] for o in ORDERS}
    diffs, count = [], 0
    for p in PEOPLE:
        fit, test = [np.flatnonzero(mask) for mask in (people!=p,people==p)]
        source = [fit[streams[fit]==s] for s in sorted(set(streams[fit]))]
        rng = np.random.default_rng(379)
        episodes = [rng.choice(source[rng.integers(len(source))],96,replace=False) for _ in range(360)]
        scale = raw[fit].square().mean().sqrt()
        proof = json.loads((root/(p+'.json')).read_text())
        assert train.digest(np.stack(episodes)) == proof['source_episodes']
        with np.load(root/(p+'.npz')) as bank:
            for arm in ARMS:
                net = DualScale(arm).eval()
                path = root/f'{p}_{arm}.pt'
                assert sha(path) == proof['checkpoints'][arm]
                net.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
                encoded = net.encode(raw[test]/scale)
                for o,order in orders.items():
                    for s in np.unique(streams[test]):
                        rows = order[np.isin(order,test[streams[test]==s])]
                        pos = np.searchsorted(test,rows)
                        labels = audit.replay(net, encoded[pos],'norm',None)
                        bad = rows[labels != bank[arm+'|'+o][pos]]
                        count += len(rows)
                        if len(bad):
                            diffs.append({'person':p,'order':o,'arm':arm,'session':s,'rows':bad.tolist()})
    assert count == 63720
    Path('audit380.json').write_text(json.dumps(dict(compared=count,checkpoints=30,differences=diffs,
        reader_sha256=sha(READER),source_sha256=sha(root/'source.sha256'),held_labels_opened=False),indent=2))
    print(count, len(diffs), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[2]))
