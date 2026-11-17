"""Descriptive in-sample fit versus held prediction in one pinned shuffle."""
import json
from pathlib import Path

import numpy as np
import torch

from train import DATA, PERM, Decoder, digest, sha


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    with np.load(DATA) as z:
        x = torch.from_numpy(z['X']).float()
        y, people, sessions, interfaces = [z[k] for k in ('y','subject','session','interface')]
    stream = np.char.add(np.char.add(people,'|'),sessions)
    with np.load(PERM) as z:
        perm = z['within|0|all_loso']
    result = {}
    for subject in sorted(set(people)):
        fit = np.flatnonzero(people != subject)
        held = subject.split('/')[-1]
        proof = json.loads(Path(held+'.json').read_text())
        assert proof['train_rows_sha256'] == digest(fit)
        scale = x[fit].square().mean().sqrt()
        result[held] = {}
        for arm in ('memory','norm','static'):
            path = Path(f'{held}_{arm}.pt')
            assert sha(path) == proof['checkpoints'][arm]
            model = Decoder().eval()
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            predictions = np.full(len(x),-1)
            for s in np.unique(stream):
                ix = perm[stream[perm]==s]
                logits,_ = model.step(x[ix]/scale,arm)
                predictions[ix] = logits.argmax(1).numpy()
            query = people == subject
            with np.load(held+'.npz') as reference:
                assert np.array_equal(predictions[query],reference[arm+'|within0'])
            cell = np.char.add(np.char.add(stream,'|'),interfaces)
            def score(mask):
                return float(np.mean([(predictions[use]==i).mean() for c in np.unique(cell[mask])
                                      for i in range(3) if (use := mask & (cell==c) & (y==i)).any()])*100)
            result[held][arm] = [score(~query),score(query)]
    Path('fit379.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
