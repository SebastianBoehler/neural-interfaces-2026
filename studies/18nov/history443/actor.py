"""Replay the coherent waveform actor with genuine, donor and age-matched source writes."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

import score379 as ref
import train
from model import Decoder

ROOTS = {379: ('439', 'b6a6397455730088ce45a041b06510abe9ab57e7266d8ecd62e6e84060e8c8f3'),
         1441: ('441', 'a82db5447b9d7d6844786f76d06af271c02f0055a04455f36738e13149825518')}


@torch.no_grad()
def replay(seed, person):
    section, manifest = ROOTS[seed]
    folder = ref.WORK / f'ni26_gain{section}_20261118/coherent'
    record = json.loads((folder / (person+'.json')).read_text())
    assert record['source_sha256'] == ref.source_manifest(folder/'source.sha256') == manifest
    assert record['held'] == person and set(record['train_people']) == set(ref.PEOPLE)-{person}
    checkpoint = folder/(person+'_memory.pt')
    archive = folder/(person+'.npz')
    assert record['checkpoints']['memory'] == ref.sha(checkpoint)
    assert record['outputs_sha256'] == ref.sha(archive)
    assert ref.sha(train.DATA) == ref.DATA_SHA and ref.sha(train.PERM) == record['order_sha256'] == ref.ORDER_SHA
    assert 'A100' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = Decoder().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    with np.load(train.DATA) as data:
        people = np.array([s.split('/')[-1] for s in data['subject']])
        streams = np.char.add(np.char.add(people, '|'), data['session'])
        x = torch.from_numpy(data['X']).float().cuda()
    src, test = np.flatnonzero(people != person), np.flatnonzero(people == person)
    assert record['train_rows_sha256'] == train.digest(src)
    x /= x[src].square().mean().sqrt()

    def encode(rows):
        return torch.cat([model.encode(x[b]) for b in np.array_split(rows, 1+len(rows)//32)])

    source = [src[streams[src] == session] for session in np.unique(streams[src])]
    means = [torch.cat(model.unpack(encode(rows)), -1).mean(0) for rows in source]
    constant = torch.stack(means).mean(0)
    donor = torch.cat(model.unpack(model.encode(x[source[0]])), -1)
    held = encode(test)
    names = ('memory', 'blank', 'donor', 'source')
    bank = {k: np.full((8,len(test),3), np.nan, np.float32) for k in names}
    with np.load(train.PERM) as perms, np.load(archive) as original:
        assert np.array_equal(original['index'], test)
        for oi, order in enumerate(train.ORDERS):
            perm = perms[f'{order[:-1]}|{order[-1]}|all_loso']
            assert np.array_equal(np.sort(perm), np.arange(len(x)))
            for session in np.unique(streams[test]):
                rows = perm[streams[perm] == session]
                pos = np.searchsorted(test, rows)
                assert np.array_equal(test[pos],rows)
                for name in names:
                    writes = None
                    if name == 'donor':
                        writes = donor[torch.arange(len(pos),device='cuda') % len(donor)]
                    elif name == 'source':
                        writes = constant.expand(len(pos), -1)
                    logits = model.sequence(held[pos], 'static' if name=='blank' else 'memory', donor=writes)[0]
                    bank[name][oi,pos] = logits.cpu().numpy()
                assert all(np.array_equal(bank[n][oi,pos[0]], bank['memory'][oi,pos[0]]) for n in names)
            for n in ('memory','donor','blank'):
                assert np.array_equal(bank[n][oi].argmax(1),original[n+'|'+order]), (person,order,n)
    if seed == 1441:
        control = folder/(person+'.controls.npz')
        proof = json.loads((folder/(person+'.controls.json')).read_text())
        assert proof['sha'] == ref.sha(control) and proof['checkpoint'] == ref.sha(checkpoint)
        with np.load(control) as z:
            assert np.array_equal(test,z['index']) and np.array_equal(bank['source'].argmax(2),z['source_write'])
    assert all(np.isfinite(v).all() for v in bank.values())
    dest = Path(f'{seed}_{person}.npz')
    np.savez_compressed(dest, index=test, **bank)
    dest.with_suffix('.json').write_text(json.dumps(dict(source=ref.sha(Path('source.sha256')),
        checkpoint=ref.sha(checkpoint), original=ref.sha(archive), output=ref.sha(dest))))


if __name__ == '__main__':
    replay(int(sys.argv[1]),sys.argv[2])
