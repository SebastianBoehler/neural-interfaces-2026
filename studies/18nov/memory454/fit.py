import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from memory import SpectralMemory
from source_labels import read_training_labels

DATA = Path('proteus_windows.npz')
PERM = Path('shuffle_perms.npz')
ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')
BANDS = ((1,4),(4,8),(8,13),(13,20),(20,30),(30,45))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def features(x):
    frequencies = torch.fft.rfftfreq(480, 1/120, device=x.device)
    power = (torch.fft.rfft(x * torch.hann_window(480, device=x.device))).abs().square()
    return torch.cat([power[:,:, (frequencies >= lo) & (frequencies < hi)].mean(-1).clamp_min(1e-8).log()
                      for lo, hi in BANDS], dim=1)


def fit(kind, seed, held, smoke, augmentation=None):
    architecture = 'kernel' if augmentation is not None else kind
    assert architecture in ('kernel', 'covariance') and 'H100' in torch.cuda.get_device_name()
    assert sha(DATA) == '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
    assert sha(PERM) == '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(DATA) as z:
        people = np.array([p.split('/')[-1] for p in z['subject']])
        streams = np.char.add(np.char.add(people, '|'), z['session'])
        interfaces = z['interface']
        x = torch.cat([features(torch.from_numpy(part).float().cuda()) for part in np.array_split(z['X'], 20)])
    src, test = np.flatnonzero(people != held), np.flatnonzero(people == held)
    assert len(set(people[src])) == 9 and len(test) and held not in people[src]
    scale = x[src].std(0).clamp_min(.05)
    standardized = (x - x[src].mean(0)) / scale
    x = standardized.clamp(-8, 8)
    labels = read_training_labels(DATA, src, len(x))
    y = torch.full((len(x),), -1, device='cuda', dtype=torch.long)
    y[src] = torch.from_numpy(labels).cuda()
    assert (y[test] == -1).all()
    sessions = [src[streams[src] == s] for s in np.unique(streams[src])]
    weights = np.zeros(len(x), np.float32)
    for rows in sessions:
        local = labels[np.searchsorted(src, rows)]
        for iface in np.unique(interfaces[rows]):
            for c in range(3):
                cell = rows[(interfaces[rows] == iface) & (local == c)]
                assert len(cell)
                weights[cell] = len(rows) / (6 * len(cell))
    weights = torch.from_numpy(weights).cuda()
    rng = np.random.default_rng(seed)
    model = SpectralMemory(architecture).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    for _ in range(4 if smoke else 400):
        rows = rng.choice(sessions[rng.integers(len(sessions))], size=96, replace=False)
        features_batch = model.encoder(x[rows])
        logits, _ = model.sequence(features_batch, model.writes(features_batch))
        loss = (F.cross_entropy(logits, y[rows], reduction='none') * weights[rows]).mean()
        if augmentation is not None:
            loss = loss + augmentation(model, kind, standardized[rows], scale, y[rows], weights[rows], logits)
        optimizer.zero_grad()
        loss.backward()
        assert torch.isfinite(loss) and model.key.weight.grad.abs().max() > 0
        optimizer.step()
    model.eval()
    with torch.no_grad(), np.load(PERM) as orders:
        encoded = model.encoder(x)
        writes = model.writes(encoded)
        centroid = torch.stack([writes[rows].mean(0) for rows in sessions]).mean(0)
        sample = src[:16]
        a = encoded[sample]
        w = writes[sample]
        full = model.sequence(a, w)[0]
        prefix, state = model.sequence(a[:5], w[:5])
        tail = model.sequence(a[5:], w[5:], state)[0]
        singles, state = [], None
        for i in range(len(a)):
            row, state = model.sequence(a[i:i+1], w[i:i+1], state)
            singles.append(row)
        assert torch.allclose(full, torch.cat([prefix, tail]), atol=1e-5)
        assert torch.allclose(full, torch.cat(singles), atol=1e-5)
        assert torch.equal(full, model.sequence(a, w)[0])
        assert (full - model.sequence(a, centroid.expand_as(w))[0]).abs().max() > 1e-5
        bank = {n:np.full((8, len(test), 3), np.nan, np.float32) for n in ('memory','donor','source')}
        for oi, order in enumerate(ORDERS):
            perm = orders[f'{order[:-1]}|{order[-1]}|all_loso']
            assert np.array_equal(np.sort(perm), np.arange(len(x)))
            for s in np.unique(streams[test]):
                rows = perm[streams[perm] == s]
                pos = np.searchsorted(test, rows)
                donor = writes[sessions[0][np.arange(len(rows)) % len(sessions[0])]]
                for arm, update in [('memory', writes[rows]), ('donor', donor), ('source', centroid.expand_as(donor))]:
                    bank[arm][oi, pos] = model.sequence(encoded[rows], update)[0].cpu().numpy()
                assert all(np.array_equal(bank[n][oi,pos[0]],bank['source'][oi,pos[0]]) for n in bank)
        assert all(np.isfinite(v).all() for v in bank.values())
        if smoke:
            print('H100 real-data source fit, prefix, step, reset and orders OK', kind, flush=True)
            return
        stem = f'{kind}_{seed}_{held}'
        path = Path(stem+'.npz')
        torch.save(model.state_dict(), stem+'.pt')
        np.savez_compressed(path, index=test, **bank)
        path.with_suffix('.json').write_text(json.dumps(dict(held=held, seed=seed, kind=kind,
            train_people=np.unique(people[src]).tolist(), source=sha(Path('source.sha256')), output=sha(path),
            checkpoint=sha(path.with_suffix('.pt')), orders=sha(PERM), data=sha(DATA)), indent=2))


if __name__ == '__main__':
    fit(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:] == ['smoke'])
