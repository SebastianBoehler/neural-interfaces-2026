"""Train the §379 static waveform model on real versus partially phase-randomized source batches."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import Decoder
from source_labels import read_training_labels
from train import DATA, PERM, ORDERS, digest, sha

ARMS = ('real', 'phase')
ARCHIVED = Path('/mnt/lustre/work/gehler/gfh527/ni26_wave379_v1_20261118')


def rotate(x, gen, identity=False):
    z = torch.fft.rfft(x)
    angle = torch.rand(len(x), 1, z.shape[-1], device=x.device, generator=gen) * (2 * torch.pi)
    angle[..., [0, -1]] = 0
    if identity:
        angle.zero_()
    return torch.fft.irfft(z * torch.polar(torch.ones_like(angle), angle), n=x.shape[-1])


def transform_checks(x):
    gen = torch.Generator(device=x.device).manual_seed(435)
    y = rotate(x, gen)
    z, s = torch.fft.rfft(x), torch.fft.rfft(y)
    cross = lambda a: a[:, :, None] * a[:, None, :].conj()
    a, b = cross(z), cross(s)
    relative = (a-b).abs().amax(-1) / a.abs().amax(-1).clamp_min(1e-12)
    rms = lambda a: a.square().mean().sqrt()
    identity = rms(rotate(x, gen, True)-x) / rms(x)
    changed = rms(y-x)/rms(x)
    kurt = lambda v: ((v-v.mean(-1, keepdim=True))/v.std(-1, keepdim=True)).pow(4).mean(-1)
    assert relative.max() < 1e-4 and identity < 1e-5 and changed > .05
    return dict(max_pair_cross_relative=float(relative.max()), identity_relative_rms=float(identity),
                relative_wave_change=float(changed), mean_absolute_kurtosis_change=float((kurt(x)-kurt(y)).abs().mean()))


def main(held, smoke):
    assert 'A100' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert sha(DATA) == '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
    assert sha(PERM) == '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'
    with np.load(DATA) as data:
        people = np.array([s.split('/')[-1] for s in data['subject']])
        streams = np.char.add(np.char.add(people, '|'), data['session'])
        interfaces = data['interface']
        x = torch.from_numpy(data['X']).float().cuda()
    fit = np.flatnonzero(people != held)
    held_rows = np.flatnonzero(people == held)
    assert len(set(people[fit])) == 9 and len(held_rows)
    x /= x[fit].square().mean().sqrt()
    proof = transform_checks(x[fit[:8]])
    labels = read_training_labels(DATA, fit, len(x))
    y = torch.full((len(x),), -1, device='cuda', dtype=torch.long)
    y[fit] = torch.from_numpy(labels).cuda()
    sessions = [fit[streams[fit] == s] for s in sorted(set(streams[fit]))]
    weight = torch.zeros(len(x), device='cuda')
    for rows in sessions:
        local = labels[np.searchsorted(fit,rows)]
        for iface in set(interfaces[rows]):
            for k in range(3):
                cell = rows[(interfaces[rows] == iface) & (local == k)]
                assert len(cell)
                weight[cell] = len(rows)/len(cell)/6
    rng = np.random.default_rng(379)
    batches = [rng.choice(sessions[rng.integers(len(sessions))], size=96, replace=False)
               for _ in range(2 if smoke else 360)]
    out = {'index': held_rows}
    for arm in ARMS:
        torch.manual_seed(379)
        gen = torch.Generator(device='cuda').manual_seed(435)
        net = Decoder().cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
        for rows in batches:
            opt.zero_grad()
            batch = x[rows]
            if arm == 'phase':
                batch[:48] = rotate(batch[:48], gen)
            logits, _ = net.step(batch, 'static')
            loss = (F.cross_entropy(logits,y[rows],reduction='none') * weight[rows]).mean()
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            active = fit[:12] if smoke else held_rows
            enc = torch.cat([net.encode(x[r]) for r in np.array_split(active,1+len(active)//32)])
            logits = net.sequence(enc,'static')[0]
            single = torch.cat([net.step(x[i:i+1],'static')[0] for i in active[:12]])
            assert torch.allclose(single,logits[:len(single)],rtol=1e-4,atol=2e-4)
            assert torch.isfinite(logits).all()
            out[arm] = logits.argmax(1).cpu().numpy()
    if smoke:
        print(proof, 'source smoke passed', flush=True)
        return
    with np.load(ARCHIVED/(held+'.npz')) as old:
        assert np.array_equal(old['index'], held_rows)
        for order in ORDERS:
            assert np.array_equal(old['static|'+order],out['real']), 'archived control mismatch'
    np.savez_compressed(held+'.npz', **out)
    proof.update(source_sha=sha(Path('source.sha256')), episode_sha=digest(np.stack(batches)),
                 train_rows=digest(fit), output_sha=sha(Path(held+'.npz')), source_people=sorted(set(people[fit])))
    Path(held+'.json').write_text(json.dumps(proof,indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:] == ['smoke'])
