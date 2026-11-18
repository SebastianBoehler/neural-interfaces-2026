"""§439 electrode gains on the source-scaled waveform at the original A6 input."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

BUNDLE = Path('../382/bundle').resolve()
sys.path.insert(0, str(BUNDLE))
from fit_bundle import sha
from kernel import schedule, sha_array, train

ARMS = ('clean', 'coherent', 'independent')
ROWS = '882b5b97ec811d697540f6152be0d02eb49448e9e4c88c3d6c28fcb1433b2f1f'


def gain_schedule(x, labels, streams, steps):
    power = np.einsum('nct,nct->nc', x, x) / x.shape[2]
    assert np.isfinite(power).all() and power.min() > 0
    rms = .5 * np.log(power)
    templates = np.stack([np.mean([rms[(streams == stream) & (labels == k)].mean(0)
                                  for k in range(3)], axis=0) for stream in np.unique(streams)])
    rng = np.random.default_rng(439)
    patterns = np.zeros((steps, 41), dtype=np.float32)
    for index in range(1, steps, 2):
        a, b = rng.choice(len(templates), 2, replace=False)
        diff = .5 * (templates[a] - templates[b])
        patterns[index] = np.clip(diff - diff.mean(), -np.log(2), np.log(2))
    gains = np.exp(patterns)
    assert np.isfinite(gains).all() and np.max(abs(gains - 1)) > .03
    coherent = np.repeat(np.arange(steps)[:, None], 48, axis=1)
    independent = coherent.copy()
    for start in range(0, steps, 12):
        block = independent[start:start + 12]
        block[:] = rng.permutation(block.flatten()).reshape(block.shape)
        assert np.array_equal(np.sort(block.ravel()), np.sort(coherent[start:start+12].ravel()))
    assert not np.array_equal(gains[independent], gains[coherent])
    return gains, coherent, independent


class Episodes:
    def __init__(self, x, batches, gain, order, arm):
        self.x, self.batches, self.gain = x, batches, gain
        self.order, self.arm = order, arm
        self.shape, self.step = x.shape, 0

    def __getitem__(self, rows):
        assert np.array_equal(rows, self.batches[self.step])
        step = self.step
        self.step += 1
        batch = self.x[rows]
        if self.arm == 'clean':
            return batch
        return batch * self.gain[self.order[step], :, None]


def main(arm, seed, smoke):
    assert arm in ARMS and 'H100' in torch.cuda.get_device_name()
    meta = json.loads((BUNDLE / 'bundle.json').read_text())
    for name, digest in meta['files'].items():
        assert sha(BUNDLE / name) == digest
    assert meta['source_only'] and meta['source_rows_sha256'] == ROWS
    assert meta['people'] == ['P04','P05','P06','P07','P08','P10']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    steps = 12 if smoke else 120
    with np.load(BUNDLE / 'source.npz', allow_pickle=False) as d:
        x, labels, streams = d['x'], d['y'], d['streams']
        assert sha_array(d['source_rows']) == ROWS
        gain, coherent, independent = gain_schedule(x, labels, streams, steps)
        x, y, w = [torch.from_numpy(v).cuda() for v in (x, labels, d['weights'])]
    batches = schedule(streams, seed, steps)
    net = torch.jit.load(str(BUNDLE / 'learned.ts'), map_location='cuda').eval()
    gain = torch.from_numpy(gain).cuda()
    assert torch.equal(x[batches[0]] * gain[0,:,None], x[batches[0]])
    with np.load(BUNDLE / 'probes.npz') as z, torch.no_grad():
        probe = x[schedule(streams,382,1)[0]]
        for _ in range(3):
            actual = net(probe)
        target = torch.from_numpy(z['learned']).cuda()
        assert (actual-target).abs().max() < .005 and torch.equal(actual.argmax(1), target.argmax(1))
    order = independent if arm == 'independent' else coherent
    episodes = Episodes(x, batches, gain, order, arm)
    rec = train(net, episodes, y, w, streams, 'J', seed, steps)
    assert episodes.step == steps
    key = ('smoke_' if smoke else '') + f'{arm}_{seed}'
    rec.update(arm=arm, rows=ROWS, source=sha(Path('source.sha256')), bundle=sha(BUNDLE/'bundle.json'),
               gain=sha_array(gain.cpu().numpy()), coherent=sha_array(coherent), independent=sha_array(independent))
    if not smoke:
        path = Path(key+'.pt')
        torch.save({k:v.cpu() for k,v in net.state_dict().items()}, path)
        rec['checkpoint'] = sha(path)
    Path(key+'.json').write_text(json.dumps(rec,indent=2))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:] == ['smoke'])
