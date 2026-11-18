"""Estimate task effects in the frozen J memory coordinates, using source sessions only."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

CHECKPOINTS = {382: '7cc45204bfed24c345c535c06a296fefaebc1e4af252b4caad599fe035a34d77',
               1382: '0185f49dad54dbc8ce8cfc2176fe6666041d829adb883050a1385a1be95b0bcf'}
BUNDLE = '06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
ROWS = '882b5b97ec811d697540f6152be0d02eb49448e9e4c88c3d6c28fcb1433b2f1f'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


@torch.no_grad()
def main(smoke):
    bundle = Path('../382/bundle')
    assert sha(bundle/'bundle.json') == BUNDLE
    meta = json.loads((bundle/'bundle.json').read_text())
    assert meta['source_only'] and meta['people'] == ['P04','P05','P06','P07','P08','P10']
    for name, expected in meta['files'].items():
        assert sha(bundle/name) == expected
    assert 'H100' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(bundle/'source.npz', allow_pickle=False) as d:
        rows, streams, y, x = (d[k] for k in ('source_rows','streams','y','x'))
    assert hashlib.sha256(rows.tobytes()).hexdigest() == ROWS == meta['source_rows_sha256']
    assert x.shape == (len(rows),41,480) and len(np.unique(rows)) == len(rows)
    rec = dict(bundle=BUNDLE, rows=ROWS, people=meta['people'], source=sha(Path('source.sha256')),
               checkpoints=CHECKPOINTS, n=len(y), sessions=len(np.unique(streams)),
               device=torch.cuda.get_device_name(), norms={})
    bank = {}
    for seed, checkpoint in CHECKPOINTS.items():
        path = Path('J_382.pt') if seed == 382 else bundle.parent/'J_1382.pt'
        assert sha(path) == checkpoint
        net = torch.jit.load(str(bundle/'learned.ts'), map_location='cuda').eval()
        net.load_state_dict(torch.load(path, weights_only=True, map_location='cuda'))
        conv = getattr(net.spatial_conv, '0')
        layer = getattr(net.spatial_conv, '1')
        chunks = []
        for batch in np.array_split(np.arange(len(y)), np.ceil(len(y)/48).astype(int)):
            h = conv(net.spectral_filtering(torch.from_numpy(x[batch]).cuda()))
            z = ((h-layer.mean[None,:,None,None])/layer.std[None,:,None,None]).squeeze(2)
            a = z.abs()
            cutoffs = layer.hinges*(.25*layer.threshold.tanh()).exp()
            codes = torch.stack([z.mean(-1),a.mean(-1),*((a-v).relu().mean(-1) for v in cutoffs)],-1)
            assert torch.isfinite(codes).all() and codes.shape[0] == len(batch)
            chunks.append(codes.cpu().numpy())
            if smoke:
                break
        if smoke:
            continue
        codes = np.concatenate(chunks)
        means = np.stack([[codes[(streams==s)&(y==c)].mean(0,dtype=np.float64)
                           for c in range(3)] for s in np.unique(streams)])
        assert np.isfinite(means).all()
        prototypes = (means-means.mean(1,keepdims=True)).mean(0)
        assert np.max(abs(prototypes.mean(0))) < 1e-9 and np.linalg.norm(prototypes) > 1e-4
        bank[str(seed)] = prototypes.astype(np.float32)
        bank[f'probe{seed}'] = codes[:8]
        rec['norms'][seed] = float(np.linalg.norm(prototypes))
    if not smoke:
        np.savez_compressed('prototypes.npz', **bank)
        rec['output'] = sha(Path('prototypes.npz'))
        Path('source.json').write_text(json.dumps(rec,indent=2))
    print(rec, flush=True)


if __name__ == '__main__':
    main(sys.argv[1] == 'smoke')
