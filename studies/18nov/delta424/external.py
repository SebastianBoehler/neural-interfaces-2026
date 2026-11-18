"""Evaluate the locked fits on the same external trials and reset horizons as §418."""
from pathlib import Path
import json
import sys

import numpy as np
import torch
from fits import WORK, models, prior

HIST = WORK/'ni26_ext418_v2_20261118'


@torch.no_grad()
def run(person):
    _,_,source_rows,runtime = prior.source()
    prior.owners(source_rows)
    assert runtime.align is False
    nets = models(runtime.net)
    native = runtime.net.eval()
    native.spatial_conv[1].k = 20.
    records = {}
    with np.load(WORK/'neural-interfaces-2026/data/shin2018/windows.npz') as z:
        rows = np.flatnonzero((z['subject']==f'Shin2018/{person}') & (z['win']==0))
        assert len(rows)==60 and len(set(z['session'][rows]))==1
        raw = z['X'][rows]
    with np.load(HIST/f'person{person}.npz') as old:
        for order in (415,1415):
            pos = np.random.default_rng([415,person,order,0]).permutation(60)
            assert np.array_equal(rows[pos],old[str(order)])
            records[str(order)] = rows[pos]
            x = torch.roll(torch.from_numpy(runtime._aligned(pos,True,src=raw)),0,-1)
            for (seed,arm),net in [((0,'native'),native),*nets.items()]:
                full = net(x)
                short = torch.cat([net(b) for b in x.split(20)])
                torch.testing.assert_close(full[:20],short[:20],atol=3e-4,rtol=1e-5)
                assert torch.isfinite(full).all() and torch.isfinite(short).all()
                pred = torch.stack((full.argmax(1),short.argmax(1))).numpy().astype('int8')
                assert np.array_equal(pred[0,:20],pred[1,:20])
                if arm=='blank':
                    assert np.array_equal(*pred)
                if arm=='native':
                    assert np.array_equal(pred,old[f'{order}|native'])
                else:
                    records[f'{order}|{arm}{seed}'] = pred
    np.savez_compressed(f'ext{person}.npz',**records)


if __name__=='__main__':
    run(int(sys.argv[1]))
