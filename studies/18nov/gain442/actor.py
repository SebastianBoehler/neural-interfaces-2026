"""Replay matched fits with genuine, blank and source-donor histories on held EEG."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

WORK = Path('/mnt/lustre/work/gehler/gfh527')
sys.path[:0] = [str(WORK/'ni26_joint382_v1_20261118'), str(WORK/'ni26_conflict388_v1_20261118')]
from actor388 import ARCH, ROWS, SEEDS, joint, prior, init

FITS = ('clean', 'coherent', 'independent')
MODES = ('real', 'blank', 'donor')
ARMS = ('A6', 'J_real', 'frozen') + tuple(f'{arm}_{mode}' for arm in FITS for mode in MODES)
SOURCE = '1a4686294eecbb064ded9954d615c244443d2bae73180f677ea7e3ebe87c4fdc'
BUNDLE = '06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'


def lock():
    records = {}
    joint.pin(Path('fits/source.sha256'), SOURCE)
    for seed in SEEDS:
        history = None
        for arm in FITS:
            path = Path('fits')/f'{arm}_{seed}.json'
            rec = json.loads(path.read_text())
            assert (rec['arm'],rec['seed'],rec['steps'],rec['rows']) == (arm,seed,120,ROWS)
            assert rec['bundle'] == BUNDLE and rec['source'] == SOURCE
            joint.pin(path.with_suffix('.pt'), rec['checkpoint'])
            matched = [rec[k] for k in ('gain','coherent','independent','before','frozen_sha','batches')]
            assert history is None or history == matched
            history = matched
            records[f'{seed}|{arm}'] = rec
    return records


@torch.no_grad()
def run(person):
    assert json.loads(Path('fits.json').read_text()) == lock()
    d, people, rows, runtime = prior.source()
    prior.owners(rows)
    streams, held, _, people2 = joint.banks(joint.verify())
    test = held[person]
    assert runtime.align is False and runtime.scale > 0 and not runtime.crops
    assert np.array_equal(people, people2) and np.array_equal(test, np.flatnonzero(people == person))
    assert joint.digest(rows) == ROWS and not np.intersect1d(test, rows).size
    assert len(np.unique(d['stream'][test])) == 1
    donor = np.flatnonzero((people == 'P04') & (d['session'].astype(str) == '1'))
    assert len(donor) == 189 and np.isin(donor, rows).all()
    donor_x = torch.from_numpy(runtime._aligned(donor,True))
    native = init(runtime.net, 'K')
    perms = joint.permutations(len(people))
    bank = {}
    for seed, (root, digest) in ARCH.items():
        joint.pin(root/'before_labels.json', digest)
        record = json.loads((root/'before_labels.json').read_text())['actors'][person]
        assert record['train_rows'] == ROWS and record['test_rows'] == joint.digest(test)
        joint.pin(root/(person+'.npz'), record['outputs_sha256'])
        nets, codes = {}, {}
        for arm in FITS:
            net = init(runtime.net, 'J')
            net.load_state_dict(torch.load(f'fits/{arm}_{seed}.pt', weights_only=True, map_location='cpu'))
            layer = net.spatial_conv[1]
            codes[arm] = layer.encode(layer.standardized(prior.preactivation(net,donor_x)))
            nets[arm] = net
        with np.load(root/(person+'.npz')) as z:
            assert np.array_equal(test, z['index'])
            for order in joint.ORDERS:
                pos = streams[person][order][2]
                rr = test[pos]
                assert np.array_equal(rr, perms[order][np.isin(perms[order],test)])
                x = torch.roll(torch.from_numpy(runtime._aligned(rr,True)),0,-1)
                undo = np.argsort(pos)
                gold = native(x).double().softmax(1).numpy()[undo]
                ref = z['A6|'+order]
                assert abs(gold-ref).max() < 1e-6 and np.array_equal(gold.argmax(1),ref.argmax(1))
                for arm in ('A6','J_real','frozen'):
                    bank[f'{seed}|{arm}|{order}'] = z[arm+'|'+order].argmax(1).astype(np.int8)
                for arm, net in nets.items():
                    layer = net.spatial_conv[1]
                    ind = np.searchsorted(donor, perms[order][np.isin(perms[order],donor)])
                    layer.donor = codes[arm][ind]
                    first = None
                    for mode in MODES:
                        layer.mode = mode
                        logits = net(x)
                        assert torch.isfinite(logits).all()
                        if first is not None:
                            torch.testing.assert_close(logits[0], first, atol=2e-5,rtol=1e-5)
                        first = logits[0]
                        if order == joint.ORDERS[0]:
                            torch.testing.assert_close(net(x[:8]),logits[:8],atol=3e-4,rtol=1e-5)
                            h = prior.preactivation(net,x[:8])
                            manual = prior.single_steps(layer,layer.standardized(h),mode,layer.donor)
                            torch.testing.assert_close(layer(h),manual,atol=2e-4,rtol=1e-5)
                        bank[f'{seed}|{arm}_{mode}|{order}'] = logits.argmax(1).numpy()[undo].astype(np.int8)
    path = Path(person+'.npz')
    np.savez_compressed(path,index=test,**bank)
    doc = dict(held=person,train=ROWS,test=joint.digest(test),output=joint.sha(path),
               fits=joint.sha(Path('fits.json')),source=joint.sha(Path('source.sha256')))
    Path(person+'.json').write_text(json.dumps(doc,indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'lock':
        Path('fits.json').write_text(json.dumps(lock(),indent=2))
    else:
        run(sys.argv[1])
