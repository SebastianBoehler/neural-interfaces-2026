"""Commit paired blind banks, then score with official cells and paired intervals."""
import json
from pathlib import Path
import sys

import numpy as np
from fusion_inputs import official
from score import interval
from inputs import inputs, ref, REPORT_SHA
import native
import terminal

PAIRS = {'native': [('prior','stock'),('reliable','stock'),('reliable','cyclic'),('stack','prior')],
         'consensus': [('consensus','stock'),('consensus','mean'),('consensus','donor_consensus'),
                       ('residual','consensus'),('residual','stock'),('residual','donor_residual'),('residual','prior')]}


def main(family, smoke):
    assert family in ('consensus','native','kernel_454','kernel_1454','covariance_454','covariance_1454','gain_454','gain_1454')
    meta, ages, old, stock = inputs()
    bank = (native.build(meta,ages,stock,smoke) if family == 'native' else
            terminal.build(family,meta,ages,old,stock,smoke))
    if smoke:
        print(f'{family} real stock, zero and historical parity OK', flush=True)
        return
    path = Path(f'{family}_decisions.npz')
    np.savez_compressed(path, **{'|'.join(k):v for k,v in bank.items()})
    proof = dict(bank=ref.sha(path), inputs=REPORT_SHA, source=ref.source_manifest(Path('source.sha256')))
    path.with_suffix('.json').write_text(json.dumps(proof, indent=2))
    assert ref.sha(ref.BASE/'data/proteus_windows.npz') == ref.DATA_SHA
    with np.load(ref.BASE/'data/proteus_windows.npz') as z:
        labels = z['y']
    metric = official()
    scores = {k[2]:np.empty((10,8)) for k in bank}
    counts = []
    for pi,p in enumerate(ref.PEOPLE):
        mask = meta['person'] == p
        counts.append(len(set(zip(meta['stream'][mask],meta['interface'][mask]))))
        for oi,o in enumerate(ref.ORDERS):
            for arm,table in scores.items():
                res = metric(labels[mask], bank[p,o,arm], meta, mask)
                assert res['n_cells'] == counts[-1]
                table[pi,oi] = 100*res['balanced_accuracy_cell_mean']
    counts = np.array(counts)
    assert counts.tolist() == [2,2,2,6,2,2,2,2,2,6]
    absolute = {arm:interval(table,counts)['pp'] for arm,table in scores.items()}
    assert abs(absolute['stock']-76.10391516713594) < 1e-10
    contrasts = {}
    pairs = PAIRS.get(family,[('memory','stock'),('memory','donor')])
    if family.startswith('gain_'):
        pairs = [('coherent_memory',arm) for arm in ('kernel_memory','scattered_memory','coherent_donor','stock')]
        pairs.extend([(actor+'_memory','stock') for actor in ('kernel','scattered')])
    for a,b in pairs:
        diff = scores[a]-scores[b]
        changed = sum(np.count_nonzero(bank[p,o,a] != bank[p,o,b]) for p in ref.PEOPLE for o in ref.ORDERS)
        contrasts[a+'-'+b] = dict(**interval(diff,counts), changed=int(changed),
                                   orders=dict(zip(ref.ORDERS,np.average(diff,weights=counts,axis=0))))
    Path(f'{family}_result.json').write_text(json.dumps(dict(**proof,absolute=absolute,contrasts=contrasts),indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:] == ['smoke'])
