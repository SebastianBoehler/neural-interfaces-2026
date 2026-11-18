"""Seal all 18 blind outputs, verify both control banks, then read Scherer labels."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path('/mnt/lustre/work/gehler/gfh527')
DATA = BASE / 'neural-interfaces-2026/data/scherer2015/windows.npz'
PRIMARY = BASE / 'ni26_qualify385_v1_20261118'
REPLICATE = BASE / 'ni26_cross411b_20261118'
ORDERS = (0,1,411,1411)
SEEDS = (454,1454)
ARMS = ('stock',) + tuple(f'{s}_{arm}' for s in SEEDS for arm in ('memory','donor'))
sys.path.insert(0,str(PRIMARY))
from data import permutation
from protocol import score


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def contrast(delta):
    """Resample participants, then days and deliveries from each sampled participant."""
    rng = np.random.default_rng(459)
    n,days,orders = delta.shape
    ii = rng.integers(n,size=(10000,n,1,1))
    jj = rng.integers(days,size=(10000,n,days,1))
    kk = rng.integers(orders,size=(10000,n,days,orders))
    draws = delta[ii,jj,kk].mean((1,2,3))
    return dict(pp=float(delta.mean()),paired80=np.percentile(draws,[10,90]).tolist(),
                people=delta.mean((1,2)).tolist(),orders=delta.mean((0,1)).tolist())


def main(work):
    assert (work/'replay.log').read_text().splitlines()[-1] == 'OK'
    assert sha(DATA) == 'a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91'
    with np.load(DATA) as z:
        subjects,days,ifs = (z[k] for k in ('subject','session','interface'))
    people = sorted(set(subjects))
    assert len(people) == 9 and len(subjects) == 2130
    streams = np.char.add(np.char.add(subjects,'|'),days)
    expected = [permutation(streams,None,'full',s) for s in ORDERS]
    labels = np.full((len(ARMS),4,len(subjects)),-1,np.int8)
    receipt = dict(actors={},scorer=sha(Path(__file__)),source=sha(work/'source.sha256'),models={})
    for pair,archive in enumerate((PRIMARY,REPLICATE)):
        for index,person in enumerate(people):
            path = work/f'external_{pair}_{index}.json'
            receipt['actors'][path.name] = sha(path)
            actor = json.loads(path.read_text())
            assert actor['source'] == receipt['source']
            prior = archive/f'cap_{index}.json'
            assert actor['archive'] == sha(prior)
            published = json.loads(prior.read_text())
            assert set(actor['sessions']) == set(published)
            for s in SEEDS:
                digest = sha(BASE/'ni26_kernel459_20261118/weights'/f'all_{s}.pt')
                assert actor['models'][str(s)] == digest
                receipt['models'][str(s)] = digest
            for k,entry in actor['sessions'].items():
                name,day,seed = k.split('|')
                oi = ORDERS.index(int(seed))
                assert name == person and oi//2 == pair
                rows = expected[oi][(subjects==person)&(days==day)]
                assert np.array_equal(rows, entry['rows']) and entry['rows'] == published[k]['rows']
                assert entry['counters'] == published[k]['activity'] and entry['arms']['stock'] == published[k]['labels']
                assert set(entry['arms']) == set(ARMS)
                for ai,arm in enumerate(ARMS):
                    assert np.all(labels[ai,oi,rows]==-1)
                    labels[ai,oi,rows] = entry['arms'][arm]
    assert np.isin(labels,(0,1,2)).all()
    (work/'external_before_labels.json').write_text(json.dumps(receipt,indent=2))
    with np.load(DATA) as z:
        y = z['y']
    cells = np.empty((len(ARMS),9,2,4))
    for pi,p in enumerate(people):
        visits = sorted(set(days[subjects==p]))
        assert len(visits) == 2
        for di,day in enumerate(visits):
            correct = []
            for c in range(3):
                mask = (subjects==p)&(days==day)&(y==c)
                assert mask.any()
                correct.append((labels[:,:,mask]==c).mean(2))
            cells[:,pi,di,:] = 100*np.mean(correct,axis=0)
    for ai,arm in enumerate(ARMS):
        for oi in range(4):
            metric = score(y,labels[ai,oi],dict(stream=streams,interface=ifs),np.ones(len(y),bool))
            assert metric['n_cells']==18 and np.isclose(cells[ai,:,:,oi].mean(),100*metric['balanced_accuracy_cell_mean'])
    groups = {'canonical':slice(0,2),'replication':slice(2,4),'four_orders':slice(0,4)}
    summaries = {}
    for group,order in groups.items():
        table = cells[:,:,:,order]
        arms = {a:float(v.mean()) for a,v in zip(ARMS,table)}
        contrasts = {}
        for seed in SEEDS:
            genuine = ARMS.index(f'{seed}_memory')
            for control in ('stock',f'{seed}_donor'):
                other = ARMS.index(control)
                a,b = labels[genuine,order],labels[other,order]
                contrasts[f'{seed}_memory-{control}'] = dict(**contrast(table[genuine]-table[other]),
                    changed=int(np.count_nonzero(a!=b)), correct=int(np.count_nonzero((a==y)&(b!=y))),
                    regressed=int(np.count_nonzero((a!=y)&(b==y))))
        summaries[group]=dict(accuracy=arms,contrasts=contrasts)
    assert abs(summaries['canonical']['accuracy']['stock']-48.45568783068783)<1e-10
    assert abs(summaries['replication']['accuracy']['stock']-48.716931216931215)<1e-10
    result=dict(people=people,orders=ORDERS,decisions_per_arm=labels[0].size,
                proof=sha(work/'external_before_labels.json'),models=receipt['models'],groups=summaries)
    (work/'external_result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
