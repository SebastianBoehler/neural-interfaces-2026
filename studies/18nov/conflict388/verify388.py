"""Pin both source-only H100 fit sets and verify all controls received identical episodes."""
import hashlib
import json
from pathlib import Path

TRAIN_SHA='4bf417f9b5dcf1ba94215b7247c80664a6217b16e0e85b72ccbe76d2b066e8ca'
BUNDLE='06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
ROWS='882b5b97ec811d697540f6152be0d02eb49448e9e4c88c3d6c28fcb1433b2f1f'
FITS=Path('fits')
SEEDS=(382,1382)
PEOPLE=('P04','P05','P06','P07','P08','P10')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    records={}
    for seed in SEEDS:
        pair={}
        for arm in ('N','H','X'):
            p=FITS/f'{arm}_{seed}.json'
            rec=json.loads(p.read_text())
            assert rec['source']==TRAIN_SHA and rec['bundle']==BUNDLE and rec['rows']==ROWS
            assert rec['arm']==arm and rec['seed']==seed and rec['steps']==120 and 'H100' in rec['device']
            assert tuple(rec['people'])==PEOPLE and rec['held_labels_opened'] is False
            assert rec['weights']==sha(p.with_suffix('.pt'))
            assert set(rec['original'])=={'body','norm','head'} and all(rec['first_grad'][g]>0 and
                rec['original'][g]!=rec['after'][g] for g in rec['original'])
            assert rec['conflicts']>0 and rec['mean_higher_loss_weight']>.5
            assert (rec['max_projection_delta']>0)==(arm=='X')
            pair[arm]=rec
        for arm in ('H','X'):
            for key in ('episodes','original','frozen'):
                assert pair[arm][key]==pair['N'][key]
        assert len({r['weights'] for r in pair.values()})==3
        records[seed]=pair
    assert records[382]['N']['episodes']!=records[1382]['N']['episodes']
    return records


if __name__=='__main__':
    fits=verify()
    with Path('fits.json').open('x') as f:
        json.dump(fits,f,indent=2)
