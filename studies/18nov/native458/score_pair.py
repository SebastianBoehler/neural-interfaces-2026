"""Decode both §454 seeds before accessing labels, using exact §456 single-seed controls."""
import json
from pathlib import Path
import sys
import numpy as np
from fusion_inputs import official
from score import interval
from stocks456 import inputs, log_probability, module, MEMORY, ref

PINS = ('233519fccdcc9d556bf12525a5896802471466b40802288ccb21c86b7c536222',
        '5e73868495f5a978baa39e31d24ec5d53267e245fdf34aea21c2eb63dd5f09d2')


def run(smoke):
    meta,ages,_,stock=inputs()
    loader=module(MEMORY,'corrections','528f76f9d69d56e4be714bb0cf6d48bbff665b46dad1f7eeddcd0cb1307e1e07')
    bank={}
    for p in ref.PEOPLE[:1] if smoke else ref.PEOPLE:
        rows=np.flatnonzero(meta['person']==p)
        pair=[]
        for seed,digest in zip((454,1454),PINS):
            path=ref.WORK/'ni26_match456_v2_20261118'/f'kernel_{seed}_decisions.npz'
            assert ref.sha(path)==digest
            delta,_=loader.load(MEMORY,'kernel',seed,p,rows)
            with np.load(path) as archive:
                for i,o in enumerate(ref.ORDERS):
                    logp=log_probability(stock[o][rows],ages[o][rows])
                    for arm in ('source','memory','donor'):
                        key=p+'|'+o+'|'
                        predicted=(logp+delta[arm][i]).argmax(1)
                        old=archive[key+('stock' if arm=='source' else arm)]
                        np.testing.assert_array_equal(predicted,old)
                        if arm=='source':
                            bank[key+'stock']=old
                        elif arm=='memory':
                            bank[key+str(seed)]=old
            pair.append(delta)
        for i,o in enumerate(ref.ORDERS):
            logp=log_probability(stock[o][rows],ages[o][rows])
            for arm in ('memory','donor'):
                averaged=(pair[0][arm][i]+pair[1][arm][i])/2
                bank['|'.join((p,o,arm))]=(logp+averaged).argmax(1)
    if smoke:
        print('Real P01, eight orders, both seeds and all original conditions match.')
        return
    path=Path('pair_predictions.npz')
    np.savez_compressed(path,**bank)
    receipt=dict(predictions=ref.sha(path),source=ref.sha(Path('source.sha256')))
    Path('blind.json').write_text(json.dumps(receipt))
    data=ref.BASE/'data/proteus_windows.npz'
    assert ref.sha(data)==ref.DATA_SHA
    with np.load(data) as z:
        y=z['y']
    metric=official()
    scores={a:np.empty((10,8)) for a in ('stock','memory','donor','454','1454')}
    counts=[]
    for pi,p in enumerate(ref.PEOPLE):
        mask=meta['person']==p
        counts.append(len(set(zip(meta['stream'][mask],meta['interface'][mask]))))
        for oi,o in enumerate(ref.ORDERS):
            for a in scores:
                result=metric(y[mask],bank['|'.join((p,o,a))],meta,mask)
                assert result['n_cells']==counts[-1]
                scores[a][pi,oi]=100*result['balanced_accuracy_cell_mean']
    counts=np.array(counts)
    absolute={a:interval(v,counts)['pp'] for a,v in scores.items()}
    assert abs(absolute['stock']-76.10391516713594)<1e-10
    contrasts={a:interval(scores['memory']-v,counts) for a,v in scores.items() if a!='memory'}
    Path('pair_result.json').write_text(json.dumps(dict(**receipt,absolute=absolute,
         memory_minus=contrasts),indent=2))


if __name__=='__main__':
    run(sys.argv[1:] == ['smoke'])
