"""Certify the held-session source masks, global deliveries and archived slots."""
import argparse
from pathlib import Path

import numpy as np

from probe import ORDERS,fingerprint,read_inputs,write_json

FAMILIES={'fbcnet_ea_tf':(1,5,'fbcnet','FBCNet'),
          'shallow_ea':(3,6,'shallow','ShallowFBCSPNet')}
PEOPLE={'P04':(2278,377,189,(2153,2383)), 'P10':(2270,385,192,(1238,1326))}
LAYOUT=('recording',*ORDERS)


def rows(root,data):
    from protocol import folds
    roster={}
    for name,tr,te in folds(data,'seen_session'):
        held=name.removesuffix('_s23')
        own=data['subject']==f'Dreyer2026Proteus/{held}'
        later=own & np.isin(data['session'],('2','3'))
        first=own & (data['session']=='1')
        expected=PEOPLE[held]
        source,test=np.flatnonzero(tr),np.flatnonzero(te)
        assert np.array_equal(te,later) and np.array_equal(tr,~later)
        assert (len(source),len(test),int((tr&first).sum()))==expected[:3]
        assert not np.intersect1d(source,test).size and not (data['split']=='external').any()
        roster[held]=source,test
    assert roster.keys()==PEOPLE.keys()
    idx=np.sort(np.concatenate([t for s,t in roster.values()]))
    assert len(idx)==762 and len(np.unique(idx))==len(idx)
    with np.load(root/'runs/shuffle_perms.npz',allow_pickle=False) as z:
        delivery={o:z[f'{o[:-1]}|{o[-1]}|seen_session'] for o in ORDERS}
    for order in delivery.values():
        assert len(order)==762 and np.array_equal(np.sort(order),idx)
    full=delivery['full0']
    for p,expected in PEOPLE.items():
        matched=[data['stream'][i]==f'Dreyer2026Proteus/{p}|{d}'
                 for i,d in zip(expected[3],('2','3'))]
        assert all(matched)
        actual=tuple(int(full[data['stream'][full]==f'Dreyer2026Proteus/{p}|{d}'][0])
                     for d in ('2','3'))
        assert actual==expected[3]
    return roster,idx,delivery


def archived(root,idx):
    with np.load(root/'runs/replay_data/inputs.npz',allow_pickle=False) as z:
        assert np.array_equal(z['idx|seen_session'],idx)
        base=z['B2_pool|Ps|seen_session']
    arrays={}
    for name,(slot,seeds,*_) in FAMILIES.items():
        bank=[]
        for seed in range(seeds):
            path=root/f'runs/experiments/probs/{name}-seen_session-s{seed}.npz'
            with np.load(path,allow_pickle=False) as z:
                j=np.argsort(z['index'])
                assert np.array_equal(z['index'][j],idx)
                p=z['proba'][j]
            assert p.shape==(762,3) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
            bank.append(p)
        assert np.array_equal(np.mean(bank,axis=0),base[slot])
        arrays[name]=np.asarray(bank)
    return arrays


def receipt(root):
    data,_=read_inputs(root)
    splits,idx,orders=rows(root,data)
    arrays=archived(root,idx)
    return dict(held_labels_opened=False,rows=len(idx),global_rows_sha=fingerprint(idx),
                orders={k:fingerprint(v) for k,v in orders.items()},
                families={k:dict(slot=FAMILIES[k][0],seeds=FAMILIES[k][1],
                                exact_archived_slot=True,recording_sha=fingerprint(v))
                          for k,v in arrays.items()},
                folds={k:dict(source_rows=len(s),held_rows=len(t),
                             source_sha=fingerprint(s),held_sha=fingerprint(t),
                             own_visit_1=PEOPLE[k][2],full0_first=list(PEOPLE[k][3]))
                       for k,(s,t) in splits.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    validation=receipt(args.root)
    write_json(args.output,validation)
    print(validation,flush=True)
