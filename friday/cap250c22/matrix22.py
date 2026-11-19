"""Seal each branch's genuine corrected count and personalized tail."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bridge_orders import write_npz, write_receipt
from bridge_owner import pin, sha, start_module, stamp
from complete_zero import FEAT, HASHES
from cross22 import run_arms
from parents22 import ARMS, C20, C21, F25, PINS, PUPIL_SOURCE

WORK = Path(__file__).resolve().parent


def helper(name, path):
    return start_module(name,path,path.read_text())


def parents(order):
    paths, proofs = {}, {}
    for label,root in (('fbl',C21),('raw',F25)):
        path, bank = root/f'{order}.json',root/f'{order}.npz'
        pin(path,PINS[order][f'{label}_proof'])
        pin(bank,PINS[order][f'{label}_array'])
        info = json.loads(path.read_text())
        assert info['stock_bitwise'] and info['native_identity'] and info['shadow_fixed']
        assert info['passed'] and info['order']==order and info['rows']==2655 and info['sessions']==14
        assert not info['held_labels_opened'] and info['output']['sha256']==sha(bank)
        paths[label],proofs[label] = bank,info
    return paths,proofs


def validate_parent_paths(order,delivery,stages,selected):
    paths,_=parents(order)
    with np.load(paths['fbl'],allow_pickle=False) as fbl,np.load(paths['raw'],allow_pickle=False) as raw:
        np.testing.assert_array_equal(fbl['delivery'],delivery)
        np.testing.assert_array_equal(raw['delivery'],delivery)
        np.testing.assert_array_equal(fbl['index'],np.arange(2655))
        np.testing.assert_array_equal(raw['index'],np.arange(2655))
        for key in ('raw','filtered','terminal'):
            stock=stages['stock'][key][selected]
            np.testing.assert_array_equal(stages['identity'][key][selected],stock)
            np.testing.assert_array_equal(fbl[f'stock|{key}'][selected],stock)
            np.testing.assert_array_equal(raw[f'stock|{key}'][selected],stock)
            np.testing.assert_array_equal(fbl[f'FBL2|{key}'][selected],stages['FBL2'][key][selected])
            np.testing.assert_array_equal(raw[f'multi|{key}'][selected],stages['multi'][key][selected])


def run(order,smoke):
    begin=perf_counter()
    source=sha(WORK/'source.sha256')
    if not smoke:
        smoke_proof=json.loads((WORK/'smoke.json').read_text())
        assert smoke_proof['passed'] and smoke_proof['scope']=='smoke' and smoke_proof['source']==source
        assert not smoke_proof['held_labels_opened']
    pin(FEAT,HASHES[FEAT])
    with np.load(FEAT,allow_pickle=False) as z:
        person=np.asarray([p.rsplit('/',1)[-1] for p in z['subject']]); visits=z['session']
    assert len(set(person))==10 and len(person)==2655
    _,receipts=parents(order)
    pupil_input=helper('c22_f25_inputs',F25/'consumer.py').source_array
    run_tail=helper('c22_parent_tail',C20/'worker.py').run_tail
    pupils,own_pins=pupil_input(order,person,PUPIL_SOURCE)
    assert own_pins==receipts['raw']['folds'] and len(own_pins)==20
    with run_arms(order,pupils) as ctx:
        assert ctx['parent']==receipts['raw']['parent']==receipts['fbl']['anchor']
        assert ctx['zero']==receipts['raw']['archive_zero']==receipts['fbl']['zero']
        count=ctx['count']
        sessions=tuple(count.SESS)
        assert len(sessions)==14 and stamp(ctx['delivery'])==receipts['fbl']['cc30']['pinned_order']
        rows={s:count.C.order_rows(order,s,count.PROTO) for s in sessions}
        p01=next(s for s,r in rows.items() if set(person[r])=={'P01'} and set(visits[r])=={'1'})
        assert len(rows[p01])==190
        latch=count.latch(order)
        peer=next(s for s in sessions if s!=p01 and latch[s] is not None and latch[s]<120)
        prefix=(p01,peer,p01)
        selected=(p01,peer) if smoke else sessions
        delivered=np.concatenate([rows[s] for s in selected])
        if not smoke:
            np.testing.assert_array_equal(delivered,ctx['delivery'])
        stages,work={},{}
        for arm in ARMS:
            arrays,work[arm]=run_tail(ctx,order,ctx['raw'][arm],selected,prefix)
            stages[arm]=arrays
            assert work[arm]['count_effect']>0 and work[arm]['csp_refits']>0
            if not smoke:
                assert (work[arm]['csp_refits'],work[arm]['csp_fusions'])==(126,1395)
            print('PASS F-C22 real count/CSP',order,arm,work[arm]['csp_refits'],flush=True)
        validate_parent_paths(order,ctx['delivery'],stages,delivered)
        for name,parent in (('joint','FBL2'),('joint','multi'),('single_joint','FBL2'),('joint','single_joint')):
            assert np.abs(stages[name]['terminal'][delivered]-stages[parent]['terminal'][delivered]).max()>1e-4
            assert any(work[name]['csp'][str(s)]['target_sha']!=work[parent]['csp'][str(s)]['target_sha'] for s in selected)
        for key in stages['stock']:
            np.testing.assert_array_equal(stages['stock'][key][delivered],ctx['reference'][key][delivered])
        proof=dict(passed=True,scope='smoke' if smoke else 'complete',order=order,source=source,
                   delivery=stamp(ctx['delivery']),rows=len(delivered),sessions=len(selected),parent=ctx['parent'],zero=ctx['zero'],
                   inputs=PINS[order],pupil_source=PUPIL_SOURCE,pupils=own_pins,work=work,mix=ctx['traces'],
                   held_labels_opened=False,parents_bitwise=True,strict_prefix=True,seconds=perf_counter()-begin)
        if not smoke:
            arrays=dict(index=np.arange(len(delivered)),delivery=delivered)
            arrays.update({f'{a}|{k}':v for a,parts in stages.items() for k,v in parts.items()})
            proof['output']=write_npz(WORK/f'{order}.npz',arrays)
        path=WORK/('smoke.json' if smoke else f'{order}.json')
        write_receipt(path,**proof)
    print('PASS F-C22',order,proof['scope'],sha(path),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('order',choices=('full0','within0'))
    parser.add_argument('mode',choices=('smoke','complete'))
    args=parser.parse_args()
    assert args.mode!='smoke' or args.order=='full0'
    run(args.order,args.mode=='smoke')
