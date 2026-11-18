"""Seal archived and fresh matched returning tail evidence without held gold."""
import json
from pathlib import Path
import sys

import numpy as np

from capture import Capture, same
from returning_owner import BASE, LEGACY_AGES, PROTO, construct, digest, inputs, sha, without_test_labels
from returning_stages import fresh, permutation_reset, prepare, run


ARCHIVE=BASE/'ni26_friday_ea_returning_bridge_v2_20261118'
TRACE=BASE/'ni26_s281_seen_prepare_20261116/full/runs/s281_seen_gate_trace/full0.npz'


def authenticated(stem):
    path=ARCHIVE/(stem+'.npz')
    meta=json.loads((ARCHIVE/(stem+'.json')).read_text())
    assert meta['status']=='passed' and meta['label_free_targets'] and not meta['scored']
    assert meta['bank']==sha(path)
    return np.load(path,allow_pickle=False)


def execute(ctx,prep,perm,stream,idx,slots):
    cap=Capture(ctx,idx)
    with cap.observe():
        outputs,tape,routing=run(ctx,prep,perm,prep[4],slots,stream)
    stages,routes=cap.seal(perm[PROTO])
    assert np.array_equal(outputs['T'],stages['staged'])
    assert {k:v['age'] for k,v in routing.items()}==LEGACY_AGES
    return stages,routes,tape


def validate(baseline,reference,tag):
    stages,routes,tape=baseline
    for k in ('T','mix','shadow'):
        actual=stages['staged'] if k=='T' else tape[k]
        assert np.array_equal(actual,reference[tag+k]),k
    return routes


def archive_guard(stages,idx):
    with np.load(TRACE,allow_pickle=False) as z:
        assert np.array_equal(z['index'],idx) and str(z['order'])=='full0'
        max_ramp=0.
        for visit in ('P04|2','P04|3','P10|2','P10|3'):
            rows=np.searchsorted(idx,stages[visit+'|rows'])
            rr=stages[visit+'|winner_r']
            old=z['B2_pool|winner_r'][rows]
            max_ramp=max(max_ramp,float(np.abs(rr-old).max()))
            assert np.array_equal(rr>0,old>0)
            old_owner=z['B2_pool|winner_person'][rows][rr>0]
            assert (old_owner==str(stages[visit+'|owner'])).all()
        assert max_ramp<=1e-9,max_ramp
    return max_ramp


def persist(mode,idx,perm,bank,result,opened):
    target=Path(mode+'.npz')
    with target.open('xb') as f:
        np.savez_compressed(f,index=idx,delivery=idx[perm[PROTO]],**bank)
    with Path(mode+'.json').open('x') as f:
        json.dump(dict(status='passed_label_blind_returning_tail',mode=mode,source=sha('source.sha256'),
                       bank=sha(target),results=result,read_inputs={str(k):v for k,v in opened.items()},
                       exact_reverse_reset=True,scope='legacy live T returning expert; no final cap250, no gold'),
                  f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')
    print(mode,result,flush=True)


def main(mode):
    assert mode in ('zero','matched')
    people,days,stream,idx,splits=inputs()
    for who,want in (('P04',189),('P10',192)):
        source=splits[who+'_s23']; own=people[source]=='Dreyer2026Proteus/'+who
        assert int(own.sum())==want and (days[source][own]=='1').all()
    if mode=='matched':
        meta=json.loads(Path('zero.json').read_text())
        assert meta['source']==sha('source.sha256') and meta['bank']==sha('zero.npz')
    with without_test_labels(splits) as opened:
        ctx=construct(people,days,splits)
        prep=prepare(ctx,idx,stream); perm=prep[1]
        reverse=permutation_reset(perm,idx,stream)
        modes={'zero':None} if mode=='zero' else dict(zip(('fresh_recording','fresh_causal'),fresh(idx,idx[perm[PROTO]])))
        snapshots={}
        for arm,slots in modes.items():
            snapshots[arm]=a=execute(ctx,prep,perm,stream,idx,slots)
            b=execute(ctx,prep,reverse,stream,idx,slots)
            assert a[1]==b[1] and a[0].keys()==b[0].keys()
            for key,v in a[0].items():
                assert same(v,b[0][key]),key
    bank,record={},{}
    ref=authenticated(mode)
    np.testing.assert_array_equal(ref['index'],idx)
    for arm,a in snapshots.items():
        tag='' if mode=='zero' else arm+'|'
        record[arm]=validate(a,ref,tag)
        bank.update({tag+k:v for k,v in a[0].items()})
    ref.close()
    if mode=='zero':
        record['legacy_winner_ramp_max_abs']=archive_guard(snapshots['zero'][0],idx)
    else:
        old,new=(snapshots[arm][0] for arm in ('fresh_recording','fresh_causal'))
        record['matched_delta']={k:float(np.abs(new[k]-old[k]).max()) for k in ('em','staged','preown')}
        record['matched_delta']['mix']=max(float(np.abs(new[k]-old[k]).max()) for k in new if k.endswith('|mix'))
        assert min(record['matched_delta'][k] for k in ('mix','staged'))>1e-5
    persist(mode,idx,perm,bank,record,opened)


if __name__=='__main__':
    main(sys.argv[1])
