"""Qualify both six-arm banks before scoring official balanced accuracy."""
import json

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import ROOT, pin, sha, stamp, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from matrix22 import WORK, validate_parent_paths
from member_inputs import ORDERS, pinned_delivery
from parents22 import ARMS, PINS, PUPIL_SOURCE

LIVE=tuple(k for k in ARMS if k!='identity')


def qualified(order,smoke):
    receipt, bank=WORK/f'{order}.json',WORK/f'{order}.npz'
    proof=json.loads(receipt.read_text())
    assert proof['passed'] and proof['scope']=='complete' and proof['order']==order
    assert proof['rows']==2655 and proof['sessions']==14 and proof['parents_bitwise'] and proof['strict_prefix']
    assert not proof['held_labels_opened'] and proof['source']==smoke['source']==sha(WORK/'source.sha256')
    assert proof['inputs']==PINS[order] and proof['pupil_source']==PUPIL_SOURCE
    assert proof['pupils']==smoke['pupils'] and len(proof['pupils'])==20
    assert proof['output']['path']==str(bank) and not (receipt.stat().st_mode|bank.stat().st_mode)&0o222
    most=pinned_delivery(ROOT/'runs/shuffle_perms.npz')[ORDERS.index(order)]
    assert proof['delivery']==stamp(most)
    for name in ('FBL2','joint','single_joint'):
        boost=proof['mix'][name]['boost']
        assert boost['share']==1/17 and boost['warm_rows']==2375
        assert boost['adapted']==proof['mix']['FBL2']['boost']['adapted']
    assert set(proof['work'])==set(ARMS)
    for work in proof['work'].values():
        assert (work['csp_refits'],work['csp_fusions'])==(126,1395) and work['count_effect']>0
    pin(bank,proof['output']['sha256'])
    with np.load(bank,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['index'],np.arange(2655))
        np.testing.assert_array_equal(z['delivery'],most)
        stages={a:{k:z[f'{a}|{k}'] for k in ('raw','filtered','terminal')} for a in ARMS}
        validate_parent_paths(order,most,stages,most)
    p=np.stack([stages[a]['terminal'] for a in LIVE])
    assert p.shape==(len(LIVE),2655,3) and np.isfinite(p).all()
    return p.argmax(-1),dict(receipt=sha(receipt),bank=proof['output']['sha256'],seconds=proof['seconds'])


def assess(guesses,truth,subjects,groups,metric):
    sums,counts,people=[],[],{}
    for who in sorted(set(subjects)):
        m=subjects==who
        cells=np.column_stack([metric.group_scores(metric.balanced_accuracy,truth[m],z[m],groups[m]) for z in guesses])
        sums.append(cells.sum(0)); counts.append(len(cells))
        people[who]=100*cells.mean(0)
    sums,counts=np.asarray(sums),np.asarray(counts)
    assert len(counts)==10 and counts.sum()==28
    draw=np.random.default_rng(460).integers(0,10,(4000,10))
    point=100*sums.sum(0)/counts.sum()
    boot=100*sums[draw].sum(1)/counts[draw].sum(1)[:,None]
    stock,fbl,multi,joint,conventional=np.eye(5)
    comparisons=dict(FBL2_minus_stock=fbl-stock,multi_minus_stock=multi-stock,
                     joint_minus_stock=joint-stock,joint_minus_FBL2=joint-fbl,joint_minus_multi=joint-multi,
                     joint_minus_single_joint=joint-conventional,interaction=joint-fbl-multi+stock)
    effects={}
    for key,coeff in comparisons.items():
        lo,hi=np.percentile(boot@coeff,(10,90)).tolist()
        effects[key]=dict(pp=float(point@coeff),paired80=[lo,hi],
                          verdict='positive' if lo>0 else 'refuted' if hi<=0 else 'undecided')
    return dict(absolute_pct=dict(zip(LIVE,point.tolist(),strict=True)),contrasts=effects,
                people_pct={p:dict(zip(LIVE,s.tolist(),strict=True)) for p,s in people.items()})


def main():
    smoke=json.loads((WORK/'smoke.json').read_text())
    assert smoke['passed'] and smoke['scope']=='smoke' and smoke['order']=='full0'
    assert smoke['parents_bitwise'] and smoke['strict_prefix'] and not smoke['held_labels_opened']
    matched={o:qualified(o,smoke) for o in ('within0','full0')}
    anchors={o:receipt for o,(_,receipt) in matched.items()}
    blind=WORK/'before_labels.json'
    write_receipt(blind,held_labels_opened=False,matched=anchors,source=sha(WORK/'source.sha256'))
    pin(METRIC,METRIC_HASH); pin(FEAT,HASHES[FEAT])
    metric=start_module('c22_official_metric',METRIC,METRIC.read_text())
    with np.load(FEAT,allow_pickle=False) as z:
        y,subjects,sessions,interface=(z[k] for k in ('y','subject','session','interface'))
    groups=np.stack((subjects,sessions,interface),axis=1)
    values={o:assess(p,y,subjects,groups,metric) for o,(p,_) in matched.items()}
    path=WORK/'score.json'
    write_receipt(path,scope='FULL-SYSTEM DEV ten reused held-new participants',orders=values,
                  before_labels=sha(blind),scorer=sha(WORK/'scoring.sha256'),
                  paired80=dict(seed=460,draws=4000,official_cells=28))
    print('PASS F-C22 score',sha(path),{o:r['contrasts'] for o,r in values.items()},flush=True)


if __name__=='__main__':
    main()
