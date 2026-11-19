"""Qualify both FBLight dose banks before computing the official metric."""
import json

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import ROOT, pin, sha, stamp, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from matrix23 import WORK
from member_inputs import ORDERS, pinned_delivery
from anchor23 import PINS, compare_parent
from weight23 import ARMS

LIVE=tuple(k for k in ARMS if k!='identity')


def qualified(order,smoke):
    receipt, bank=WORK/f'{order}.json',WORK/f'{order}.npz'
    proof=json.loads(receipt.read_text())
    assert proof['passed'] and proof['stage']=='complete' and proof['order']==order
    assert proof['rows']==2655 and proof['sessions']==14 and proof['native_identity'] and proof['stock_bitwise'] and proof['strict_past']
    assert not proof['held_labels_opened'] and proof['source']==smoke['source']==sha(WORK/'source.sha256')
    assert tuple(proof['parent']) == PINS[order] and proof['shadow_fixed']
    assert proof['output']['path']==str(bank) and not (receipt.stat().st_mode|bank.stat().st_mode)&0o222
    most=pinned_delivery(ROOT/'runs/shuffle_perms.npz')[ORDERS.index(order)]
    assert proof['delivery']==stamp(most)
    for name, share in (("FBL2", 1/17), ("frozen2", 1/17), ("FBL3", 1/18)):
        mix = proof["mixer"][name]
        assert mix["share"] == share and mix["warm"] == 2375 and mix["movement"] > .001
    assert proof["mixer"]["frozen2"]["extra"] != proof["mixer"]["FBL2"]["extra"]
    assert set(proof['work'])==set(ARMS)
    for work in proof['work'].values():
        assert (work['csp_refits'],work['csp_fusions'])==(126,1395) and work['count_effect']>0
    pin(bank,proof['output']['sha256'])
    with np.load(bank,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['index'],np.arange(2655))
        np.testing.assert_array_equal(z['delivery'],most)
        stages={a:{k:z[f'{a}|{k}'] for k in ('raw','filtered','terminal')} for a in ARMS}
        compare_parent(order,most,most,stages)
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
    stock, fbl, frozen, triple = np.eye(4)
    comparisons = dict(FBL2_minus_stock=fbl-stock, frozen_minus_stock=frozen-stock,
                       FBL3_minus_stock=triple-stock, FBL2_minus_frozen=fbl-frozen,
                       FBL3_minus_FBL2=triple-fbl)
    effects={}
    for key,coeff in comparisons.items():
        lo,hi=np.percentile(boot@coeff,(10,90)).tolist()
        effects[key]=dict(pp=float(point@coeff),paired80=[lo,hi],
                          verdict='positive' if lo>0 else 'refuted' if hi<=0 else 'undecided')
    return dict(absolute_pct=dict(zip(LIVE,point.tolist(),strict=True)),contrasts=effects,
                people_pct={p:dict(zip(LIVE,s.tolist(),strict=True)) for p,s in people.items()})


def main():
    smoke=json.loads((WORK/'smoke.json').read_text())
    assert smoke['passed'] and smoke['stage']=='smoke' and smoke['order']=='full0'
    assert smoke['native_identity'] and smoke['stock_bitwise'] and smoke['strict_past'] and not smoke['held_labels_opened']
    matched={o:qualified(o,smoke) for o in ('within0','full0')}
    anchors={o:receipt for o,(_,receipt) in matched.items()}
    blind=WORK/'before_labels.json'
    write_receipt(blind,held_labels_opened=False,matched=anchors,source=sha(WORK/'source.sha256'))
    pin(METRIC,METRIC_HASH); pin(FEAT,HASHES[FEAT])
    metric=start_module('c23_official_metric',METRIC,METRIC.read_text())
    with np.load(FEAT,allow_pickle=False) as z:
        y,subjects,sessions,interface=(z[k] for k in ('y','subject','session','interface'))
    groups=np.stack((subjects,sessions,interface),axis=1)
    values={o:assess(p,y,subjects,groups,metric) for o,(p,_) in matched.items()}
    path=WORK/'score.json'
    write_receipt(path,scope='FULL-SYSTEM DEV ten reused held-new participants',orders=values,
                  before_labels=sha(blind),scorer=sha(WORK/'scoring.sha256'),
                  paired80=dict(seed=460,draws=4000,official_cells=28))
    print('PASS F-C23 score',sha(path),{o:r['contrasts'] for o,r in values.items()},flush=True)


if __name__=='__main__':
    main()
