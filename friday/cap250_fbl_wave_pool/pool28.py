"""Compare fixed arithmetic and geometric pooling of two complete native paths."""
import json
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import pin, sha, stamp, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from score22 import WORK as PARENT, LIVE, qualified

NAMES = ("stock","FBL2","hard","early_joint","arithmetic","geometric")
COMPARED = {
    "arithmetic_minus_stock":(4,0), "arithmetic_minus_FBL2":(4,1), "arithmetic_minus_hard":(4,2),
    "arithmetic_minus_early_joint":(4,3), "geometric_minus_early_joint":(5,3),
    "arithmetic_minus_geometric":(4,5),
}
SEAL = "082f66359da762e29d3a72583ded36ba50cfb4d4742ce2cb16c69036274f9822"


def probability(raw):
    assert np.isfinite(raw).all() and raw.shape == (2655,3)
    fused = (raw < 0).all(1)
    assert fused.sum() == 1395 and (raw[~fused] >= 0).all()
    np.testing.assert_allclose(raw[~fused].sum(1),1,rtol=0,atol=1e-12)
    out = raw.copy()
    out[fused] = np.exp(raw[fused]-raw[fused].max(1,keepdims=True))
    out[fused] /= out[fused].sum(1,keepdims=True)
    np.testing.assert_array_equal(out.argmax(1),raw.argmax(1))
    return out


def acquire(order, control, smoke):
    old, checked = qualified(order,smoke)
    assert all(control[k] == checked[k] for k in ("receipt","bank","seconds"))
    source = PARENT/f"{order}.npz"
    with np.load(source,allow_pickle=False) as f:
        arrays = [f[f"{name}|terminal"] for name in ("stock","FBL2","hard","joint")]
        delivery = f["delivery"]
    stock,fbl,hard,joint = [probability(raw) for raw in arrays]
    agree = fbl.argmax(1)==hard.argmax(1)
    ar = (fbl+hard)*.5
    ge = np.sqrt(np.maximum(fbl,1e-12)*np.maximum(hard,1e-12))
    ge /= ge.sum(1,keepdims=True)
    answers = np.stack([p.argmax(1) for p in (stock,fbl,hard,joint,ar,ge)])
    for base,name in enumerate(("stock","FBL2","hard","joint")):
        np.testing.assert_array_equal(answers[base],old[LIVE.index(name)])
    for branch in (4,5):
        np.testing.assert_array_equal(answers[branch,agree],answers[1,agree])
    # Pooling is per current terminal row. Recompute separately on delivered prefixes.
    for bound in (1,31,120):
        rows=delivery[:bound]
        np.testing.assert_array_equal((fbl[rows]+hard[rows])*.5,ar[rows])
        np.testing.assert_array_equal(((1-0)*stock[rows]+0*hard[rows]).argmax(1),answers[0,rows])
    assert np.any(answers[4]!=answers[3]) and np.any(answers[4]!=answers[5])
    return answers


def score(answers,y,people,groups,metric):
    sums, sizes = [], []
    for person in sorted(set(people)):
        chosen=people==person
        cells=np.stack([metric.group_scores(metric.balanced_accuracy,y[chosen],p[chosen],groups[chosen])
                        for p in answers])
        sums.append(100*cells.sum(1))
        sizes.append(cells.shape[1])
    sums,sizes = np.array(sums),np.array(sizes)
    assert len(sizes)==10 and sizes.sum()==28
    draw=np.random.default_rng(460).integers(10,size=(4000,10))
    point=sums.sum(0)/28
    boot=sums[draw].sum(1)/sizes[draw].sum(1)[:,None]
    effects={}
    for name,(a,b) in COMPARED.items():
        pp=float(point[a]-point[b])
        lo,hi=np.percentile(boot[:,a]-boot[:,b],[10,90]).tolist()
        effects[name]=dict(pp=pp,paired80=[lo,hi],verdict="supported" if lo>0 else "refuted" if hi<=0 else "undecided")
    win,loss = (answers[4]==y)&(answers[3]!=y), (answers[4]!=y)&(answers[3]==y)
    classes={c:dict(rescues=int((win&(y==c)).sum()),regressions=int((loss&(y==c)).sum())) for c in range(3)}
    return dict(absolute_pct=dict(zip(NAMES,point.tolist(),strict=True)),contrasts=effects,versus_early=classes)


def run():
    pin(PARENT/"before_labels.json",SEAL)
    blind=json.loads((PARENT/"before_labels.json").read_text())
    smoke=json.loads((PARENT/"smoke.json").read_text())
    assert not blind["held_labels_opened"] and smoke["passed"] and smoke["strict_prefix"]
    complete={order:acquire(order,witness,smoke) for order,witness in blind["matched"].items()}
    path=Path("before_labels.json")
    write_receipt(path,held_labels_opened=False,parent=SEAL,outputs={k:stamp(v) for k,v in complete.items()})
    pin(FEAT,HASHES[FEAT]); pin(METRIC,METRIC_HASH)
    metric=start_module("c28_official",METRIC,METRIC.read_text())
    with np.load(FEAT,allow_pickle=False) as d:
        gold,people,visit,interface = (d[k] for k in ("y","subject","session","interface"))
    cells=np.stack((people,visit,interface),axis=1)
    result=Path("result.json")
    results={k:score(v,gold,people,cells,metric) for k,v in complete.items()}
    write_receipt(result,scope="FULL-SYSTEM DEV fixed pooling of corrected held-new tails",
                  paired_people=10,official_cells=28,before_labels=sha(path),orders=results)
    print("PASS F-C28 score",sha(result),results,flush=True)


if __name__=="__main__":
    run()
