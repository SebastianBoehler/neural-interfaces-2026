"""Check all corrected factor banks before scoring any held classes."""

import json
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import ROOT, pin, sha, stamp, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from inputs import FULL, FULL_BANK, FULL_PROOF
from member_inputs import ORDERS, pinned_delivery
from upstream import PARENT, PARENT_ARRAYS, PARENT_RECEIPT

WORK=Path(__file__).resolve().parent
INSTRUMENTS=dict(within0=(PARENT/"matched_within0.npz",PARENT_ARRAYS,PARENT_RECEIPT,78.32414737307604),
                 full0=(FULL/"matched_full0.npz",FULL_BANK,FULL_PROOF,73.73897960185187))
LIVE=("stock","SS","FBL2","JS")
NAMES=(*LIVE,"oracle")


def qualified(order,smoke):
    path,bank=WORK/f"{order}.json",WORK/f"{order}.npz"
    proof=json.loads(path.read_text())
    reference,target,anchor,expected=INSTRUMENTS[order]
    assert proof["passed"] and proof["stage"]=="complete" and proof["order"]==order
    assert proof["rows"]==2655 and proof["sessions"]==14 and proof["anchor"]==anchor
    assert proof["stock_bitwise"] and proof["native_identity"] and proof["shadow_fixed"] and proof["strict_past"]
    assert not proof["held_labels_opened"] and proof["source"]==smoke["source"]==sha(WORK/"experiment.sha256")
    assert proof["cc30"]["folds"]==smoke["cc30"]["folds"] and len(proof["cc30"]["streams"])==6
    assert proof["cc30"]["joint_members"]==6 and len(proof["cc30"]["folds"])==60
    assert proof["output"]["path"]==str(bank) and not (path.stat().st_mode|bank.stat().st_mode)&0o222
    pin(bank,proof["output"]["sha256"])
    pin(reference,target)
    p=pinned_delivery(ROOT/"runs/shuffle_perms.npz")[ORDERS.index(order)]
    assert proof["cc30"]["pinned_order"]==stamp(p)
    with np.load(bank,allow_pickle=False) as z,np.load(reference,allow_pickle=False) as s:
        np.testing.assert_array_equal(z["index"],np.arange(2655))
        np.testing.assert_array_equal(z["delivery"],p)
        for k in ("raw","filtered","terminal"):
            np.testing.assert_array_equal(z["stock|"+k],s["corrected|cap250_"+k])
            np.testing.assert_array_equal(z["stock|"+k],z["identity|"+k])
        preds=np.column_stack([z[k+"|terminal"].argmax(1) for k in LIVE])
    for arm,item in proof["work"].items():
        assert item["csp_refits"]==126 and item["csp_fusions"]==1395 and item["count_effect"]>0
    return preds,dict(expected=expected,proof=sha(path),bank=proof["output"]["sha256"],seconds=proof["seconds"],
                      gates=proof["mixer"],work=proof["work"])


def assess(pred,proof,y,subjects,groups,metric):
    stock,js=pred[:,0],pred[:,3]
    oracle=np.where((stock!=y)&(js==y),js,stock)
    pred=np.column_stack((pred,oracle))
    sums,weights,view,fixes=[],[],{},{}
    for p in sorted(set(subjects)):
        m=subjects==p
        cells=np.column_stack([metric.group_scores(metric.balanced_accuracy,y[m],arm[m],groups[m])
                                for arm in pred.T])
        sums.append(cells.sum(0))
        weights.append(len(cells))
        view[p]=dict(zip(NAMES,(100*cells.mean(0)).tolist(),strict=True))
        fixes[p]={k:changes(stock,v,y,m) for k,v in zip(NAMES,pred.T,strict=True)}
    sums,weights=np.asarray(sums),np.asarray(weights)
    assert len(weights)==10 and sum(weights)==28
    draw=np.random.default_rng(460).integers(0,10,(4000,10))
    point=100*sums.sum(0)/weights.sum()
    boot=100*sums[draw].sum(1)/weights[draw].sum(1)[:,None]
    assert abs(point[0]-proof["expected"])<1e-8
    a,b,c,d,e=np.eye(5)
    formula=dict(SS_minus_stock=b-a,FBL2_minus_stock=c-a,JS_minus_stock=d-a,JS_minus_SS=d-b,
                  JS_minus_FBL2=d-c,interaction=d-b-c+a,oracle_minus_stock=e-a)
    stats={}
    for key,v in formula.items():
        lo,hi=np.percentile(boot@v,(10,90)).tolist()
        stats[key]=dict(pp=float(point@v),paired80=[lo,hi],
                         verdict="positive" if lo>0 else "refuted" if hi<=0 else "undecided")
    classes={str(k):{a:changes(stock,v,y,y==k) for a,v in zip(NAMES,pred.T,strict=True)} for k in (0,1,2)}
    return dict(proof,accuracy_pct=dict(zip(NAMES,point.tolist(),strict=True)),
                contrasts=stats,per_person_pct=view,per_person_changes=fixes,classes=classes)


def changes(stock,other,y,mask):
    fixes=int((mask&(stock!=y)&(other==y)).sum())
    losses=int((mask&(stock==y)&(other!=y)).sum())
    return dict(fixes=fixes,losses=losses,net=fixes-losses)


def main():
    smoke=json.loads((WORK/"smoke.json").read_text())
    assert smoke["passed"] and smoke["order"]=="full0" and smoke["stage"]=="smoke"
    assert not smoke["held_labels_opened"] and smoke["stock_bitwise"] and smoke["native_identity"]
    qualified_orders={k:qualified(k,smoke) for k in INSTRUMENTS}
    print("F-C21 sealed both orders before gold", {k:(v["proof"],v["bank"]) for k,(_,v) in qualified_orders.items()},
          flush=True)
    pin(METRIC,METRIC_HASH)
    pin(FEAT,HASHES[FEAT])
    metric=start_module("fc21_official",METRIC,METRIC.read_text())
    with np.load(FEAT,allow_pickle=False) as z:
        y,subjects,visits,iface=[z[k] for k in ("y","subject","session","interface")]
    groups=np.stack((subjects,visits,iface),axis=1)
    orders={k:assess(p,v,y,subjects,groups,metric) for k,(p,v) in qualified_orders.items()}
    advance=all(m["contrasts"]["JS_minus_stock"]["paired80"][0]>0 and
                min(m["contrasts"][k]["pp"] for k in ("JS_minus_SS","JS_minus_FBL2"))>0 for m in orders.values())
    path=WORK/"score.json"
    write_receipt(path,scope="FULL-SYSTEM DEV, ten reused held-new people, two corrected orders",
                  orders=orders,advance=advance,scoring=sha(WORK/"scoring.sha256"),
                  paired_person_bootstrap=dict(draws=4000,seed=460,interval=.80))
    print("PASS F-C21 score",sha(path),"advance",advance,flush=True)
    for k,v in orders.items():
        print(k,v["accuracy_pct"],v["contrasts"],flush=True)


if __name__=="__main__":
    main()
