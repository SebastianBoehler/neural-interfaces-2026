"""Score corrected full-system stock, source-sham and jitter arms after sealing."""

import json
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import ROOT, pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from inputs import FULL, FULL_BANK, FULL_PROOF
from member_inputs import ORDERS, pinned_delivery
from upstream import PARENT, PARENT_ARRAYS, PARENT_RECEIPT
from worker import PUPILS, WORK

INSTRUMENTS = dict(within0=(PARENT/"matched_within0.npz",PARENT_ARRAYS,PARENT_RECEIPT,78.32414737307604),
                   full0=(FULL/"matched_full0.npz",FULL_BANK,FULL_PROOF,73.73897960185187))
ARMS = ("stock","hard","jitter")


def qualified(order, smoke):
    path, bank = WORK/f"{order}.json",WORK/f"{order}.npz"
    proof = json.loads(path.read_text())
    stock_bank,digest,parent,reference=INSTRUMENTS[order]
    assert proof["passed"] and proof["scope"]=="complete" and proof["order"]==order
    assert not proof["held_labels_opened"] and proof["actual_replay_rows"]==proof["rows"]==2655
    assert proof["stock_bitwise"] and proof["native_identity"] and proof["sessions"]==14
    assert proof["shadow_fixed"] and proof["new_csp_targets"] and proof["parent"]==parent
    assert proof["source"]==smoke["source"]==sha(WORK/"consumer.sha256")
    pupil_release=sha(PUPILS/"source.sha256")
    assert smoke["pupil_source"]==proof["pupil_source"]==pupil_release and proof["folds"]==smoke["folds"]
    assert len(proof["folds"])==20 and all(sha(PUPILS/"pupils"/k)==v for k,v in proof["folds"].items())
    assert proof["output"]["path"]==str(bank) and not (path.stat().st_mode & 0o222 or bank.stat().st_mode & 0o222)
    pin(bank,proof["output"]["sha256"])
    pin(stock_bank,digest)
    with np.load(bank,allow_pickle=False) as b, np.load(stock_bank,allow_pickle=False) as s:
        order_rows=pinned_delivery(ROOT/"runs/shuffle_perms.npz")[ORDERS.index(order)]
        np.testing.assert_array_equal(b["index"],np.arange(2655))
        np.testing.assert_array_equal(b["delivery"],order_rows)
        np.testing.assert_array_equal(s["delivery"],order_rows)
        for k in ("raw","filtered","terminal"):
            np.testing.assert_array_equal(s["corrected|cap250_"+k],b["stock|"+k])
            np.testing.assert_array_equal(b["stock|"+k],b["identity|"+k])
        p = np.column_stack([b[f"{arm}|terminal"].argmax(1) for arm in ARMS])
        assert np.isin(p,(0,1,2)).all()
    return p,dict(proof=sha(path),output=proof["output"],reference=reference,
                  work=proof["work"],outer=proof["outer"],seconds=proof["seconds"],raw_changes=proof["raw_changes"])


def metrics(est, evidence, truth, subject, groups, scorer):
    people,sums,weights,fixes={},[],[],{}
    labels=sorted(set(subject))
    for name in labels:
        selected=subject==name
        matrix=np.column_stack([scorer.group_scores(scorer.balanced_accuracy, truth[selected], pred[selected],
                               groups[selected]) for pred in est.T])
        sums.append(matrix.sum(0))
        weights.append(len(matrix))
        people[name]=dict(zip(ARMS,(100*matrix.mean(0)).tolist(),strict=True))
        fixes[name]={arm:changes(est[:,0],est[:,i],truth,selected) for i,arm in enumerate(ARMS)}
    sums,weights=np.asarray(sums),np.asarray(weights)
    assert len(labels)==10 and sum(weights)==28
    draw=np.random.default_rng(460).integers(0,len(labels),(4000,len(labels)))
    point=100*sums.sum(0)/weights.sum()
    boot=100*sums[draw].sum(1)/weights[draw].sum(1)[:,None]
    assert abs(point[0]-evidence["reference"])<1e-8
    contrasts={}
    for a,b in ((2,0),(2,1),(1,0)):
        lo,hi=np.percentile(boot[:,a]-boot[:,b],(10,90)).tolist()
        verdict="positive" if lo>0 else "refuted" if hi<=0 else "undecided"
        contrasts[f"{ARMS[a]}_minus_{ARMS[b]}"]=dict(pp=float(point[a]-point[b]),paired80=[lo,hi],verdict=verdict)
    return dict(evidence,absolute_pct=dict(zip(ARMS,point.tolist(),strict=True)),contrasts=contrasts,
                people_pct=people,people_changes=fixes,class_changes={
                    str(c):{arm:changes(est[:,0],est[:,i],truth,truth==c) for i,arm in enumerate(ARMS)}
                    for c in (0,1,2)})


def changes(stock, candidate, true, selected):
    gain=int(np.count_nonzero(selected & (stock!=true) & (candidate==true)))
    loss=int(np.count_nonzero(selected & (stock==true) & (candidate!=true)))
    return dict(fixes=gain,regressions=loss,net=gain-loss)


def main():
    smoke=json.loads((WORK/"smoke.json").read_text())
    assert smoke["passed"] and smoke["scope"]=="smoke" and smoke["order"]=="full0"
    assert smoke["rows"]==190 and smoke["sessions"]==2 and not smoke["held_labels_opened"]
    bank={o:qualified(o,smoke) for o in INSTRUMENTS}
    print("F-C20 locked complete predictions before gold", {o:(v[1]["proof"],v[1]["output"]["sha256"]) for o,v in bank.items()},
          flush=True)
    pin(FEAT,HASHES[FEAT])
    pin(METRIC,METRIC_HASH)
    metric=start_module("f20_official",METRIC,METRIC.read_text())
    with np.load(FEAT,allow_pickle=False) as z:
        y,subjects,visits,interfaces=[z[k] for k in ("y","subject","session","interface")]
    groups=np.stack((subjects,visits,interfaces),axis=1)
    results={o:metrics(est,ev,y,subjects,groups,metric) for o,(est,ev) in bank.items()}
    advance=all(instrument["contrasts"][k]["paired80"][0]>0 for instrument in results.values()
                for k in ("jitter_minus_stock","jitter_minus_hard"))
    out=WORK/"score.json"
    write_receipt(out,scope="FULL-SYSTEM DEV, corrected PROTEUS ten held-new people in both orders",
                  orders=results,source=sha(WORK/"scoring.sha256"),advance=advance,
                  paired_person_bootstrap=dict(draws=4000,seed=460,interval=.80))
    print("PASS F-C20 score",sha(out), "advance",advance,flush=True)
    for o,r in results.items():
        print(o,r["absolute_pct"],r["contrasts"],flush=True)


if __name__=="__main__":
    main()
