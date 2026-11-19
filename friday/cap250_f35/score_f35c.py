"""Score the two complete, authenticated blind F35 cap250 banks."""
import json

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import ROOT, pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from member_inputs import ORDERS as ORIGINAL_ORDERS, pinned_delivery
from pins_f35c import F35, ORDERS, PUPIL_ARMS, REF_BANKS, REF_RESULTS, REFERENCE, WORK
from score_f25c import SCORED, metrics
from whole_f35c import exact_hard


ARMS = ("stock","identity",*PUPIL_ARMS)
NAMES = ("stock",*PUPIL_ARMS)
DISPLAY = dict(zip(SCORED,NAMES,strict=True))


def held_bank(order,source,smoke):
    proof, path = WORK/f"{order}.json", WORK/f"{order}.npz"
    receipt = json.loads(proof.read_text())
    expected = dict(passed=True,scope="complete",order=order,source=source,rows=2655,sessions=14,
                    stock_bitwise=True,native_identity=True,actual_warm_delta=True,original_shadow=True,
                    new_csp_targets=True,hard_reference_bitwise=True,held_labels_opened=False,
                    folds=smoke["folds"],fit_receipts=smoke["fit_receipts"])
    if (any(receipt[k] != v for k,v in expected.items()) or len(receipt["folds"]) != 20
            or receipt["output"]["path"] != str(path) or (proof.stat().st_mode | path.stat().st_mode) & 0o222
            or any((receipt["work"][k]["csp_refits"],receipt["work"][k]["csp_fusions"]) != (126,1395) for k in ARMS)):
        raise ValueError("prior hard, causal work or complete blind bank authentication failed")
    pin(path,receipt["output"]["sha256"])
    for key,d in receipt["folds"].items():
        pin(WORK/"pupils"/key,d)
        person,seed = key[:-4].split("_s")
        for arm,fit in receipt["fit_receipts"][key].items():
            pin(F35/"checkpoints"/f"{arm}_{seed}_{person}.json",fit)
    with np.load(path,allow_pickle=False) as a:
        index = np.arange(2655)
        delivery = pinned_delivery(ROOT/"runs/shuffle_perms.npz")[ORIGINAL_ORDERS.index(order)]
        np.testing.assert_array_equal(a["index"],index)
        np.testing.assert_array_equal(a["delivery"],delivery)
        tables = {k:{s:a[f"{k}|{s}"] for s in ("raw","filtered","terminal")} for k in ARMS}
        exact_hard(order,tables,index)
        for stage,p in tables["stock"].items():
            np.testing.assert_array_equal(p,tables["identity"][stage])
        p = np.stack([tables[k]["terminal"] for k in NAMES],axis=1)
        assert p.shape == (2655,4,3) and np.isfinite(p).all()
    return p.argmax(2), receipt, sha(proof)


def describe(result):
    for field in ("people_pct","net"):
        result[field] = {key:{DISPLAY[k]:v for k,v in group.items()} for key,group in result[field].items()}
    result["absolute_pct"] = {DISPLAY[k]:v for k,v in result["absolute_pct"].items()}
    contrasts = {}
    for name,v in result["contrasts"].items():
        a,b = name.split("_minus_")
        contrasts[f"{DISPLAY[a]}_minus_{DISPLAY[b]}"] = v
        if (a,b) == ("multi","single"):
            lo,hi = v["paired80"]
            contrasts["balance_minus_mirrored"] = dict(pp=-v["pp"], paired80=[-hi,-lo],
                                                       verdict="positive" if hi<0 else "refuted" if lo>=0 else "undecided")
    result["contrasts"] = contrasts
    return result


def main():
    source = {k:sha(WORK/f"{k}.sha256") for k in ("pupil","whole")}
    smoke_path = WORK/"whole-smoke.json"
    smoke = json.loads(smoke_path.read_text())
    assert smoke["passed"] and smoke["source"] == source and smoke["scope"] == "smoke"
    assert smoke["order"] == "full0" and smoke["rows"] == 379 and smoke["sessions"] == 2
    assert smoke["hard_reference_bitwise"] and not smoke["held_labels_opened"] and not smoke_path.stat().st_mode & 0o222
    banks = {o:held_bank(o,source,smoke) for o in ORDERS}
    blind = WORK/"before-labels.json"
    write_receipt(blind,science="F35c",source=source,smoke=sha(smoke_path),held_labels_opened=False,
                  bank={o:dict(proof=h,output=r["output"],inputs=r["folds"]) for o,(_,r,h) in banks.items()},
                  comparator=REF_BANKS)
    pin(REFERENCE/"results.json",REF_RESULTS)
    old = json.loads((REFERENCE/"results.json").read_text())["orders"]
    pin(FEAT,HASHES[FEAT])
    pin(METRIC,METRIC_HASH)
    scorer = start_module("f35c_official",METRIC,METRIC.read_text())
    with np.load(FEAT,allow_pickle=False) as d:
        gold,subject,visit,interface = (d[k] for k in ("y","subject","session","interface"))
    groups = np.stack((subject,visit,interface),axis=1)
    results = {}
    for order,(p,r,proof) in banks.items():
        result = describe(metrics(p,gold,subject,groups,scorer))
        if any(abs(result["absolute_pct"][k]-old[order]["absolute_pct"][k]) > 1e-8 for k in ("stock","hard")):
            raise ValueError("exact stock/hard official comparator score differs")
        results[order] = dict(result,seconds=r["seconds"],work=r["work"],proof=proof)
    supports = all(results[o]["contrasts"][f"balance_minus_{k}"]["paired80"][0]>0
                   for o in ORDERS for k in ("hard","mirrored"))
    out = WORK/"results.json"
    write_receipt(out,scope="FULL-SYSTEM DEV, corrected PROTEUS held-new",source=source,blind=sha(blind),orders=results,
                  supports_balance=supports,paired_draws=4000,seed=460,interval=.80)
    print("PASS F35c",sha(out),supports,flush=True)


if __name__ == "__main__":
    main()
