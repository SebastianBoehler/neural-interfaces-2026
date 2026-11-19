"""Open the held gold only after authenticating both complete F-C22h banks."""
import json

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import ROOT, pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from member_inputs import ORDERS as ORIGINAL_ORDERS, pinned_delivery
from pins import ARMS, C20, C20_ARRAYS, C20_SCORE, ORDERS
from score_f25c import SCORED, metrics
from whole_f25a import WORK


def authenticate(order, witness, source):
    path, npz = WORK / f"{order}.json", WORK / f"{order}.npz"
    receipt = json.loads(path.read_text())
    expected = dict(passed=True, scope="complete", source=source, order=order, rows=2655, sessions=14,
                    held_labels_opened=False, stock_bitwise=True, native_identity=True, actual_warm_delta=True,
                    original_shadow=True, new_csp_targets=True, folds=witness["folds"],
                    fit_receipts=witness["fit_receipts"])
    if (any(receipt[k] != v for k,v in expected.items()) or len(receipt["folds"]) != 20
            or not all(receipt["bounded_clips"].values()) or receipt["output"]["path"] != str(npz)
            or path.stat().st_mode & 0o222 or npz.stat().st_mode & 0o222):
        raise ValueError("missing authorized complete blind prediction proof")
    pin(npz, receipt["output"]["sha256"])
    for name, digest in receipt["folds"].items():
        f = WORK / "pupils" / name
        pin(f, digest)
        base = f.stem.split("_s")
        for kind, receipt_sha in receipt["fit_receipts"][name].items():
            pin(WORK / "checkpoints" / f"{kind}_{base[1]}_{base[0]}.json", receipt_sha)
    if any((receipt["work"][k]["csp_refits"],receipt["work"][k]["csp_fusions"]) != (126,1395) for k in ARMS):
        raise ValueError("native count/CSP work incomplete")
    original = C20 / f"{order}.npz"
    pin(original, C20_ARRAYS[order])
    with np.load(npz, allow_pickle=False) as current, np.load(original, allow_pickle=False) as prior:
        delivery = pinned_delivery(ROOT / "runs/shuffle_perms.npz")[ORIGINAL_ORDERS.index(order)]
        np.testing.assert_array_equal(current["index"], np.arange(2655))
        np.testing.assert_array_equal(current["delivery"], delivery)
        np.testing.assert_array_equal(prior["delivery"], delivery)
        for stage in ("raw", "filtered", "terminal"):
            np.testing.assert_array_equal(current[f"stock|{stage}"], prior[f"stock|{stage}"])
            np.testing.assert_array_equal(current[f"identity|{stage}"], current[f"stock|{stage}"])
        trajectories = np.stack([current[f"{k}|terminal"] for k in SCORED], axis=1)
        if trajectories.shape != (2655,len(SCORED),3) or not np.isfinite(trajectories).all():
            raise ValueError("invalid locked complete trajectories")
        estimate = trajectories.argmax(2)
    return estimate, receipt, sha(path)


def main():
    source = {k:sha(WORK / f"{k}.sha256") for k in ("actor","pupil","whole")}
    smoke_path = WORK / "whole-smoke.json"
    smoke = json.loads(smoke_path.read_text())
    if (not smoke["passed"] or smoke["scope"] != "smoke" or smoke["order"] != "full0"
            or smoke["sessions"] != 2 or not 240 <= smoke["rows"] < 2655 or smoke["source"] != source
            or smoke["held_labels_opened"] or smoke_path.stat().st_mode & 0o222):
        raise ValueError("the genuine two-session held smoke did not pass")
    signed = {order:authenticate(order, smoke, source) for order in ORDERS}
    blind = WORK / "before-labels.json"
    write_receipt(blind, source=source, smoke=sha(smoke_path),
                  orders={k:dict(proof=d, bank=r["output"], folds=r["folds"]) for k,(_,r,d) in signed.items()},
                  held_labels_opened=False)
    pin(C20 / "score.json", C20_SCORE)
    control = json.loads((C20 / "score.json").read_text())["orders"]
    pin(FEAT,HASHES[FEAT])
    pin(METRIC,METRIC_HASH)
    scorer = start_module("f25a_official", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        gold, subject, visit, interface = (z[k] for k in ("y","subject","session","interface"))
    groups = np.stack((subject,visit,interface), axis=1)
    results = {}
    for order,(prediction,receipt,proof) in signed.items():
        results[order] = result = metrics(prediction,gold,subject,groups,scorer)
        if abs(result["absolute_pct"]["stock"] - control[order]["absolute_pct"]["stock"]) > 1e-8:
            raise ValueError("official stock comparator differs")
        result.update(proof=proof, work=receipt["work"], seconds=receipt["seconds"])
    supports = all(results[o]["contrasts"][f"single_minus_{k}"]["paired80"][0] > 0
                   for o in ORDERS for k in ("stock","hard"))
    path = WORK / "results.json"
    write_receipt(path, scope="FULL-SYSTEM DEV, corrected PROTEUS held-new", fit="matched A100 native F25",
                  supports_conventional=supports, source=source, blind=sha(blind), orders=results,
                  paired_draws=4000, seed=460, interval=.80)
    print("PASS F-C22h paired score", sha(path), supports, flush=True)


if __name__ == "__main__":
    main()
