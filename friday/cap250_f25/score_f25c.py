"""Open labels only after sealing both observed corrected trajectories."""
import json

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import ROOT, pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from consumer import WORK
from member_inputs import ORDERS as ORIGINAL_ORDERS, pinned_delivery
from pins import ARMS, C20, C20_ARRAYS, C20_SCORE, ORDERS, PUPIL_ARMS


SCORED = ("stock", *PUPIL_ARMS)
PAIRS = (("multi", "stock"), ("multi", "hard"), ("multi", "single"),
         ("single", "hard"), ("hard", "stock"), ("single", "stock"))


def qualify(order, smoke):
    path, bank = WORK / f"{order}.json", WORK / f"{order}.npz"
    receipt = json.loads(path.read_text())
    assert receipt["passed"] and receipt["scope"] == "complete" and receipt["order"] == order
    assert not receipt["held_labels_opened"] and receipt["rows"] == receipt["actual_replay_rows"] == 2655
    assert receipt["stock_bitwise"] and receipt["native_identity"] and receipt["sessions"] == 14
    assert receipt["shadow_fixed"] and receipt["new_csp_targets"]
    assert receipt["source"] == smoke["source"] == sha(WORK / "consumer.sha256")
    assert receipt["pupil_source"] == smoke["pupil_source"] == sha(WORK / "source.sha256")
    assert receipt["folds"] == smoke["folds"] and len(receipt["folds"]) == 20
    assert all(sha(WORK / "pupils" / k) == d for k, d in receipt["folds"].items())
    assert receipt["output"]["path"] == str(bank) and not (path.stat().st_mode & 0o222 or bank.stat().st_mode & 0o222)
    assert all((receipt["work"][k]["csp_refits"], receipt["work"][k]["csp_fusions"]) == (126, 1395) for k in ARMS)
    pin(bank, receipt["output"]["sha256"])
    original = C20 / f"{order}.npz"
    pin(original, C20_ARRAYS[order])
    with np.load(bank, allow_pickle=False) as current, np.load(original, allow_pickle=False) as prior:
        delivery = pinned_delivery(ROOT / "runs/shuffle_perms.npz")[ORIGINAL_ORDERS.index(order)]
        np.testing.assert_array_equal(current["index"], np.arange(2655))
        np.testing.assert_array_equal(current["delivery"], delivery)
        np.testing.assert_array_equal(prior["delivery"], delivery)
        for stage in ("raw", "filtered", "terminal"):
            for arm in ("stock", "hard"):
                np.testing.assert_array_equal(current[f"{arm}|{stage}"], prior[f"{arm}|{stage}"])
            np.testing.assert_array_equal(current[f"identity|{stage}"], current[f"stock|{stage}"])
        predict = np.stack([current[f"{k}|terminal"].argmax(1) for k in SCORED], axis=1)
    assert np.isin(predict, (0, 1, 2)).all()
    return predict, receipt, sha(path)


def metrics(est, truth, subject, groups, scorer):
    people = sorted(set(subject))
    sums, weights, individual = [], [], {}
    for name in people:
        m = subject == name
        v = np.column_stack([scorer.group_scores(scorer.balanced_accuracy, truth[m], p[m], groups[m]) for p in est.T])
        sums.append(v.sum(0))
        weights.append(len(v))
        individual[name] = dict(zip(SCORED, (100*v.mean(0)).tolist(), strict=True))
    sums, weights = np.asarray(sums), np.asarray(weights)
    assert len(people) == 10 and weights.sum() == 28
    drawn = np.random.default_rng(460).integers(0, len(people), (4000, len(people)))
    point = 100*sums.sum(0)/weights.sum()
    boot = 100*sums[drawn].sum(1)/weights[drawn].sum(1)[:, None]
    contrasts = {}
    for a, b in PAIRS:
        i, j = SCORED.index(a), SCORED.index(b)
        lo, hi = np.percentile(boot[:, i] - boot[:, j], [10, 90]).tolist()
        contrasts[f"{a}_minus_{b}"] = dict(pp=float(point[i] - point[j]), paired80=[lo, hi],
                                   verdict="positive" if lo > 0 else "refuted" if hi <= 0 else "undecided")
    correct = est == truth[:, None]
    changes = {}
    for entity, mask in [(p, subject == p) for p in people] + [(str(c), truth == c) for c in range(3)]:
        changes[entity] = {}
        for k in PUPIL_ARMS:
            j = SCORED.index(k)
            fix = int(np.count_nonzero(mask & ~correct[:, 0] & correct[:, j]))
            loss = int(np.count_nonzero(mask & correct[:, 0] & ~correct[:, j]))
            changes[entity][k] = dict(fixes=fix, losses=loss, net=fix-loss)
    return dict(absolute_pct=dict(zip(SCORED, point.tolist(), strict=True)), contrasts=contrasts,
                people_pct=individual, net=changes)


def main():
    smoke = json.loads((WORK / "smoke.json").read_text())
    assert smoke["passed"] and smoke["order"] == "full0" and smoke["rows"] == 190 and smoke["sessions"] == 2
    assert not smoke["held_labels_opened"]
    signed = {order: qualify(order, smoke) for order in ORDERS}
    lock = WORK / "before_labels.json"
    write_receipt(lock, **{o: dict(proof=p, output=r["output"]) for o, (_, r, p) in signed.items()})
    pin(C20 / "score.json", C20_SCORE)
    reference = json.loads((C20 / "score.json").read_text())["orders"]
    pin(FEAT, HASHES[FEAT])
    pin(METRIC, METRIC_HASH)
    scorer = start_module("f25c_official", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        truth, subject, visit, interface = (z[k] for k in ("y","subject","session","interface"))
    groups = np.stack((subject, visit, interface), axis=1)
    results = {}
    for order, (est, receipt, proof) in signed.items():
        results[order] = result = metrics(est, truth, subject, groups, scorer)
        assert all(abs(result["absolute_pct"][k] - reference[order]["absolute_pct"][k]) <= 1e-8 for k in ("stock","hard"))
        result.update(seconds=receipt["seconds"], proof=proof, stage=receipt["work"])
    advance = all(r["contrasts"][f"multi_minus_{k}"]["paired80"][0] > 0
                  for k in ("stock","hard","single") for r in results.values())
    path = WORK / "score.json"
    write_receipt(path, scope="FULL-SYSTEM DEV, corrected PROTEUS held-new", advance=advance, orders=results,
                  before_labels=sha(lock), pair_draws=4000, seed=460, interval=.80)
    print("PASS F25c",sha(path),advance,flush=True)


if __name__ == "__main__":
    main()
