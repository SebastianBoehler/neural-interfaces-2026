"""Score sealed whole-path replays and compute the two label-aware ceilings."""

import json
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from inputs import FULL, FULL_BANK, FULL_PROOF, WITHIN, WITHIN_BANK, WITHIN_PROOF
from member_inputs import ORDERS, pinned_delivery
from bridge_owner import ROOT

WORK = Path(__file__).resolve().parent
INSTRUMENTS = dict(within0=(WITHIN/"complete.npz", WITHIN_BANK, WITHIN_PROOF, "stock"),
                   full0=(FULL/"matched_full0.npz", FULL_BANK, FULL_PROOF, "corrected|cap250"))
REAL = ("stock", "challenger", "gate", "sham", "blend")
KEYS = (*REAL, "oracle_unrestricted", "oracle_eligible")
EXPECTED_STOCK = dict(within0=78.3241479921539, full0=73.73898)


def qualified(order, smoke):
    path, bank = WORK/f"{order}.json", WORK/f"{order}.npz"
    proof = json.loads(path.read_text())
    stock_bank, digest, parent, stock_key = INSTRUMENTS[order]
    assert proof["passed"] and not proof["held_labels_opened"] and proof["order"] == order
    assert proof["rows"] == 2655 and proof["sessions"] == 14
    assert proof["stock_bitwise"] and proof["raw_frozen_zero"] and proof["strict_past"] and proof["no_recognition"]
    assert smoke["source"] == proof["source"] == sha(WORK/"source.sha256") and proof["parent"] == parent
    assert proof["warm_steps"] == 9600 and len(proof["prefix"]) == 3
    assert proof["prefix"][0] == proof["prefix"][-1] != proof["prefix"][1]
    assert proof["output"]["path"] == str(bank)
    pin(bank, proof["output"]["sha256"])
    pin(stock_bank, digest)
    assert not (path.stat().st_mode & 0o222 or bank.stat().st_mode & 0o222)
    with np.load(bank, allow_pickle=False) as f, np.load(stock_bank, allow_pickle=False) as s:
        want = pinned_delivery(ROOT / "runs/shuffle_perms.npz")[ORDERS.index(order)]
        np.testing.assert_array_equal(f["index"], np.arange(2655))
        np.testing.assert_array_equal(f["delivery"], want)
        np.testing.assert_array_equal(s["delivery"], want)
        for k in ("raw", "filtered", "terminal"):
            ref = s[f"{stock_key}{'|' if order == 'within0' else '_'}{k}"]
            np.testing.assert_array_equal(f["stock|"+("scores" if k == "terminal" else k)], ref)
        stock, challenger = f["stock|scores"], f["no_ft|scores"]
        eligible, live, sham = [f[k] for k in ("eligible", "live", "sham_mask")]
        assert not ((live | sham) & ~eligible).any()
        assert proof["activity"] == dict(eligible=int(eligible.sum()), live=int(live.sum()), sham_mask=int(sham.sum()))
        for name, mask in (("gate", live), ("sham", sham), ("blend", eligible)):
            expected = .5 * (stock + challenger) if name == "blend" else challenger
            np.testing.assert_array_equal(f[name][mask], expected[mask])
            np.testing.assert_array_equal(f[name][~mask], stock[~mask])
        scores = np.stack([stock, challenger, f["gate"], f["sham"], f["blend"]], axis=1)
        decisions = scores.argmax(2)
        assert scores.shape == (2655, len(REAL), 3) and np.isfinite(scores).all()
        assert not (eligible & (decisions[:,0] == decisions[:,1])).any()
    return dict(order=order, predictions=decisions, eligible=eligible, proof=sha(path),
                output=proof["output"], activity=proof["activity"], seconds=proof["seconds"])


def analyze(item, names, y, subject, groups, metric):
    p = item["predictions"]
    baseline, challenger = p[:,0], p[:,1]
    possible = (baseline != y) & (challenger == y)
    available = np.where(possible, challenger, baseline)
    masked = np.where(possible & item["eligible"], challenger, baseline)
    predictions = np.column_stack((p, available, masked))
    people, sums, weights, rescues = {}, [], [], {}
    for name in names:
        ix = subject == name
        cells = np.column_stack([metric.group_scores(metric.balanced_accuracy, y[ix], pred[ix], groups[ix])
                                for pred in predictions.T])
        weights.append(len(cells))
        sums.append(cells.sum(0))
        people[str(name)] = dict(zip(KEYS, (100*cells.mean(0)).tolist(), strict=True))
        rescues[str(name)] = {}
        for key, v in zip(KEYS, predictions.T, strict=True):
            positive = int(np.count_nonzero(ix & (baseline != y) & (v == y)))
            negative = int(np.count_nonzero(ix & (baseline == y) & (v != y)))
            rescues[str(name)][key] = dict(fixes=positive, regressions=negative, net=positive-negative)

    sums, weights = np.asarray(sums), np.asarray(weights)
    assert weights.sum() == 28 and len(names) == 10
    draw = np.random.default_rng(460).integers(0,len(names),(4000,len(names)))
    points = 100*sums.sum(0)/weights.sum()
    boot = 100*sums[draw].sum(1)/weights[draw].sum(1)[:,None]
    np.testing.assert_allclose(points[0], EXPECTED_STOCK[item["order"]], atol=.0001, rtol=0)
    pairs = [("challenger","stock"),("gate","stock"),("sham","stock"),("blend","stock"),("gate","sham"),
             ("gate","blend"),("oracle_unrestricted","stock"),("oracle_eligible","stock")]
    results = {}
    for a, b in pairs:
        i, j = KEYS.index(a), KEYS.index(b)
        lo, hi = np.percentile(boot[:,i] - boot[:,j], (10,90)).tolist()
        name = f"{a}_minus_{b}"
        results[name] = dict(pp=float(points[i]-points[j]), paired80=[lo,hi],
                             verdict="positive" if lo > 0 else "refuted" if hi <= 0 else "undecided")
    effects = {}
    for label in range(3):
        hit = (y == label)
        effects[str(label)] = {key:dict(
            fixes=int(np.count_nonzero(hit & (baseline != y) & (pred == y))),
            regressions=int(np.count_nonzero(hit & (baseline == y) & (pred != y))))
            for key,pred in zip(KEYS,predictions.T,strict=True)}
    metadata = {k:v for k,v in item.items() if k not in ("predictions","eligible")}
    return dict(**metadata, absolute_pct=dict(zip(KEYS,points.tolist(),strict=True)), contrasts=results,
                people_pct=people, people_changes=rescues, class_changes=effects)


def main():
    smoke = json.loads((WORK/"smoke.json").read_text())
    assert smoke["passed"] and smoke["order"]=="full0" and smoke["rows"]==190
    assert smoke["activity"]["eligible"] > 0 and not smoke["held_labels_opened"]
    results = {k:qualified(k, smoke) for k in INSTRUMENTS}
    print("F-C19 locked predictions before gold", {k:(v["proof"],v["output"]["sha256"]) for k,v in results.items()},
          flush=True)
    pin(METRIC, METRIC_HASH)
    pin(FEAT, HASHES[FEAT])
    metric = start_module("f19_official", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        y, subject, visit, interface = [z[k] for k in ("y","subject","session","interface")]
    names = np.unique(subject)
    groups = np.stack((subject,visit,interface),axis=1)
    order = {k:analyze(v,names,y,subject,groups,metric) for k,v in results.items()}
    robust = all(v["contrasts"][k]["paired80"][0]>0 for v in order.values()
                 for k in ("gate_minus_stock", "gate_minus_sham"))
    ceiling = max(v["contrasts"]["oracle_eligible_minus_stock"]["pp"] for v in order.values())
    out = WORK/"score.json"
    write_receipt(out, scope="FULL-SYSTEM DEV held-new PROTEUS, both orders on the same people",
                  official_cells_per_order=28, people=len(names), orders=order, scoring=sha(WORK/"scoring.sha256"),
                  eligible_oracle_max_pp=ceiling, advances_to_registered_external=robust and ceiling>=1.70,
                  paired_person_bootstrap=dict(draws=4000,seed=460,interval=.80))
    print("PASS F-C19 score", sha(out), "advance", robust and ceiling >= 1.70, flush=True)
    for k,v in order.items():
        print(k, json.dumps(v["contrasts"],sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
