"""Lane H: seal per-arm ensemble banks (laneC format) and score standalone held-new LOPO, paired B-A, C-A.

python laneH/collect.py ROOT WORK      ROOT = checkout with data/proteus_windows.npz + runs/shuffle_perms.npz
Writes WORK/ensemble_{A,B,C}_held_new.npz (index, full0-7, full*_delivery, full*_members) and WORK/laneH_standalone.json.
A bank is refused if any receipt, seed, arm, held person, trunk or row count is wrong (no partial fold sets)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneC"))
from oracle import ORDER_SHA, metadata, metric_weights, require, sha  # noqa: E402

ARMS, SEEDS = "ABC", (0, 1, 2)
ALL = tuple(f"P{i:02}" for i in range(1, 11))
ORDERS = [f"full{i}" for i in range(8)]


def fold(work, arm, seed, held, d, perms):
    stem = work / f"h_{arm}_{seed}_{held}"
    r = json.loads(stem.with_suffix(".json").read_text())
    require((r["arm"], r["seed"], r["held"]) == (arm, seed, held) and (r["init"] is None) == (arm == "A"), f"bad receipt {stem}")
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    query = np.flatnonzero(person == held)
    with np.load(stem.with_suffix(".npz")) as z:
        np.testing.assert_array_equal(z["index"], query)
        np.testing.assert_array_equal(z["train"], np.flatnonzero(person != held))
        out = {}
        for o in ORDERS:
            ref = perms[f"full|{o[-1]}|all_loso"]
            np.testing.assert_array_equal(z[o + "_delivery"], ref[np.isin(ref, query)])
            p = z[o]
            require(p.shape == (len(query), 3) and np.isfinite(p).all() and (p >= 0).all(), f"bad posterior {stem}")
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-5, rtol=0)
            out[o] = p
    return query, out


def ba(d, rows, pred):
    mask = np.zeros(len(d["y"]), bool); mask[rows] = True
    w, _ = metric_weights(d, mask)
    return float(100 * w[rows] @ (pred == d["y"][rows]))


def person_delta(d, pa, pb):
    y = d["y"]; res = {}
    for s in np.unique(d["subject"]):
        w, cells = metric_weights(d, d["subject"] == s)
        res[s.rsplit("/", 1)[-1]] = (float(100 * w @ ((pb == y).astype(int) - (pa == y))), cells)
    return res


def interval(per_order, seed=20261119):
    """per_order: list over orders of {person: (delta_pp, cells)}; resamples people and orders (10k draws), 80 %."""
    people = list(per_order[0]); rng = np.random.default_rng(seed)
    dl = np.array([[o[p][0] for p in people] for o in per_order]); ce = np.array([per_order[0][p][1] for p in people])
    pi = rng.integers(len(people), size=(10000, len(people))); oi = rng.integers(len(per_order), size=10000)
    ch = np.array([np.average(dl[oi[k]][pi[k]], weights=ce[pi[k]]) for k in range(10000)])
    return float(np.average(dl.mean(0), weights=ce)), np.quantile(ch, [0.1, 0.9]).tolist()


def main():
    root, work = Path(sys.argv[1]), Path(sys.argv[2])
    d = metadata(root / "data/proteus_windows.npz")
    require(sha(root / "runs/shuffle_perms.npz") == ORDER_SHA, "orders changed")
    rep, banks = {}, {}
    with np.load(root / "runs/shuffle_perms.npz", allow_pickle=False) as perms:
        for arm in ARMS:
            folds = {h: [fold(work, arm, s, h, d, perms) for s in SEEDS] for h in ALL}
            rows = np.concatenate([v[0][0] for v in folds.values()]); idx = np.argsort(rows)
            np.testing.assert_array_equal(rows[idx], np.arange(len(d["y"])))
            want = dict(index=rows[idx])
            for o in ORDERS:
                mem = np.stack([np.concatenate([v[s][1][o] for v in folds.values()])[idx] for s in range(len(SEEDS))])
                want[o], want[o + "_members"] = mem.mean(0), mem
                want[o + "_delivery"] = perms[f"full|{o[-1]}|all_loso"]
            np.savez(work / f"ensemble_{arm}_held_new.npz", **want)
            banks[arm] = want
            rep[arm] = dict(sha=sha(work / f"ensemble_{arm}_held_new.npz"),
                            ba_ensemble={o: ba(d, want["index"], want[o].argmax(1)) for o in ORDERS},
                            ba_single_seed_mean={o: float(np.mean([ba(d, want["index"], want[o + "_members"][s].argmax(1)) for s in range(3)])) for o in ORDERS})
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        per = [person_delta(d, banks[a][o].argmax(1), banks[b][o].argmax(1)) for o in ORDERS]
        mean, iv = interval(per)
        rep[f"{b}-{a}"] = dict(delta_pp=mean, paired_people_orders_80=iv, per_person={p: float(np.mean([x[p][0] for x in per])) for p in per[0]})
    (work / "laneH_standalone.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if "-" in k and "per_person" not in k} | {k: v["ba_ensemble"]["full0"] for k, v in rep.items() if k in ARMS}, indent=1))


if __name__ == "__main__":
    main()
