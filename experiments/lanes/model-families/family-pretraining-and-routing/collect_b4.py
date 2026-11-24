"""Lane B4 collector.  python laneB4/collect_b4.py banks ROOT WORK   |   python laneB4/collect_b4.py pair ROOT RESULT_A.npz RESULT_B.npz ...
banks: assemble per-seed held-new banks (laneH bank format: index, full0-7, full*_delivery) per family, standalone seed variance
       (cell-weighted BA, all ten people, mean over the eight pinned full orders), K-seed-mean curve, and the ensemble banks R3/R6/R12
       (raw family, seeds 0-2 / 0-5 / 0-11) for laneH/full_replay.py. Refuses partial seed or person sets.
pair:  the first result is the reference; prints candidate-minus-reference (pp) from the saved terminal predictions with
       people-resampled paired 80 % intervals (10k draws, cell-weighted, laneH/full_replay.paired); also key 'standalone'."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneC"))
from oracle import ORDER_SHA, metadata, metric_weights, require, sha  # noqa: E402

FAMS = {"raw": 12, "eafbc": 12, "fblight": 12, "sinc": 6}
ALL = tuple(f"P{i:02}" for i in range(1, 11))
ORDERS = [f"full{i}" for i in range(8)]


def seed_bank(work, fam, seed, d, perms):
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    rows, post = [], {o: [] for o in ORDERS}
    for h in ALL:
        stem = work / f"b4_{fam}_{seed}_{h}"
        r = json.loads(stem.with_suffix(".json").read_text())
        require((r["fam"], r["seed"], r["held"]) == (fam, seed, h) and r["epochs"] == r["epochs_shipped"], f"bad receipt {stem}")
        q = np.flatnonzero(person == h)
        with np.load(stem.with_suffix(".npz")) as z:
            np.testing.assert_array_equal(z["index"], q)
            np.testing.assert_array_equal(z["train"], np.flatnonzero(person != h))
            for o in ORDERS:
                ref = perms[f"full|{o[-1]}|all_loso"]
                np.testing.assert_array_equal(z[o + "_delivery"], ref[np.isin(ref, q)])
                p = z[o]
                require(p.shape == (len(q), 3) and np.isfinite(p).all() and (p >= 0).all(), f"bad posterior {stem}")
                np.testing.assert_allclose(p.sum(1), 1, atol=1e-5, rtol=0)
                post[o].append(p)
        rows.append(q)
    rows = np.concatenate(rows); idx = np.argsort(rows)
    np.testing.assert_array_equal(rows[idx], np.arange(len(d["y"])))
    out = dict(index=rows[idx])
    for o in ORDERS:
        out[o] = np.concatenate(post[o])[idx]; out[o + "_delivery"] = perms[f"full|{o[-1]}|all_loso"]
    return out


def ba(d, pred):
    w, _ = metric_weights(d, np.ones(len(d["y"]), bool))
    return float(100 * w @ (pred == d["y"]))


def banks(root, work):
    d = metadata(root / "data/proteus_windows.npz")
    require(sha(root / "runs/shuffle_perms.npz") == ORDER_SHA, "orders changed")
    rep = {}
    with np.load(root / "runs/shuffle_perms.npz", allow_pickle=False) as perms:
        for fam, n in FAMS.items():
            seeds = [s for s in range(n) if all((work / f"b4_{fam}_{s}_{h}.npz").exists() for h in ALL)]
            require(seeds == list(range(len(seeds))) and len(seeds) >= 3, f"{fam}: seeds present {seeds}")
            sb = {s: seed_bank(work, fam, s, d, perms) for s in seeds}
            for s, b in sb.items():
                np.savez(work / f"seed_{fam}_{s}.npz", **b)
            per = np.array([[ba(d, sb[s][o].argmax(1)) for o in ORDERS] for s in seeds])  # seeds x orders
            curve = {}
            for K in range(1, len(seeds) + 1):
                v = [np.mean([ba(d, np.mean([sb[s][o] for s in c], 0).argmax(1)) for o in ORDERS]) for c in itertools.islice(itertools.combinations(seeds, K), 60)]
                curve[K] = dict(mean=float(np.mean(v)), sd_over_subsets=float(np.std(v)))
            rep[fam] = dict(seeds=seeds, ba_per_seed_mean_over_orders=per.mean(1).round(3).tolist(),
                            sd_seed_ba=float(per.mean(1).std(ddof=1)), sd_seed_ba_within_order_mean=float(per.std(0, ddof=1).mean()), curve=curve)
            tag = "R" if fam == "raw" else fam  # raw keeps the R3/R6/R12 names used by the first replays
            for K in (3, 6, 12):
                if len(seeds) >= K:
                    e = dict(index=sb[0]["index"])
                    for o in ORDERS:
                        e[o + "_members"] = np.stack([sb[s][o] for s in range(K)]); e[o] = e[o + "_members"].mean(0); e[o + "_delivery"] = sb[0][o + "_delivery"]
                    np.savez(work / f"ensemble_{tag}{K}_held_new.npz", **e)
                    rep[fam][f"E{K}_sha"] = sha(work / f"ensemble_{tag}{K}_held_new.npz")
    (work / "laneB4_standalone.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({f: {k: v for k, v in r.items() if k != "curve"} | {"curve": {K: round(c["mean"], 3) for K, c in r["curve"].items()}} for f, r in rep.items()}, indent=1))


def pair(root, results):
    d = metadata(root / "data/proteus_windows.npz"); y = d["y"]
    ref = np.load(results[0]); rng = np.random.default_rng(20261119)
    people = np.unique(d["subject"]); wc = {p: metric_weights(d, d["subject"] == p) for p in people}
    samples = rng.integers(len(people), size=(10000, len(people)))
    for path in results[1:]:
        cand = np.load(path)
        for key in ("laneC", "standalone"):
            a, b = (ref[key] == y).astype(int), (cand[key] == y).astype(int)
            per = np.array([100 * wc[p][0] @ (b - a) for p in people]); cells = np.array([wc[p][1] for p in people])
            ch = np.average(per[samples], weights=cells[samples], axis=1)
            w, _ = metric_weights(d, np.ones(len(y), bool))
            print(f"{Path(path).stem} - {Path(results[0]).stem} [{key}] {100 * w @ (b - a):+.3f} pp  80% {np.quantile(ch, [.1, .9]).round(3).tolist()}")


if __name__ == "__main__":
    {"banks": lambda: banks(Path(sys.argv[2]), Path(sys.argv[3])), "pair": lambda: pair(Path(sys.argv[2]), sys.argv[3:])}[sys.argv[1]]()
