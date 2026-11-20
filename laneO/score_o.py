"""Lane O proxy instrument: slot proposals of the final-stack candidates, STANDALONE, on eight full orders x ten held-new people.
NOT the corrected two-order full-system chain (no inner route, mix, EM/TSEM, count/CSP): posterior-level balanced accuracy of the raw-slot proposal.
usage: score_o.py BANKDIR HARDDIR OUT.json OUT.md
Arms (slot proposals, MASS ignored): stock = R6 raw-FBC six-seed bank mean (the cap250 raw group), hard = mean of the two F32c pair seeds,
cb = CBraMod-C 3-seed mean, hardcb = 0.8 hard + 0.2 cb, X, hcX = 0.4h + 0.4X + 0.2cb, hXc = 0.8h + 0.2X.
Intervals (80 %, 10000 draws): people+orders (both resampled), people only, orders only."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import metadata, metric_weights, require  # noqa: E402

PEOPLE = [f"P{i:02}" for i in range(1, 11)]
ORDERS = [f"full{k}" for k in range(8)]
CHAIN = {"hard": (0.331, 0.0), "cb": (0.263, 0.0), "hardcb": (0.514, 0.0)}  # corrected chain, full0 (final_stack.md), for reference only


def load(bank, o):
    with np.load(bank, allow_pickle=False) as z:
        return z[o].copy(), z[o + "_delivery"].copy()


def main(bankdir, harddir, out_json, out_md):
    bankdir, harddir = Path(bankdir), Path(harddir)
    d = metadata(Path(bankdir) / "proteus_windows.npz")
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    n = len(d["y"])
    post = {a: {} for a in ("stock", "hard", "cb", "X")}
    for o in ORDERS:
        stock, delivery = load(bankdir / "ensemble_R6_held_new.npz", o)
        for a, f in (("cb", "ensemble_C_held_new.npz"), ("X", "ensemble_X_held_new.npz")):
            p, dl = load(bankdir / f, o)
            np.testing.assert_array_equal(dl, delivery)
            post[a][o] = p
        post["stock"][o] = stock
        h = np.full((2, n, 3), np.nan)
        for who in PEOPLE:
            rows = np.flatnonzero(person == who)
            for s in (0, 1):
                with np.load(harddir / f"hard_{who}_s{s}.npz", allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["index"], rows)
                    np.testing.assert_array_equal(z[o + "_delivery"], delivery[np.isin(delivery, rows)])
                    h[s, rows] = z[o]
        require(np.isfinite(h).all(), "missing hard rows")
        post["hard"][o] = h.mean(0)
    for a in post:
        for o in ORDERS:
            p = post[a][o]
            require(p.shape == (n, 3) and np.isfinite(p).all() and (p >= 0).all(), f"bad bank {a} {o}")
    h, c, x = post["hard"], post["cb"], post["X"]
    post["hardcb"] = {o: 0.8 * h[o] + 0.2 * c[o] for o in ORDERS}
    post["hcX"] = {o: 0.4 * h[o] + 0.4 * x[o] + 0.2 * c[o] for o in ORDERS}
    post["hXc"] = {o: 0.8 * h[o] + 0.2 * x[o] for o in ORDERS}
    W = {p: metric_weights(d, person == p) for p in PEOPLE}
    ok = {a: {o: post[a][o].argmax(1) == d["y"] for o in ORDERS} for a in post}
    cells = np.array([W[p][1] for p in PEOPLE], float)
    ba = {a: np.array([[100 * W[p][0] @ ok[a][o] for p in PEOPLE] for o in ORDERS]) for a in post}  # [order, person]

    def pooled(m):  # m [order, person] -> weighted mean over people of order mean
        return float(np.average(m.mean(0), weights=cells))

    rng = np.random.default_rng(20261120)
    po = rng.integers(len(PEOPLE), size=(10000, len(PEOPLE)))
    oo = rng.integers(len(ORDERS), size=(10000, len(ORDERS)))

    def interval(m):
        both = np.array([np.average(m[np.ix_(oo[i], po[i])].mean(0), weights=cells[po[i]]) for i in range(10000)])
        ppl = np.array([np.average(m[:, po[i]].mean(0), weights=cells[po[i]]) for i in range(10000)])
        ord_ = m[oo].mean(1) @ cells / cells.sum()
        q = lambda v: np.quantile(v, [0.1, 0.9]).round(3).tolist()
        return q(both), q(ppl), q(ord_)

    names = list(post)
    res = dict(instrument="STANDALONE raw-slot proposals, 8 full orders x 10 held-new people; not the full-system chain", absolute={a: pooled(ba[a]) for a in names},
               per_order={a: ba[a].mean(1).round(3).tolist() for a in names}, pairs={}, chain_full0_reference=CHAIN)
    rows = []
    for a, b in [(a, "stock") for a in names if a != "stock"] + [("hardcb", "hard"), ("hardcb", "cb"), ("hard", "cb"), ("hcX", "hardcb"), ("hXc", "hardcb")]:
        m = ba[a] - ba[b]
        both, ppl, ordr = interval(m)
        pair = dict(delta=pooled(m), people_orders80=both, people80=ppl, orders80=ordr, per_order=(m @ cells / cells.sum()).round(3).tolist(),
                    full0=float(m[0] @ cells / cells.sum()), n_positive_orders=int(((m @ cells) > 0).sum()))
        res["pairs"][f"{a}-{b}"] = pair
        rows.append(f"| {a} - {b} | {pair['delta']:+.3f} | [{both[0]:+.3f}, {both[1]:+.3f}] | [{ppl[0]:+.3f}, {ppl[1]:+.3f}] | [{ordr[0]:+.3f}, {ordr[1]:+.3f}] | {pair['full0']:+.3f} | {pair['n_positive_orders']}/8 |")
    Path(out_json).write_text(json.dumps(res, indent=2) + "\n")
    Path(out_md).write_text("| contrast | pooled pp | people+orders 80 % | people 80 % | orders 80 % | full0 | orders > 0 |\n|---|---:|---|---|---|---:|---:|\n" + "\n".join(rows) + "\n\nabsolute BA: " +
                            ", ".join(f"{a} {v:.2f}" for a, v in res["absolute"].items()) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main(*sys.argv[1:5])
