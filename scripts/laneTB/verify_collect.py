"""Collect Lane TB-verify: paired Team B deltas per base on the fresh orders, pooled with the Lane TB orders for hardcb.

python verify_collect.py OUT_DIR --data NPZ --orders 71124,...,71131 --tb TB_STREAMS --json RESULT
"""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect as C  # noqa: E402  (adds the harness to sys.path)

D = C.D
ARMS = ("base", "nine15", "nine25")


def merge(parts, out, orders, tb=None):
    """One pred_<mode>_<held>.npz per fold and arm from the order chunks (and the Lane TB streams when given)."""
    for mode, person in C.FOLDS:
        fold = f"{mode}_{person}"
        weights = set()
        for arm in ARMS:
            files = sorted((parts / fold / arm).glob(f"pred_{mode}_{person}_*.npz"))
            if tb is not None:
                files.append(tb / fold / arm / f"pred_{mode}_{person}.npz")
            res = {}
            for f in files:
                with np.load(f, allow_pickle=False) as z:
                    weights.add(str(z["weights_sha"]))
                    for k in z.files:
                        if k.startswith(("rows_", "labels_")):
                            assert k not in res, (f, k)
                            res[k] = z[k]
            got = {int(k.split("_")[1]) for k in res if k.startswith("rows_")}
            assert got == set(orders), (fold, arm, sorted(got))
            (out / fold / arm).mkdir(parents=True)
            np.savez(out / fold / arm / f"pred_{mode}_{person}.npz", source_sha="merged", weights_sha="merged", **res)
        assert len(weights) == 1, (fold, weights)  # every arm and chunk used the same Team B fold weights


def contrast(a, b, orders):
    """Paired a - b on shared cells, people then orders resampled (Lane TB collector code)."""
    assert a.keys() == b.keys()
    delta = {k: a[k] - b[k] for k in a}
    rng = np.random.default_rng(20261121)
    idx = rng.integers(len(orders), size=(20_000, len(orders)))
    pr = np.random.default_rng(71120)
    new, nd = C.by_cohort(delta, "new", D.PEOPLE, idx, pr)
    ret, rd = C.by_cohort(delta, "ret", D.RETURNING, idx, pr)
    rp = ret["people"]
    return dict(new=new, returning=ret, j=0.5 * (new["mean"] + ret["mean"]), j_ci80=C.interval(0.5 * (nd + rd)),
                proxy_57_43_unverified=0.57 * new["mean"] + 0.43 * (0.2 * rp["P04"]["mean"] + 0.8 * rp["P10"]["mean"]))


def load(root, orders, data):
    C.ORDERS = tuple(orders)
    return {arm: C.load_arm(root, arm, data)[0] for arm in ARMS}


def line(name, r):
    n, t = r["new"], r["returning"]
    f = lambda x: f"{x:+.2f}"  # noqa: E731
    return (f"| {name} | {f(n['mean'])} [{f(n['ci80'][0])}, {f(n['ci80'][1])}] | {f(t['mean'])} [{f(t['ci80'][0])}, "
            f"{f(t['ci80'][1])}] | {f(t['people']['P04']['mean'])} / {f(t['people']['P10']['mean'])} | "
            f"{f(r['j'])} | {f(r['proxy_57_43_unverified'])} | {' '.join(f(o) for o in n['orders'])} |")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--data", required=True)
    p.add_argument("--orders", required=True)
    p.add_argument("--tb", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    a = p.parse_args()
    data = D.load(a.data)
    fresh = [int(s) for s in a.orders.split(",")]
    pooled = sorted(fresh + list(C.ORDERS))
    result, rows = {}, []
    with TemporaryDirectory() as tmp:
        cells = {}
        for base in ("hardcb", "hard"):
            merge(a.out / base, Path(tmp) / base, fresh)
            cells[base] = load(Path(tmp) / base, fresh, data)
        merge(a.out / "hardcb", Path(tmp) / "pooled", pooled, a.tb)
        cells["pooled"] = load(Path(tmp) / "pooled", pooled, data)
    for base, orders in (("hardcb", fresh), ("hard", fresh), ("pooled", pooled)):
        for arm in ARMS[1:]:
            key = f"{base}_{arm}"
            result[key] = contrast(cells[base][arm], cells[base]["base"], orders)
            rows.append(line(f"{base} {arm} − {base} base ({len(orders)} orders)", result[key]))
    for arm in ARMS:
        key = f"hard_{arm}_vs_hardcb_base"
        result[key] = contrast(cells["hard"][arm], cells["hardcb"]["base"], fresh)
        rows.append(line(f"hard {arm} − hardcb base", result[key]))
    a.json.write_text(json.dumps(dict(fresh_orders=fresh, pooled_orders=pooled, contrasts=result), indent=1) + "\n")
    print("| Contrast | Held-new Δ [80%] | Returning Δ [80%] | P04 / P10 | J | 57/43* | held-new per order |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
