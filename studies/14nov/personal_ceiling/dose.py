"""DOSE-RESPONSE on TRUE labels (ledger s252 amendment 2a): classic aE trained on n of the four training runs' windows at
label purity p, the corruption drawn at random or chain-like. Per split: one random subset of size n (each class >= 2);
one corruption ranking per window (Efraimidis-Spirakis keys, weights = the class error rate) and one wrong label per window,
so the corrupted set at a lower purity contains the one at a higher purity (nested cells, less draw noise along p).
  rand   weights 1, wrong label uniform over the other two classes
  chain  per person the r15 chain's confusion over all 20 orders (plabels_r15 '<order>|raw'): weights = class error rate,
         wrong label drawn from the class's off-diagonal row
Output runs/personal_ceiling/dose_<kind>_<n>.npz: '<session>|<pair>|p<pct>' -> aE (n_test, 3), '<session>|<pair>|idx'.
Usage: python studies/14nov/personal_ceiling/dose.py KIND N   (KIND rand | chain; N 20 | 40 | 70 | 100 | all)"""
import sys, time, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import classic as CL  # noqa: E402
import splits as S  # noqa: E402

PUR = (100, 95, 90, 85, 80, 75)
ORD = [f"within{i}" for i in range(12)] + [f"full{i}" for i in range(4)] + [f"runs{i}" for i in range(4)]


def confusion(m):
    """-> {person: (3, 3) row-normalised confusion of the r15 chain's argmax, pooled over the 20 orders}"""
    PL = np.load(S.OUT / "plabels_r15.npz"); out = {}
    for p in S.PEOPLE:
        r = m["person"] == p; C = np.zeros((3, 3))
        for o in ORD:
            np.add.at(C, (m["y"][r], PL[f"{o}|raw"][r].argmax(1)), 1)
        out[p] = (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
    return out


def corrupt_plan(y, kind, cm, rng):
    """-> (ranking of windows to corrupt, wrong label per window)"""
    if kind == "rand":
        w = np.ones(len(y)); wrong = (y + rng.integers(1, 3, len(y))) % 3
    else:
        e = 1 - np.diag(cm); w = e[y]; wrong = np.empty(len(y), int)
        for i, c in enumerate(y):
            q = cm[c].copy(); q[c] = 0; wrong[i] = rng.choice(3, p=q / q.sum())
    key = rng.random(len(y)) ** (1 / w); return np.argsort(-key), wrong


def subset(trn, y, n, rng):
    if n >= len(trn):
        return trn
    for _ in range(1000):
        s = np.sort(rng.choice(trn, n, replace=False))
        if np.bincount(y[s], minlength=3).min() >= 2:
            return s
    raise RuntimeError("no stratified subset")


def main():
    kind, ns = sys.argv[1], sys.argv[2]; n = 10 ** 6 if ns == "all" else int(ns); assert kind in ("rand", "chain")
    f = S.OUT / f"dose_{kind}_{ns}.npz"
    if f.exists():
        return print("exists", f)
    m = S.meta(); F = CL.features(m); y = m["y"]; cms = confusion(m); res = {}; t0 = time.time()
    for s in S.sessions(m):
        for name, trn, tst in S.pairs(m, s):
            rng = np.random.default_rng(zlib.crc32(f"{s}|{name}|{ns}|{kind}".encode())); sub = subset(trn, y, n, rng)
            order, wrong = corrupt_plan(y[sub], kind, cms[s.split("_")[0]], rng); res[f"{s}|{name}|idx"] = tst
            for p in PUR:
                yy = y.copy(); k = int(round((100 - p) / 100 * len(sub))); c = order[:k]; yy[sub[c]] = wrong[c]
                assert abs(np.mean(yy[sub] == y[sub]) - p / 100) <= 0.5 / len(sub) + 1e-9
                res[f"{s}|{name}|p{p}"] = CL.fit_predict(F, yy, sub, tst)["aE"]
        print(kind, ns, s, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(f, **res); print("saved", f, flush=True)


if __name__ == "__main__":
    main()
