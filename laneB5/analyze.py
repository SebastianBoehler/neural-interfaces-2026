"""Lane B5: curve fits and people-resampled intervals, standalone and full system.

python laneB5/analyze.py ROOT BANKDIR RESULTDIR OUT.json    BANKDIR has standalone.json, RESULTDIR result_N{n}s{s}.npz (laneC arm = swapped family)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneC"))
from oracle import metadata, metric_weights  # noqa: E402

NS, S, PEOPLE = (5, 6, 7, 8, 9), 5, [f"P{i:02}" for i in range(1, 11)]
EXTRA = (5, 10, 20)


def fit(y, ns=NS):
    """-> function N -> BA for the two fits: log-linear and saturating power law a - b N^-c (c in [0.2, 2], b >= 0)."""
    x = np.log(ns); b, a = np.polyfit(x, y, 1)
    best = None
    for c in np.linspace(0.2, 2, 37):
        t = np.array(ns, float) ** -c
        B, A = np.polyfit(t, y, 1)
        if B > 0:  # y = A + B N^-c rises with N only for B < 0
            continue
        r = ((A + B * t - y) ** 2).sum()
        if best is None or r < best[0]:
            best = (r, A, B, c)
    if best is None:  # no saturating form fits a non-rising curve: fall back to flat at the mean
        best = (0, y.mean(), 0.0, 1.0)
    _, A, B, c = best
    return (lambda n: a + b * np.log(n)), (lambda n: A + B * np.asarray(n, float) ** -c)


def summarise(per_person, cells, draws=10000, seed=20261120):
    """per_person: (people, N) BA matrix. Bootstrap people, cell weighted."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(cells), size=(draws, len(cells)))
    rows = []
    for k in range(draws):
        y = np.average(per_person[idx[k]], axis=0, weights=cells[idx[k]])
        ll, sat = fit(y)
        rows.append([(y[4] - y[2]) / 2, y[4] - y[0], *[ll(9 + e) - ll(9) for e in EXTRA], *[sat(9 + e) - sat(9) for e in EXTRA],
                     (ll(9.5) - ll(8.5)), (sat(9.5) - sat(8.5))])
    rows = np.array(rows)
    point = np.average(per_person, axis=0, weights=cells)
    ll, sat = fit(point)
    names = ["slope_7to9_per_person", "gain_5to9", *[f"loglin_+{e}" for e in EXTRA], *[f"satpow_+{e}" for e in EXTRA], "loglin_slope_at9", "satpow_slope_at9"]
    pt = [(point[4] - point[2]) / 2, point[4] - point[0], *[ll(9 + e) - ll(9) for e in EXTRA], *[sat(9 + e) - sat(9) for e in EXTRA], ll(9.5) - ll(8.5), sat(9.5) - sat(8.5)]
    return dict(mean_by_N=point.tolist(), **{n: dict(point=float(p), lo80=float(np.quantile(rows[:, i], .1)), hi80=float(np.quantile(rows[:, i], .9)))
                                              for i, (n, p) in enumerate(zip(names, pt))})


def main(root, bankdir, resdir, out):
    d = metadata(root / "data/proteus_windows.npz")
    person = d["subject"]; y = d["y"]
    st = json.loads((bankdir / "standalone.json").read_text())
    cells = np.array([st["N9s0"]["people"][h]["cells"] for h in PEOPLE], float)
    cfg = lambda n, s: f"N{n}s{0 if n == 9 else s}"
    stand = np.array([[np.mean([st[cfg(n, s)]["people"][h]["ens"] for s in range(1 if n == 9 else S)]) for n in NS] for h in PEOPLE])
    full = np.zeros_like(stand); stock = np.zeros(len(PEOPLE))
    w = {h: metric_weights(d, np.array([s.rsplit("/", 1)[-1] == h for s in person]))[0] for h in PEOPLE}
    for j, n in enumerate(NS):
        for i, h in enumerate(PEOPLE):
            v = []
            for s in range(1 if n == 9 else S):
                z = np.load(resdir / f"result_{cfg(n, s)}.npz")
                v.append(100 * w[h] @ (z["laneC"] == y)); stock[i] = 100 * w[h] @ (z["stock"] == y)
            full[i, j] = np.mean(v)
    res = dict(standalone=summarise(stand, cells), full_system=summarise(full, cells), stock_ba=float(np.average(stock, weights=cells)),
               per_person_standalone=stand.tolist(), per_person_full=full.tolist(), cells=cells.tolist())
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    for k in ("standalone", "full_system"):
        print(k, {n: (round(v["point"], 2), round(v["lo80"], 2), round(v["hi80"], 2)) if isinstance(v, dict) else [round(x, 2) for x in v] for n, v in res[k].items()})
    print("stock", res["stock_ba"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
