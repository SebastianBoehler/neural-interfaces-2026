"""Team C collector: official cell-mean BA per variant and order over held-new people, people-resampled intervals,
paired delta vs cap250 (order 0 = cap250's full0 bank, rows indexed by data row).

python cleanslate/teamC/score.py DIR [--cap250 full0.npz] [--json OUT]   (DIR holds eval_<held>.npz from stream_eval)
Refuses partial person sets unless --allow-partial.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))
from benchmark_utils.metrics import balanced_accuracy, group_scores  # noqa: E402

PEOPLE = [f"P{i:02}" for i in range(1, 11)]


def cells(d, rows, pred):
    """{(stream, interface): BA} for the given data rows and labels."""
    st = np.char.add(np.char.add(d["subject"][rows], "|"), d["session"][rows])
    keys = np.stack([st, d["interface"][rows]], 1)
    out = {}
    for k in dict.fromkeys(map(tuple, keys)):
        m = (keys[:, 0] == k[0]) & (keys[:, 1] == k[1])
        out[k] = balanced_accuracy(d["y"][rows][m], pred[m])
    return out


def boot(per_person, n=10000, seed=0):
    """per_person: list of arrays of cell values (one array per person); cell-weighted mean, people resampled."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        pick = rng.integers(0, len(per_person), len(per_person))
        pts.append(np.concatenate([per_person[i] for i in pick]).mean())
    return np.percentile(pts, [10, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--cap250")
    ap.add_argument("--json")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    d = np.load(ROOT / "data/proteus_windows.npz")
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    files = {p: Path(a.dir) / f"eval_{p}.npz" for p in PEOPLE}
    have = [p for p in PEOPLE if files[p].exists()]
    assert a.allow_partial or have == PEOPLE, f"missing {sorted(set(PEOPLE) - set(have))}"
    Z = {p: np.load(files[p]) for p in have}
    keys = sorted({k.rsplit("|", 1)[0] for z in Z.values() for k in z.files if k.endswith("|label")})
    cap = None
    if a.cap250:
        cz = np.load(a.cap250)
        assert (cz["index"] == np.arange(len(person))).all()
        cap = cz["stock|terminal"].argmax(1)
    res = {}
    for key in keys:
        v, k = key.split("|")
        pp, dp, ms = [], [], []
        for p in have:
            if f"{key}|label" not in Z[p].files:
                continue
            rows = Z[p][f"rows|{k}"]
            c = cells(d, rows, Z[p][f"{key}|label"])
            pp.append(np.array(list(c.values())))
            ms.append(Z[p][f"{key}|sec"])
            if cap is not None and k == "0":
                cc = cells(d, rows, cap[rows])
                dp.append(np.array([c[q] - cc[q] for q in c]))
        allc = np.concatenate(pp)
        r = dict(ba=100 * allc.mean(), ci=(100 * boot(pp)).round(2).tolist(), n_people=len(pp), n_cells=len(allc),
                 ms_per_window=1e3 * float(np.concatenate(ms).mean()))
        if dp:
            r["delta_cap250"] = 100 * np.concatenate(dp).mean()
            r["delta_ci"] = (100 * boot(dp)).round(2).tolist()
        res[key] = r
        print(f"{key:10s} BA {r['ba']:6.2f} {r['ci']}  cells {r['n_cells']:2d} people {r['n_people']:2d} "
              f"ms/win {r['ms_per_window']:7.1f}" + (f"  d_cap250 {r['delta_cap250']:+6.2f} {r['delta_ci']}" if dp else ""))
    for v in sorted({k.split("|")[0] for k in keys}):
        ks = [k for k in keys if k.startswith(v + "|")]
        if len(ks) > 1:
            print(f"{v:10s} mean over {len(ks)} orders: {np.mean([res[k]['ba'] for k in ks]):.2f}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
