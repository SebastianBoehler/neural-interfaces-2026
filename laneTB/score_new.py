"""B-new paired delta TB25 vs hardcbB: BA per (person, scheme, seed, visit) from out/new_<v>_k<k>.npz; person-resampled 80 % interval.
usage: score_new.py OUT_DIR WINDOWS_NPZ"""
import sys
import numpy as np

out, win = sys.argv[1], sys.argv[2]
y = np.load(win, allow_pickle=False)["y"]
names = sorted(set(np.load(win, allow_pickle=False)["subject"]))
rng = np.random.default_rng(191)


def ba(v):
    B = np.full((2, 8, 2, 9), np.nan)  # scheme, seed, visit, person
    for k in range(9):
        try:
            d = np.load(f"{out}/new_{v}_k{k}.npz")
        except FileNotFoundError:
            continue
        for si, s in enumerate(("full", "within")):
            for i in range(8):
                for vi in (1, 2):
                    tag = f"{s}|s{71008 + i}|{names[k]}|{vi}"
                    rows, lab = d[tag + "|rows"], d[tag + "|labels"]
                    t = y[rows]
                    B[si, i, vi - 1, k] = 100 * np.mean([(lab[t == c] == c).mean() for c in range(3)])
    return B


a, b = ba("hardcbB"), ba("hardcbB_TB25")
ok = ~np.isnan(a).any(axis=(0, 1, 2)) & ~np.isnan(b).any(axis=(0, 1, 2))
d = (b - a)[..., ok]
print("people", ok.sum(), "abs hardcbB", np.nanmean(a[..., ok]).round(3))
for name, per in (("full-session (both visits)", d[0].mean((0, 1))), ("within", d[1].mean((0, 1))), ("pooled", d.mean((0, 1, 2))),
                  ("full visit1", d[0, :, 0].mean(0)), ("full visit2", d[0, :, 1].mean(0))):
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(10000)]
    print(f"{name:28s} {per.mean():+.3f} [{np.percentile(bs, 10):+.3f},{np.percentile(bs, 90):+.3f}] by_person {per.round(2).tolist()}")
