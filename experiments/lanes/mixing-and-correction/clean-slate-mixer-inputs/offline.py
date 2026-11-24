"""Offline mix-level look (no chain): stand-alone BA of clean-slate posterior sources and their 50/50 blends with the cap250 mix.
usage: offline.py PROF_NPZ TEAMC_DIR ORDER_KEY_IDX  (TEAMC eval_P*.npz with 'rows|k' and 'V|k|post'). Diagnostic only; decisions come from the chain."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m", ROOT / "2026-competition/tracks/bci_decoding/benchmark_utils/metrics.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
D = np.load(ROOT / "data/proteus_windows.npz")
Y, SUBJ = D["y"].astype(int), np.array([s.rsplit("/", 1)[-1] for s in D["subject"]])
G = np.stack((D["subject"], D["session"], D["interface"]), 1)
PEOPLE = sorted(set(SUBJ))


def cell_ba(post):
    g = post.argmax(1)
    s, n = 0.0, 0
    per = {}
    for p in PEOPLE:
        m = SUBJ == p
        v = M.group_scores(M.balanced_accuracy, Y[m], g[m], G[m])
        per[p] = 100 * float(np.mean(v)); s += float(np.sum(v)); n += len(v)
    return 100 * s / n, per


def teamc(dirpath, k, variant):
    out = np.full((len(Y), 3), np.nan)
    for p in PEOPLE:
        with np.load(Path(dirpath) / f"eval_{p}.npz") as z:
            out[z[f"rows|{k}"]] = z[f"{variant}|{k}|post"]
    assert np.isfinite(out).all()
    return out


if __name__ == "__main__":
    prof, tdir, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    z = np.load(prof)
    mix, term = z["stock|base|mix"].astype(float), z["stock|base|terminal"].astype(float)
    print("cap250 mix %.2f terminal %.2f" % (cell_ba(mix)[0], cell_ba(term)[0]))
    for v in ("A0", "A1", "A2", "A2n"):
        try:
            t = teamc(tdir, k, v)
        except KeyError:
            continue
        line = [f"{v} alone {cell_ba(t)[0]:.2f}"]
        for w in (0.25, 0.5):
            line.append(f"w{w} {cell_ba((1 - w) * mix + w * t)[0]:.2f}")
        agree = (t.argmax(1) == mix.argmax(1)).mean()
        line.append(f"argmax agree {100*agree:.1f}")
        print(" ".join(line))
