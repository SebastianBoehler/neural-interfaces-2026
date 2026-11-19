"""Assemble worker outputs into per-cell scores (official cell metric) for one tag.

python bank.py OUT_DIR/TAG DATA_NPZ OUT_SCORES.npz [--cohort scherer_new|scherer_ret]
Labels are opened here, never by the worker. Cell = person x visit (Scherer has one interface). Score = official
balanced accuracy (mean recall over the union of true and predicted classes), in percent.
Output arrays: cell (scheme, order, person, visit) in %, rec (same shape, first-recognition window, -1 never),
routed (same shape), pred (scheme, order, 2130) int8 label by data row (-1 missing), person names.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orders import SCHEMES  # noqa: E402

NORD = 8


def balanced_accuracy(y, p):
    classes = np.union1d(y, p)
    return float(np.mean([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in classes]))


def load_tag(directory):
    """Merge every p*_c*.npz; keys like 'full|3|Scherer2015/G|1|labels'."""
    merged = {}
    for f in sorted(Path(directory).glob("p*_c*.npz")):
        with np.load(f, allow_pickle=False) as z:
            for k in z.files:
                assert k not in merged, f"duplicate {k}"
                merged[k] = z[k]
    return merged


def assemble(directory, data, only_visits=None):
    merged = load_tag(directory)
    with np.load(data, allow_pickle=False) as z:
        y, subject = z["y"], z["subject"]
    people = sorted(set(subject))
    shape = (len(SCHEMES), NORD, len(people), 2)
    cell, rec, routed = (np.full(shape, np.nan) for _ in range(3))
    pred = np.full((len(SCHEMES), NORD, len(y)), -1, np.int8)
    for si, scheme in enumerate(SCHEMES):
        for k in range(NORD):
            for pi, who in enumerate(people):
                for vi, v in enumerate((1, 2)):
                    tag = f"{scheme}|{k}|{who}|{v}"
                    if tag + "|labels" not in merged:
                        continue
                    rows, labels = merged[tag + "|rows"], merged[tag + "|labels"].astype(int)
                    cell[si, k, pi, vi] = 100 * balanced_accuracy(y[rows], labels)
                    meta = merged[tag + "|meta"]
                    rec[si, k, pi, vi], routed[si, k, pi, vi] = meta[0], meta[1]
                    pred[si, k, rows] = labels
    return dict(cell=cell, rec=rec, routed=routed, pred=pred, people=np.array(people))


def main():
    directory, data, out = sys.argv[1:4]
    r = assemble(directory, data)
    n_done = int(np.isfinite(r["cell"]).sum())
    print(f"{directory}: {n_done} of {r['cell'].size} cells; mean {np.nanmean(r['cell']):.3f} %")
    np.savez(out, **r)


if __name__ == "__main__":
    main()
