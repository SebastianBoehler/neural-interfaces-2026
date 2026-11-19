"""Build team B's external bank for the class-agnostic clusterability loss (no PROTEUS, no Scherer: that stays NI-V).

python extbank.py DATA_ROOT OUT.npz [MAX_PER_SESSION]   Reads the 120 Hz (n,41,480) exports of the public corpora listed in
CORPORA, keeps per (corpus, subject, session) up to MAX_PER_SESSION windows, class-balanced, sessions with at least two
classes of at least 12 windows. Stores float16 signals scaled per session to unit median channel std (alignment removes
scale anyway), class index within the corpus, a session id, and the 1-45 Hz covariance is recomputed at training time.
Labels of different corpora are never matched to PROTEUS classes: they only say which windows of one session belong together.
"""
import glob
import sys

import numpy as np

CORPORA = ("dreyer2023", "lee2019_mi", "cho2017_m1", "physionetmi_m1", "schirrmeister2017_m1", "stieger2021", "weibo2014",
           "zhou2020", "kumar2024", "chang2025", "yang2025", "guttmannflury2025_mi", "cmx7mm", "shin2017ab", "shin2018wg")


def main(root, out, cap=96):
    rng = np.random.default_rng(20261120)
    X, y, sid, names = [], [], [], []
    for corpus in CORPORA:
        files = sorted(f for f in glob.glob(f"{root}/{corpus}/windows*.npz") if not f.endswith("/windows.npz")
                       or len(glob.glob(f"{root}/{corpus}/windows*.npz")) == 1)
        n0 = len(names)
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                x, lab = z["X"], z["y"]
                key = np.char.add(np.char.add(z["subject"].astype(str), "|"), z["session"].astype(str))
            for k in dict.fromkeys(key.tolist()):
                rows = np.flatnonzero(key == k)
                classes = [c for c in np.unique(lab[rows]) if (lab[rows] == c).sum() >= 12]
                if len(classes) < 2:
                    continue
                per = cap // len(classes)
                keep = np.concatenate([rng.permutation(rows[lab[rows] == c])[:per] for c in classes])
                xs = x[keep].astype(np.float64)
                xs -= xs.mean(-1, keepdims=True)
                scale = np.median(xs.std(-1))
                if not np.isfinite(scale) or scale <= 0:
                    continue
                X.append((xs / scale).clip(-60, 60).astype(np.float16))
                y.append(np.searchsorted(classes, lab[keep]).astype(np.int8))
                sid.append(np.full(len(keep), len(names), np.int32))
                names.append(f"{corpus}|{k}")
        print(corpus, len(files), "files,", len(names) - n0, "sessions", flush=True)
    X, y, sid = np.concatenate(X), np.concatenate(y), np.concatenate(sid)
    assert not any(n.lower().startswith(("scherer", "dreyer2026")) for n in names)
    np.savez(out, X=X, y=y, session=sid, names=np.array(names))
    print("bank", X.shape, len(names), "sessions", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:]))
