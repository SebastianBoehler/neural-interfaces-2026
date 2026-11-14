"""External corpora for the harness (ledger s216): windows on the 41 PROTEUS channels at 120 Hz (scripts/export_moabb.py
or the Scherer export), reduced to K balanced classes, then CAR, a filter bank and log-Euclidean tangent vectors.

feats(name) -> dict(V (n, nb*861) tangent vectors, uncentred; y; person; session; run; classes), cached in
runs/extharness/feat_<name>.npz under a key that holds every setting below (a changed setting fails loudly)."""
import glob, hashlib, json, os
import numpy as np
from scipy.signal import butter, sosfiltfilt

REPO = os.path.abspath(__file__ + "/../../../..")
OUT = REPO + "/runs/extharness"
BANDS = ((4, 8), (8, 13), (13, 20), (20, 30))
SHRINK = 0.05  # covariance shrinkage towards trace/41 (CAR leaves rank 40; interpolated channels lower it further)
# corpus -> (file glob, classes kept by name, in this label order). Class names come from the export's `classes`.
CORPORA = {
    "physionetmi": ("data/physionetmi/windows_*.npz", ("left_hand", "right_hand", "feet")),
    "schirrmeister2017": ("data/schirrmeister2017/windows_*.npz", ("left_hand", "right_hand", "feet")),
    "scherer2015": ("data/scherer2015/windows.npz", None),  # already 3 balanced classes (the PROTEUS tasks)
}


def _classes(w):
    for k in ("classes", "class_names", "events"):
        if k in w.files:
            return [str(c) for c in w[k]]
    return None


def load(name):
    pat, keep = CORPORA[name]; files = sorted(glob.glob(REPO + "/" + pat)); assert files, f"no files for {name}: {pat}"
    cols = {k: [] for k in ("X", "y", "person", "session", "run")}; names = None
    for f in files:
        with np.load(f, allow_pickle=True) as w:
            cn = _classes(w)
            if keep is not None:
                assert cn is not None, f"{f}: no class names in the export, cannot select {keep}"
                assert names in (None, cn), f"{f}: class list differs between files: {cn} vs {names}"
                names = cn; lut = np.full(len(cn), -1); lut[[cn.index(c) for c in keep]] = np.arange(len(keep))
                y = lut[w["y"]]; m = y >= 0; y = y[m]
            else:
                m = np.ones(len(w["y"]), bool); y = w["y"]
            cols["X"].append(w["X"][m].astype(np.float32)); cols["y"].append(y.astype(int))
            cols["person"].append(w["subject"][m].astype(str)); cols["session"].append(w["session"][m].astype(str))
            cols["run"].append(w["run"][m].astype(str))
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d["classes"] = np.array(keep if keep is not None else ["c0", "c1", "c2"])
    for k in np.unique(d["y"]):
        assert (d["y"] == k).sum() > 0
    assert set(np.unique(d["y"])) == set(range(len(d["classes"]))), "class mapping broken"
    return d


def tangent(X):
    """(n, 41, T) microvolts -> (n, nb * 861): CAR, Butterworth-4 band (zero-phase inside the window), shrunk covariance,
    matrix log, upper triangle with off-diagonals x sqrt(2)."""
    X = X - X.mean(1, keepdims=True); c = X.shape[1]; iu = np.triu_indices(c); wgt = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    out = []
    for lo, hi in BANDS:
        sos = butter(4, [lo, hi], "bandpass", fs=120, output="sos"); vb = []
        for a in range(0, len(X), 2000):
            Xb = sosfiltfilt(sos, X[a:a + 2000], axis=-1)
            C = np.einsum("nct,ndt->ncd", Xb, Xb) / Xb.shape[-1]; tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / c
            C = (1 - SHRINK) * C + SHRINK * tr * np.eye(c); w, U = np.linalg.eigh(C)
            Lg = np.einsum("ncd,nd,ned->nce", U, np.log(w), U); vb.append(Lg[:, iu[0], iu[1]] * wgt)
        out.append(np.concatenate(vb))
    return np.concatenate(out, 1).astype(np.float32)


def feats(name):
    pat, keep = CORPORA[name]
    key = hashlib.sha256(json.dumps([name, pat, keep, BANDS, SHRINK, "car", "logeuclid", 1]).encode()).hexdigest()[:12]
    path = f"{OUT}/feat_{name}.npz"
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:
            if str(z["key"]) == key:
                return {k: z[k] for k in z.files}
            print(f"feature cache {path} has key {z['key']}, settings give {key}: rebuilding", flush=True)
    d = load(name); V = tangent(d.pop("X")); os.makedirs(OUT, exist_ok=True)
    np.savez(path, V=V, key=key, **d)
    return dict(V=V, key=key, **d)


if __name__ == "__main__":
    import sys, collections as C
    for nm in sys.argv[1:]:
        d = feats(nm); print(nm, d["V"].shape, "people", len(np.unique(d["person"])), "classes", list(d["classes"]),
                            "counts", np.bincount(d["y"]).tolist(), "sessions/person",
                            C.Counter(C.Counter(zip(d["person"], d["session"])).values()).most_common(4))
