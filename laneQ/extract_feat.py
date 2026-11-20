"""Lane Q: per-window IdentityGate spectral feature (1599 dims, verbatim IdentityGate.feat) for external corpora.
Usage: extract_feat.py OUT.npz IN1.npz [IN2.npz ...]   (numpy only; windows (n,41,480) as exported by export_mi84.py)"""
import sys
import numpy as np

WIN = np.hanning(120)


def feat(x):  # == submissions/lean_v66 IdentityGate.feat (checked in laneQ/check_feat.py)
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    x = x - x.mean(1, keepdims=True)
    seg = np.stack([x[:, s:s + 120] for s in range(0, 361, 60)], 1) * WIN
    L = np.log((np.abs(np.fft.rfft(seg, axis=-1)) ** 2).mean(1)[:, 2:41] + 1e-12)
    return (L - L.mean(1, keepdims=True)).ravel()


if __name__ == "__main__":
    out, ins = sys.argv[1], sys.argv[2:]
    cols = {k: [] for k in ("F", "subject", "session", "run", "onset", "y")}
    for f in ins:
        with np.load(f, allow_pickle=True) as z:
            X = z["X"]
            for i in range(len(X)):
                cols["F"].append(feat(np.nan_to_num(X[i])).astype(np.float32))
            for k in ("subject", "session", "run", "onset", "y"):
                cols[k].append(np.asarray(z[k]))
        print(f, len(X), flush=True)
    d = {k: np.concatenate(v) if k != "F" else np.stack(v) for k, v in cols.items()}
    np.savez(out, **d)
    print(out, d["F"].shape, {k: d[k][:3].tolist() for k in ("subject", "session", "run")})
