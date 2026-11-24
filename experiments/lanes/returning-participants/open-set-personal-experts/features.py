"""Fixed per-window EEG measurements. Each row is independent of all other rows."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneQ"))
from extract_feat import feat as spectral_identity

BANDS = ((0.5, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 44), (44, 58))
TRIU = np.triu_indices(41)
UPPER = np.triu_indices(41, 1)
WINDOW = np.hanning(480)
SEG_WINDOW = np.hanning(120)
VIEW_SIZES = (len(BANDS) * len(TRIU[0]), 41 * 16, 41 * 4 * len(BANDS))


def extract(x):
    x = np.nan_to_num(x.astype(np.float64), nan=0, posinf=0, neginf=0)
    if x.shape != (41, 480):
        raise ValueError(f"unexpected EEG shape {x.shape}")
    ident = spectral_identity(x)
    car = x - x.mean(0)
    ft = np.fft.rfft((car - car.mean(1, keepdims=True)) * WINDOW)
    seg = car.reshape(41, 4, 120)
    power = np.abs(np.fft.rfft((seg - seg.mean(-1, keepdims=True)) * SEG_WINDOW)) ** 2
    logcov, envelope, connect = [], [], []
    for lo, hi in BANDS:
        v = ft[:, int(lo * 4):int(hi * 4)]
        c = (v @ v.conj().T).real / v.shape[1]
        # Absolute scale survives in the log-covariance diagonal.
        c += np.eye(41) * max(c.trace() * 0.03 / 41, 1e-12)
        w, u = np.linalg.eigh(c)
        logcov.append(((u * np.log(w)) @ u.T)[TRIU])
        envelope.append(np.log(1e-12 + power[:, :, max(1, int(np.ceil(lo))):int(hi)].mean(-1)).ravel())
        if lo in (4, 8, 13, 20):
            sd = np.sqrt(c.diagonal())
            connect.append((c / sd[:, None] / sd[None, :])[UPPER])
    slow = car.reshape(41, 16, 30).mean(-1)
    values = [np.concatenate([*logcov, slow.ravel(), *envelope]), ident, np.concatenate(connect)]
    if any(not np.isfinite(a).all() for a in values):
        raise ValueError("nonfinite EEG feature")
    return tuple(a.astype(np.float32) for a in values)


def save(path, x, idx, day, **meta):
    result = [extract(w) for w in x]
    f, spec, conn = map(np.stack, zip(*result))
    assert sum(VIEW_SIZES) == f.shape[1] and len(day) == len(f)
    # Probe labels stay in the source archive; only visit-1 training labels enter a feature file.
    y = meta.pop("y")
    np.savez(path, f=f, spec=spec, conn=conn, idx=idx, day=day, y1=y[day == 1], **meta)


def main(args):
    args.out.mkdir(exist_ok=True, parents=True)
    with np.load(args.data, allow_pickle=False) as z:
        x, y = z["X"], z["y"]
        if args.corpus == "net":
            day = z["day"]
            first = np.flatnonzero(day == 1)
            first = first[np.lexsort((z["onset"][first], z["run"][first]))]
            labels = np.full(len(y), -1)
            labels[first] = np.where(y[first] == 1, 0, 1 + np.cumsum(y[first] == 0) % 2)
            subjects = np.repeat(args.data.stem, len(day))
        else:
            subjects, day, labels = z["subject"], z["session"].astype(int), y
        for s in dict.fromkeys(subjects):
            idx = np.flatnonzero(subjects == s)
            if args.smoke:
                a, b = extract(x[idx[0]]), extract(x[idx[-1]])
                assert any(np.any(i != j) for i, j in zip(a, b))
                for i, j in zip(a, extract(x[idx[0]])):
                    np.testing.assert_array_equal(i, j)
                print("FEATURE SMOKE OK", x[idx[0]].shape, [v.shape for v in a], flush=True)
                return
            save(args.out / f"{args.corpus}_{s.split('/')[-1]}.npz", x[idx], idx, day[idx],
                 y=labels[idx], channels=z["ch_names"], name=s)
            print(s, len(idx), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus", choices=("pub", "sch", "net"))
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
