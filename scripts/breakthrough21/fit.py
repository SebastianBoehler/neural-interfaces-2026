"""Fit only the spectral expert on the declared per-fold training rows."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cleanslate/harness"))
import data as D
from spectral import SpectralGeometry

DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
TRAIN_ORDERS = (15101, 15102, 15103, 15104)


def roster(fold, data):
    if fold == "all":
        people, first = D.PEOPLE, []
    else:
        mode, person = fold.split("_")
        if person not in D.PEOPLE or mode not in ("new", "ret") or (mode == "ret" and person not in D.RETURNING):
            raise ValueError(f"invalid training fold {fold}")
        people = D.PEOPLE if mode == "ret" else [p for p in D.PEOPLE if p != person]
        first = [person] if mode == "ret" else []
    return D.train_rows(data, people, first), people, first


def fit(data, fold, out):
    rows, people, first = roster(fold, data)
    shape = SpectralGeometry(data["ch_names"])
    x = np.stack([shape.features(data["X"][i]) for i in rows])
    y = data["y"][rows]
    stream = data["stream"][rows]
    cells = np.char.add(np.char.add(stream, "|"), data["interface"][rows])
    keys = np.char.add(np.char.add(cells, "|"), y.astype(str))
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inv]
    weight /= weight.mean()
    center = np.average(x, axis=0, weights=weight)
    x -= center
    scale = np.sqrt(np.average(x ** 2, axis=0, weights=weight)).clip(1e-5)
    copies = [x]
    for seed in TRAIN_ORDERS:
        sample = x.copy()
        delivery = D.delivery(stream, seed)
        for s in dict.fromkeys(stream):
            p = delivery[stream[delivery] == s]
            prefix = np.cumsum(x[p], axis=0)
            sample[p] -= shape.CENTER * prefix / (shape.PRIOR + np.arange(1, len(p) + 1)[:, None])
        copies.append(sample)
    design = np.clip(np.concatenate(copies) / scale, -6, 6)
    optim = LogisticRegression(C=0.04, max_iter=500, tol=1e-7)
    optim.fit(design, np.tile(y, len(copies)), sample_weight=np.tile(weight, len(copies)) / len(copies))
    if optim.n_iter_.max() >= optim.max_iter or tuple(optim.classes_) != (0, 1, 2):
        raise RuntimeError("spectral fit failed to converge on the three classes")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as f:
        np.savez_compressed(f, center=center, scale=scale, weight=optim.coef_, bias=optim.intercept_,
                            channels=data["ch_names"])
    return dict(fold=fold, people=people, first_only=first, streams=sorted(set(stream)),
                rows=len(rows), spectral_sha=sha256(out.read_bytes()).hexdigest(), data_sha=DATA_SHA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("fold")
    p.add_argument("out", type=Path)
    a = p.parse_args()
    if sha256(a.data.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("unexpected PROTEUS export")
    receipt = fit(D.load(a.data), a.fold, a.out)
    a.out.with_suffix(".json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
