"""Compute an equal-session covariance prior from eligible source EEG only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fit import DATA_SHA, D, roster
from patch_terminal import verify_parent
from verify import B, digest


def population(data, fold):
    rows, people, first = roster(fold, data)
    source = {}
    exec(B.encoder_source(), source)
    mask = source["fmask"](1.0, 45.0)
    session = data["stream"][rows]
    covariances = []
    for stream in sorted(set(session)):
        x = torch.from_numpy(data["X"][rows[session == stream]])
        x = source["detrend"](x)
        filt = source["bandpass"](x, mask).double()
        cov = (filt @ filt.transpose(-1, -2)).mean(0).numpy() / x.shape[-1]
        energy = np.trace(cov)
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError(f"invalid source covariance from {stream}")
        covariances.append((cov + cov.T) / (2 * energy))
    prior = np.stack(covariances).mean(axis=0)
    if prior.shape != (41, 41) or np.linalg.eigvalsh(prior)[0] <= 0:
        raise ValueError("positive-definite 41-channel covariance required")
    return prior, dict(data_sha=DATA_SHA, fold=fold, people=people, first_only=first,
                       streams=sorted(set(session)), rows=len(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("fold")
    p.add_argument("out", type=Path)
    a = p.parse_args()
    verify_parent(a.data, DATA_SHA)
    torch.set_num_threads(2)
    prior, receipt = population(D.load(a.data), a.fold)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("xb") as f:
        np.save(f, prior, allow_pickle=False)
    receipt["prior_sha"] = digest(a.out)
    a.out.with_suffix(".json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
