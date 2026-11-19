"""Inspect open-set recognition distances without using labels or making predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eeg import center, covariance, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("data", type=Path)
    a = p.parse_args()
    with np.load(a.data, allow_pickle=False) as z:
        raw = center(z["X"])
    identities = np.concatenate([fingerprint(x, covariance(x)) for x in np.array_split(raw, 12)])
    result = []
    for line in (a.run / "jobs.tsv").read_text().splitlines():
        role, held, train, _, _ = line.split("\t")
        people = sorted(train.split(","))
        bank = torch.load(a.run / f"{role}_{held}" / "weights.pt", map_location="cpu", weights_only=True, mmap=True)["bank"]
        g = bank["gallery"].numpy()
        scale = bank["identity_scale"].numpy()
        with np.load(a.run / f"pred_{role}_{held}.npz", allow_pickle=False) as archive:
            for key in archive.files:
                if not key.startswith(("rows_71000_", "rows_71001_")):
                    continue
                rows = archive[key]
                means = np.cumsum(identities[rows], axis=0, dtype=np.float64)
                ages = np.array([16, 32, 64, 128, len(rows)])
                d = (g[None] - means[ages - 1, None] / ages[:, None, None]) * scale
                distance = (d ** 2).mean(-1) ** 0.5
                rank = np.argsort(distance, axis=-1)
                best = rank[:, 0]
                idx = np.arange(len(ages))
                ratio = distance[idx, best] / distance[idx, rank[:, 1]]
                result.append(dict(
                    role=role, person=held, tag=key, age=ages.tolist(),
                    best=[people[i] for i in best], distance=distance[idx, best].round(4).tolist(),
                    gap=ratio.round(4).tolist(), radius=bank["identity_radius"],
                    true_rank=None if held not in people else (rank == people.index(held)).argmax(-1).tolist(),
                ))
    out = a.run / "gallery_diagnostics.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
