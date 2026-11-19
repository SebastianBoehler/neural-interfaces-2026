"""Choose paired local and dispersed sensor gaps using only source waveform energy."""
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
MONTAGE = "71d07f1263c348b64990c2d1c329cd30fc21281583563be5a8629ed9fcf281a9"
ROW_TOL = .006
MEAN_TOL = .001


def geometry(channels):
    path = ROOT / "montage.tsv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != MONTAGE:
        raise ValueError("frozen standard_1020 montage differs")
    rows = [line.split("\t") for line in path.read_text().splitlines()[1:]]
    if [entry[0] for entry in rows] != list(channels) or len(rows) != 41:
        raise ValueError("montage channel order differs from the official tensor")
    xyz = np.asarray([[float(q) for q in r[1:]] for r in rows])
    distance = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
    local = np.argsort(distance, axis=1, kind="stable")[:, :3]
    pairs = np.asarray(list(combinations(range(41), 2)))
    controls = {}
    for site, patch in enumerate(local):
        diameter = distance[patch[:, None], patch].max()
        outside = ~np.isin(pairs, patch).any(1)
        for shared in patch:
            ds = np.column_stack((distance[shared, pairs], distance[pairs[:, 0], pairs[:, 1]]))
            valid = outside & (ds.min(1) >= .055) & (ds.max(1) >= diameter + .025)
            controls[site, shared] = pairs[valid]
            if not valid.any():
                raise ValueError("a local montage patch has no physically dispersed control")
    return local, controls, distance


def plan(x, channels, batches, held, seed):
    if x.shape[1:] != (41, 480) or x.dtype != np.float32:
        raise ValueError("expected raw 41×480 source windows")
    local, choices, distance = geometry(channels)
    weights = np.einsum("rct,rct->rc", x, x, dtype=np.float64)
    totals = weights.sum(1)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("invalid source sensor energies")
    weights /= totals[:, None]
    rng = np.random.default_rng(18018 + 100 * int(held[1:]) + seed)
    order = np.argsort(rng.random((len(x), 41)), axis=1, kind="stable")
    nearby = np.full((len(x), 3), -1, dtype=np.int16)
    spread = nearby.copy()
    remaining = np.arange(len(x))
    for attempt in range(41):
        site = order[remaining, attempt]
        patch = local[site]
        w = weights[remaining[:, None], patch]
        shared = patch[np.arange(len(patch)), w.argmax(1)]
        target = w.sum(1) - weights[remaining, shared]
        for key in np.unique(np.stack((site, shared), axis=1), axis=0):
            k, a = (int(v) for v in key)
            use = np.flatnonzero((site == k) & (shared == a))
            ids = remaining[use]
            options = choices[k, a]
            pair_energy = weights[ids[:, None], options[:, 0]] + weights[ids[:, None], options[:, 1]]
            cost = np.abs(pair_energy - target[use, None])
            best = cost.argmin(1)
            matching = cost[np.arange(len(best)), best] <= ROW_TOL
            ids = ids[matching]
            nearby[ids] = local[k]
            spread[ids, 0] = a
            spread[ids, 1:] = options[best[matching]]
        remaining = remaining[nearby[remaining, 0] == -1]
        if not len(remaining):
            break
    if len(remaining):
        raise ValueError(f"{len(remaining)} source windows lack an energy-matched candidate")
    ix = np.arange(len(x))[:, None]
    a, b = (weights[ix, group].sum(1) for group in (nearby, spread))
    error = np.abs(a - b)
    overlap = (nearby[:, :, None] == spread[:, None, :]).sum((1, 2))
    shared_energy = weights[np.arange(len(x)), spread[:, 0]]
    local_span = distance[nearby[:, :, None], nearby[:, None, :]].max((1, 2))
    off_local = distance[spread[:, :, None], spread[:, None, :]]
    margin = off_local.max((1, 2)) - local_span
    tri = off_local[:, [0, 0, 1], [1, 2, 2]].min()
    if (error.max() > ROW_TOL + 1e-12 or error.mean() > MEAN_TOL or not (overlap == 1).all() or
            not np.array_equal(shared_energy, weights[ix, nearby].max(1)) or
            (a <= 0).any() or (b <= 0).any() or tri < .055 or margin.min() < .025 - 1e-12):
        raise ValueError(f"spatial severity control failed: max {error.max():.6g}, mean {error.mean():.6g}")
    digest = hashlib.sha256(np.asarray([nearby, spread], dtype="<i2").tobytes())
    masks = []
    for rows in (r for epoch in batches for r in epoch):
        if len(rows) % 4:
            raise ValueError("original batch cannot receive an exact quarter treatment")
        mask = np.zeros(len(rows), bool)
        mask[rng.permutation(len(rows))[:len(rows) // 4]] = True
        digest.update(mask.tobytes())
        masks.append(mask)
    evidence = dict(energy_error_max=float(error.max()), energy_error_mean=float(error.mean()),
                    local_energy=float(a.mean()), dispersed_energy=float(b.mean()), separation_min=float(tri),
                    diameter_margin_min=float(margin.min()), second_choice_rows=int(np.count_nonzero(
                        (nearby != local[order[:, 0]]).any(1))))
    return nearby, spread, masks, digest.hexdigest(), evidence


@torch.no_grad()
def zero_channels(x, selections):
    if selections.shape != (len(x), 3):
        raise ValueError("training requires three whole channels per selected row")
    how = torch.as_tensor(selections.astype(np.int64), device=x.device)
    mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device).scatter_(1, how, True)
    out = x.masked_fill(mask[:, :, None], 0)
    if (not (mask.sum(1) == 3).all() or torch.count_nonzero(out[mask]) or
            not torch.equal(out[~mask], x[~mask]) or torch.equal(out, x)):
        raise ValueError("actual waveform masking did not withhold only the intended channels")
    return out
