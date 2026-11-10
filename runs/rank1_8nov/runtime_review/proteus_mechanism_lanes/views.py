"""Signal-processing views for three source-only PROTEUS screens."""

import numpy as np
from scipy.fft import dct
from scipy.special import ndtri
from scipy.stats import rankdata

from experiments.features import bandpass, covariances, expm_sym, logm, whiten_tangent

BANDS = ((4, 8), (8, 13), (13, 25), (25, 40))


def correlation(x):
    return covariances(x / np.maximum(x.std(axis=-1, keepdims=True), 1e-8), shrink=.05)


def copula_views(raw):
    blocks = {name: [] for name in ("linear", "copula", "linear_compressed")}
    for low, high in BANDS:
        band = bandpass(raw, low, high)
        rank = rankdata(band, axis=-1)
        normal = ndtri((rank - .5) / band.shape[-1]).astype(np.float32)
        # Strictly monotone soft compression leaves within-channel ranks intact.
        scale = np.maximum(2 * band.std(axis=-1, keepdims=True), 1e-8)
        wide = band.astype(np.float64)
        compressed = wide / (1 + np.abs(wide) / scale)
        np.testing.assert_array_equal(rank, rankdata(compressed, axis=-1))
        blocks["linear"].append(correlation(band))
        blocks["copula"].append(correlation(normal))
        blocks["linear_compressed"].append(correlation(compressed))
    return {name: np.stack(parts, axis=1) for name, parts in blocks.items()}


def tangent_views(covs, rows):
    out = {name: [] for name in covs}
    for name in ("linear", "copula"):
        for i in range(len(BANDS)):
            band = covs[name][:, i]
            ref = expm_sym(logm(band[rows]).mean(axis=0))
            out[name].append(whiten_tangent(band, ref))
            if name == "linear":
                out["linear_compressed"].append(whiten_tangent(covs["linear_compressed"][:, i], ref))
    return {name: np.column_stack(parts) for name, parts in out.items()}


def nuisance_views(x, draws, y, person, interface, seed, fold):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x[draws])
    z = scaler.transform(x)
    observed, placebo = [], []
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 27]))
    for who in np.unique(person[draws]):
        for label in (0, 1, 2):
            cells = [draws[(person[draws] == who) & (interface[draws] == k) & (y[draws] == label)]
                     for k in ("BrainHero", "Graz")]
            if min(map(len, cells)) < 2:
                raise ValueError("Missing source context samples")
            observed.append(z[cells[0]].mean(0) - z[cells[1]].mean(0))
            halves = [np.array_split(rng.permutation(rows), 2) for rows in cells]
            placebo.append(np.mean([z[a].mean(0) - z[b].mean(0) for a, b in halves], axis=0))

    def remove(deltas):
        _, _, axes = np.linalg.svd(np.array(deltas), full_matrices=False)
        directions = axes[:3]
        return z - (z @ directions.T) @ directions

    return {"plain": z, "nuisance_removed": remove(observed), "placebo_removed": remove(placebo)}


def ordered_views(raw):
    taper = np.hanning(120)
    frequencies = np.fft.rfftfreq(120, 1/120)
    slices = [(frequencies >= a) & (frequencies < b) for a, b in BANDS]
    tiles = []
    for start in range(0, 361, 60):
        x = raw[..., start:start+120]
        power = np.abs(np.fft.rfft((x - x.mean(axis=-1, keepdims=True))*taper, axis=-1))**2
        x = np.log(np.stack([power[..., f].mean(axis=-1) for f in slices], axis=-1).clip(1e-12))
        tiles.append(x - x.mean(axis=1, keepdims=True))
    x = np.stack(tiles, axis=1)
    order = np.random.default_rng(7802).permuted(np.broadcast_to(np.arange(x.shape[1]), x.shape[:2]), axis=1)
    shuffled = np.take_along_axis(x, order[..., None, None], axis=1)
    mean = x.mean(axis=1).reshape(len(x), -1)

    def harmonics(value):
        return dct(value, axis=1, norm="ortho")[:, 1:3].reshape(len(x), -1)

    return {"mean_only": mean, "ordered": np.column_stack((mean, harmonics(x))),
            "shuffled_order": np.column_stack((mean, harmonics(shuffled)))}
