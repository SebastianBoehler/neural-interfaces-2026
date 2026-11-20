"""Refit the cap250 CountFilter, including the nested detector threshold."""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import family_args, load_source

BANDS = ((4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 58))
CHECKS = (40, 50, 60, 80, 100, 150)
SCHEMES = {"full": 0, "runs": 1, "within": 2}
FS = 120.0


def log_variance(x):
    bands = []
    for band in BANDS:
        sos = butter(4, band, "bandpass", fs=FS, output="sos")
        bands.append(np.concatenate([
            np.log(sosfiltfilt(sos, chunk, axis=-1).var(-1) + 1e-12)
            for chunk in np.array_split(x, (len(x) + 63) // 64)
        ]))
    return np.concatenate(bands, axis=1).astype(np.float32)


def residual(x, groups):
    out = np.empty_like(x)
    for group in dict.fromkeys(groups):
        keep = groups == group
        out[keep] = x[keep] - x[keep].mean(0)
    return out


def projection(features, runs):
    within = residual(features, runs)
    sw = np.cov(within.T)
    sb = np.cov((features - within).T)
    sw += np.trace(sw) / len(sw) * np.eye(len(sw))
    eigenvalues, vectors = eigh(sb, sw)
    return vectors[:, np.argsort(eigenvalues)[::-1][:6]]


def run_names(data):
    return np.char.add(np.char.add(data["task"].astype(str), "|"), data["run"].astype(str))


def orders(stream, run):
    sessions = [np.flatnonzero(stream == s) for s in dict.fromkeys(stream)]
    yield True, sessions
    for scheme, offset in SCHEMES.items():
        for seed in range(12):
            rng = np.random.default_rng([seed, 0, offset])
            deliveries = []
            for rows in sessions:
                if scheme == "full":
                    deliveries.append(rows[rng.permutation(len(rows))])
                    continue
                blocks = [rows[run[rows] == tag] for tag in dict.fromkeys(run[rows])]
                if scheme == "runs":
                    deliveries.append(np.concatenate([blocks[i] for i in rng.permutation(len(blocks))]))
                else:
                    deliveries.append(np.concatenate([block[rng.permutation(len(block))] for block in blocks]))
            yield scheme != "full", deliveries


def detector(projected, stream, run):
    values, contiguous = [], []
    for contig, deliveries in orders(stream, run):
        for rows in deliveries:
            ratios = []
            for n in CHECKS:
                if len(rows) < n:
                    raise ValueError(f"CountFilter calibration visit has fewer than {n} windows")
                i, j = np.triu_indices(n, 1)
                distances = pdist(projected[rows[:n]])
                lag = j - i
                ratios.append(distances[lag <= 3].mean() / distances[lag >= 15].mean())
            values.append(ratios)
            contiguous.append(contig)
    same = np.asarray(contiguous)
    return np.array([threshold(value, same) for value in np.asarray(values).T])


def threshold(value, same):
    ordered = np.sort(value)
    candidates = (ordered[1:] + ordered[:-1]) * .5
    error = ((value[same, None] >= candidates).mean(0) +
             (value[~same, None] < candidates).mean(0)) * .5
    return candidates[error.argmin()]


def design_prior(stream, task):
    observed, first = [], []
    for s in dict.fromkeys(stream):
        bh = task[stream == s] == "AcquisitionBH"
        observed.append(bh.sum())
        first.append(bh[0])
    lengths = range(30, 37)
    prior = np.array([observed.count(length) + .5 for length in lengths])
    prior /= prior.sum()
    bh_first = (sum(first) + 1) / (len(first) + 2)
    hypotheses = [[a, b, 30, 30, 30, 30] for length in lengths
                  for a, b in ((36, length), (length, 36))]
    log_prior = (np.log(prior)[:, None] + np.log((1 - bh_first, bh_first))).ravel()
    return np.array(hypotheses), log_prior, np.tile([12, 12, 10, 10, 10, 10, 12, 12], (14, 1))


def fit(data):
    names = np.unique(data["person"])
    if len(names) < 3:
        raise ValueError("nested CountFilter calibration requires at least three source people")
    stream = data["stream"]
    local_run = run_names(data)
    runs = np.char.add(np.char.add(stream, "|"), local_run)
    features = log_variance(data["X"])
    centered = residual(features, stream)
    masks = [data["person"] == name for name in names]

    def estimate(excluded):
        keep = ~np.logical_or.reduce([masks[i] for i in excluded])
        return projection(centered[keep], runs[keep])

    def within_scale(w, held):
        error = residual(centered[held] @ w, runs[held])
        return (error ** 2).mean(0)

    n = range(len(names))
    single = [estimate((i,)) for i in n]
    pair = {(i, j): estimate((i, j)) for i in n for j in range(i + 1, len(names))}
    base_scale = np.sqrt(np.mean([within_scale(single[i], masks[i]) for i in n], axis=0))
    projected = np.empty((len(features), 6))
    for i in n:
        variance = [within_scale(pair[min(i, j), max(i, j)], masks[j]) for j in n if j != i]
        projected[masks[i]] = (features[masks[i]] @ single[i]) / np.sqrt(np.mean(variance, axis=0))
    lengths, prior, caps = design_prior(stream, data["task"])
    result = dict(cf_W=projection(centered, runs), cf_sw=base_scale, cf_bands=np.array(BANDS, float),
                  cf_fs=FS, cf_checks=CHECKS, cf_thr=detector(projected, stream, local_run),
                  cf_Ls=lengths, cf_lp=prior, cf_cap=caps, cf_tau=2.0, cf_floor=.5, cf_K=13)
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("nonfinite fitted CountFilter parameter")
    return result


if __name__ == "__main__":
    args = family_args(__doc__, (0,))
    data = load_source(args.people, args.first_session_only, args.data)
    np.savez(args.out / "cf_params.npz", **fit(data))
