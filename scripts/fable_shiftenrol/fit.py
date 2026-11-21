"""Fit the person-indexed parameters consumed by the signed lead."""
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("", "experiments", "cleanslate/cap250/families")]
from _id_own import KEYS, Space
from bio_gate import Bio
from features import expm_sym, logm, slow_waveform, whiten_tangent
from geometry import CONFIG, contrast, standardize
from lean.src66.tangent import TangentCore
from ts_core import blocks, parse

SEED = 20261121
SLOTS = ("bio_gal", "route_gal", "route_ps1_W", "route_ps1_c", "v6_tpri", *(f"own_{k}" for k in KEYS))


def fit_views(x, ref, z, pc):
    """Use the geometry family's offline PCs for PS1 and its live low PCs for TSEM."""
    car = x - x.mean(1, keepdims=True)
    specific = []
    alignment = np.arange(len(x)) if ref is None else ref
    for cov in blocks(car, parse(CONFIG)):
        specific.append(whiten_tangent(cov, expm_sym(logm(cov[alignment]).mean(0))))
    slow = slow_waveform(car, cutoff=1).astype(np.float64)
    specific.append(slow - slow[alignment].mean(0))
    scaled = (np.concatenate(specific, axis=1) - z["ts_mu"]) / z["ts_sd"]
    offline = (scaled - z["tsem_m"]) @ pc

    core = TangentCore({k[3:]: v for k, v in z.items() if k.startswith("ts_")})
    if ref is not None:
        for r in ref:
            core.features(x[r])
        reference = (core.n, core.log_sum.copy(), np.array(core.slow_sum, copy=True))
    live = []
    for window in x:
        if ref is not None:
            core.n, log_sum, slow_sum = reference
            core.log_sum, core.slow_sum = log_sum.copy(), slow_sum.copy()
        live.append((core.step(np.nan_to_num(window))[1] - z["tsem_m"]) @ pc[:, :128])
    return offline, standardize(np.asarray(live))


def copies(x, names, person, visit):
    from ts_core2 import laplacian

    if len(x) < 40:
        raise ValueError("augmented enrolment needs at least 40 windows per visit")
    rng = np.random.default_rng([SEED, int(person[1:]), int(visit)])
    neighbor = np.eye(len(names)) - laplacian(names, k=4)
    np.testing.assert_allclose(neighbor.sum(1), 1, atol=1e-12, rtol=0)
    result = [(x, None)]
    for _ in range(8):
        gain = np.exp(rng.normal(0, 0.1, size=len(names)))
        mixing = np.eye(len(names)) + rng.uniform(0, 0.3) * neighbor
        shifted = ((mixing * gain[None]).astype(x.dtype) @ x).astype(x.dtype)
        ref = np.sort(rng.choice(len(x), size=rng.integers(20, 41), replace=False))
        if not np.isfinite(shifted).all() or np.array_equal(x, shifted):
            raise ValueError("cap transform did not produce a distinct finite copy")
        result.append((shifted, ref))
    return result


def fit_person(data, visits, person, z, augmented=False):
    pc = np.ascontiguousarray(np.concatenate((z["tsem_V"], z["route_V_hi"]), 1), dtype=np.float64)
    bio = Bio.__new__(Bio)
    bio.win = np.hanning(120)
    gallery, identity, tpri, views, signals, labels, refs, groups = [], [], [], [], [], [], [], []
    n = 0
    for visit, rows in visits.items():
        x, y = data["X"][rows], data["y"][rows]
        if (not np.all(data["stream"][rows] == f"{person}|{visit}") or
                not np.array_equal(np.unique(y), [0, 1, 2]) or len(rows) != len(np.unique(rows))):
            raise ValueError("invalid enrolment rows")
        samples = copies(x, data["ch_names"], person, visit) if augmented else [(x, None)]
        for wave, ref in samples:
            offline, live = fit_views(wave, ref, z, pc)
            gallery.append(contrast(offline, np.eye(3)[y]))
            identity.append(np.stack([bio.feat(w) for w in wave]).mean(0))
            tpri.append(np.stack([live[y == i].mean(0) for i in range(3)]))
            views.append(standardize(offline))
            signals.append(wave)
            labels.append(y)
            groups.append(np.arange(n, n + len(y)))
            refs.append(groups[-1] if ref is None else n + ref)
            n += len(y)
    y = np.concatenate(labels)
    own = Space(np.concatenate(signals), y).person(groups, refs)
    ps1 = LogisticRegression(C=.1, max_iter=3000).fit(np.concatenate(views), y)
    result = {f"own_{key}": value for key, value in own.items()}
    result.update(bio_gal=((np.mean(identity, 0) - z["bio_mu"]) / z["bio_sd"]) @ z["bio_V"],
                  route_gal=np.mean(gallery, 0), route_ps1_W=ps1.coef_, route_ps1_c=ps1.intercept_,
                  v6_tpri=np.mean(tpri, 0))
    if set(result) != set(SLOTS) or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("incomplete or non-finite enrolment")
    return result


def half(data, person, visit):
    """Fixed stratified enrolment and test halves, with source rows in recording order."""
    rng = np.random.default_rng([SEED, int(person[1:]), int(visit), 1])
    visit_rows = np.flatnonzero(data["stream"] == f"{person}|{visit}")
    train, test = [], []
    for interface in sorted(set(data["interface"][visit_rows])):
        for label in range(3):
            rows = visit_rows[(data["interface"][visit_rows] == interface) & (data["y"][visit_rows] == label)]
            if len(rows) < 2:
                raise ValueError("cannot stratify an empty or singleton task/interface")
            rows = rng.permutation(rows)
            train.extend(rows[:len(rows) // 2])
            test.extend(rows[len(rows) // 2:])
    np.testing.assert_array_equal(np.sort(train + test), visit_rows)
    return np.sort(train), np.sort(test)
