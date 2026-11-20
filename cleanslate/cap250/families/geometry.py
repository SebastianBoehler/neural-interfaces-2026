"""Refit the real tangent core, PC basis, task gallery, PS1 and TSEM personal prior."""

import os
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import ROOT, family_args, load_source

sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import session_em
import ts_core
from export_tsem import pc_basis

CONFIG = "car_ov17_fs2_t0.1_le_k2"


def fit_basis(data, *, need_views):
    cfg = ts_core.parse(CONFIG)
    features = ts_core.Feats(data, cfg)
    n = len(data["y"])
    head = ts_core.Core(features, ["lr10"], {"lr10": np.full((n, 3), np.nan)})
    head.fit(np.arange(n))
    scaler, lr = [model for _, model in head.models[0].steps]
    if not np.array_equal(lr.classes_, (0, 1, 2)):
        raise ValueError("real tangent head missing an official task class")
    scaled = scaler.transform(features.F)
    mean, vectors, _ = pc_basis(scaled, 256)
    if vectors.shape != (44239, 256):
        raise ValueError("real tangent PC basis must span all 256 cap250 coordinates")
    vectors = vectors.astype(np.float32)
    core = dict(bands=np.array(ts_core.BANDSETS[cfg["bands"]], dtype=np.float64),
                log_glob=np.stack(head.priors), k=np.float64(cfg["k"]), shrink=np.float64(.1),
                mu=scaler.mean_, sd=scaler.scale_, W=lr.coef_, b=lr.intercept_,
                slow_cutoff=np.float64(1), slow_step=np.int64(60), car=np.int64(1), nsub=np.int64(2))
    event_scale = session_em.global_mad(data["X"].astype(np.float64), data["ch_names"].tolist())
    params = {f"ts_{key}": value for key, value in core.items()}
    params.update(tsem_m=mean, tsem_V=np.ascontiguousarray(vectors[:, :128]),
                  route_V_hi=np.ascontiguousarray(vectors[:, 128:]), em_ev_g=event_scale,
                  tsem_lam=np.float64(.15), tsem_rho=np.float64(.7), tsem_shrink=np.float64(.7),
                  tsem_m0=np.int64(20), tsem_n0=np.int64(60), tsem_nmin=np.int64(15))
    if not need_views:
        return params, None, None

    source_view = (scaled - mean) @ vectors.astype(np.float64)
    del scaled, features
    sys.path.insert(0, str(ROOT))
    from lean.src66.tangent import TangentCore

    live_core = TangentCore(core)
    causal_low = np.empty((n, 128))
    low = params["tsem_V"].astype(np.float64)
    for visit in dict.fromkeys(data["stream"]):
        live_core.reset_state()
        for i in np.flatnonzero(data["stream"] == visit):
            _, row = live_core.step(np.nan_to_num(data["X"][i]))
            causal_low[i] = (row - mean) @ low
    return params, source_view, causal_low


def standardize(x):
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)


def contrast(x, weights):
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0) * 1.4826
    normalized = np.clip((x - median) / np.maximum(mad, 1e-6), -4., 4.)
    mass = weights.sum(0)
    if (mass == 0).any():
        raise ValueError("route calibration has a visit with an empty contrast class")
    means = (weights.T @ normalized) / mass[:, None]
    return means - means.mean(0)


def confident_targets(p):
    weights = np.zeros_like(p)
    confidence, decision = p.max(1), p.argmax(1)
    quota = max(2, int(.7 * len(p) / 3))
    for task in range(3):
        rows = np.flatnonzero(decision == task)
        weights[rows[np.argsort(-confidence[rows])[:quota]], task] = 1
    return weights


def fit_personal(data, people, source_view, causal_low, oof):
    gallery, debias, heads, intercepts, tpri, visits = [], [], [], [], [], []
    for person in people:
        groups = [np.flatnonzero(data["stream"] == visit)
                  for visit in dict.fromkeys(data["stream"][data["person"] == person])]
        contrast_true, contrast_error, prior = [], [], []
        for rows in groups:
            y = data["y"][rows]
            actual = contrast(source_view[rows], np.eye(3)[y])
            pseudo = contrast(source_view[rows], confident_targets(oof[rows]))
            contrast_true.append(actual)
            contrast_error.append(pseudo - actual)
            causal = standardize(causal_low[rows])
            prior.append(np.stack([causal[y == i].mean(0) for i in range(3)]))
        lr = LogisticRegression(C=.1, max_iter=3000)
        lr.fit(np.concatenate([standardize(source_view[r]) for r in groups]),
               np.concatenate([data["y"][r] for r in groups]))
        gallery.append(np.mean(contrast_true, axis=0))
        debias.append(np.mean(contrast_error, axis=0))
        heads.append(lr.coef_)
        intercepts.append(lr.intercept_)
        tpri.append(np.mean(prior, axis=0))
        visits.append(len(groups))
    pop, bias = np.mean(gallery, axis=0), np.mean(debias, axis=0)
    q, r = np.linalg.qr(np.stack([pop.ravel(), bias.ravel()], axis=1))
    if np.min(np.abs(np.diag(r))) < 1e-9:
        raise ValueError("route task contrast and debias directions have insufficient rank")
    params = dict(route_gal=np.stack(gallery), route_pop=pop, route_b=bias, route_Q=q,
                  route_names=np.array(people), route_ps1_W=np.stack(heads), route_ps1_c=np.stack(intercepts),
                  route_clip=np.float64(4), route_rho=np.float64(.7), route_t_start=np.int64(60),
                  route_tau=np.float64(.22), route_ramp=np.float64(.05), route_slot_g=np.float64(3),
                  route_slot_a=np.float64(.06666666666666668), route_v3=np.int64(1))
    return params, dict(tpri=np.stack(tpri), names=np.array(people), sessions=np.array(visits))


def fit(args):
    data = load_source(args.people, args.first_session_only, args.data)
    base_only = os.environ.get("CAP250_BASE_ONLY") == "1"
    if not base_only:
        from route_inputs import load_oof
        oof = load_oof(Path(os.environ["CAP250_ROUTE_OOF"]), data["index"], args.people, args.first_session_only, args.data)
    params, training, causal = fit_basis(data, need_views=not base_only)
    if not base_only:
        personal, prior = fit_personal(data, args.people, training, causal, oof)
        params.update(personal)
        np.savez(args.out / "tpri.npz", **prior)
    np.savez(args.out / "weights.npz", **params)


if __name__ == "__main__":
    fit(family_args(__doc__, (0,)))
