"""Recompute the route4 contrast and spectral identity ramps from live shadows."""
from pathlib import Path

import numpy as np

from bridge_owner import PROTO, ROOT, STACK, pin, sha

REFERENCE = ROOT / "runs/s281_v66_route/full0.json"
PREFIX = 80


def source_fold(path, fold, stream, people, opened):
    with np.load(path, allow_pickle=False) as data:
        source = data["idx_tr"]
        if not np.array_equal(np.sort(source), np.flatnonzero(people != fold)):
            raise ValueError(f"{fold}: recognition gallery includes held rows or misses source rows")
        pcs = data["pc_tr"].astype(np.float64)
        # The fold's source labels are opened only after authenticating the row exclusion.
        y = data.raw["y_tr"]
        if y.shape != source.shape or not np.isin(y, [0, 1, 2]).all():
            raise ValueError("invalid source-only gallery classes")
        sessions = stream[source]
        train = {s: (pcs[sessions == s], y[sessions == s]) for s in dict.fromkeys(sessions)}
        test = {}
        for i, name in enumerate(data["sessions"]):
            rows = data[f"{i}|idx"]
            if not np.array_equal(rows, np.flatnonzero(stream == name)) or not np.all(people[rows] == fold):
                raise ValueError(f"{fold}: recognition test rows differ from the original session")
            test[str(name)] = (data[f"{i}|pc"].astype(np.float64), None, rows)
    if not np.array_equal(np.sort(np.concatenate([v[2] for v in test.values()])), np.flatnonzero(people == fold)):
        raise ValueError(f"{fold}: recognition missed held rows")
    opened[fold] = dict(source_rows=len(source), held_rows=sum(len(v[2]) for v in test.values()),
                        filesha256=sha(path))
    return train, test


def bio_trace(bio, windows, rows):
    records = []
    for row in rows:
        candidate, ramp = bio.step(windows[row])
        records.append((candidate, ramp, bio.last[1] if bio.last else np.nan))
    return np.array(records)


def contrast_trace(ctx, pc, shadow, Gn, Q):
    scores = ctx.A.R2.scores_session(pc, shadow, Gn, Q)
    ramp, who = ctx.A.R3.rmap(scores, "lin05")
    return ramp, who, scores


def recognise(ctx, bridge, archived_route):
    from bio_gate import Bio, fold_params

    p = bridge["delivery"]
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as data:
        X, subject, visit = (data[k] for k in ("X", "subject", "session"))
    stream = np.char.add(np.char.add(subject, "|"), visit)
    people = np.array([v.rsplit("/", 1)[1] for v in subject])
    if not np.array_equal(p, ctx.Pperm) or not np.array_equal(np.arange(len(X)), bridge["index"]):
        raise ValueError("recognition order differs from the incumbent shadow")
    ctx.S.apply({q: np.arange(ctx.S.N[q]) for q in ctx.S.PR})
    OOF = ctx.A.shadow(ctx.A.mix(STACK, PROTO, ctx.A.FAMS)[0], PROTO)[1]
    with np.load(ROOT / "runs/replay_data/bio_model_psd_d64.npz", allow_pickle=False) as z:
        model = dict(z)
    feature_only = Bio.__new__(Bio)
    feature_only.win = np.hanning(120)
    features = np.stack([feature_only.feat(x) for x in X])
    stage = {"bio_ramp": np.full(len(X), np.nan), "bio_who": np.full(len(X), -1),
             "bio_score": np.full(len(X), np.nan)}
    for key in ("historical", "corrected"):
        stage[f"{key}|contrast_ramp"] = np.full(len(X), np.nan)
        stage[f"{key}|contrast_who"] = np.full(len(X), -1)
        stage[f"{key}|contrast_score"] = np.full((len(X), 9), np.nan)
    opened, count, prefix = {}, 0, 0
    RL = ctx.A.R3.L
    original = RL.fold

    def verified_fold(proto, fold):
        if proto != PROTO:
            raise ValueError("unregistered recognition protocol")
        return source_fold(Path(RL.FE) / f"{proto}-{fold}.npz", fold, stream, people, opened)

    RL.fold = verified_fold
    try:
        for fold in sorted(set(people)):
            test, names, gallery, Q, _, _ = ctx.A.R3.gate_model(PROTO, fold, OOF)
            if len(names) != 9 or fold in names:
                raise ValueError("fold-specific contrast gallery does not exclude the held person")
            with np.load(Path(RL.FE) / f"{PROTO}-{fold}.npz", allow_pickle=False) as z:
                train = z["idx_tr"]
            params = fold_params(model, features[train], stream[train], list(names), t0=10, tau=0.26)
            bio = Bio(params)
            for session, (pcs, _, index) in test.items():
                delivered = p[stream[p] == session]
                pc = pcs[np.searchsorted(index, delivered)]
                n = min(len(delivered), PREFIX)
                bio.reset_state()
                full = bio_trace(bio, X, delivered)
                bio.reset_state()
                first = bio_trace(bio, X, delivered[:n])
                other = p[stream[p] != session][:n]
                bio.reset_state()
                bio_trace(bio, X, other)
                bio.reset_state()
                if (not np.array_equal(full[:n], first, equal_nan=True)
                        or not np.array_equal(first, bio_trace(bio, X, delivered[:n]), equal_nan=True)):
                    raise ValueError("Bio recognition prefix/reset changed")
                stage["bio_who"][delivered] = full[:, 0].astype(int)
                stage["bio_ramp"][delivered] = full[:, 1]
                stage["bio_score"][delivered] = full[:, 2]
                for arm in ("historical", "corrected"):
                    sh = bridge[f"{arm}|shadow"][delivered]
                    contrast = contrast_trace(ctx, pc, sh, gallery, Q)
                    cut = contrast_trace(ctx, pc[:n], sh[:n], gallery, Q)
                    if any(not np.array_equal(a[:n], b, equal_nan=True) for a, b in zip(contrast, cut)):
                        raise ValueError("contrast identity ramp differs on a strict prefix")
                    for name, value in zip(("ramp", "who", "score"), contrast):
                        stage[f"{arm}|contrast_{name}"][delivered] = value
                    prefix += 1
                count += 1
    finally:
        RL.fold = original
    if count != 14 or len(opened) != 10:
        raise ValueError("recognition did not execute every original fold and session")
    for key in (k for k in stage if k.endswith("_ramp")):
        value = stage[key]
        if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError(f"invalid recognition ramp {key}")
    stats = {}
    for arm in ("historical", "corrected"):
        contrast = stage[f"{arm}|contrast_ramp"] > 0
        bio = stage["bio_ramp"] > 0
        routed = [{"session": str(s), "first": int(np.flatnonzero((contrast | bio)[p[stream[p] == s]])[0])}
                  for s in dict.fromkeys(stream[p]) if np.any((contrast | bio)[p[stream[p] == s]])]
        stats[arm] = dict(sessions=count, routed_sessions=len(routed), contrast_windows=int(contrast.sum()),
                          bio_windows=int(bio.sum()), routed=routed)
    if stats["historical"] != archived_route["bases"][STACK]:
        raise ValueError("live historical route4 ramps differ from the pinned archive")
    return stage, dict(routing=stats, gallery_folds=opened, source_calibration_sha256=sha(REFERENCE),
                       contrast_prefix_checks=prefix, bio_prefix_checks=count, bio_reset_checks=count)
