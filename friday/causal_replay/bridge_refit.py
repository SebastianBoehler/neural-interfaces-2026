"""Reconstruct source-independent EM/TSEM pseudo-labels at each strict refit prefix."""
import numpy as np

FIRST_REFIT, STRIDE = 20, 10
TOL = 1e-12


def norm(p):
    return p / p.sum(axis=1, keepdims=True)


def em_state(gm, view, rebalance, gate, t, actor):
    train, b = view[:t], rebalance[:t]
    m, sd = train.mean(0), np.maximum(train.std(0), gm.SE.SD_FLOOR)
    standard = (train - m) / sd
    q = b
    for _ in range(actor.iters):
        mu = (q / (q.sum(0, keepdims=True) + actor.m0)).T @ standard
        covariance = np.cov((standard - q @ mu).T)
        cov = (1 - actor.shrink) * covariance + actor.shrink * np.eye(len(covariance)) * np.trace(covariance) / len(covariance)
        inverse = np.linalg.inv(cov)
        q = norm(b * gm.SE._softmax_lda(standard, mu, inverse) ** actor.lam)
    exponent = actor.lam * t / (t + actor.n0) * (1 + actor.bem * (gm.M.h(t, 30) * gate[t]))
    likelihood = gm.SE._softmax_lda((view[:t + 1] - m) / sd, mu, inverse)
    return norm(rebalance[:t + 1] * likelihood ** exponent)


def tsem_state(gm, view, emitted, past_em, gate, t):
    train = view[:t]
    m, sd = train.mean(0), np.maximum(train.std(0), 1e-3)
    standard = (train - m) / sd
    labels, confidence = emitted[:t].argmax(1), emitted[:t].max(1)
    limit = max(2, int(0.7 * t / 3))
    seeds = []
    for k in range(3):
        rows = np.flatnonzero(labels == k)
        seeds.append(rows[np.argsort(-confidence[rows])[:limit]])
    if min(map(len, seeds)) < 2:
        return past_em, emitted[t], False
    mu = np.stack([standard[rows].sum(0) / (len(rows) + 20) for rows in seeds])
    residual = np.concatenate([standard[rows] - mu[k] for k, rows in enumerate(seeds)])
    covariance = np.cov(residual.T)
    cv = (1 - 0.7) * covariance + 0.7 * np.eye(len(covariance)) * np.trace(covariance) / len(covariance)
    likelihood = gm.L.G.T._lda((view[:t + 1] - m) / sd, mu, np.linalg.inv(cv))
    exponent = 0.15 * t / (t + 60) * (1 + gm.BTS * (gm.M.h(t, 30) * gate[t]))
    current = emitted[t] * likelihood[t] ** exponent
    return norm(past_em * likelihood[:t] ** exponent), current / current.sum(), True


def session_refits(gm, ge, pc, rebalance, gate, emitted_em, shadow, tgrid):
    actor = gm.EMg(gm.BEM, lam=0.5, n0=60, m0=20, shrink=0.7, nmin=gm.NMIN)
    if actor.nmin != FIRST_REFIT:
        raise ValueError("historical EM activation differs from the registered refit start")
    result = np.full((len(ge), len(tgrid), 3), np.nan)
    errors = []
    for column, t in enumerate(tgrid):
        if t >= len(ge):
            break
        em = em_state(gm, ge, rebalance, gate, t, actor)
        past, current, tsem_active = tsem_state(gm, pc, emitted_em, em[:t], gate, t)
        em_error = float(np.abs(em[t] - emitted_em[t]).max())
        ts_error = float(np.abs(current - shadow[t]).max())
        if not np.isfinite(past).all() or max(em_error, ts_error) > TOL:
            raise ValueError(f"reconstructed state differs from stock emission at age {t}")
        result[:t, column] = past
        errors.append((em_error, ts_error, tsem_active))
    return result, errors


def relabel(ctx, bridge):
    p, sess = bridge["delivery"], bridge["session"]
    session_rows = [p[sess[p] == s] for s in dict.fromkeys(sess[p])]
    tgrid = np.arange(FIRST_REFIT, max(map(len, session_rows)), STRIDE)
    if not len(tgrid) or len(session_rows) != 14:
        raise ValueError("unexpected reference grid or sessions")
    position = np.full(len(p), -1, dtype=int)
    for rows in session_rows:
        position[rows] = np.arange(len(rows))

    def for_rows(arm, rows):
        return (bridge["ge"][rows], bridge["pc128"][rows],
                *(bridge[f"{arm}|{key}"][rows] for key in ("rebal", "gate", "em", "shadow")))

    result = dict(index=bridge["index"], delivery=p, pos=position, tgrid=tgrid)
    stats = {}
    for arm in ("historical", "corrected"):
        relab = np.full((len(p), len(tgrid), 3), np.nan)
        errors = []
        for rows in session_rows:
            relab[rows], local = session_refits(ctx.gm, *for_rows(arm, rows), tgrid)
            errors.extend(local)
            for column, t in enumerate(tgrid):
                want = np.arange(len(rows)) < t if t < len(rows) else np.zeros(len(rows), bool)
                if not np.array_equal(np.isfinite(relab[rows, column]).all(1), want):
                    raise ValueError("refit includes current/future or leaves out a prior row")
        for rows in (session_rows[0], session_rows[-1], session_rows[0]):
            t = FIRST_REFIT + 3 * STRIDE
            crop = rows[:t + 1]
            again, _ = session_refits(ctx.gm, *for_rows(arm, crop), tgrid)
            active = tgrid <= t
            if not np.array_equal(again[:, active], relab[crop][:, active], equal_nan=True):
                raise ValueError("refit labels depend on future windows or another session")
        valid = relab[np.isfinite(relab).all(2)]
        if (valid < 0).any() or not np.allclose(valid.sum(1), 1, rtol=0, atol=TOL):
            raise ValueError("invalid strict-prefix pseudo-labels")
        result[f"{arm}|relab"] = relab
        result[f"{arm}|all_loso"] = bridge[f"{arm}|shadow"]
        stats[arm] = dict(refits=len(errors), tsem_active=sum(int(v[2]) for v in errors),
                          em_emission_max_abs=max(v[0] for v in errors), tsem_emission_max_abs=max(v[1] for v in errors),
                          past_rows=len(valid), refit_prefix_checks=3)
    both = np.isfinite(result["historical|relab"]).all(2) & np.isfinite(result["corrected|relab"]).all(2)
    delta = float(np.abs(result["historical|relab"][both] - result["corrected|relab"][both]).max())
    if delta <= 1e-5:
        raise ValueError("corrected incumbents did not propagate to the refit label bank")
    return result, dict(arms=stats, relab_delta_max_abs=delta)
