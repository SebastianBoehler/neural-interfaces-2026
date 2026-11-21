"""Live mix6 replay and taps from the exact nine25 solver."""
from dataclasses import dataclass

import numpy as np
import torch

from blend import round3_log_post as log_post, round3_mix as mix
from settings import D


@dataclass(frozen=True)
class RebalancedMix:
    weight: float
    start: int


def rebalance(settings, raw, start=1):
    rb = type(settings)({"rb_alpha": settings.alpha, "rb_k": settings.k, "rb_w": settings.W})
    out = np.stack([rb.step(p) for p in raw])
    out[:start - 1] = raw[:start - 1]
    return out


def predict(model, window):
    label = model.predict(window)
    if label.shape != (1,) or label.dtype != torch.int64 or int(label[0]) not in (0, 1, 2):
        raise ValueError(f"invalid terminal label {label}")
    if model.dead or model.log6["err"] or model.routeB.log["err"] or model.cp.dead or model.cf.dead:
        raise RuntimeError(f"degraded solver at {model.log6}")
    if model.cp.log["err"] or model.cf.log["err"]:
        raise RuntimeError("terminal stage failed")
    return int(label[0])


def replay(template, windows, *, raw=None, variant=None, hard=False, capture=False, endpoint=False):
    model = D.official()[1].fresh_model(template)
    if template.team_b.windows or model.team_b.windows:
        raise RuntimeError("Team B state crossed a session boundary")
    if hard and raw is not None:
        raise ValueError("hard-only replay does not consume Team B tapes")
    pre_em, terminal, team = [], [], []
    if hard:
        model.team_b = None
    else:
        encoder = model.team_b.posterior
        def posterior(window):
            q = encoder(window) if raw is None else torch.from_numpy(raw[len(team)])
            team.append(q.cpu().numpy().copy())
            return q

        if raw is not None or capture:
            model.team_b.posterior = posterior
        if variant is not None:
            if raw is None:
                raise ValueError("grid replay requires the paired Team B tape")
            balanced = rebalance(model.routeB.tail.rb, raw, variant.start) if variant.start else raw
            original_hard = super(type(model), model).mix6

            def mixer(p, members, p_ft):
                m = original_hard(p, members, p_ft)
                if not members:
                    return m
                # Use the current delivery exactly once, just as the native mix6 does.
                q = balanced[len(team)]
                team.append(raw[len(team)])
                return mix(m, q, variant.weight)

            model.mix6 = mixer

    if capture:
        em = model.routeB.tail.em
        original_em, original_cp = em.step, model.cp.step

        def tap_em(x, p):
            pre_em.append(p.copy())
            return original_em(x, p)

        def tap_cp(*args):
            count = model.cp.fused
            out = original_cp(*args)
            terminal.append(out.copy() if count != model.cp.fused else log_post(out))
            return out

        em.step, model.cp.step = tap_em, tap_cp

    labels = np.fromiter((predict(model, x[None]) for x in windows), dtype=np.int8, count=len(windows))
    if model.log6["v6"] != len(windows):
        raise ValueError("mix6 path did not process the entire stream")
    if not hard:
        if capture or variant is not None or raw is not None:
            assert len(team) == len(windows)
        if raw is None and torch.device(model.team_b.device).type == "cuda":
            graphs = model.team_b.feature_graphs
            assert not any(g.failed for g in graphs.values())
            assert graphs[1].replays > 0 and (len(windows) < 64 or graphs[64].replays > 0)
    if capture:
        del model.routeB.tail.em.step, model.cp.step
    if not hard and (raw is not None or capture):
        del model.team_b.posterior
    if variant is not None:
        del model.mix6
    result = {"labels": labels}
    if capture:
        for key, taps in (("pre_em", pre_em), ("terminal", terminal)):
            result[key] = np.asarray(taps)
            assert result[key].shape == (len(windows), 3) and np.isfinite(result[key]).all()
        np.testing.assert_array_equal(result["terminal"].argmax(1), labels)
        if not hard:
            result["team"] = np.asarray(team)
            assert result["team"].shape == (len(windows), 3) and np.isfinite(result["team"]).all()
            np.testing.assert_allclose(result["team"].sum(1), 1.0, rtol=0, atol=1e-6)
            result["team_rb"] = rebalance(model.routeB.tail.rb, result["team"])
    if endpoint:
        assert not hard and raw is None and variant is None
        return result, model
    return result
