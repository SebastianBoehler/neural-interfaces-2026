"""Replay the actual count and CSP consumers with isolated native session state."""

from contextlib import contextmanager
import hashlib
from time import perf_counter

import numpy as np

from complete_tail import count_filter, limit_session
from within0_pair_inputs import ORDER


ARMS = {"stock": (True, True), "no_ft": (False, True), "no_csp": (True, False), "both_off": (False, False)}


@contextmanager
def scopes(count, sessions):
    before = count.SESS
    count.SESS = tuple(sessions)
    try:
        yield
    finally:
        count.SESS = before


def personalize(final, windows, raw, filtered, rows):
    cp = final.CP.CausalPers(final.CONFIG)
    assert (cp.t0, cp.k, cp.w0) == (20, 20, 90)
    influence = hashlib.sha256()
    times, original = [], cp.refit

    def strict_fit(at):
        assert len(cp.C) == len(cp.P) == at+1
        expected = raw[rows[:at]]
        np.testing.assert_array_equal(np.asarray(cp.P[:at]), expected)
        class_id, confidence = expected.argmax(1), expected.max(1)
        select = cp.select
        calls = []

        def check_selection(y, c):
            np.testing.assert_array_equal(y, class_id)
            np.testing.assert_array_equal(c, confidence)
            take = select(y,c)
            assert len(take) and take.min() >= 0 and take.max() < at
            influence.update(np.asarray(rows[:at][take], dtype="<i8").tobytes())
            influence.update(np.asarray(expected[take], dtype="<f8").tobytes())
            calls.append(len(take))
            return take

        cp.select = check_selection
        try:
            result = original(at)
        finally:
            cp.select = select
        assert len(calls) == 1
        times.append(at)
        return result

    cp.refit = strict_fit
    out = np.stack([cp.step(windows[r],raw[r],filtered[r],False) for r in rows])
    assert np.isfinite(out).all() and cp.log["err"] == 0 and not cp.dead
    assert cp.stopped is None and cp.t == len(rows)
    assert times == list(range(20,len(rows),20))
    assert cp.refits == len(times) and cp.fused == max(0,len(rows)-90)
    np.testing.assert_array_equal(out[:90],filtered[rows[:90]])
    return out, dict(feats=cp.t,refits=cp.refits,fusions=cp.fused,single_class=cp.nfail,
                     target_sha=influence.hexdigest(),refit_ages=times)


def run_arm(ctx, ft, cp, sessions):
    count, final = ctx["count"], ctx["final"]
    raw = ctx["raw"][ft]
    with scopes(count,sessions):
        start = perf_counter()
        filtered, latch = count_filter(count,raw,ORDER)
        count_s = perf_counter()-start
        rows = [count.C.order_rows(ORDER,s,count.PROTO) for s in sessions]
    delivery = np.concatenate(rows)
    assert len(set(delivery)) == len(delivery)
    assert np.array_equal(delivery,ctx["delivery"][np.isin(ctx["delivery"],delivery)])
    assert np.isfinite(filtered[delivery]).all()
    active = int(np.count_nonzero(np.any(filtered[delivery] != raw[delivery],axis=1)))
    assert active > 0 and all(t is not None for t in latch.values())
    start = perf_counter()
    work, terminal = {}, filtered.copy()
    if cp:
        for session, row in zip(sessions,rows,strict=True):
            terminal[row], work[str(session)] = personalize(final,ctx["windows"],raw,filtered,row)
    csp_s = perf_counter()-start
    affected = int(np.count_nonzero(np.any(terminal[delivery] != filtered[delivery],axis=1)))
    assert (affected > 0) == cp
    if not cp:
        np.testing.assert_array_equal(terminal,filtered)
    return dict(raw=raw, filtered=filtered, terminal=terminal), dict(
        count_s=count_s, csp_s=csp_s, count_windows=len(delivery), count_effect=active,
        csp_effect=affected, csp_refits=sum(v["refits"] for v in work.values()),
        csp_fusions=sum(v["fusions"] for v in work.values()),
        csp_features=sum(v["feats"] for v in work.values()),
        latch={str(s):t for s,t in latch.items()}, work=work)


def exact_prefix(ctx, ft, cp, stage, session):
    count, final = ctx["count"], ctx["final"]
    with limit_session(count,session):
        filtered, _ = count_filter(count,ctx["raw"][ft],ORDER)
        rows = count.C.order_rows(ORDER,session,count.PROTO)
    assert len(rows) == 120
    np.testing.assert_array_equal(filtered[rows],stage["filtered"][rows])
    if cp:
        scores, proof = personalize(final,ctx["windows"],ctx["raw"][ft],filtered,rows)
        assert proof["refits"] == 5 and proof["fusions"] == 30
    else:
        scores = filtered[rows]
    np.testing.assert_array_equal(scores,stage["terminal"][rows])
    return int(np.count_nonzero(np.any(filtered[rows] != stage["raw"][rows],axis=1)))
