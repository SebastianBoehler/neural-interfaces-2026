"""Call the original qualified count filter and CausalPers on the live posteriors."""
from contextlib import contextmanager
import sys

import numpy as np

from bridge_owner import ORDER, ROOT, replace_once, start_module

PREFIX = 120


def load_personalization():
    source = ROOT / "studies/14nov/newmember/s281_cp.py"
    final = start_module("bridge_s281_cp", source, source.read_text())
    final.verify_package_config()
    return final


def load_count(current):
    current.S.apply({p: np.arange(current.S.N[p]) for p in current.S.PR})
    folder = ROOT / "studies/14nov/rundetect"
    sys.path.insert(0, str(folder))
    source = folder / "common.py"
    text = replace_once(source.read_text(), 'SES, PER, Y = Z[f"sess|{p0}"], Z[f"person|{p0}"], Z[f"y|{p0}"].astype(int)',
                        'SES, PER = Z[f"sess|{p0}"], Z[f"person|{p0}"]')
    if not (ROOT / "runs/rundetect/proj_d6_r1.npz").is_file():
        raise ValueError("pinned source-excluded projection is unavailable")
    start_module("common", source, text)
    source = folder / "package.py"
    original = next(line for line in source.read_text().splitlines() if line.startswith('assert (C.A.L.Z[f"y|'))
    text = replace_once(source.read_text(), original, "assert np.array_equal(ROWS, C.IDX)")
    return start_module("bridge_package_count", source, text)


def count_filter(count, raw):
    latched = count.latch(ORDER)
    hc = count.J.hyp_cache(ORDER, "wprior", count.X, count.LAM, sessions=count.SESS, proto=count.PROTO)
    return count.gate(raw, count.J.filt_joint(raw, hc, "urn"), ORDER, latched), latched


@contextmanager
def limit_session(count, session):
    original_sessions, original_order = count.SESS, count.C.order_rows
    count.SESS = (session,)
    count.C.order_rows = lambda *args, **kwargs: original_order(*args, **kwargs)[:PREFIX]
    try:
        yield
    finally:
        count.SESS, count.C.order_rows = original_sessions, original_order


def live_tail(count, final, raw, windows, delivery, verify_prefix=True):
    filtered, latch = count_filter(count, raw)
    sessions = [count.C.order_rows(ORDER, s, count.PROTO) for s in count.SESS]
    if not np.array_equal(np.concatenate(sessions), delivery):
        raise ValueError("terminal stage did not see the same pinned delivery")
    terminal = final.apply_sessions(windows, raw, filtered, sessions)
    filter_rows = int(np.count_nonzero(np.any(filtered != raw, axis=1)))
    personalized = int(np.count_nonzero(np.any(terminal != filtered, axis=1)))
    if not filter_rows or not personalized or not any(w is not None for w in latch.values()):
        raise ValueError("the archived count/CSP stages did not activate")
    prefix_count = 0
    if verify_prefix:
        first, last = count.SESS[0], count.SESS[-1]
        for session in (first, last, first):
            with limit_session(count, session):
                prefix, _ = count_filter(count, raw)
                rows = count.C.order_rows(ORDER, session, count.PROTO)
            actor = final.CP.CausalPers(final.CONFIG)
            scores = np.stack([actor.step(windows[r], raw[r], prefix[r], False) for r in rows])
            if (len(rows) != PREFIX or not actor.refits or not actor.fused or actor.log["err"]
                    or not np.array_equal(prefix[rows], filtered[rows]) or not np.array_equal(scores, terminal[rows])):
                raise ValueError("truncated/interleaved count or CSP prefix differs")
            prefix_count += 1
    return dict(raw=raw, filtered=filtered, terminal=terminal), dict(
        latch={str(s): t for s, t in latch.items()}, filter_affected_rows=filter_rows,
        personalization_affected_rows=personalized, exact_interleaved_prefix_checks=prefix_count)
