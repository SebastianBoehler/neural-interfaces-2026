"""s266 part (a): the package's label-free identity gate (experiments/bio_gate.Bio + fold_params, routing v4.1's bio stream: model
bio_model_psd_d64, tau 0.26, ramp 0.05, t0 10 as route4's bio_stream builds them) on the within-session proxy.
Own session of person q, session s: the gate sees the ONLINE rows of s in delivery order (acquisition rows play the offline public
session); gallery = the nine other people's public rows of the LOSO fold of q + q's acquisition rows of s as a one-session
entry (m = 1). False routes: every session of another person p (all rows, delivered in the order) against fold p's nine-person
gallery with q's full rows replaced by q's acquisition rows of the session s_q.
Not applied: the v3 half of v4.1's max (needs labelled gallery contrasts of the acquisition rows); the gate here is the bio half,
whose ramp is a lower bound on the merged r. Bio.step is called unchanged; only its window feature is served from a cache.
step() raw results are stored (top person, top score) so tau / ramp can be varied exactly as Bio does: r = clip((s - tau) / ramp, 0, 1).
Usage: [ORDERS=within0,..] [TAG=_smoke] python gate266.py -> runs/pview/gate266<TAG>.npz"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import data as D  # noqa: E402
sys.path.insert(0, D.REPO + "/experiments")
from bio_gate import Bio, fold_params  # noqa: E402
TAU, RAMP, T0 = 0.26, 0.05, 10  # the shipped v4.1 constants (route4: TAU default 0.26, fold_params ramp 0.05, t0 10)
FE = D.REPO + "/runs/replay_data/feats_exact"
MODEL = dict(np.load(D.REPO + "/runs/replay_data/bio_model_psd_d64.npz"))
_X = D.W["X"]; _B0 = Bio.__new__(Bio); _B0.win = np.hanning(120)
FB = np.stack([_B0.feat(x) for x in _X])
STR = np.char.add(np.char.add(D.PER, "|"), D.SES)
TASK = D.W["task"].astype(str); ONL = np.char.startswith(TASK, "Online")
SESSIONS = list(dict.fromkeys(D.SESS))  # 14 sessions; a session's person is D.PER of its rows
_G = {}


def gate_for(fold, q=None, acq=None):
    """Bio for a test session of person `fold` (fold p's LOSO training rows); q, acq: person whose acquisition rows `acq` form their
    one-session gallery entry (q == fold: own session; q != fold: q's full rows are replaced)."""
    key = (fold, q, None if acq is None else int(acq[0]))
    if key not in _G:
        itr = np.load(f"{FE}/all_loso-{fold}.npz")["idx_tr"].astype(int)
        if q is not None and q != fold:
            itr = itr[D.PER[itr] != q]
        names = list(dict.fromkeys(D.PER[itr])); feats, strm = FB[itr], STR[itr]
        if q is not None:
            feats = np.concatenate([feats, FB[acq]]); strm = np.concatenate([strm, np.full(len(acq), f"{q}|acq")]); names.append(q)
        _G[key] = (fold_params(MODEL, feats, strm, names, tau=TAU, ramp=RAMP, t0=T0), names)
    return _G[key]


def stream(par, gi):
    """-> top person index (n,), top score (n,), step's r at the shipped tau / ramp (n,); j = -1, NaN before t0."""
    b = Bio(par); b.feat = lambda g: FB[g]  # window feature from the cache (Bio.feat on the same X)
    j, s, r = np.full(len(gi), -1), np.full(len(gi), np.nan), np.zeros(len(gi))
    for t, g in enumerate(gi):
        _, r[t] = b.step(int(g))
        if b.last is not None:
            j[t], s[t] = b.last[0], b.last[1]
    return j, s, r


def acq_rows(s):
    return np.flatnonzero((D.SESS == s) & ~ONL)


def own_session(s, gi):
    """gate on the online rows `gi` (global indices, delivery order) of session s. -> j, score, r, names"""
    q = D.PER[gi[0]]; par, names = gate_for(q, q, acq_rows(s)); return (*stream(par, gi), names)


def cross_session(s, gi):
    """returning person with several public sessions (P04, P10): gallery = the nine others + ALL rows of q's other sessions as the
    package holds them (real cross-session recognition, no same-session gallery). -> j, score, r, names"""
    q = D.PER[gi[0]]; key = (q, "x", s)
    if key not in _G:
        itr = np.load(f"{FE}/all_loso-{q}.npz")["idx_tr"].astype(int); oth = np.flatnonzero((D.PER == q) & (D.SESS != s)); assert len(oth)
        names = list(dict.fromkeys(D.PER[itr])) + [q]
        _G[key] = (fold_params(MODEL, np.concatenate([FB[itr], FB[oth]]), np.concatenate([STR[itr], STR[oth]]), names, tau=TAU, ramp=RAMP, t0=T0), names)
    return (*stream(_G[key][0], gi), _G[key][1])


def false_routes(sq, sp, gi):
    """session sp of person p (rows gi in delivery order) against the gallery with q's acquisition rows of sq. -> j, score, r, names"""
    q, p = D.PER[acq_rows(sq)[0]], D.PER[gi[0]]; par, names = gate_for(p, q, acq_rows(sq)); return (*stream(par, gi), names)


if __name__ == "__main__":
    t0 = time.time(); out = {}
    PERMS = np.load(D.REPO + "/runs/shuffle_perms.npz")
    for order in os.environ.get("ORDERS", ",".join(o for o in D.ORDERS if o != "rec")).split(","):
        sc, sd = order.rstrip("0123456789"), order[len(order.rstrip("0123456789")):]
        idx = PERMS[f"{sc}|{sd}|all_loso"].astype(int)  # delivery order of the global rows
        for s in SESSIONS:
            gs = idx[(D.SESS[idx] == s)]; gon = gs[ONL[gs]]
            j, sco, r, names = own_session(s, gon); q = D.PER[gon[0]]
            out[f"own|{order}|{s}|j"], out[f"own|{order}|{s}|s"], out[f"own|{order}|{s}|r"] = j, sco, r
            out[f"own|{order}|{s}|names"] = np.array(names)
            k = np.flatnonzero(r > 0); print(f"{order} {s} ({q}): first routed online window {k[0] if len(k) else -1}, share {(r > 0).mean():.2f}, wrong-person {((j >= 0) & (np.array(names + ['x'])[j] != q)).sum()} [{time.time() - t0:.0f}s]", flush=True)
            if q in ("P04", "P10"):
                j, sco, r, names = cross_session(s, gon)
                out[f"cross|{order}|{s}|j"], out[f"cross|{order}|{s}|s"], out[f"cross|{order}|{s}|r"] = j, sco, r
            for sq in SESSIONS:
                if D.PER[acq_rows(sq)[0]] == q:
                    continue
                j, sco, r, names = false_routes(sq, s, gs)
                out[f"false|{order}|{sq}|{s}|j"], out[f"false|{order}|{sq}|{s}|s"], out[f"false|{order}|{sq}|{s}|r"] = j, sco, r
                out[f"false|{order}|{sq}|{s}|names"] = np.array(names)
    np.savez(D.OUT + "/gate266" + os.environ.get("TAG", "") + ".npz", **out); print("written", flush=True)
