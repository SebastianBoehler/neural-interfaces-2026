"""WARM (ledger s64): personal head standardised with a public-session prior (k pseudo-windows) instead of cold.
Usage: python studies/12nov/warm.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
exec(compile(open(REPO + "/studies/12nov/v5_replay.py").read().split("\nres = []")[0], "v5_replay_head", "exec"))  # ns, A, R3, SC, staged, REC, NM, O
FOLD = {}; _gm2 = R3.gate_model


def gate_model2(proto, f, oof):
    out = _gm2(proto, f, oof); FOLD[(proto, f)] = (out[0], list(out[1]), out[4]); return out


R3.gate_model = gate_model2
real_bio = ns["bio_stream"]


def oracle_bio(proto, f, s, gi, names, t0):
    r, j, sc = np.zeros(len(gi)), np.full(len(gi), -1), np.full(len(gi), np.nan); hit = [i for i, q in enumerate(names) if q.endswith(O.PER[int(gi[0])])]
    if hit:
        r[:], j[:] = 1.0, hit[0]
    return r, j, sc


def warm_std(F, m0, v0, k):
    n = np.arange(1, len(F) + 1)[:, None]; m = (k * m0 + np.cumsum(F, 0)) / (k + n); e2 = (k * (v0 + m0 ** 2) + np.cumsum(F ** 2, 0)) / (k + n)
    return (F - m) / np.sqrt(np.maximum(e2 - m ** 2, 1e-6))


def set_heads(k):
    """k None: the shipped cold standardisation; else the warm one, per gallery person."""
    A._H.clear()
    if k is None:
        return
    for (proto, f), (te, names, per) in FOLD.items():
        hk = f"{proto}|{f}"; mom = {q: (np.concatenate([X[:, :256] for X, _ in per[q]]).mean(0), np.concatenate([X[:, :256] for X, _ in per[q]]).var(0)) for q in names}
        for s, (Xs, y, gi) in te.items():
            A._H[(proto, s)] = np.stack([A._HEADS[hk][q].predict_proba(warm_std(Xs[:, :256], *mom[q], k)) for q in names])


for bn in ("B2_pool", "B2_s345"):
    out = {}
    for gate, bio, mode in (("real", real_bio, "max"), ("oracle", oracle_bio, "bio")):
        for k in (None, 5, 20):
            ns["bio_stream"] = bio; set_heads(None if not FOLD else k); REC.clear(); o, _ = ns["system4"](bn, mode, 10)
            if k is not None and not A._H:
                set_heads(k); REC.clear(); o, _ = ns["system4"](bn, mode, 10)
            out[(gate, k)] = (o, staged(o)[0])
    ref = out[("real", None)][0]
    for (gate, k), (o, os_) in out.items():
        r, rs = SC.compare(o, ref), SC.compare(os_, ref)
        print(f"{bn} {gate:6s} gate, head {'cold' if k is None else f'warm k={k}':9s}: seen {r['seen']:+.2f} (P04 {r['ds'][0]:+.2f}, P10 {r['ds'][1]:+.2f}) | with stage {rs['seen']:+.2f} (P04 {rs['ds'][0]:+.2f}, P10 {rs['ds'][1]:+.2f}) | new {r['new']:+.3f}", flush=True)
ns["bio_stream"] = real_bio; A._H.clear()
