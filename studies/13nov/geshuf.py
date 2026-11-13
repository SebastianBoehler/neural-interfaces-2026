"""GESHUF (ledger s184): the event-rate half of SessionEM's GE view (experiments/session_em.py EventRates, a running
median per session) recomputed with each session's windows in a delivery order of runs/shuffle_perms.npz. The gaze half
(gaze_features) is per window and does not depend on order. The recording-order rebuild must equal the stored EV
(runs/diag/base_cache.npz) exactly. Saves runs/geshuf.npz: EV per order and protocol, keyed by global window index
(rec, <scheme><seed>), all 2,655 windows.
Usage: [ORDERS=within:0,...] python studies/13nov/geshuf.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/experiments", REPO + "/replay"]
import protocol  # noqa: E402
import session_em as SE  # noqa: E402
ORDERS = os.environ.get("ORDERS", "within:0,within:1,within:2,within:3,full:0,full:1,runs:0,runs:1").split(",")
d = protocol.load(); T = SE.event_traces(d["X"], d["ch_names"]); g = SE.global_mad(d["X"], d["ch_names"]); st = d["stream"]
PERMS = np.load(REPO + "/runs/shuffle_perms.npz"); EV0 = np.load(REPO + "/runs/diag/base_cache.npz")["EV"]


def ev(order):
    """EventRates per session with the session's windows in the order given (global indices)."""
    out = np.full((len(st), 16), np.nan)
    for s in dict.fromkeys(st[order]):
        er = SE.EventRates(g)
        for r in order[st[order] == s]:
            out[r] = er.step(T[:, r])
    return out


rec = ev(np.arange(len(st))); assert np.allclose(rec, EV0), f"recording-order rebuild differs from the stored EV ({np.abs(rec - EV0).max()})"
OUT = {"rec": rec}
for o in ORDERS:
    sc, sd = o.split(":")
    for p in ("all_loso", "seen_session"):
        order = PERMS[f"{sc}|{sd}|{p}"]; e = ev(order); OUT[f"{sc}{sd}|{p}"] = e
        m = ~np.isnan(e[:, 0]); print(f"{sc}{sd} {p}: {m.sum()} windows, mean |change| against recording order {np.abs(e[m] - rec[m]).mean():.4f} (sd of EV {rec.std():.3f})", flush=True)
np.savez_compressed(REPO + "/runs/geshuf.npz", **OUT); print("saved runs/geshuf.npz")
