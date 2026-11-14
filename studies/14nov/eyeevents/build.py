"""s240: eye-event view (panel neuroscience proposal 4) as SessionEM view columns for shufcombo's EXTRA_VIEW hook.
Traces as studies/13nov/ocushuf.py traces() before its standardisation: h = SG(F5/AF3/F3 - F6/AF4/F4), v = SG(AF3/AF4 -
P1/P2/Pz), Savitzky-Golay (25, 3); velocities = gradient. Event scale m = running session median of the per-window median
|velocity| over windows delivered so far (current included), blended with the all-window prior as K windows (as
experiments/session_em.py EventRates), so each delivery order has its own rows. Columns (ev8):
  0,1 signed horizontal event balance per half (0-2 s, 2-4 s): share(dh > 4m) - share(dh < -4m)
  2,3 the same at 2m
  4   latency of the first blink (first dv > 8 m_v), seconds, censored at 4.0
  5   vertical level late minus early (mean v 2-4 s minus 0-2 s)
  6   vertical event share (|dv| > 4 m_v) late minus early
  7   horizontal level late minus early
Keys: 'rec' (recording order) and '<scheme><seed>' for ORDERS, rows in replay row order (= window index). No label is read.
Usage: python studies/14nov/eyeevents/build.py  -> runs/views_eye/ev8.npz"""
import os
import numpy as np
from scipy.signal import savgol_filter
REPO = os.path.abspath(__file__ + "/../../../..")
OUT = REPO + "/runs/views_eye"
ORDERS = [f"{s}{k}" for s in ("within", "full", "runs") for k in range(2)]
FS, HALF, K, QH, QL, QB, QV = 120, 240, 5, 4.0, 2.0, 8.0, 4.0


def traces(W):
    X = W["X"].astype(np.float64); ix = {str(c): i for i, c in enumerate(W["ch_names"])}
    sel = lambda cs: X[:, [ix[c] for c in cs]].mean(1)  # noqa: E731
    sg = lambda a: savgol_filter(a, 25, 3, axis=-1)  # noqa: E731
    h = sg(sel(("F5", "AF3", "F3")) - sel(("F6", "AF4", "F4"))); v = sg(sel(("AF3", "AF4")) - sel(("P1", "P2", "Pz")))
    return h, v, np.gradient(h, axis=-1), np.gradient(v, axis=-1)


def window_feats(h, v, dh, dv, mh, mv):
    e, l = slice(0, HALF), slice(HALF, None); f = np.empty(8)
    for j, q in enumerate((QH, QL)):
        for k, s in enumerate((e, l)):
            f[2 * j + k] = np.mean(dh[s] > q * mh) - np.mean(dh[s] < -q * mh)
    b = np.flatnonzero(dv > QB * mv); f[4] = b[0] / FS if len(b) else len(dv) / FS
    f[5] = v[l].mean() - v[e].mean(); f[6] = np.mean(np.abs(dv[l]) > QV * mv) - np.mean(np.abs(dv[e]) > QV * mv)
    f[7] = h[l].mean() - h[e].mean()
    return f


def view(order, h, v, dh, dv, sess, gh, gv):
    out = np.full((len(sess), 8), np.nan); acc = {}
    mdh, mdv = np.median(np.abs(dh), 1), np.median(np.abs(dv), 1)
    for i in order:
        n, sh, sv = acc.get(sess[i], (0, [], [])); sh.append(mdh[i]); sv.append(mdv[i]); acc[sess[i]] = (n + 1, sh, sv)
        mh = (K * gh + np.sum(sh)) / (K + len(sh)); mv = (K * gv + np.sum(sv)) / (K + len(sv))  # running mean of medians, as EventRates
        out[i] = window_feats(h[i], v[i], dh[i], dv[i], mh, mv)
    assert not np.isnan(out).any(); return out


def main():
    W = np.load(REPO + "/data/proteus_windows.npz"); h, v, dh, dv = traces(W)
    sess = np.char.add(W["subject"].astype(str), np.char.add("|", W["session"].astype(str))); n = len(sess)
    gh, gv = np.median(np.abs(dh)), np.median(np.abs(dv))
    pz = np.load(REPO + "/runs/shuffle_perms.npz"); O = {"rec": np.arange(n)}
    for k in ORDERS:
        s = k.rstrip("0123456789"); O[k] = pz[f"{s}|{k[len(s):]}|all_loso"].astype(int); assert sorted(O[k]) == list(range(n)), k
    os.makedirs(OUT, exist_ok=True); res = {k: view(o, h, v, dh, dv, sess, gh, gv).astype(np.float32) for k, o in O.items()}
    for k, a in res.items():
        print(f"{k:8s} mean " + " ".join(f"{x:+.3f}" for x in a.mean(0)) + " | sd " + " ".join(f"{x:.3f}" for x in a.std(0)), flush=True)
    np.savez(f"{OUT}/ev8.npz", **res, cols=np.array(["hbal4_e", "hbal4_l", "hbal2_e", "hbal2_l", "blink_lat", "v_le", "vev_le", "h_le"]))
    print(f"written {OUT}/ev8.npz keys {sorted(res)}")


if __name__ == "__main__":
    main()
