"""Lane K test 4 step 2: compact band-power head with fixed vs individual-alpha-peak-relative bands, causal on the delivered prefix.
python laneK/alpha_head.py DATA.npz PERMS.npz OUT.npz [smoke]
Per held person h (LOPO over P01-P10, full0 delivery): head trained on the other nine people (their own whole-history peak offsets),
tested causally: the offset at delivered window n uses only windows < n (shrunk n/(n+20), clipped +-1 Hz, 0 if unreliable or n < 20).
Arms: fixed (head trained and applied at offset 0), gen (peak head, own prefix offset), don (peak head, offset of the next person
cyclically at the same delivered index n). Registered in evidence/wave_20nov/laneK_council_tests.md."""
import json
import sys

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression

from alpha_peak import peak

FS, F0_MARGIN, CLIP, N0 = 120, 1.0, 1.0, 20
EDGES = np.array([4, 7, 8, 10, 10, 13, 13, 20, 20, 30], float).reshape(-1, 2)
POST = 9


def prefix_offsets(lp, f0):
    """lp: delivered-order mean-over-posterior-channel log PSD (n x freq). Offset used at window n from windows < n."""
    n = len(lp)
    cs = np.cumsum(lp, 0)
    off = np.zeros(n)
    for t in range(N0, n, 1):
        pk, h = peak(cs[t - 1] / t, FREQ)
        if h >= 0.1:
            off[t] = np.clip((t / (t + N0)) * (pk - f0), -CLIP, CLIP)
    return off


def band_feats(psd, shift):
    """psd: n x ch x freq (linear power), shift: n (Hz). Mean power in each shifted band by linear interpolation of the cumulative PSD."""
    df = FREQ[1] - FREQ[0]
    cum = np.concatenate([np.zeros(psd.shape[:2] + (1,)), np.cumsum(psd, -1) * df], -1)  # integral up to the lower edge of bin i
    grid = np.concatenate([FREQ - df / 2, [FREQ[-1] + df / 2]])
    out = []
    for lo, hi in EDGES:
        def at(e):
            e = np.clip(e + shift, grid[0], grid[-1])
            i = np.clip(np.searchsorted(grid, e, side="right") - 1, 0, len(grid) - 2)
            w = (e - grid[i]) / df
            r = np.arange(len(e))
            return cum[r, :, i] * (1 - w)[:, None] + cum[r, :, i + 1] * w[:, None]
        out.append(np.log((at(hi) - at(lo)) / (hi - lo) + 1e-12))
    return np.concatenate(out, 1)  # n x (ch * bands)


def standardize(x, g_mean, g_var):
    n = np.arange(1, len(x) + 1)[:, None]
    cs, cq = np.cumsum(x, 0), np.cumsum(x ** 2, 0)
    m = (cs + N0 * g_mean) / (n + N0)
    v = (cq + N0 * (g_var + g_mean ** 2)) / (n + N0) - m ** 2
    return (x - m) / np.sqrt(np.maximum(v, 1e-6))


def main(data, perms, out, smoke=False):
    global FREQ
    z = np.load(data, allow_pickle=False)
    names = [str(c) for c in z["ch_names"]]
    post = [i for i, c in enumerate(names) if c.startswith(("P", "O"))]
    person = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]])
    y = z["y"]
    perm = np.load(perms)["full|0|all_loso"]
    people = [f"P{i:02}" for i in range(1, 11)]
    if smoke:
        people = people[:4]
    rows = {p: perm[person[perm] == p] for p in people}
    psd, lp = {}, {}
    for p in people:
        FREQ, pw = welch(z["X"][rows[p]].astype(np.float64), fs=FS, nperseg=240, noverlap=120, axis=-1)
        psd[p] = pw
        lp[p] = np.log10(pw[:, post].mean(1) + 1e-30)

    whole = {p: peak(lp[p].mean(0), FREQ)[0] for p in people}
    f0 = float(np.mean(list(whole.values())))
    pre = {p: prefix_offsets(lp[p], f0) for p in people}
    full_off = {p: np.clip(whole[p] - f0, -CLIP, CLIP) for p in people}
    feat0 = {p: band_feats(psd[p], np.zeros(len(rows[p]))) for p in people}
    # source features: fixed offset 0; peak: own whole-history offset (source people are not causal; their offset is a constant)
    featS = {p: band_feats(psd[p], np.full(len(rows[p]), full_off[p])) for p in people}
    featG = {p: band_feats(psd[p], pre[p]) for p in people}
    nxt = {p: people[(i + 1) % len(people)] for i, p in enumerate(people)}
    featD = {p: band_feats(psd[p], np.interp(np.arange(len(rows[p])), np.arange(len(pre[nxt[p]])), pre[nxt[p]])) for p in people}
    post_out = {k: np.zeros((len(y), 3)) for k in ("fixed", "gen", "don")}
    meta = dict(f0=f0, whole=whole, mean_abs_offset={p: float(np.abs(pre[p][N0:]).mean()) for p in people})
    for h in people:
        src = [p for p in people if p != h]
        for arm, tr, te in (("fixed", feat0, feat0[h]), ("peak", featS, None)):
            Xs = np.concatenate([tr[p] for p in src]); gm, gv = Xs.mean(0), Xs.var(0)
            Xz = np.concatenate([standardize(tr[p], gm, gv) for p in src])
            ys = np.concatenate([y[rows[p]] for p in src])
            clf = LogisticRegression(C=0.05, max_iter=300).fit(Xz, ys)
            if arm == "fixed":
                post_out["fixed"][rows[h]] = clf.predict_proba(standardize(te, gm, gv))
            else:
                post_out["gen"][rows[h]] = clf.predict_proba(standardize(featG[h], gm, gv))
                post_out["don"][rows[h]] = clf.predict_proba(standardize(featD[h], gm, gv))
        acc = {k: float((v[rows[h]].argmax(1) == y[rows[h]]).mean()) for k, v in post_out.items()}
        print(h, acc, flush=True)
        meta.setdefault("acc", {})[h] = acc
    np.savez(out, delivery=perm, **post_out)
    json.dump(meta, open(str(out).replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4], smoke=len(sys.argv) > 4)
