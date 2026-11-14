"""SOFTFP weights (ledger s251). Soft fingerprint neighbourhoods, nothing grouped: for window t of a session in delivery
order, every earlier window j < t gets the weight w_tj = P(same run | relative fingerprint distance), causal (only
windows <= t are read; nothing crosses sessions; no labels at use).
View: the s248 hardware fingerprint (runs/fingerprint/hw.npz via fingerprint/metric.py: log power 45-58 / 48-52 Hz,
channel correlations 30-58 Hz), z-scored with the OTHER nine people's mean / SD (a population constant).
Relative distance: r_tj = d2_tj / median_{j' < t} d2_tj' (scale-free in the session's own spread; the held-out person's
scale differs, s250 amendment). Calibration: P(same run | log r) in 40 quantile bins of log r, monotone (isotonic in the
bin rates), fitted per held-out person on the nine others' sessions under the full orders CAL_ORDERS with their TRUE runs
(labels used offline only, on other people).
weights(order, h) -> {session: (rows in delivery order, W (T, T) strictly lower-triangular)} for person h's sessions."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../fingerprint")
import metric as M  # noqa: E402
C = M.C
CAL_ORDERS = ("full4", "full5", "full6", "full7")  # calibration orders: disjoint from the scored full 0-3 (only permutations)
NB = 40
_F = M.features(); _CACHE = {}


def zview(h):
    m = C.PER != h; mu, sd = _F[m].mean(0), _F[m].std(0) + 1e-9; return (_F - mu) / sd


def reldist(Zs):
    """Zs (T, D) in delivery order -> R (T, T): r_tj for j < t (log), nan elsewhere."""
    sq = (Zs ** 2).sum(1); D = np.maximum(sq[:, None] + sq[None] - 2 * Zs @ Zs.T, 0); T = len(Zs); R = np.full((T, T), np.nan)
    for t in range(1, T):
        d = D[t, :t]; R[t, :t] = np.log(np.maximum(d, 1e-9) / max(np.median(d), 1e-9))
    return R


def calib(h):
    """-> (bin edges, monotone same-run rate per bin) from the nine others under CAL_ORDERS."""
    if ("cal", h) in _CACHE:
        return _CACHE["cal", h]
    Zv = zview(h); xs, ys = [], []
    for q in [q for q in C.PEOPLE if q != h]:
        for s in [s for s in C.SESSIONS if C.PER[np.flatnonzero(C.SES == s)[0]] == q]:
            for o in CAL_ORDERS:
                rows = C.order_rows(o, s); R = reldist(Zv[rows]); r = C.RUNI[rows]; ii, jj = np.tril_indices(len(rows), -1)
                xs.append(R[ii, jj]); ys.append(r[ii] == r[jj])
    x, y = np.concatenate(xs), np.concatenate(ys).astype(float); edges = np.quantile(x, np.linspace(0, 1, NB + 1)); edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(edges, x, "right") - 1, 0, NB - 1); rate = np.bincount(b, y, NB) / np.maximum(np.bincount(b, minlength=NB), 1)
    from sklearn.isotonic import IsotonicRegression
    rate = IsotonicRegression(increasing=False).fit_transform(np.arange(NB), rate, sample_weight=np.bincount(b, minlength=NB))
    _CACHE["cal", h] = (edges, rate); return edges, rate


def weights(order, h):
    key = (order, h)
    if key in _CACHE:
        return _CACHE[key]
    edges, rate = calib(h); Zv = zview(h); out = {}
    for s in [s for s in C.SESSIONS if C.PER[np.flatnonzero(C.SES == s)[0]] == h]:
        rows = C.order_rows(order, s); R = reldist(Zv[rows]); W = np.zeros_like(R); m = ~np.isnan(R)
        W[m] = rate[np.clip(np.searchsorted(edges, R[m], "right") - 1, 0, NB - 1)]; out[s] = (rows, W)
    _CACHE[key] = out; return out


def all_weights(order):
    """{session: (rows, W)} for every public session (each person with its own nested calibration)."""
    out = {}
    for h in C.PEOPLE:
        out.update(weights(order, h))
    return out


if __name__ == "__main__":  # diagnostic: calibration curve and how much true same-run mass the weights carry (full 0-3)
    for h in C.PEOPLE[:3]:
        e, r = calib(h); print(h, "same-run rate by log-r bin (first 8, last 3):", np.round(r[:8], 3), np.round(r[-3:], 3))
    for o in ("full0", "within0", "runs0"):
        a, b, n1 = [], [], []
        for s, (rows, W) in all_weights(o).items():
            r = C.RUNI[rows]; S = (r[:, None] == r[None]) & np.tri(len(rows), k=-1, dtype=bool)
            a.append((W * S).sum() / max(W.sum(), 1e-9)); b.append(S.sum() / np.tri(len(rows), k=-1).sum())
            Wn = np.where(np.tri(len(rows), k=-1, dtype=bool), W, -1); n1.append(np.mean([r[t] == r[Wn[t].argmax()] for t in range(1, len(rows))]))
        print(f"{o}: weighted same-run share {np.mean(a):.3f} (base {np.mean(b):.3f}); top-weight earlier window same run {np.mean(n1):.3f}")
