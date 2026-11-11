"""WSP: profile core1's weights (what transfers to unseen people) and score weight-space edits.

Reads the cache of wsp_build.py. Every edit of a fold's head uses only that fold's training people (through the inner
leave-one-more-person-out heads); the held-out rows are used for scoring only. An edited head is rescaled by one scalar
so that its logit size on inner held-out rows equals the unedited head's (its weight in the member mix stays the same).
Usage: python experiments/wsp_core.py CACHE_DIR [profile] [edits] [chain]
"""
import glob, os, sys
import numpy as np

CACHE = sys.argv[1]
C = np.load(f"{CACHE}/common.npz", allow_pickle=True)
F, Y, PER, SL = C["F"], C["y"], C["person"], C["slices"]
D, NB = F.shape[1], len(SL) - 1  # NB tangent blocks (band x {full, half 1, half 2}), then the slow block
IU = np.triu_indices(41)
BLOCK = np.concatenate([np.full(b - a, i) for i, (a, b) in enumerate(SL)])  # feature -> block id (slow = NB)
BAND, WIN = BLOCK // 3, BLOCK % 3
DIAG = np.concatenate([np.tile(IU[0] == IU[1], NB), np.zeros(D - NB * 861, bool)])
SLOW = BLOCK == NB
BANDS = C["bands"]
FOLDS = {os.path.basename(f)[:-4]: f for f in sorted(glob.glob(f"{CACHE}/*-*.npz"))}


def load(name):
    z = np.load(FOLDS[name], allow_pickle=True); o = dict(rows=z["rows"], train=z["train"], Fc=z["Fc"], proba=z["proba"])
    o["out"] = {k: z[f"o_{k}"] for k in ("mean", "scale", "coef", "b")}
    o["inner"] = {q: {k: z[f"i_{q}_{k}"] for k in ("mean", "scale", "coef", "b")} for q in z["inner_people"]}
    return o


def logits(m, X):
    return ((X - m["mean"]) / m["scale"]) @ m["coef"].T + m["b"]


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def contrib(m, X, y):
    """Per-feature mean contribution to the true-class margin (logit of y minus the mean of the other two)."""
    V = 1.5 * m["coef"] - 0.5 * m["coef"].sum(0)  # row k: w_k - mean of the other two
    out = np.zeros(D)
    for k in range(3):
        r = y == k
        out += (((X[r] - m["mean"]) / m["scale"]).sum(0) * V[k])
    return out / len(y)


def inner_rows(f, q):
    tr = f["train"]; return tr[PER[tr] == q], tr[PER[tr] != q]


def inner_stats(f):
    """-> held-out and in-sample per-feature margin contributions of the inner heads (mean over left-out people),
    the inner coefficient stack, and the RMS logit of the inner heads on their held-out rows."""
    ho, ins, coefs = np.zeros(D), np.zeros(D), []
    for q, m in f["inner"].items():
        te, tr = inner_rows(f, q)
        ho += contrib(m, F[te], Y[te]); ins += contrib(m, F[tr], Y[tr]); coefs.append(m["coef"] - m["coef"].mean(0))
    n = len(f["inner"]); return ho / n, ins / n, np.stack(coefs)


def rms_inner(f, edit):
    v = []
    for q, m in f["inner"].items():
        te, _ = inner_rows(f, q); z = logits(edit(m), F[te]); v.append(((z - z.mean(1, keepdims=True)) ** 2).mean())
    return np.sqrt(np.mean(v))


def by(v, key, n):
    return np.bincount(key, weights=v, minlength=n)


# ---------------------------------------------------------------- edits: fold stats -> function(model) -> model
def e_scale(phi):
    def f(m):
        W = m["coef"] - m["coef"].mean(0); return dict(m, coef=(W * phi).astype(np.float32))
    return f


def e_group(st, perm=None):
    ho, ins, _ = st; g = np.maximum(by(ho, BLOCK, NB + 1), 0) / np.maximum(by(ins, BLOCK, NB + 1), 1e-9)
    g = g / g.mean()
    if perm is not None:
        g = g[perm.permutation(len(g))]
    return e_scale(g[BLOCK])


def e_jack(st, perm=None):
    co = st[2]; n = len(co); mu = co.mean(0); v = (n - 1) / n * ((co - mu) ** 2).sum(0); phi = mu ** 2 / (mu ** 2 + v + 1e-12)
    if perm is not None:
        phi = np.stack([p[perm.permutation(D)] for p in phi])
    return e_scale(phi)


def e_smooth(rho=0.5):
    def f(m):
        W = (m["coef"] - m["coef"].mean(0)).copy(); T = W[:, :NB * 861].reshape(3, NB // 3, 3, 861)
        nb = np.concatenate([T[:, 1:2], (T[:, :-2] + T[:, 2:]) / 2, T[:, -2:-1]], 1)
        W[:, :NB * 861] = ((1 - rho) * T + rho * nb).reshape(3, -1); return dict(m, coef=W)
    return f


def e_tie(rho=0.5):
    def f(m):
        W = (m["coef"] - m["coef"].mean(0)).copy(); T = W[:, :NB * 861].reshape(3, NB // 3, 3, 861)
        h = T[:, :, 1:].mean(2, keepdims=True); T[:, :, 1:] = (1 - rho) * T[:, :, 1:] + rho * h; return dict(m, coef=W)
    return f


def e_prune(q):
    def f(m):
        W = m["coef"] - m["coef"].mean(0); return dict(m, coef=np.where(np.abs(W) >= np.quantile(np.abs(W), q), W, 0).astype(np.float32))
    return f


def e_soup(f_):
    ms = list(f_["inner"].values()); avg = {k: np.mean([m[k] for m in ms], 0) for k in ("mean", "scale", "coef", "b")}
    return lambda m: avg


def apply(f, edit):
    """Edited outer head, rescaled to the unedited logit size on inner held-out rows -> posteriors of the held-out rows."""
    c = rms_inner(f, lambda m: m) / rms_inner(f, edit); m = edit(f["out"])
    return softmax(c * (logits(m, f["Fc"]) - m["b"]) + m["b"] * c)


EDITS = {
    "E1_group": lambda f, st: e_group(st), "E1_perm": lambda f, st: e_group(st, np.random.default_rng(0)),
    "E2_jack": lambda f, st: e_jack(st), "E2_perm": lambda f, st: e_jack(st, np.random.default_rng(0)),
    "E3_smooth": lambda f, st: e_smooth(), "E4_tie": lambda f, st: e_tie(),
    "E6_prune50": lambda f, st: e_prune(0.5), "E6_prune90": lambda f, st: e_prune(0.9), "E8_soup": lambda f, st: e_soup(f),
}


def profile(names):
    tot = dict(w2=np.zeros(D), ho=np.zeros(D), ins=np.zeros(D), iho=np.zeros(D), unstable=0.0)
    for nm in names:
        f = load(nm); m = f["out"]; W = m["coef"] - m["coef"].mean(0); st = inner_stats(f)
        tot["w2"] += (W ** 2).sum(0); tot["ho"] += contrib(m, f["Fc"], Y[f["rows"]]); tot["ins"] += contrib(m, F[f["train"]], Y[f["train"]])
        tot["iho"] += st[0]; co = st[2]; mu = co.mean(0); sd = co.std(0) * np.sqrt(len(co) - 1)
        tot["unstable"] += ((W ** 2)[np.abs(mu) < sd]).sum() / (W ** 2).sum()
        print(f"  {nm}: held-out margin {tot_s(contrib(m, f['Fc'], Y[f['rows']])):.3f}, in-sample {tot_s(contrib(m, F[f['train']], Y[f['train']])):.3f}", flush=True)
    n = len(names); w2, ho, ins, iho = (tot[k] / n for k in ("w2", "ho", "ins", "iho"))
    print(f"held-out / in-sample margin {ho.sum() / ins.sum():.3f}; inner held-out / in-sample {iho.sum() / ins.sum():.3f}")
    print(f"unstable-sign weights (jackknife |mean| < sd): {tot['unstable'] / n:.3f} of weight energy")
    sh = lambda v, m: v[m].sum() / v.sum()  # noqa: E731
    print("group         weight-energy  held-out-margin  inner-held-out  in-sample")
    for lab, m in (("off-diagonal", ~DIAG & ~SLOW), ("diagonal", DIAG), ("slow block", SLOW), ("full window", (WIN == 0) & ~SLOW),
                   ("half 1", (WIN == 1) & ~SLOW), ("half 2", (WIN == 2) & ~SLOW)):
        print(f"{lab:13s} {sh(w2, m):13.3f} {sh(ho, m):16.3f} {sh(iho, m):15.3f} {sh(ins, m):10.3f}")
    for b, (lo, hi) in enumerate(BANDS):
        m = (BAND == b) & ~SLOW
        print(f"{lo:4.0f}-{hi:<4.0f} Hz  {sh(w2, m):13.3f} {sh(ho, m):16.3f} {sh(iho, m):15.3f} {sh(ins, m):10.3f}")
    np.savez(f"{CACHE}/profile.npz", w2=w2, ho=ho, ins=ins, iho=iho)


def tot_s(v):
    return float(v.sum())


def run_edits(names):
    out = {k: {} for k in ["base"] + list(EDITS)}
    for nm in names:
        f = load(nm); st = inner_stats(f); out["base"][nm] = (f["rows"], f["proba"])
        for k, mk in EDITS.items():
            out[k][nm] = (f["rows"], apply(f, mk(f, st)))
        print("edited", nm, flush=True)
    np.savez(f"{CACHE}/edits.npz", **{f"{k}|{nm}|{t}": v[i] for k, d in out.items() for nm, v in d.items() for i, t in enumerate(("rows", "proba"))})


if __name__ == "__main__":
    names = list(FOLDS); todo = sys.argv[2:] or ["profile", "edits"]
    if "profile" in todo:
        profile([n for n in names if n.startswith("all_loso")])
    if "edits" in todo:
        run_edits(names)
