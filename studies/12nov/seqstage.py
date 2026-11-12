"""SEQ (ledger s70): a learned sequence stage. For each window, attention over the session's earlier windows in a
learned projection of the stage's views (128 components + 48 gaze/event features) pools their class posteriors; the
output combines the window's own log posterior, the pooled context and a class-share correction. Two rounds.
Trained on the other nine people with true labels. Usage: python studies/12nov/seqstage.py [steps=400] [d=24]"""
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z; p = "all_loso"; torch.set_num_threads(4)
kw = dict(a.split("=") for a in sys.argv[1:]); STEPS = int(kw.get("steps", 400)); D = int(kw.get("d", 24)); SEEDS = (0, 1, 2)
sess, per, y = Z[f"sess|{p}"], SC.B[f"person|{p}"], SC.B[f"y|{p}"].astype(int); assert (Z[f"idx|{p}"] == np.arange(len(y))).all()
X0 = np.concatenate([Z[f"pc128|{p}"], Z[f"GE|{p}"]], 1).astype(np.float64); rng = np.random.default_rng(0)


def causal_z(X, m0=10.0):  # each row standardised by the earlier rows of its session (pseudo-counts: zero mean, unit sd)
    n = np.arange(len(X))[:, None]; cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)[:-1]]); c2 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X ** 2, 0)[:-1]])
    mu = cs / (n + m0); var = (c2 + m0) / (n + m0) - mu ** 2; return (X - mu) / np.sqrt(np.maximum(var, 1e-4))


G = np.nan_to_num(X0 / np.maximum(X0.std(0), 1e-9)); XS = np.zeros_like(G); ROWS = {}
for s in np.unique(sess):
    r = np.flatnonzero(sess == s); assert (np.diff(r) == 1).all(); ROWS[s] = r; XS[r] = causal_z(G[r])
XS = torch.tensor(np.clip(XS, -6, 6), dtype=torch.float32); Y = torch.tensor(y)


class Stage(torch.nn.Module):
    def __init__(s, F, d):
        super().__init__(); s.W = torch.nn.Linear(F, d, bias=False); s.t = torch.nn.Parameter(torch.tensor([2.0, 2.0])); s.g = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        s.a = torch.nn.Parameter(torch.tensor([1.0, 1.0])); s.b = torch.nn.Parameter(torch.tensor([0.5, 0.5])); s.k = torch.nn.Parameter(torch.tensor([0.5, 0.5])); s.n0 = torch.nn.Parameter(torch.tensor(3.0))

    def forward(s, X, LP):
        n = len(X); f = torch.nn.functional.normalize(s.W(X), dim=1); S = f @ f.T; past = torch.tril(torch.ones(n, n, dtype=torch.bool), -1)
        tt = torch.arange(n, dtype=torch.float32); g = (tt / (tt + torch.exp(s.n0)))[:, None]; L = LP
        for i in range(2):
            V = torch.softmax(torch.exp(s.g[i]) * L, 1); att = torch.softmax((torch.exp(s.t[i]) * S).masked_fill(~past, -1e9), 1) * past.any(1, keepdim=True)
            ctx = att @ V; run = (torch.cumsum(V, 0) - V + 1.0) / (tt[:, None] + 3.0)
            L = s.a[i] * LP + s.b[i] * g * torch.log(ctx + 1e-3) - s.k[i] * g * torch.log(run)
            L = L - torch.logsumexp(L, 1, keepdim=True)
        return L


def fit_predict(LPnp, target, seed):
    torch.manual_seed(seed); r_ = np.random.default_rng(seed); LP = torch.tensor(LPnp, dtype=torch.float32); m = Stage(XS.shape[1], D)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2); train = [s for s in ROWS if per[ROWS[s][0]] != target]
    for step in range(STEPS):
        r = ROWS[train[r_.integers(len(train))]]; r = r[np.sort(r_.choice(len(r), int(len(r) * r_.uniform(0.7, 1.0)), replace=False))]
        out = m(XS[r], LP[r]); loss = torch.nn.functional.nll_loss(out[5:], Y[r][5:]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return {s: m(XS[ROWS[s]], LP[ROWS[s]]).numpy() for s in ROWS if per[ROWS[s][0]] == target}


def boot(d):
    b = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]; return f"{d.mean():+.2f} [{np.percentile(b, 10):+.2f}, {np.percentile(b, 90):+.2f}]"


pp = lambda P: 100 * np.asarray(SC.person(P, p))  # noqa: E731
for bn in env.EXPL:
    mixP, _ = A.mix(bn, p, A.FAMS); chain = A.system(bn, A.FAMS, routing=False)["out"][p]; ref = pp(chain); rm = pp(mixP)
    print(f"{bn}: mix {rm.mean():.2f} | hand-built stages {ref.mean():.2f} (P04 {ref[3]:.1f}) | steps {STEPS}, d {D}", flush=True)
    for tag, src in (("learned stage on the mix", mixP), ("learned stage on the chain's output", chain)):
        LPn = np.log(np.clip(src, 1e-6, 1)); out = np.zeros((len(y), 3))
        for q in SC.PEOPLE:
            for seed in SEEDS:
                for s, L in fit_predict(LPn, q, seed).items():
                    out[ROWS[s]] += np.exp(L) / len(SEEDS)
        r = pp(out); d = r - ref; prod = np.log(np.clip(out, 1e-9, 1)) + np.log(np.clip(chain, 1e-9, 1)); d2 = pp(prod) - ref
        print(f"  {tag}: new {r.mean():.2f}, against the chain {boot(d)} | P04 {d[3]:+.1f} | people >= 0: {(d >= 0).sum()}/10 | " + " ".join(f"{v:+.1f}" for v in d))
        print(f"     product with the chain: {boot(d2)} | P04 {d2[3]:+.1f} | people >= 0: {(d2 >= 0).sum()}/10", flush=True)
