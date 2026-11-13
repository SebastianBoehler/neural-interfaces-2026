"""SEQ2 (ledger s141): the learned sequence stage of s70 in a fold-consistent basis. s70 pooled each person's 128
components from that person's own fold (52-54 of 128 components flip sign between folds), so the stage learned on
nine people saw coordinates that meant something else for the tenth (review 13 Nov). Here, for held-out person q,
every row (the nine training people's and q's) is expressed in fold q's basis (feats_exact all_loso-q: pc_tr for the
training rows, <s>|pc for q's sessions). Model, training and scoring as s70 (scores in percent; s70's pp() printed
100x percent). Usage: [STEPS=400] [D=24] [SMOKE=1] python studies/13nov/seqstage2.py"""
import glob, os, sys
import numpy as np
import torch
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z; p = "all_loso"; torch.set_num_threads(int(os.environ.get("THREADS", 4)))
STEPS = int(os.environ.get("STEPS", 400)); D = int(os.environ.get("D", 24)); SMOKE = os.environ.get("SMOKE") == "1"; SEEDS = (0,) if SMOKE else (0, 1, 2)
sess, per, y = Z[f"sess|{p}"], SC.B[f"person|{p}"], SC.B[f"y|{p}"].astype(int); N = len(y); assert (Z[f"idx|{p}"] == np.arange(N)).all()
GE = Z[f"GE|{p}"].astype(np.float64); rng = np.random.default_rng(0); Y = torch.tensor(y)
ROWS = {s: np.flatnonzero(sess == s) for s in np.unique(sess)}; assert all((np.diff(r) == 1).all() for r in ROWS.values())
print(f"SEQ2: steps {STEPS}, d {D}, seeds {SEEDS}; basis per held-out fold; scored against A.system(routing=False) on new people", flush=True)


def causal_z(X, m0=10.0):  # each row standardised by the earlier rows of its session (pseudo-counts: zero mean, unit sd)
    n = np.arange(len(X))[:, None]; cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)[:-1]]); c2 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X ** 2, 0)[:-1]])
    mu = cs / (n + m0); var = (c2 + m0) / (n + m0) - mu ** 2; return (X - mu) / np.sqrt(np.maximum(var, 1e-4))


def fold_features(q):
    """(N, 128 + GE) features of every row in fold q's basis, causally standardised per session"""
    z = np.load(f"{REPO}/runs/replay_data/feats_exact/all_loso-{q}.npz", allow_pickle=True); X = np.full((N, 128), np.nan)
    X[z["idx_tr"].astype(int)] = z["pc_tr"][:, :128]
    for k in [k for k in z.files if k.endswith("|pc")]:
        X[z[k.replace("|pc", "|idx")].astype(int)] = z[k][:, :128]
    assert not np.isnan(X).any(), f"fold {q}: a row has no coordinates"
    assert set(np.unique(per[z["idx_tr"].astype(int)])) == set(SC.PEOPLE) - {q}, f"fold {q}: training rows include the held-out person"
    G = np.c_[X, GE]; G = np.nan_to_num(G / np.maximum(G.std(0), 1e-9)); XS = np.zeros_like(G)
    for r in ROWS.values():
        XS[r] = causal_z(G[r])
    return torch.tensor(np.clip(XS, -6, 6), dtype=torch.float32)


class Stage(torch.nn.Module):  # s70's model, unchanged
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


def fit_predict(XS, LPnp, target, seed):
    torch.manual_seed(seed); r_ = np.random.default_rng(seed); LP = torch.tensor(LPnp, dtype=torch.float32); m = Stage(XS.shape[1], D)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2); train = [s for s in ROWS if per[ROWS[s][0]] != target]
    for step in range(STEPS):
        r = ROWS[train[r_.integers(len(train))]]; r = r[np.sort(r_.choice(len(r), int(len(r) * r_.uniform(0.7, 1.0)), replace=False))]
        out = m(XS[r], LP[r]); loss = torch.nn.functional.nll_loss(out[5:], Y[r][5:]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return {s: m(XS[ROWS[s]], LP[ROWS[s]]).numpy() for s in ROWS if per[ROWS[s][0]] == target}


def boot(d):
    b = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]; return f"{d.mean():+.2f} [{np.percentile(b, 10):+.2f}, {np.percentile(b, 90):+.2f}]"


FEAT = {q: fold_features(q) for q in (SC.PEOPLE[:1] if SMOKE else SC.PEOPLE)}
for bn in env.EXPL[:1] if SMOKE else env.EXPL:
    mixP, _ = A.mix(bn, p, A.FAMS); chain = A.system(bn, A.FAMS, routing=False)["out"][p]; ref = SC.person(chain, p); rm = SC.person(mixP, p)
    print(f"{bn}: mix {rm.mean():.2f} | hand-built stages (grid chain, no routing) {ref.mean():.2f} (P04 {ref[3]:.1f})", flush=True)
    for tag, src in (("learned stage on the mix", mixP), ("learned stage on the chain's output", chain)):
        LPn = np.log(np.clip(src, 1e-6, 1)); out = src.copy()
        for q in FEAT:
            got = [fit_predict(FEAT[q], LPn, q, seed) for seed in SEEDS]
            for s in got[0]:
                out[ROWS[s]] = np.mean([np.exp(g[s]) for g in got], 0)
        r = SC.person(out, p); d = r - ref; prod = np.log(np.clip(out, 1e-9, 1)) + np.log(np.clip(chain, 1e-9, 1)); d2 = SC.person(prod - prod.max(1, keepdims=True), p) - ref
        print(f"  {tag}: new {r.mean():.2f}, against the chain {boot(d)} | P04 {d[3]:+.1f} | people >= 0: {(d >= 0).sum()}/10 | " + " ".join(f"{v:+.1f}" for v in d))
        print(f"     product with the chain: {boot(d2)} | P04 {d2[3]:+.1f} | people >= 0: {(d2 >= 0).sum()}/10", flush=True)
