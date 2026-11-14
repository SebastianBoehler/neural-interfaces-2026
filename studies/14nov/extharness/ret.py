"""External harness v2 (ledger s227): the returning path on multi-session corpora, leave-one-person-out.

Per held-out person p (population member, views, stages as harness.py; stages = kstages.chain, K-generic, parity-tested):
  personal heads   logistic regression (C 0.1, as rlib3.ps1_heads) on the session-1 PCs (stdz over the session) of p and
                   of 9 gallery people drawn from the other people (seeded); at test the PCs are causally standardised
                   (replay/route.causal_std);
  identity gate    bio_id.TwoCov (psd view) fitted on the other people's sessions; gallery = the 10 people's session-1
                   signatures; probe = running mean of the delivered windows; from window 10 on, the top gallery person
                   is routed when its score is >= tau (99th percentile of impostor maxima at n 20 among the other
                   people); a wrong top person routes the wrong head, as in the package;
  blend            kstages.slot_blend: mix = (1 - s r) member + s r head, then the chain (rebal, SessionEM, TSEM);
  LORO             on session 1: head trained on the other runs, blended with the population member on the held-out
                   run (member level, r 1), balanced accuracy per s; advantage = head alone - member.
Test sessions 2.. in delivery orders rec, full x NSEED, within x NSEED.
Usage: [PEOPLE=n] [NSEED=2] [NJ=6] [TAG=] python ret.py CORPUS -> runs/extharness/ret_<CORPUS><TAG>.pkl"""
import os, sys, time, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import corpus as CO  # noqa: E402
import kstages as KS  # noqa: E402
sys.path.insert(0, CO.REPO + "/experiments")
from bio_id import TwoCov  # noqa: E402
import route as RT  # noqa: E402  (replay/route.py: causal_std)
from harness import order, cell_ba  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

S_GRID = (0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0)
C_POP, C_HEAD, K0, NPC, NGE, T0, GAL, NHEAD = 0.05, 0.1, 5.0, 128, 48, 10, 10, 60
NSEED = int(os.environ.get("NSEED", 2))
ORDERS = [("rec", 0)] + [(sc, s) for sc in ("full", "within") for s in range(NSEED)]
D = {}


def stdz(X):
    return (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)


def causal_std(F):  # = replay/route.causal_std, vectorised (asserted once in main)
    n = np.arange(1, len(F) + 1)[:, None]; m = np.cumsum(F, 0) / n; v = np.maximum(np.cumsum(F ** 2, 0) / n - m ** 2, 0)
    out = (F - m) / np.maximum(np.sqrt(v), 1e-3); out[:4] = (F - m)[:4]; return out


def sess_order(labels):
    u = list(dict.fromkeys(labels))
    try:
        return sorted(u, key=lambda s: int(s))
    except ValueError:
        return sorted(u)


def ba(P, y):
    return cell_ba(y, P.argmax(1))


def fold(pi):
    t0 = time.time(); per = D["people"][pi]; V, Vc, y, B = D["V"], D["Vc"], D["y"], D["B"]
    person, session, run = D["person"], D["session"], D["run"]; K = int(y.max()) + 1
    tr = person != per; mu, sd = Vc[tr].mean(0), Vc[tr].std(0) + 1e-6; Ztr = (Vc[tr] - mu) / sd
    clf = LogisticRegression(C=C_POP, max_iter=500, tol=1e-4).fit(Ztr, y[tr])
    pca = PCA(NPC, svd_solver="randomized", random_state=0).fit(Ztr); del Ztr
    rng = np.random.default_rng([pi, 77]); others = [q for q in D["people"] if q != per]
    gal = [per] + list(rng.choice(others, min(GAL, len(others) + 1) - 1, replace=False))  # Scherer: 9 people

    def s1rows(q):
        r = np.flatnonzero(person == q); return r[session[r] == sess_order(session[r])[0]]
    heads = {}
    for q in gal:  # session-1 heads (training-side features: whole-session centred, stdz over the session)
        r = s1rows(q); Hq = stdz(pca.transform((Vc[r] - mu) / sd))
        heads[q] = LogisticRegression(C=C_HEAD, max_iter=3000).fit(Hq, y[r])
    # LORO on p's session 1
    r1 = s1rows(per); Z1 = (Vc[r1] - mu) / sd; P1 = clf.predict_proba(Z1); H1 = stdz(pca.transform(Z1)); y1 = y[r1]
    ru = run[r1]
    if len(np.unique(ru)) < 2:
        ru = np.array([str(i * 4 // len(r1)) for i in range(len(r1))])  # no run ids: 4 contiguous blocks
    cv, adv = np.zeros(len(S_GRID)), []
    us = [u for u in dict.fromkeys(ru) if len(np.unique(y1[ru != u])) == K]
    for u in us:
        m = ru == u; h = LogisticRegression(C=C_HEAD, max_iter=3000).fit(H1[~m], y1[~m]); Ph = h.predict_proba(H1[m])
        cv += np.array([ba((1 - s) * P1[m] + s * Ph, y1[m]) for s in S_GRID]) / len(us); adv.append(ba(Ph, y1[m]) - ba(P1[m], y1[m]))
    # identity model on the other people
    St = []
    for q in others:
        rq = np.flatnonzero(person == q)
        for s in dict.fromkeys(session[rq]):
            rs = rq[session[rq] == s]; St.append(dict(corpus="c", person=q, session=s, full=B[rs].mean(0), head=B[rs][:NHEAD]))
    Mid = TwoCov().fit(St, "psd"); muc = np.mean([s["full"] for s in St], 0)
    imp = []
    multi = [q for q in others if len({s["session"] for s in St if s["person"] == q}) > 1]
    for k in range(min(40, len(multi))):
        g = list(rng.choice([q for q in others if q != multi[k]], min(GAL, len(others) - 1), replace=False))
        G = [Mid.proj(next(s["full"] for s in St if s["person"] == q), muc) for q in g]
        pr = Mid.proj(next(s for s in St if s["person"] == multi[k])["head"][:20].mean(0), muc)
        imp.append(max(Mid.llr(gq, 1, pr, 20) for gq in G))
    tau = float(np.percentile(imp, 99))
    Gp = [Mid.proj(B[s1rows(q)].mean(0), muc) for q in gal]
    rp = np.flatnonzero(person == per); tests = sess_order(session[rp])[1:]; res = dict(person=per, cv=cv,
                                                                                      adv=float(np.mean(adv)), tau=tau, n1=len(r1), test={})
    for gi, ts in enumerate(tests):
        rt = rp[session[rp] == ts]
        for sc, seed in ORDERS:
            o = rt[order(session[rt], run[rt], sc, seed, pi * 10 + gi)]; Vo = V[o].astype(np.float64)
            Xc = Vo - np.cumsum(Vo, 0) / (K0 + 1 + np.arange(len(o)))[:, None]
            Zt = (Xc - mu) / sd; P = clf.predict_proba(Zt); pc = pca.transform(Zt); Hc = causal_std(pc)
            Hs = np.stack([heads[q].predict_proba(Hc) for q in gal])
            cb = np.cumsum(B[o].astype(np.float64), 0); r, j = np.zeros(len(o)), np.full(len(o), -1)
            for t in range(T0, len(o)):
                x = Mid.proj(cb[t] / (t + 1), muc); scs = np.array([Mid.llr(g, 1, x, t + 1) for g in Gp])
                j[t] = int(scs.argmax()); r[t] = float(scs[j[t]] >= tau)
            Hr = np.where((j >= 0)[:, None], Hs[np.maximum(j, 0), np.arange(len(o))], P)
            so, yo = session[o], y[o]; row = []
            for s in S_GRID:
                st = KS.chain(KS.slot_blend(P, Hr, r, s), so, pc[:, :NGE].copy(), pc)
                row.append([ba(st[k], yo) for k in ("member", "tsem")])
            res["test"][(gi + 1, sc, seed)] = dict(ba=np.array(row), routed=float(r.mean()), right=float(((j == 0) & (r > 0)).mean()),
                                                  wrong=float(((j > 0) & (r > 0)).mean()), own_head=ba(Hs[0], yo))
    k0 = (1, "rec", 0)
    print(f"{per}: s1 n {len(r1)} LORO adv {res['adv']:+.1f} | gap1 rec member {res['test'][k0]['ba'][0, 0]:.1f} chain s0 "
          f"{res['test'][k0]['ba'][0, 1]:.1f} s.25 {res['test'][k0]['ba'][4, 1]:.1f} routed {res['test'][k0]['routed']:.2f} "
          f"(right {res['test'][k0]['right']:.2f}) | {len(tests)} test sessions ({time.time() - t0:.0f} s)", flush=True)
    return res


def main(name):
    d = CO.feats(name); D.update(d); D["B"] = CO.bio(name)
    key = np.char.add(d["person"], np.char.add("|", d["session"])); Vc = d["V"].astype(np.float64)
    for k in np.unique(key):
        m = key == k; Vc[m] -= Vc[m].mean(0)
    D["Vc"] = Vc.astype(np.float32)
    multi = [q for q in dict.fromkeys(d["person"]) if len(np.unique(d["session"][d["person"] == q])) > 1]
    D["people"] = list(dict.fromkeys(d["person"]))
    F = np.random.default_rng(0).normal(size=(30, 5)); assert np.allclose(causal_std(F), RT.causal_std(F)), "causal_std"
    n = int(os.environ.get("PEOPLE", len(multi))); nj = int(os.environ.get("NJ", 6))
    print(f"RET {name}: {len(D['people'])} people, {len(multi)} with >= 2 sessions ({n} scored), classes {list(d['classes'])}, "
          f"orders {ORDERS}, s {S_GRID}", flush=True)
    idx = [D["people"].index(q) for q in multi[:n]]
    with mp.get_context("fork").Pool(nj) as pool:
        out = pool.map(fold, idx, chunksize=1)
    mem = os.environ.get("MEMBER", "v1"); path = f"{CO.OUT}/ret_{name}{'' if mem == 'v1' else '_' + mem}{os.environ.get('TAG', '')}.pkl"
    with open(path, "wb") as f:
        pickle.dump(dict(corpus=name, s=S_GRID, orders=ORDERS, res=out), f)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
