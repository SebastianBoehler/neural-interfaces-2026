"""Diagnostic: for NEW people, does spectral-identity similarity to a public person predict how well that person's
personal head (PS1) works on them? If yes, similarity-weighted person models could help the 57 % of sealed that is new."""
import sys
import numpy as np
from scipy.stats import spearmanr
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/experiments"]
import env
from bio_gate import Bio, fold_params
A = env.A; R3, RR, RL, pl, SC = A.R3, A.RR, A.RL, A.pl, A.SC
z = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True); X = z["X"]; Y = z["y"]
PER = np.array([s[-3:] for s in z["subject"]]); STR = np.char.add(np.char.add(PER, "|"), z["session"].astype(str))
B0 = Bio.__new__(Bio); B0.win = np.hanning(120); FB = np.stack([B0.feat(x) for x in X]); MODEL = dict(np.load(sys.argv[1]))
bn = "B2_pool"; mixes = {p: A.mix(bn, p, A.FAMS)[0] for p in A.PROTOS}; sh = A.shadow(mixes["all_loso"], "all_loso")[1]
proto = "all_loso"; idx = pl.B[f"idx|{proto}"]; pos = {int(g): k for k, g in enumerate(idx)}; c1 = A.members(bn, proto)["c1"]
ba = lambda y, p: np.mean([np.mean(p[y == k] == k) for k in range(3)])  # noqa: E731
rows = []
for f in RR.FOLDS[proto]:
    te, names, Gn, Q, per, _ = R3.gate_model(proto, f, sh); heads = R3.ps1_heads(per, names)
    itr = np.load(f"{RL.FE}/{proto}-{f}.npz")["idx_tr"]
    for s, (Xs, y, gi) in te.items():
        gi = np.asarray(gi); r = np.array([pos[int(g)] for g in gi]); Zc = RR.causal_std(Xs[:, :256])
        H = np.stack([heads[q].predict_proba(Zc) for q in names]); acc = np.array([ba(y, h.argmax(1)) for h in H])
        b = Bio(fold_params(MODEL, FB[itr], STR[itr], list(names), t0=0)); S = []
        for g in gi:
            b.step(X[g]); S.append(b.scores())
        S = np.array(S); fin = S[-1]; early = S[19]
        w = np.exp((early - early.max()) / 0.05); w /= w.sum(); mixh = np.einsum("p,pnk->nk", w, H)
        rows.append(dict(s=s.split("/")[-1], rho_final=spearmanr(fin, acc)[0], rho_early=spearmanr(early, acc)[0], core1=ba(y, c1[r].argmax(1)),
                         best_head=acc.max(), mean_head=acc.mean(), head_of_most_similar=acc[int(early.argmax())], sim_weighted_heads=ba(y, mixh.argmax(1)),
                         uniform_heads=ba(y, H.mean(0).argmax(1)), top=names[int(early.argmax())], top_score=float(early.max())))
print("session   rho(final) rho(n=20) | BA: core1  best-head mean-head most-similar-head sim-weighted uniform-mix | most similar (score)")
for r in rows:
    print(f"{r['s']:8s} {r['rho_final']:+9.2f} {r['rho_early']:+9.2f} | {100*r['core1']:9.1f} {100*r['best_head']:9.1f} {100*r['mean_head']:9.1f} {100*r['head_of_most_similar']:12.1f} {100*r['sim_weighted_heads']:12.1f} {100*r['uniform_heads']:11.1f} | {r['top']} ({r['top_score']:+.2f})")
m = lambda k: 100 * np.mean([r[k] for r in rows])  # noqa: E731
print(f"mean      {np.mean([r['rho_final'] for r in rows]):+9.2f} {np.mean([r['rho_early'] for r in rows]):+9.2f} | {m('core1'):9.1f} {m('best_head'):9.1f} {m('mean_head'):9.1f} {m('head_of_most_similar'):12.1f} {m('sim_weighted_heads'):12.1f} {m('uniform_heads'):11.1f}")
