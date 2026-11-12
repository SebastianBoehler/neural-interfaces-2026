"""BANDMAP addendum (ledger s77): confusion matrices of below-chance transfers. Usage: python studies/12nov/bandconf.py"""
import sys, itertools, numpy as np
__file__ = "/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026/studies/12nov/bandmap.py"
src = open(__file__).read().split("keys = [")[0]; exec(src)
def probs(M, prior, r, key):
    Cw, _ = O.recentred(r, prior)
    if key == "S": return M["S"].predict_proba(O.causal_z(O.SL[r]))
    f, b = key
    if f == "T": return M["T", b].predict_proba(O.tang([Cw[b]]))
    W, m = M["C", b]; return m.predict_proba(np.log(np.einsum("cf,ncd,df->nf", W, Cw[b], W)))
for t, keys in (("P04", ["S", ("C", 7), ("C", 3)]), ("P05", [("C", 2), ("C", 7), "S"]), ("P03", [("T", 4)])):
    M, pr = fit([r for q in PEOPLE if q != t for r in sess(q)])
    for k in keys:
        for r in sess(t):
            P = probs(M, pr, r, k); cm = np.zeros((3, 3))
            for a, b in zip(O.y[r], P.argmax(1)): cm[a, b] += 1
            cm = cm / cm.sum(1, keepdims=True); best = max(itertools.permutations(range(3)), key=lambda pm: sum(cm[i, pm[i]] for i in range(3)))
            print(t, O.SES[r[0]], k, "ba %.0f" % O.ba(r, P), "| confusion (rows true MI/SUB/WORD):", np.round(100 * cm).astype(int).tolist(), "| best permutation", best, "-> %.0f" % (100 * np.mean([cm[i, best[i]] for i in range(3)])), flush=True)
