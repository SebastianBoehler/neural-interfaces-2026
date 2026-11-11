"""Score optimize.run_session on the dev sessions. Printed (search) metric: balanced accuracy on unseen people of half
G1, mean of two bases. Hidden from the optimiser (history.jsonl only): half G2, seen people, a fresh base.
A candidate that uses future windows is rejected: every session is also run on a prefix and must give the same rows."""
import importlib.util, json, os, sys, time
import numpy as np
TASK = os.path.dirname(os.path.abspath(__file__)); t0 = time.time()
spec = importlib.util.spec_from_file_location("opt", TASK + "/optimize.py"); opt = importlib.util.module_from_spec(spec); spec.loader.exec_module(opt)
D = np.load(TASK + "/pack/data.npz", allow_pickle=True)
PEOPLE = [f"P{i:02d}" for i in range(1, 11)]; G1 = ("P01", "P02", "P05", "P07", "P09"); G2 = tuple(p for p in PEOPLE if p not in G1)


def run(bn, p, check):
    ss = D[f"sess|{p}"]; M = D[f"mix|{bn}|{p}"]; X = D[f"X|{p}"]; F = D[f"F|{p}"]; out = np.empty_like(M); rng = np.random.default_rng(0)
    for s in np.unique(ss):
        r = np.flatnonzero(ss == s); o = np.asarray(opt.run_session(M[r].copy(), X[r].copy(), F[r].copy()), dtype=np.float64)
        assert o.shape == (len(r), 3) and np.isfinite(o).all(), "run_session must return finite (n, 3)"
        if check:
            c = int(rng.integers(25, len(r) - 5)); o2 = np.asarray(opt.run_session(M[r[:c]].copy(), X[r[:c]].copy(), F[r[:c]].copy()), dtype=np.float64)
            assert np.allclose(o[:c], o2, atol=1e-9), "NOT CAUSAL: rows change when later windows are removed"
        out[r] = o
    return out


def person_ba(P, p):
    y, cell, per = D[f"y|{p}"], D[f"cell|{p}"], D[f"person|{p}"]; pred = P.argmax(1); acc = {}
    for c in np.unique(cell):
        m = cell == c; acc.setdefault(per[m][0], []).append(100 * np.mean([np.mean(pred[m][y[m] == k] == k) for k in np.unique(y[m])]))
    return {q: float(np.mean(v)) for q, v in acc.items()}


try:
    new, seen = [], []
    for bn in ("B2_pool", "B2_s345"):
        new.append(person_ba(run(bn, "all_loso", bn == "B2_pool"), "all_loso")); seen.append(person_ba(run(bn, "seen_session", False), "seen_session"))
    g1 = np.mean([[d[q] for q in G1] for d in new]); g2 = np.mean([[d[q] for q in G2] for d in new]); sn = np.mean([list(d.values()) for d in seen])
    fr = person_ba(run("B2_s012", "all_loso", False), "all_loso"); frs = person_ba(run("B2_s012", "seen_session", False), "seen_session")
    rec = dict(time=time.strftime("%H:%M:%S"), g1=float(g1), g2=float(g2), seen=float(sn), J=float(0.5 * (g1 + g2) / 2 * 2 * 0.5 + 0.5 * sn) if False else float(0.25 * g1 + 0.25 * g2 + 0.5 * sn),
               fresh_new=float(np.mean(list(fr.values()))), fresh_seen=float(np.mean(list(frs.values()))), secs=round(time.time() - t0, 1))
    with open(TASK + "/history.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"g1_new_ba: {rec['g1']:.4f}")
except Exception as e:
    print("ERROR:", repr(e)[:300]); print("g1_new_ba: 0.0")
