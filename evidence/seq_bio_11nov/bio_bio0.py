import numpy as np
from scipy.signal import welch
z = np.load("<repo>/data/proteus_windows.npz", allow_pickle=True)
X = z["X"]; per = np.array([s[-3:] for s in z["subject"]]); ses = z["session"].astype(str)
X = X - X.mean(1, keepdims=True)
f, P = welch(X, fs=120, nperseg=120, noverlap=60, axis=-1)
m = (f >= 2) & (f <= 40); F = np.log(P[:, :, m] + 1e-12); F = F - F.mean(2, keepdims=True); F = F.reshape(len(F), -1).astype(np.float64)
sess = sorted(set(zip(per, ses))); people = sorted(set(per))
def sig(p, s, n=None):
    r = np.flatnonzero((per == p) & (ses == s)); r = r if n is None else r[:n]; return F[r].mean(0)
for variant in ("shape", "shape+rank"):
    print("==", variant)
    for n in (3, 5, 10, 20, 60, None):
        gen, impmax, rank1, newmax = [], [], [], []
        for (p, s) in sess:
            G = {}
            for q in people:
                ss = [x for x in sess if x[0] == q and x != (p, s)]
                if ss: G[q] = np.mean([sig(*x) for x in ss], 0)
            names = list(G); M = np.stack([G[q] for q in names]); mu, sd = M.mean(0), M.std(0) + 1e-6
            a = (sig(p, s, n) - mu) / sd; B = (M - mu) / sd
            if variant == "shape+rank":
                a = np.argsort(np.argsort(a)).astype(float); B = np.argsort(np.argsort(B, 1), 1).astype(float)
            sc = np.array([np.corrcoef(a, b)[0, 1] for b in B])
            if p in names:
                g = sc[names.index(p)]; im = np.max(np.delete(sc, names.index(p))); gen.append(g); impmax.append(im); rank1.append(g > im)
            else:
                newmax.append(sc.max())
        gen, impmax, newmax = np.array(gen), np.array(impmax), np.array(newmax)
        thr = np.quantile(newmax, 0.9)
        print(f" n={str(n):>4}: genuine {np.round(gen, 2).tolist()} | best impostor for those probes {np.round(impmax, 2).tolist()} | rank-1 {int(np.sum(rank1))}/6 "
              f"| new-person max {np.round(np.sort(newmax), 2).tolist()} | genuine > p90(new) {int((gen > thr).sum())}/6, > max(new) {int((gen > newmax.max()).sum())}/6")
