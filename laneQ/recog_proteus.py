"""Lane Q test 1, in-lab arm: P04/P10 enrolled on visit 1 only (m=1), visits 2/3 streamed (8 shuffled orders, 150 windows), spectral gate stock vs S-norm
(cohort = NETBCI + ds007221 sessions, threshold = stock's wrong-latch rate on the other 8 people vs a gallery of the 10 public rows minus the person (new-person streams)).
python laneQ/recog_proteus.py PKGDIR FEATDIR"""
import sys
import numpy as np
sys.path.insert(0, "laneQ"); sys.path.insert(0, "2026-competition")
from recog_lib import Gate
pk, fd = sys.argv[1:3]

gate = Gate(dict(np.load(pk + "/weights.npz", allow_pickle=False)))
import importlib.util
spec = importlib.util.spec_from_file_location("ef", "laneQ/extract_feat.py"); ef = importlib.util.module_from_spec(spec); spec.loader.exec_module(ef)
z = np.load("data/proteus_windows.npz", allow_pickle=True)
sub, ses = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]]), z["session"].astype(str)
F = gate.proj(np.stack([ef.feat(np.nan_to_num(x)) for x in z["X"]]))
T, NORD = 150, 8
names = ["P%02d" % i for i in range(1, 11)]
print("sessions per person", {p: sorted(set(ses[sub == p])) for p in ("P04", "P10", "P01")})
ck = [(c, f"{fd}/q_{c}_feat.npz") for c in ("netbci", "ds7g", "ds7lr")]
cm, cp = [], []
for c, fn in ck:
    with np.load(fn, allow_pickle=True) as d:
        Fc, s1, s2 = gate.proj(d["F"]), d["subject"].astype(str), d["session"].astype(str)
    for k in sorted(set(zip(s1, s2))):
        ix = np.flatnonzero((s1 == k[0]) & (s2 == k[1])); cm.append(Fc[ix].mean(0))
        if len(ix) >= T:
            p = np.random.default_rng([99, len(cp)]).permutation(len(ix))[:T]; cp.append(np.cumsum(Fc[ix[p]], 0) / np.arange(1, T + 1)[:, None])
cm, cp = np.stack(cm), np.stack(cp)
ses_ids = sorted(set(ses))


def sess_ix(p, s): return np.flatnonzero((sub == p) & (ses == s))


def stream(ix, o):
    p = np.random.default_rng([5, o, int(ix[0])]).permutation(len(ix))[:T]
    return np.cumsum(F[ix[p]], 0) / np.arange(1, T + 1)[:, None]


def scores(G, m, PM):
    """PM (R,T,d) -> raw, snorm (R,T,M)."""
    R = len(PM); raw = np.zeros((R, T, len(G))); sn = np.zeros_like(raw)
    for n in range(1, T + 1):
        x = PM[:, n - 1]
        raw[:, n - 1] = gate.score(G, m, x, n)
        ct = gate.score(cm, np.ones(len(cm), int), x, n); cz = gate.score(G, m, cp[:, n - 1], n)
        sn[:, n - 1] = 0.5 * ((raw[:, n - 1] - cz.mean(0)) / cz.std(0) + (raw[:, n - 1] - ct.mean(1)[:, None]) / ct.std(1)[:, None])
    return raw, sn


def latch(sc, thr):
    mx = sc.max(2); ok = mx > thr; ok[:, :10] = False
    hit = ok.any(1); f = ok.argmax(1)
    return np.where(hit, f + 1, 0), sc.argmax(2)[np.arange(len(sc)), f]


# gallery: public rows; P04/P10 replaced by visit-1-only means (m=1)
G = gate.G0.copy(); m = gate.m0.copy()
first = {p: sorted(set(ses[sub == p]))[0] for p in names}
for p in ("P04", "P10"):
    G[names.index(p)] = F[sess_ix(p, first[p])].mean(0); m[names.index(p)] = 1
res = {}
for p in ("P04", "P10"):
    for s in sorted(set(ses[sub == p]))[1:]:
        ix = sess_ix(p, s); PM = np.stack([stream(ix, o) for o in range(NORD)])
        raw, sn = scores(G, m, PM); res[(p, s)] = (raw, sn, names.index(p))
# impostor reference: other 8 people's first sessions (new-person streams) vs the same gallery
imp = [(p, s) for p in names if p not in ("P04", "P10") for s in sorted(set(ses[sub == p]))[:1]]
IR, IS = [], []
for p, s in imp:
    ix = sess_ix(p, s); PM = np.stack([stream(ix, o) for o in range(NORD)])
    raw, sn = scores(G, m, PM); raw[:, :, names.index(p)] = -1e9; sn[:, :, names.index(p)] = -1e9  # the person's own row is absent for a NEW person
    IR.append(raw); IS.append(sn)
IR, IS = np.concatenate(IR), np.concatenate(IS)
print(f"impostor streams {len(IR)} (8 new people x {NORD} orders), gallery 10 rows with P04/P10 = visit 1 only")
print("stock impostor latch rate", float((latch(IR, .26)[0] > 0).mean()))
for name, sc_i, key in (("stock raw .26", IR, 0), ("S-norm", IS, 1)):
    if key == 0:
        thr = .26
    else:  # smallest thr with impostor latch rate <= stock's
        tgt = float((latch(IR, .26)[0] > 0).mean()); grid = np.quantile(IS.max(2)[:, 10:], np.linspace(0, .9999, 400))
        thr = next((g for g in grid if (latch(IS, g)[0] > 0).mean() <= tgt + 1e-12), IS.max() + 1e-6)  # target 0: above every impostor score
    print(f"{name}: thr {thr:.3f} impostor FA {float((latch(sc_i, thr)[0] > 0).mean()):.3f}")
    for (p, s), v in res.items():
        n, j = latch(v[key], thr); ok = (n > 0) & (j == v[2])
        print(f"   {p} visit {s}: TA {ok.mean():.2f} misID {((n > 0) & (j != v[2])).mean():.2f}  ages {sorted(n[ok].tolist())}")
