"""Independent pre-registered check of R19e = R19 gaze view + 16 causal ocular-event rates
(experiments/session_em.py EventRates), frozen EM config lam .5 n0 60 m0 20 shrink .7 it 2 nmin 10,
events frozen (8 Hz low-pass derivative, q 4/8 x running MAD, k 5, 4 bins).

PRE-REGISTERED GATES (written before running; increments are R19e minus R19 = gaze only):
 G1 dla six-seed pooled paired increment (frozen config, global g) >= +0.4, >= 6/8 participants >= 0;
    in-fold g (g from the fold's training rows only) >= +0.3.
 G2 nested LOPO (12-config grid, chosen per view on the other participants) paired increment pooled >= +0.4.
 G3 dla increment above the max of 20 event-only circular shifts and 20 noise pads, on both seed halves.
 G4 first-60-window dla increment >= 0 on both halves.
 G5 harm: no dla participant other than P08 below -2.0; P08 not below -4.0 (pooled).
 Diagnostic only (no gate on its own): ds per half; al; seen_session proxy vs R19 (pooled; P04 >= -2.0 flagged).
"""
import sys, itertools, json
from pathlib import Path
S = str(Path(__file__).resolve().parent)  # scripts/diag; outputs go to runs/diag
sys.path.insert(0, S)
OUTD = Path(S).parents[1] / "runs/diag"; OUTD.mkdir(parents=True, exist_ok=True)
from diag_common import *  # d, load, mixp, rebal, cells, cm
import session_em as SE
import protocol
from scipy.signal import sosfiltfilt

X = d["X"].astype(np.float64); ch = list(d["ch_names"]); st = d["stream"]
G = SE.gaze_features(X, ch)
T = SE.event_traces(X, ch)


def evrates(g, rows=None):
    out = np.empty((len(st), 16))
    for s in dict.fromkeys(st):
        er = SE.EventRates(g)
        for r in np.flatnonzero(st == s):
            out[r] = er.step(T[:, r])
    return out


g_all = SE.global_mad(X, ch)
EV = evrates(g_all)

# --- reference: probe formula verbatim (neuro2/mixev.py) ---
I = ch.index
L = [I(c) for c in ["F5", "AF3", "F3"]]; R = [I(c) for c in ["F6", "AF4", "F4"]]
dh = np.diff(sosfiltfilt(SE.EV_SOS, X[:, L].mean(1) - X[:, R].mean(1), axis=-1), axis=-1)[:, :476]
dv = np.diff(sosfiltfilt(SE.EV_SOS, X[:, [I("AF3"), I("AF4")]].mean(1) - X[:, [I("C1"), I("C2"), I("Cz")]].mean(1), axis=-1), axis=-1)[:, :476]


def causal_mad(A, k=5):
    out = np.zeros(len(A)); g = np.median(np.abs(A))
    for s in dict.fromkeys(st):
        rows = np.flatnonzero(st == s); hist = []
        for r in rows:
            hist.append(np.median(np.abs(A[r]))); out[r] = (k * g + np.sum(hist)) / (k + len(hist))
    return out


mh, mv = causal_mad(dh), causal_mad(dv); Fr = []
for A, m in ((np.abs(dh), mh), (np.abs(dv), mv)):
    for q in (4, 8):
        Fr.append((A > q * m[:, None]).reshape(len(A), 4, -1).mean(-1))
EVref = np.log(np.concatenate(Fr, 1) + 1e-3)
print(f"event features vs probe formula: max abs diff {np.abs(EV - EVref).max():.2e}", flush=True)
del X, dh, dv

FROZEN = dict(lam=0.5, n0=60, m0=20)
GRID = [dict(lam=l, n0=n, m0=m) for l, n, m in itertools.product((0.3, 0.5, 0.7), (30, 60), (0, 20))]


def spec(a, b, tfb):
    return ["v2", f"fbcnet_ea_tf:{a}-{tfb}*0.5", f"fbcnet_raw_tf:{a}-{tfb}*0.5"] + [f"{n}:{a}-{b}" for n in (
        "shallow_ea", "shallow_raw", "fbcnet_raw", "fbcnet_ea_rot05", "sincshallow_ea_ptm7r_rot05", "fblight_ea_rot05")]


def pp(idx, o):
    c = cells(idx, o.argmax(1)); per = {}
    for k, v in c.items():
        per.setdefault(k[0].split("/")[-1].split("|")[0], []).append(v)
    return {p: float(np.mean(v)) for p, v in per.items()}


def em(F, base, idx, **kw):
    return SE.apply(F[idx], base, st[idx], **{**FROZEN, **kw})


def infold_ev(proto):
    src = "all_loso" if proto == "dev_loso_all" else proto
    out = np.empty_like(EV)
    for _, tr, te in protocol.folds(d, src):
        e = evrates(np.median(np.abs(T[:, tr]), axis=(1, 2)))
        out[te] = e[te]
    return out


res = {}
rng = np.random.default_rng(7)
for tag, sp in (("s0-2", spec(0, 2, 1)), ("s3-5", spec(3, 5, 4))):
    for proto in ("dev_loso_all", "dev_session", "all_loso"):
        _, Ps, w, idx, _ = load(proto, sp)
        base = rebal(mixp(Ps, w), idx)
        V = {"R15": base, "R19": em(G, base, idx), "R19e": em(np.c_[G, EV], base, idx),
             "R19e_infold_g": em(np.c_[G, infold_ev(proto)], base, idx)}
        sc = {k: cm(idx, o.argmax(1)) for k, o in V.items()}
        r = {"score": sc, "pp": {k: pp(idx, o) for k, o in V.items()}}
        line = f"{tag} {proto:13s} " + " ".join(f"{k} {v:.2f}" for k, v in sc.items())
        if proto == "dev_loso_all":
            nul_sh, nul_nz = [], []
            for i in range(20):
                Es = EV.copy()
                for s in dict.fromkeys(st):
                    q = np.flatnonzero(st == s); Es[q] = EV[np.roll(q, rng.integers(15, len(q) - 15))]
                nul_sh.append(cm(idx, em(np.c_[G, Es], base, idx).argmax(1)) - sc["R19"])
                Z = (EV - EV.mean(0)) / EV.std(0)
                nul_nz.append(cm(idx, em(np.c_[G, rng.standard_normal(Z.shape)], base, idx).argmax(1)) - sc["R19"])
            r["null_shift"], r["null_noise"] = nul_sh, nul_nz
            line += f" | inc {sc['R19e'] - sc['R19']:+.2f} shift max {max(nul_sh):+.2f} mean {np.mean(nul_sh):+.2f} noise max {max(nul_nz):+.2f} mean {np.mean(nul_nz):+.2f}"
            nest = {}
            for vn, F in (("R19", G), ("R19e", np.c_[G, EV])):
                Rg = [pp(idx, em(F, base, idx, **c)) for c in GRID]
                parts = list(Rg[0]); nest[vn] = {}
                for p in parts:
                    best = max(range(len(GRID)), key=lambda i: np.mean([Rg[i][q] for q in parts if q != p]))
                    nest[vn][p] = Rg[best][p]
            r["nested"] = nest
            for n in (60, 90):
                keep = np.zeros(len(idx), bool)
                for s in dict.fromkeys(st[idx]):
                    keep[np.flatnonzero(st[idx] == s)[:n]] = True
                r[f"trunc{n}"] = {k: cm(idx[keep], V[k][keep].argmax(1)) for k in ("R15", "R19", "R19e")}
                line += f" | first{n} R19e-R19 {r[f'trunc{n}']['R19e'] - r[f'trunc{n}']['R19']:+.2f}"
        print(line, flush=True)
        res[f"{tag} {proto}"] = r

# seen_session proxy (all seeds; R10-like members)
_, Ps, w, idx, _ = load("seen_session")
base = rebal(mixp(Ps, w), idx)
V = {"R15p": base, "R19": em(G, base, idx), "R19e": em(np.c_[G, EV], base, idx)}
res["seen_session"] = {"score": {k: cm(idx, o.argmax(1)) for k, o in V.items()}, "pp": {k: pp(idx, o) for k, o in V.items()}}
print("seen_session proxy", res["seen_session"]["score"], res["seen_session"]["pp"], flush=True)
json.dump(res, open(OUTD / "verify_r19e.json", "w"), indent=1)

# ---- gates ----
def inc_pp(key, a="R19e", b="R19"):
    P = res[key]["pp"]; return {p: P[a][p] - P[b][p] for p in P[a]}


halves = ("s0-2", "s3-5")
dla = [inc_pp(f"{h} dev_loso_all") for h in halves]
pooled = {p: np.mean([x[p] for x in dla]) for p in dla[0]}
v = np.array(list(pooled.values()))
dla_inf = np.mean([res[f"{h} dev_loso_all"]["score"]["R19e_infold_g"] - res[f"{h} dev_loso_all"]["score"]["R19"] for h in halves])
nest = {p: np.mean([res[f"{h} dev_loso_all"]["nested"]["R19e"][p] - res[f"{h} dev_loso_all"]["nested"]["R19"][p] for h in halves]) for p in pooled}
nv = np.array(list(nest.values()))
print("\n== per participant dla R19e-R19 pooled:", {p: round(x, 2) for p, x in pooled.items()})
print(f"G1 pooled dla inc {v.mean():+.2f} (se {v.std(ddof=1) / np.sqrt(len(v)):.2f}, t {v.mean() / (v.std(ddof=1) / np.sqrt(len(v))):.2f}) n>=0 {int((v >= 0).sum())}/{len(v)}; in-fold g {dla_inf:+.2f}"
      f" -> {'PASS' if v.mean() >= 0.4 and (v >= 0).sum() >= 6 and dla_inf >= 0.3 else 'FAIL'}")
print(f"G2 nested paired inc {nv.mean():+.2f} (se {nv.std(ddof=1) / np.sqrt(len(nv)):.2f}) -> {'PASS' if nv.mean() >= 0.4 else 'FAIL'}")
g3 = [res[f'{h} dev_loso_all']['score']['R19e'] - res[f'{h} dev_loso_all']['score']['R19'] > max(max(res[f'{h} dev_loso_all']['null_shift']), max(res[f'{h} dev_loso_all']['null_noise'])) for h in halves]
print(f"G3 above null max both halves: {g3} -> {'PASS' if all(g3) else 'FAIL'}")
g4 = [res[f'{h} dev_loso_all']['trunc60']['R19e'] - res[f'{h} dev_loso_all']['trunc60']['R19'] for h in halves]
print(f"G4 first-60 inc {[round(x, 2) for x in g4]} -> {'PASS' if min(g4) >= 0 else 'FAIL'}")
harm = [p for p, x in pooled.items() if (x < -2.0 and p != 'P08') or (p == 'P08' and x < -4.0)]
print(f"G5 harm violations {harm} -> {'PASS' if not harm else 'FAIL'}")
for h in halves:
    s = res[f"{h} dev_session"]["score"]; a = res[f"{h} all_loso"]["score"]
    print(f"diag {h}: ds R19e-R19 {s['R19e'] - s['R19']:+.2f} (R19e-R15 {s['R19e'] - s['R15']:+.2f}); al R19e-R19 {a['R19e'] - a['R19']:+.2f} (R19e-R15 {a['R19e'] - a['R15']:+.2f})")
ss = res["seen_session"]
print("diag ss R19e-R19 {:+.2f} | per part {}".format(ss["score"]["R19e"] - ss["score"]["R19"], {p: round(ss['pp']['R19e'][p] - ss['pp']['R19'][p], 2) for p in ss['pp']['R19']}))
