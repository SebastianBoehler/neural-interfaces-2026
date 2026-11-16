"""s290 part 1: first-principles core-level designs P1/P2/P3 on core1's features (LOSO over ten people, LR C=10 = 'lr10'), config only.
Features: s263 variant Z (core1: CAR, trace shrinkage 0.1, log-Euclidean reference k 2, ov17 x [full, 0-2 s, 2-4 s] + slow block) read from runs/s263/Z;
P1 regional views S1 (sensorimotor) / S2 (frontal, parietal, left fronto-temporal) are built with ts_core2.Feats on the channel subset (own CAR) and cached.
Arms (one task = one held-out person, all arms; outputs runs/s290/<arm>_fold<I>.json/.npz):
  P1a hierarchical, stage 1 MI vs {SUB,WORD} on S1 (mu+beta), stage 2 SUB vs WORD on S2 (theta+alpha); P1b same with stage 1 on all channels (Z columns, mu+beta);
  P1c flat core1 (asserted equal to the stored s263 Z lr10 of the same person); P2a/P2b/P2c pairwise LRs (MI-SUB, MI-WORD, SUB-WORD) with own bands / segments / both,
  fused by pairwise coupling; P3a one-vs-rest LRs with Platt calibration (inner leave-people-out on the training people); P3b = geometric mean of P3a and P2c.
Usage: s290.py prep | fold I [OUTDIR] (the smoke runs folds 0 and 1 with OUTDIR=runs/s290_smoke)."""
import json, os, pickle, sys, time
import numpy as np
REPO = os.environ.get("NI26_ROOT") or os.path.abspath(__file__ + "/../../../..")
os.environ["NI26_ROOT"] = REPO
sys.path[:0] = [REPO + "/experiments", REPO + "/studies/14nov/s263"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
import s263  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OV17 = T.BANDSETS["ov17"]
NSEG = 3  # full, 0-2 s, 2-4 s (core1 fs2 order)
SEGSETS = {"all": (0, 1, 2), "early": (1,), "late": (2,)}
S1_CH = "C1 C2 C3 C4 C5 C6 Cz CP1 CP2 CP3 CP4 CPz FC1 FC2 FC3 FC4 FCz".split()
S2_CH = "Fz F1 F2 F3 F4 F5 F6 AF3 AF4 Pz P1 P2 P3 P4 P5 P6 PO7 PO8 FC5 TP7 CP5".split()


def band_idx(lo, hi):  # ov17 bands whose geometric centre lies in [lo, hi) Hz
    return tuple(i for i, (a, b) in enumerate(OV17) if lo <= (a * b) ** 0.5 < hi)


MB = band_idx(8, 30)  # mu + beta
TA = tuple(sorted(set(band_idx(4, 13)) | set(band_idx(30, 45))))  # theta + alpha + gamma 30-45
THA = band_idx(4, 13)  # theta + alpha (P1 stage 2)
assert (len(MB), len(TA), len(THA)) == (8, 8, 5) and not set(S1_CH) & set(S2_CH)
VIEWS = {"S1": dict(ch=S1_CH, bands=MB), "S2": dict(ch=S2_CH, bands=THA)}
ARMS = ["P1a", "P1b", "P1c", "P2a", "P2b", "P2c", "P3a", "P3b"]


def view_settings(v):
    s = VIEWS[v]
    return dict(view=v, ch=s["ch"], bands=[list(OV17[i]) for i in s["bands"]], segs=s263.SEG3, est="t0.1", k=2, cov="fs2", ref="car")


def view_path(v):
    return f"{REPO}/runs/s290/feats_{v}.pkl"


def build_view(d, v):
    s = view_settings(v)
    ix = [list(map(str, d["ch_names"])).index(c) for c in s["ch"]]
    assert len(ix) == len(set(ix)) == len(s["ch"])
    T.BANDSETS["s290" + v] = [tuple(b) for b in s["bands"]]
    d2 = dict(d, X=d["X"][:, ix], ch_names=np.asarray(s["ch"]))
    fe = T.Feats(d2, T.parse(f"car_s290{v}_fs2_t0.1_le_k2"))
    assert len(fe.C) == NSEG * len(s["bands"]) and len(fe.slices) == len(fe.C) + 1, "view block count"
    return fe


def load_view(v):
    with open(view_path(v), "rb") as f:
        st = pickle.load(f)
    assert st["settings"] == json.loads(json.dumps(view_settings(v))), f"cache of view {v} does not match its settings"
    fe = T.Feats.__new__(T.Feats)
    fe.__dict__.update(st["state"])
    return fe


def cols(fe, bands, segs, nb_total):
    """column index of the blocks (band i, segment j) = block i * 3 + j of a Feats with nb_total bands, plus the slow block."""
    assert len(fe.slices) == nb_total * NSEG + 1
    sl = [fe.slices[i * NSEG + j] for i in bands for j in segs] + [fe.slices[-1]]
    return np.concatenate([np.arange(s.start, s.stop) for s in sl])


def lr(C=10.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def couple(r):
    """Price et al. pairwise coupling: r[(i, j)] = P(i | i or j) (n,), i < j; returns (n, 3) posteriors."""
    R = {}
    for (i, j), v in r.items():
        R[(i, j)], R[(j, i)] = v, 1 - v
    p = np.stack([1.0 / (sum(1.0 / np.clip(R[(i, j)], 1e-6, 1) for j in range(3) if j != i) - 1.0) for i in range(3)], 1)
    return p / p.sum(1, keepdims=True)


def hier(p1, q):  # p1 = P(MI), q = P(SUB | not MI)
    return np.stack([p1, (1 - p1) * q, (1 - p1) * (1 - q)], 1)


class S290:
    """protocol model: fit(train_idx); predict_proba_stream(one session) fills store[arm] for every arm."""

    def __init__(self, FE, y, subj, store):
        self.FE, self.y, self.subj, self.store, self.spec = FE, y, subj, store, {}
        z = FE["Z"]
        assert z.cfg["bands"] == "ov17" and z.cfg["cov"] == "fs2" and not z.cfg["ct"] and not z.cfg["cb"] and not z.sz
        self.c = {"mb": cols(z, MB, SEGSETS["all"], 17), "ta": cols(z, TA, SEGSETS["all"], 17), "all17e": cols(z, range(17), SEGSETS["early"], 17),
                  "all17l": cols(z, range(17), SEGSETS["late"], 17), "mbe": cols(z, MB, SEGSETS["early"], 17), "tal": cols(z, TA, SEGSETS["late"], 17)}
        self.pairs = {  # pair -> (arm -> column key); P2a bands only, P2b segments only, P2c both
            (0, 1): dict(P2a="mb", P2b="all17e", P2c="mbe"), (0, 2): dict(P2a="mb", P2b="all17e", P2c="mbe"), (1, 2): dict(P2a="ta", P2b="all17l", P2c="tal")}

    def fit(self, idx):
        y, FE = self.y, self.FE
        self.state = {k: (dict(pri=[fe.prior(b, idx) for b in range(len(fe.C))], d2p=[d2[idx].mean() for d2 in fe.d2], szp=fe.sz_prior(idx))) for k, fe in FE.items()}
        self.m = {}
        z = FE["Z"].F[idx]; yi = y[idx]
        self.m["flat"] = T.fit_head("lr10", z, yi, FE["Z"].slices)  # core1 as is (P1c)
        mk = lambda key, X, yy: self.m.__setitem__(key, lr().fit(X, yy))
        s1, s2 = FE["S1"].F[idx], FE["S2"].F[idx]
        n2 = yi != 0
        mk("p1a_1", s1, (yi == 0).astype(int)); mk("p1b_1", z[:, self.c["mb"]], (yi == 0).astype(int))
        mk("p1_2", s2[n2], (yi[n2] == 1).astype(int))
        for (i, j), arms in self.pairs.items():
            r = (yi == i) | (yi == j)
            for a, k in arms.items():
                mk((a, i, j), z[r][:, self.c[k]], (yi[r] == i).astype(int))
        self.platt, self.m["ovr"] = {}, [lr().fit(z, (yi == c).astype(int)) for c in range(3)]
        people = sorted(set(self.subj[idx])); fold = np.array([people.index(s) % 3 for s in self.subj[idx]])
        for c in range(3):  # Platt on out-of-fold scores of leave-three-people-out fits of the training people only
            oof = np.zeros(len(yi))
            for k in range(3):
                oof[fold == k] = lr().fit(z[fold != k], (yi[fold != k] == c).astype(int)).decision_function(z[fold == k])
            self.platt[c] = LogisticRegression(C=1e4).fit(oof[:, None], (yi == c).astype(int))
        self.spec = {"S1": dict(ch=S1_CH, bands=[list(OV17[i]) for i in MB], dim=int(FE["S1"].F.shape[1])), "S2": dict(ch=S2_CH, bands=[list(OV17[i]) for i in THA], dim=int(FE["S2"].F.shape[1])),
                     "mb": [list(OV17[i]) for i in MB], "ta": [list(OV17[i]) for i in TA], "segs": {k: list(v) for k, v in SEGSETS.items()},
                     "cols": {k: len(v) for k, v in self.c.items()}, "pairs": {f"{i}{j}": a for (i, j), a in self.pairs.items()}}
        assert not any(str(p).endswith(self.test_person) for p in people), "held-out person among the training people"

    def posteriors(self, F):
        z, s1, s2, c = F["Z"], F["S1"], F["S2"], self.c
        P = {"P1c": self.m["flat"].predict_proba(z)}
        q = self.m["p1_2"].predict_proba(s2)[:, 1]
        P["P1a"] = hier(self.m["p1a_1"].predict_proba(s1)[:, 1], q)
        P["P1b"] = hier(self.m["p1b_1"].predict_proba(z[:, c["mb"]])[:, 1], q)
        for a in ("P2a", "P2b", "P2c"):
            P[a] = couple({(i, j): self.m[(a, i, j)].predict_proba(z[:, c[k[a]]])[:, 1] for (i, j), k in self.pairs.items()})
        o = np.stack([self.platt[k].predict_proba(self.m["ovr"][k].decision_function(z)[:, None])[:, 1] for k in range(3)], 1)
        P["P3a"] = o / o.sum(1, keepdims=True)
        g = np.sqrt(P["P3a"] * P["P2c"])
        P["P3b"] = g / g.sum(1, keepdims=True)
        return P

    def predict_proba_stream(self, idx):
        st = self.state
        F = {k: fe.causal(idx, st[k]["pri"], st[k]["d2p"], st[k]["szp"]) for k, fe in self.FE.items()}
        for a, p in self.posteriors(F).items():
            assert p.shape == (len(idx), 3) and np.isfinite(p).all() and np.allclose(p.sum(1), 1)
            self.store[a][idx] = p


def ref_ba(i):
    return json.load(open(f"{REPO}/runs/s263/Z/fold{i}.json"))["ba"]["lr10"]


def main():
    cmd = sys.argv[1]
    d = protocol.load()
    os.makedirs(f"{REPO}/runs/s290", exist_ok=True)
    if cmd == "prep":
        for v in VIEWS:
            t0 = time.time(); fe = build_view(d, v)
            with open(view_path(v) + ".tmp", "wb") as f:
                pickle.dump(dict(settings=json.loads(json.dumps(view_settings(v))), state=fe.__dict__), f, protocol=4)
            os.replace(view_path(v) + ".tmp", view_path(v)); load_view(v)
            print(f"prep {v} dim {fe.F.shape[1]} blocks {len(fe.C)} {time.time() - t0:.0f}s", flush=True)
        return
    i = int(sys.argv[2]); out = sys.argv[3] if len(sys.argv) > 3 else f"{REPO}/runs/s290"
    os.makedirs(out, exist_ok=True)
    FE = {"Z": s263.load_feats("Z"), "S1": load_view("S1"), "S2": load_view("S2")}
    name, tr, te = list(protocol.folds(d, "all_loso"))[i]
    store = {a: np.full((len(d["y"]), 3), np.nan) for a in ARMS}
    m = S290(FE, d["y"], np.asarray(d["subject"]), store); m.test_person = name
    t0 = time.time(); m.fit(np.flatnonzero(tr)); fit = time.time() - t0
    protocol.predict_streams(m, d, te, proba=True)
    ba = {a: protocol.score(d["y"][te], store[a][te].argmax(1), d, te)["balanced_accuracy_cell_mean"] for a in ARMS}
    assert abs(ba["P1c"] - ref_ba(i)) < 1e-9, f"flat arm {ba['P1c']} != stored s263 Z lr10 {ref_ba(i)} for {name}"
    for a in ARMS:
        json.dump(dict(arm=a, person=name, fold=i, ba={"lr10": ba[a]}, spec=m.spec, fit_s=fit, total_s=time.time() - t0), open(f"{out}/{a}_fold{i}.json", "w"))
        np.savez_compressed(f"{out}/{a}_fold{i}.npz", idx=np.flatnonzero(te), lr10=store[a][te])
    print(name, {a: round(100 * b, 2) for a, b in ba.items()}, f"flat == stored Z lr10 {ref_ba(i):.6f}", f"fit {fit:.0f}s total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
