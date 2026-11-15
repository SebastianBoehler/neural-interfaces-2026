"""s278 step 3: Shin 2018 word generation (WG) as extra WORD-class people for core1, core level, 16-channel intersect with PROTEUS.
Mirrors studies/14nov/dreyer/dreyer.py (core1 recipe lr10, LOSO over the ten PROTEUS people via all_loso). Shin windows: the whole recording of a
person is one session (WG + baseline mixed) and is Feats-centred as such; the baseline rows are then dropped, WG rows enter as class 2 (WORD).
Arms (one fit each, per held-out person): a0 PROTEUS only (0 Shin rows); a1w<w> + Shin WG as WORD, session-centred; a2w<w> + Shin shifted by
m_WORD (mean feature vector of the fold's PROTEUS training WORD windows); w in 0.1, 0.3, 1. a1w0 (weight 0) must equal a0 (zero-change guard).
Usage: shin278.py prep | fold I. SHIN_SMOKE=n uses only the first n Shin people (own key, own dir)."""
import glob, hashlib, json, os, pickle, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
REPO = os.environ.get("NI26_ROOT") or os.path.abspath(__file__ + "/../../../..")
os.environ["NI26_ROOT"] = REPO
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402

SFILE = "data/shin2018/windows.npz"
CFG = "car_ov17_fs2_t0.1_le_k2"
WORD = 2  # PROTEUS class 2 = word generation (OFFICIAL_CONTRACT); Shin y 1 = WG, 0 = baseline
SMOKE = int(os.environ.get("SHIN_SMOKE", "0"))
ARMS = {"a0": dict(dr=False, shift=False, w=0.0), "a1w0": dict(dr=True, shift=False, w=0.0)}
for _w in (0.1, 0.3, 1):
    ARMS[f"a1w{_w}"] = dict(dr=True, shift=False, w=float(_w)); ARMS[f"a2w{_w}"] = dict(dr=True, shift=True, w=float(_w))
SETTINGS = dict(cfg=CFG, sfile=SFILE, word=WORD, head="lr10", smoke=SMOKE, arms=ARMS)


def key(chans):
    return hashlib.sha1(json.dumps([list(chans), SETTINGS, os.path.getsize(f"{REPO}/{SFILE}"), T.BANDSETS["ov17"]]).encode()).hexdigest()[:12]


def out_dir():
    return f"{REPO}/runs/shin278/S16{'_smoke%d' % SMOKE if SMOKE else ''}"


def load_shin():
    """Shin windows on the 16 recorded channels (PROTEUS order), WG and baseline kept, one stream per person; layout asserted identical to PROTEUS."""
    pro = [str(c) for c in np.load(protocol.DATA)["ch_names"]]
    with np.load(f"{REPO}/{SFILE}", allow_pickle=True) as w:
        assert [str(c) for c in w["ch_names"]] == pro, "channel layout differs from PROTEUS"
        assert w["X"].shape[-1] == 480 and [str(c) for c in w["classes"]] == ["baseline", "WG"]
        rec = {str(c) for c in w["recorded"]}; chans = [i for i, c in enumerate(pro) if c in rec]
        subj = w["subject"].astype(str); keep = np.ones(len(subj), bool)
        if SMOKE:
            keep = np.isin(subj, sorted(set(subj))[:SMOKE])
        assert len(set(w["session"].astype(str))) == 1
        d = dict(X=w["X"][keep][:, chans].astype(np.float32), subject=subj[keep], wg=w["y"][keep] == 1)
    assert len(chans) == 16
    d["y"] = np.full(len(d["X"]), WORD); d["ch_names"] = np.array([pro[i] for i in chans]); d["stream"] = d["subject"]
    return d, chans


def prep():
    t0 = time.time(); d = protocol.load(); cfg = T.parse(CFG); sh, chans = load_shin()
    assert [d["ch_names"][i] for i in chans] == list(sh["ch_names"]), "channel intersect differs between corpora"
    k = key(chans); os.makedirs(out_dir(), exist_ok=True)
    fe = T.Feats(dict(d, X=d["X"][:, chans], ch_names=d["ch_names"][chans]), cfg)
    with open(f"{out_dir()}/pro_{k}.pkl", "wb") as f:
        pickle.dump(dict(key=k, state=fe.__dict__), f, protocol=4)
    fs = T.Feats(sh, cfg); assert fs.F.shape[1] == fe.F.shape[1]
    np.save(f"{out_dir()}/shin_{k}.npy", fs.F[sh["wg"]])  # centred with baseline mixed in, baseline rows then dropped
    print(f"prep key {k} dim {fe.F.shape[1]} chans 16 pro {len(fe.F)} shin WG {int(sh['wg'].sum())} of {len(fs.F)} people {len(set(sh['subject']))} {time.time() - t0:.0f}s", flush=True)


class Model:
    """One arm: weighted scaler + LR C 10 on PROTEUS training rows (+ Shin WG rows as WORD); causal test features as core1's Core."""

    def __init__(self, fe, idx, arm, Fs, m):
        self.fe = fe
        self.priors = [fe.prior(b, idx) for b in range(len(fe.C))]  # PROTEUS training rows in every arm
        self.d2p = [d2[idx].mean() for d2 in fe.d2]; self.szp = fe.sz_prior(idx)
        X, y, w = fe.F[idx], fe.y[idx], np.ones(len(idx)); self.n_sh = 0
        if arm["dr"]:
            assert Fs is not None
            X = np.concatenate([X, Fs + (m if arm["shift"] else 0.0)]); y = np.concatenate([y, np.full(len(Fs), WORD)])
            w = np.concatenate([w, np.full(len(Fs), arm["w"])]); self.n_sh = len(Fs)
        else:
            assert len(X) == len(idx), "arm without Shin carries Shin rows"
        self.sc = StandardScaler().fit(X, sample_weight=w)
        self.lr = LogisticRegression(C=10.0, max_iter=2000).fit(self.sc.transform(X), y, sample_weight=w)
        assert self.lr.n_iter_.max() < 2000, "LR did not converge in 2000 iterations"

    def predict_proba_stream(self, idx):
        return self.lr.predict_proba(self.sc.transform(self.fe.causal(idx, self.priors, self.d2p, self.szp)))


def fold(i):
    d = protocol.load(); od = out_dir(); fe = T.Feats.__new__(T.Feats)
    cands = glob.glob(f"{od}/pro_*.pkl"); assert len(cands) == 1, f"expected one PROTEUS cache in {od}: {cands}"
    with open(cands[0], "rb") as f:
        st = pickle.load(f)
    fe.__dict__.update(st["state"]); k = st["key"]
    _, chans = load_shin()
    assert key(chans) == k, "cache key does not match the current settings (arms, smoke, data, bands)"
    Fs = np.load(f"{od}/shin_{k}.npy")
    name, tr, te = list(protocol.folds(d, "all_loso"))[i]
    idx = np.flatnonzero(tr); held = d["subject"][te][0]
    assert not np.isin(d["subject"][idx], [held]).any() and (d["split"][idx] != "external").all()
    rows = idx[d["y"][idx] == WORD]; assert len(rows) > 0 and not (d["subject"][rows] == held).any()
    m = fe.F[rows].mean(0)  # fold's training people only
    res = dict(key=k, person=name, arms=ARMS, ba={}, word_recall={}, pred_share={}, n_train={}, fit_s={}, m_norm=float(np.linalg.norm(m)))
    probs = {}; yt = d["y"][te]
    for a, arm in ARMS.items():
        t0 = time.time(); mod = Model(fe, idx, arm, Fs, m)
        P = protocol.predict_streams(mod, d, te, proba=True); pr = P.argmax(1)
        res["ba"][a] = protocol.score(yt, pr, d, te)["balanced_accuracy_cell_mean"]
        res["word_recall"][a] = float((pr[yt == WORD] == WORD).mean())
        res["pred_share"][a] = np.bincount(pr, minlength=3).tolist(); res["n_train"][a] = [len(idx), mod.n_sh]
        res["fit_s"][a] = time.time() - t0; probs[a] = P
        print(name, a, f"BA {100 * res['ba'][a]:.2f} WORDrec {100 * res['word_recall'][a]:.1f} pred {res['pred_share'][a]} n {res['n_train'][a]} {res['fit_s'][a]:.0f}s", flush=True)
    res["w0_maxdiff"] = float(np.abs(probs["a1w0"] - probs["a0"]).max())
    json.dump(res, open(f"{od}/fold{i}.json", "w"))
    np.savez_compressed(f"{od}/fold{i}.npz", idx=np.flatnonzero(te), **probs)


if __name__ == "__main__":
    {"prep": prep, "fold": lambda: fold(int(sys.argv[2]))}[sys.argv[1]]()
