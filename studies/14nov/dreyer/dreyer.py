"""s270: Dreyer 2023 as extra MI people for core1, core level, 27-channel intersect (ledger s270).
Core1's recipe unchanged (ts_core2 Feats: CAR, trace shrinkage 0.1, log-Euclidean session reference k 2, tangent space, slow block;
head lr10 = scaler + LR C 10), LOSO over the ten PROTEUS people (all_loso). Both corpora on the 27 channels Dreyer recorded
(the other 14 of the 41-channel layout are interpolated in the export); CAR over those 27. Dreyer rows: class 0 (MI), each
session recentred at its own mean (the Feats whitening), so arm 1 replicates the 19 Oct collapse set-up.
Arms (one fit each, per held-out person):
  a0     PROTEUS only (reference, 0 Dreyer rows)
  a1     + Dreyer as MI, session-centred, weight 1
  a2w<w> + Dreyer shifted by m_MI (mean of the fold's PROTEUS training people's MI-window feature vectors), weight w in 1, 0.3, 0.1
Test-time causal references use the PROTEUS training rows' prior in every arm (Dreyer enters as training rows only).
Variants: Z41 = PROTEUS only on all 41 channels (zero-change against core1's stored all_loso lr10), D27 = the study.
Usage: dreyer.py prep VARIANT | fold VARIANT I. DREYER_SMOKE=n uses only the first n Dreyer people (own key, own dir)."""
import glob, hashlib, json, os, pickle, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
REPO = os.environ.get("NI26_ROOT") or os.path.abspath(__file__ + "/../../../..")
os.environ["NI26_ROOT"] = REPO
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402

DFILES = ["data/dreyer2023/windows_1-44.npz", "data/dreyer2023/windows_45-87.npz"]
CFG = "car_ov17_fs2_t0.1_le_k2"
MI = 0  # PROTEUS class 0 = kinesthetic motor imagery (OFFICIAL_CONTRACT)
SMOKE = int(os.environ.get("DREYER_SMOKE", "0"))
ARMS = {"a0": dict(dr=False, shift=False, w=0.0), "a1": dict(dr=True, shift=False, w=1.0),
        "a2w1": dict(dr=True, shift=True, w=1.0), "a2w0.3": dict(dr=True, shift=True, w=0.3),
        "a2w0.1": dict(dr=True, shift=True, w=0.1)}
SETTINGS = dict(cfg=CFG, dfiles=DFILES, mi=MI, head="lr10", smoke=SMOKE)


def key(variant, chans):
    sz = [(f, os.path.getsize(f"{REPO}/{f}")) for f in DFILES] if variant == "D27" else []
    return hashlib.sha1(json.dumps([variant, list(chans), SETTINGS, sz, T.BANDSETS["ov17"]]).encode()).hexdigest()[:12]


def out_dir(variant):
    return f"{REPO}/runs/dreyer270/{variant}{'_smoke%d' % SMOKE if SMOKE else ''}"


def load_dreyer():
    """Dreyer windows from both exports, 27 recorded channels; asserts the intersect is identical in both corpora."""
    pro_names = [str(c) for c in np.load(protocol.DATA)["ch_names"]]
    cols, interp = {k: [] for k in ("X", "subject", "session")}, None
    for f in DFILES:
        with np.load(f"{REPO}/{f}", allow_pickle=True) as w:
            assert [str(c) for c in w["ch_names"]] == pro_names, f"{f}: channel layout differs from PROTEUS"
            ip = sorted(str(c) for c in w["interpolated"])
            assert interp in (None, ip), "interpolated sets differ between files"
            interp = ip
            assert w["X"].shape[-1] == 480, f"{f}: window length {w['X'].shape[-1]} != 480"
            print(f, "classes", [str(c) for c in w["classes"]], "labels", np.unique(w["y"]).tolist(), flush=True)
            chans = [i for i, c in enumerate(pro_names) if c not in interp]
            subj = w["subject"].astype(str)
            keep = np.ones(len(subj), bool)
            if SMOKE:
                keep = np.isin(subj, sorted(set(subj))[:SMOKE])
            cols["X"].append(w["X"][keep][:, chans].astype(np.float32))
            cols["subject"].append(subj[keep]); cols["session"].append(w["session"][keep].astype(str))
    names = [pro_names[i] for i in chans]
    assert len(names) == 27 and len(interp) == 14 and set(names).isdisjoint(interp)
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d["y"] = np.full(len(d["X"]), MI); d["ch_names"] = np.array(names)
    d["stream"] = np.char.add(np.char.add(d["subject"], "|"), d["session"])
    return d, chans


def prep(variant):
    t0 = time.time(); d = protocol.load(); cfg = T.parse(CFG)
    if variant == "Z41":
        chans, dr = list(range(41)), None
    else:
        dr, chans = load_dreyer()
        assert [d["ch_names"][i] for i in chans] == list(dr["ch_names"]), "channel intersect differs between corpora"
    k = key(variant, chans); os.makedirs(out_dir(variant), exist_ok=True)
    dp = dict(d, X=d["X"][:, chans], ch_names=d["ch_names"][chans])
    fe = T.Feats(dp, cfg)
    with open(f"{out_dir(variant)}/pro_{k}.pkl", "wb") as f:
        pickle.dump(dict(key=k, state=fe.__dict__), f, protocol=4)
    nd = 0
    if dr is not None:
        fd = T.Feats(dr, cfg)
        assert fd.F.shape[1] == fe.F.shape[1]
        np.save(f"{out_dir(variant)}/dr_{k}.npy", fd.F); nd = len(fd.F)
        print(f"Dreyer: {nd} windows, {len(set(dr['subject']))} people, {len(set(dr['stream']))} sessions", flush=True)
    print(f"prep {variant} key {k} dim {fe.F.shape[1]} chans {len(chans)} pro {len(fe.F)} dr {nd} {time.time() - t0:.0f}s", flush=True)


class Model:
    """One arm: weighted scaler + LR C 10 on PROTEUS training rows (+ Dreyer rows); causal test features as core1's Core."""

    def __init__(self, fe, idx, arm, Fd, m):
        self.fe = fe
        self.priors = [fe.prior(b, idx) for b in range(len(fe.C))]  # PROTEUS training rows in every arm
        self.d2p = [d2[idx].mean() for d2 in fe.d2]; self.szp = fe.sz_prior(idx)
        X, y, w = fe.F[idx], fe.y[idx], np.ones(len(idx))
        self.n_dr = 0
        if arm["dr"]:
            assert Fd is not None and arm["w"] > 0
            X = np.concatenate([X, Fd + (m if arm["shift"] else 0.0)]); y = np.concatenate([y, np.full(len(Fd), MI)])
            w = np.concatenate([w, np.full(len(Fd), arm["w"])]); self.n_dr = len(Fd)
        else:
            assert len(X) == len(idx), "arm without Dreyer carries Dreyer rows"
        self.sc = StandardScaler().fit(X, sample_weight=w)
        self.lr = LogisticRegression(C=10.0, max_iter=2000).fit(self.sc.transform(X), y, sample_weight=w)
        assert self.lr.n_iter_.max() < 2000, "LR did not converge in 2000 iterations"

    def predict_proba_stream(self, idx):
        return self.lr.predict_proba(self.sc.transform(self.fe.causal(idx, self.priors, self.d2p, self.szp)))


def fold(variant, i):
    d = protocol.load(); od = out_dir(variant)
    fe = T.Feats.__new__(T.Feats)
    cands = glob.glob(f"{od}/pro_*.pkl"); assert len(cands) == 1, f"expected one PROTEUS cache in {od}: {cands}"
    with open(cands[0], "rb") as f:
        st = pickle.load(f)
    fe.__dict__.update(st["state"]); k = st["key"]
    chans = list(range(41)) if variant == "Z41" else [j for j, c in enumerate(d["ch_names"]) if c in set(load_dreyer_names())]
    assert key(variant, chans) == k, "cache key does not match the current settings"
    Fd = np.load(f"{od}/dr_{k}.npy") if variant == "D27" else None
    name, tr, te = list(protocol.folds(d, "all_loso"))[i]
    idx = np.flatnonzero(tr); held = d["subject"][te][0]
    assert not np.isin(d["subject"][idx], [held]).any() and (d["split"][idx] != "external").all()
    rows = idx[d["y"][idx] == MI]; assert len(rows) > 0 and not (d["subject"][rows] == held).any()
    m = fe.F[rows].mean(0)  # fold's training people only
    arms = {"a0": ARMS["a0"]} if variant == "Z41" else ARMS
    res = dict(variant=variant, key=k, person=name, arms=arms, ba={}, pred_share={}, n_train={}, fit_s={}, m_norm=float(np.linalg.norm(m)))
    probs = {}
    for a, arm in arms.items():
        t0 = time.time(); mod = Model(fe, idx, arm, Fd, m)
        P = protocol.predict_streams(mod, d, te, proba=True)
        res["ba"][a] = protocol.score(d["y"][te], P.argmax(1), d, te)["balanced_accuracy_cell_mean"]
        res["pred_share"][a] = np.bincount(P.argmax(1), minlength=3).tolist(); res["n_train"][a] = [len(idx), mod.n_dr]
        res["fit_s"][a] = time.time() - t0; probs[a] = P
        print(variant, name, a, f"BA {100 * res['ba'][a]:.2f} pred {res['pred_share'][a]} n {res['n_train'][a]} {res['fit_s'][a]:.0f}s", flush=True)
    json.dump(res, open(f"{od}/fold{i}.json", "w"))
    np.savez_compressed(f"{od}/fold{i}.npz", idx=np.flatnonzero(te), **probs)


def load_dreyer_names():
    pro = [str(c) for c in np.load(protocol.DATA)["ch_names"]]
    with np.load(f"{REPO}/{DFILES[0]}", allow_pickle=True) as w:
        ip = {str(c) for c in w["interpolated"]}
    return [c for c in pro if c not in ip]


if __name__ == "__main__":
    {"prep": lambda: prep(sys.argv[2]), "fold": lambda: fold(sys.argv[2], int(sys.argv[3]))}[sys.argv[1]]()
