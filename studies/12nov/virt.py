"""SYNTH (ledger s47): core1 with virtual copies of every training session. MODE=none|rot|cap|mix|gmix|fsc PARAM=<float> [COPIES=2]
Usage: MODE=cap PARAM=6 python studies/12nov/virt.py"""
import os, sys, time, warnings
import numpy as np
from scipy.linalg import expm as sexpm
from scipy.spatial.transform import Rotation
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
EXP = REPO + "/studies/12nov/exp"
sys.path[:0] = [EXP if os.path.isdir(EXP) else REPO + "/experiments"]
import protocol  # noqa: E402
import features as FT  # noqa: E402
import ts_core2 as T  # noqa: E402
MODE, PARAM, COPIES = os.environ.get("MODE", "none"), float(os.environ.get("PARAM", 0)), int(os.environ.get("COPIES", 2))
d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); y = d["y"]; F32 = fe.F.astype(np.float32)
per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str); itf = d["interface"]; sid = np.char.add(per, ses)
PEOPLE = sorted(set(per)); NB = len(fe.C); IU = np.triu_indices(41); WT = np.where(IU[0] == IU[1], 1.0, np.sqrt(2.0)).astype(np.float32); T0 = time.time()
if MODE == "fsc":  # the same sessions with every band edge scaled: a person whose rhythms run slower / faster
    FSC = []
    for tag, f in (("fslo", 1 / (1 + PARAM)), ("fshi", 1 + PARAM)):
        T.BANDSETS[tag] = [(lo * f, min(hi * f, 59.0)) for lo, hi in T.BANDSETS["ov17"]]
        g = T.Feats(d, T.parse(f"car_{tag}_fs2_t0.1_le_k2")); FSC.append(g.F.astype(np.float32)); del g
if MODE == "cap":
    import mne
    from mne.channels.interpolation import _make_interpolation_matrix
    names = [str(c) for c in np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)["ch_names"]]
    cp = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]; allp = np.array(list(cp.values()))
    Aq = np.c_[2 * allp, np.ones(len(allp))]; cen = np.linalg.lstsq(Aq, (allp ** 2).sum(1), rcond=None)[0][:3]  # sphere fit
    POS = np.array([cp[n] for n in names]) - cen; POS /= np.linalg.norm(POS, axis=1, keepdims=True)


def rot_copy(rows, rng):
    G = rng.standard_normal((41, 41)); R = sexpm(PARAM * (G - G.T) / np.sqrt(2 * 41)).astype(np.float32); out = F32[rows].copy()
    for b in range(NB):
        sl = fe.slices[b]; L = np.zeros((len(rows), 41, 41), np.float32); V = out[:, sl] / WT
        L[:, IU[0], IU[1]] = V; L[:, IU[1], IU[0]] = V; L = R @ L @ R.T; out[:, sl] = L[:, IU[0], IU[1]] * WT
    return out


def cap_copy(rows, rng):
    A = _make_interpolation_matrix(POS, POS @ Rotation.from_rotvec(rng.standard_normal(3) * np.deg2rad(PARAM)).as_matrix().T, alpha=1e-11)  # exact at zero shift
    out = np.empty((len(rows), F32.shape[1]), np.float32)
    for b in range(NB):
        C = A @ fe.C[b][rows] @ A.T; out[:, fe.slices[b]] = FT.whiten_tangent(C, T.expm(T.logm(C).mean(0)))
    M = np.einsum("ij,njt->nit", A, fe.M[rows].reshape(len(rows), 41, -1)).reshape(len(rows), -1); out[:, fe.slices[NB]] = M - M.mean(0)
    return out


def ba(r, pred):
    return 100 * np.mean([np.mean([np.mean(pred[(itf[r] == i) & (y[r] == c)] == c) for c in range(3)]) for i in np.unique(itf[r])])


def fold(p):
    idx = np.flatnonzero(per != p); Xtr, ytr = [F32[idx]], [y[idx]]; rng = np.random.default_rng(abs(hash((p, MODE, PARAM))) % 2 ** 32)
    if MODE != "none":
        mu = {(q, k): F32[(per == q) & (y == k)].mean(0) for q in PEOPLE if q != p for k in range(3)} if MODE in ("mix", "gmix") else None
        for s in sorted(set(sid[idx])):
            r = np.flatnonzero(sid == s)
            for ci in range(COPIES):
                if MODE == "fsc":
                    x = FSC[ci % 2][r]
                elif MODE == "gmix":  # a random member of the between-person family: all three class deviations share the weights
                    a = per[r[0]]; others = [q for q in PEOPLE if q != p]; w = rng.standard_normal(len(others)) * PARAM / np.sqrt(len(others) - 1); x = F32[r].copy()
                    for k in range(3):
                        g = np.mean([mu[(q, k)] for q in others], 0); x[y[r] == k] += sum(wi * (mu[(q, k)] - g) for wi, q in zip(w, others)) - (mu[(a, k)] - g)
                elif MODE == "mix":
                    a = per[r[0]]; b = rng.choice([q for q in PEOPLE if q not in (p, a)]); lam = rng.uniform(0, PARAM); x = F32[r].copy()
                    for k in range(3):
                        x[y[r] == k] += lam * (mu[(b, k)] - mu[(a, k)])
                else:
                    x = (rot_copy if MODE == "rot" else cap_copy)(r, rng)
                Xtr.append(x); ytr.append(y[r])
    core = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000)).fit(np.concatenate(Xtr), np.concatenate(ytr))
    pri = [fe.prior(b, idx) for b in range(NB)]; d2p = [q[idx].mean() for q in fe.d2]; a = []
    for s in sorted(set(ses[per == p])):
        r = np.flatnonzero((per == p) & (ses == s)); a.append(ba(r, core.predict_proba(fe.causal(r, pri, d2p, None)).argmax(1)))
    return np.mean(a)


if MODE == "cap":  # sanity: no shift must reproduce the stored features
    r = np.flatnonzero(sid == sid[0]); P0 = PARAM; PARAM = 0.0; e = np.abs(cap_copy(r, np.random.default_rng(0)) - F32[r]).max(); PARAM = P0
    print(f"cap sanity (sigma 0): max abs feature difference {e:.2e}", flush=True)
v = []
for p in PEOPLE:
    v.append(fold(p)); print(f"  {MODE} {PARAM} {p} {v[-1]:.1f} ({time.time() - T0:.0f}s)", flush=True)
print(f"RESULT {MODE} {PARAM} copies {COPIES}: mean {np.mean(v):.2f} | " + " ".join(f"{p[1:]}:{x:.1f}" for p, x in zip(PEOPLE, v)), flush=True)
