"""MI-transfer arm, step 2 (1 Nov): the external-head teacher as one more R19e member.

Usage: python scripts/diag/eval_external_teacher.py BASE_CACHE HEADS_NPZ OUT_JSON
HEADS_NPZ comes from screen_external_heads.py (per-checkpoint log-probs of the external heads,
causal EA). Teacher = logistic map (C 0.1, balanced) from seed-averaged head log-probs to the
3 classes, fit on each fold's training participants only (dla: the other nine; ds: dev rows
outside P10 sessions 2-3; al: the other nine). No session state, so it is a stateless member,
packageable as TorchScript (trunk + heads + map). It enters the member mix with weight w (the
plain members have 1), then the unchanged rebalancing and R19e stage (lam 0.5, n0 60, m0 20).
Pre-registered primary: family sinc+fbcnet, w 1. Seed halves use the matching pretraining seeds.
Env (2 Nov): NI26_HALVES "name=lo-hi,..." (default s02/s35/s05), NI26_C (default 0.1), NI26_WTS,
NI26_MW (member weights with the teacher; deltas stay against the unchanged R19e), and
NI26_PROBE=1 for trunk embeddings: one standardised map per net, probabilities averaged.
NI26_ADD "spec,spec" (mix_probs specs, e.g. eegnex_ea_ptm7r_rot05:0-5*1): extra members on
top of R15, mixed with the teacher; w 0 then gives the members' effect alone.
NI26_T2 "prefix+prefix@w2" (wave 8): a second teacher with its own logistic map, mixed as
its own member at weight w2 (late fusion, as two Jit members would be packaged).
NI26_VIEW (2 Nov, code-freeze exploration): the R19e stage also sees the teacher, per
session and label-free. "heads" appends the teacher's head log-probs to the gaze/event
view, "tmap" its 3-class log-probs, "heads2" runs a second EM on the head log-probs alone;
NI26_EMLAM sets the stage's lam (default 0.5, as R19e).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts/diag")]
from diag_common import R15, cells, cm, d, load, mixp, rebal  # noqa: E402
from mix_probs import parse  # noqa: E402
import session_em as SE  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

WS = [1, 0.5, 0.5, 1, 1, 1, 1, 1, 1]  # R15 member weights (base_cache.py spec order)
HALF = {h: range(int(a), int(b) + 1) for h, a, b in (
    (k, *r.split("-")) for k, r in (x.split("=") for x in os.environ.get(
        "NI26_HALVES", "s02=0-2,s35=3-5,s05=0-5").split(",")))}
C, PROBE = float(os.environ.get("NI26_C", 0.1)), bool(os.environ.get("NI26_PROBE"))
WTS = tuple(float(w) for w in os.environ.get("NI26_WTS", "0.5,1,2,3,4").split(","))
MW = [float(w) for w in os.environ["NI26_MW"].split(",")] if "NI26_MW" in os.environ else WS
ADD = [s for s in os.environ.get("NI26_ADD", "").split(",") if s]
T2F, _, T2W = os.environ.get("NI26_T2", "").partition("@")
VIEW, EMLAM = os.environ.get("NI26_VIEW", ""), float(os.environ.get("NI26_EMLAM", 0.5))
NCOL = int(os.environ.get("NI26_NCOL", 0)) or None  # wave 10m: first heads only (19 = the m7 corpora)
FAMS = {"sinc": ["sincshallow_ea-m7r"], "fbcnet": ["fbcnet_ea-m7r"],
        "sinc+fbcnet": ["sincshallow_ea-m7r", "fbcnet_ea-m7r"]}
st, y, subj = d["stream"], d["y"], d["subject"]
DEV = ~np.isin(subj, [f"Dreyer2026Proteus/{p}" for p in ("P04", "P09")])


def features(H, fams, seeds):
    if PROBE:  # embeddings of different nets do not share axes: one map per net
        return [H[f"{f}-s{s}"] for f in fams for s in seeds]
    return [np.concatenate([np.mean([H[f"{f}-s{s}"][:, :NCOL] for s in seeds], 0) for f in fams], 1)]


def fit_predict(Fs, tr, te):
    lr = lambda: LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    m = (lambda: make_pipeline(StandardScaler(), lr())) if PROBE else lr
    return np.mean([m().fit(F[tr], y[tr]).predict_proba(F[te]) for F in Fs], 0)


def teacher(Fs, proto, idx):
    T = np.zeros((len(y), 3))
    if proto == "dev_session":
        tr = DEV & ~np.isin(np.arange(len(y)), idx)
        T[idx] = fit_predict(Fs, tr, idx)
        return T[idx]
    for p in sorted(set(subj[idx])):
        te = subj == p
        T[te] = fit_predict(Fs, ~te, te)
    return T[idx]


def by_participant(idx, pred):
    out = {}
    for (s, _), v in cells(idx, pred).items():
        out.setdefault(s.split("|")[0].split("/")[-1], []).append(v)
    return {p: float(np.mean(v)) for p, v in out.items()}


def main():
    B, H, out = np.load(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3])
    fams = dict(a.split("=") for a in sys.argv[4:]) or FAMS  # NAME=prefix+prefix (controls)
    fams = {k: v.split("+") if isinstance(v, str) else v for k, v in fams.items()}
    GE = np.c_[B["G"], B["EV"]]
    res = {}
    for fam, fl in fams.items():
        missing = [f"{f}-s{s}" for f in fl for s in HALF["s05" if "s05" in HALF else next(iter(HALF))]
                   if f"{f}-s{s}" not in H.files]
        if missing:  # e.g. an architecture whose pretraining failed
            print(fam, "skipped, missing", missing[:3], flush=True)
            continue
        for half, seeds in HALF.items():
            for proto in ("dev_loso_all", "dev_session", "all_loso"):
                k = f"{half if f'{half}_{proto}_idx' in B.files else 's05'}_{proto}"  # s611: s05 base
                idx, M, R19e = B[k + "_idx"], list(B[k + "_members"]), B[k + "_R19e"]
                base = rebal(mixp(M, WS), idx)
                assert np.allclose(base, B[k + "_R15"]), "member weights do not reproduce R15"
                MT = MW
                if ADD:  # extra members after R15's, same rows (load asserts the order)
                    _, Ps, _, ix, _ = load(proto, R15 + ADD)
                    assert (ix == idx).all(), "added members do not cover the cached rows"
                    M, MT = Ps, MW + [parse(s)[2] for s in ADD]
                T = teacher(features(H, fl, seeds), proto, idx)
                if T2F:  # second teacher, own map, own member slot
                    M, MT = M + [teacher(features(H, T2F.split("+"), seeds), proto, idx)], MT + [float(T2W)]
                r = res.setdefault(fam, {}).setdefault(half, {}).setdefault(proto, {})
                r["R15"], r["R19e"] = round(cm(idx, base.argmax(1)), 2), round(cm(idx, R19e.argmax(1)), 2)
                r["teacher"] = round(cm(idx, T.argmax(1)), 2)
                ref = by_participant(idx, R19e.argmax(1))
                V, Fh = GE[idx], features(H, fl, seeds)[0][idx] if VIEW.startswith("heads") else None
                if VIEW in ("heads", "tmap"):
                    V = np.c_[V, Fh if VIEW == "heads" else np.log(T + 1e-6)]
                for w in WTS:
                    b = rebal(mixp(M + [T], MT + [w]), idx)
                    e = SE.apply(V, b, st[idx], lam=EMLAM, n0=60, m0=20)
                    if VIEW == "heads2":
                        e = SE.apply(Fh, e, st[idx], lam=EMLAM, n0=60, m0=20)
                    pp = by_participant(idx, e.argmax(1))
                    r[f"w{w:g}"] = {"R15+T": round(cm(idx, b.argmax(1)), 2),
                                    "R19e+T": round(cm(idx, e.argmax(1)), 2),
                                    "per_participant_delta": {p: round(pp[p] - ref[p], 2) for p in ref}}
            for w in (f"w{w:g}" for w in WTS):
                s = {p: res[fam][half][p] for p in ("dev_loso_all", "dev_session")}
                sel0 = np.mean([s[p]["R19e"] for p in s])
                sel1 = np.mean([s[p][w]["R19e+T"] for p in s])
                res[fam][half][f"sel_delta_{w}"] = round(float(sel1 - sel0), 2)
            ws = [f"w{w:g}" for w in WTS]
            print(fam, half, {w: res[fam][half][f"sel_delta_{w}"] for w in ws},
                  "dla", res[fam][half]["dev_loso_all"]["R19e"], "->",
                  {w: res[fam][half]["dev_loso_all"][w]["R19e+T"] for w in ws},
                  "ds", res[fam][half]["dev_session"]["R19e"], "->",
                  {w: res[fam][half]["dev_session"][w]["R19e+T"] for w in ws}, flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
