"""WSP cache: everything needed to profile and edit core1's weights without refitting.

core1 = car_ov17_fs2_t0.1_le_k2 + lr10 (17 overlapping bands x {full, half 1, half 2} tangent blocks + slow block,
StandardScaler + logistic regression C 10). For each fold of all_loso (10) and seen_session (2) this saves:
  outer model (scaler mean/scale, coef, intercept), the causal test features of the held-out rows and its posteriors;
  inner models: the same head refitted with one more training person left out (for edits that must not see the
  outer held-out person: transfer of feature groups, weight stability across people).
Also saved once: whole-session (training) features F, labels, person, stream, block slices.
Usage: python experiments/wsp_build.py OUT_DIR
"""
import os, sys, time
import numpy as np
from joblib import Parallel, delayed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402

CFG, HEAD = "car_ov17_fs2_t0.1_le_k2", "lr10"


def pack(m):
    sc, lr = m.steps[0][1], m.steps[1][1]
    return dict(mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32), coef=lr.coef_.astype(np.float32),
                b=lr.intercept_.astype(np.float32), n_iter=int(lr.n_iter_[0]))


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); t0 = time.time()
    d = protocol.load(); fe = T.Feats(d, T.parse(CFG)); y = d["y"]; per = np.array([s[-3:] for s in d["subject"]])
    sl = np.array([[s.start, s.stop] for s in fe.slices])
    np.savez(f"{out}/common.npz", F=fe.F.astype(np.float32), y=y, person=per, stream=d["stream"], slices=sl,
             bands=np.array(T.bandset("ov17"), float))
    print(f"features {fe.F.shape} {time.time() - t0:.0f}s", flush=True)
    for proto in ("all_loso", "seen_session"):
        for name, tr, te in protocol.folds(d, proto):
            idx, rows = np.flatnonzero(tr), np.flatnonzero(te); store = {HEAD: np.full((len(y), 3), np.nan)}
            core = T.Core(fe, [HEAD], store); core.fit(idx)
            Fc = np.zeros((len(rows), fe.F.shape[1]), np.float32); P = np.zeros((len(rows), 3)); st = d["stream"][rows]
            for s in dict.fromkeys(st):
                m = st == s; f64 = fe.causal(rows[m], core.priors, core.d2p, core.szp)
                Fc[m] = f64; P[m] = core.models[0].predict_proba(f64)
            people = sorted(set(per[idx]))
            inner = Parallel(n_jobs=4, prefer="threads")(
                delayed(T.fit_head)(HEAD, fe.F[idx[per[idx] != q]], y[idx[per[idx] != q]], fe.slices) for q in people)
            o = pack(core.models[0]); save = dict(rows=rows, train=idx, Fc=Fc, proba=P.astype(np.float32),
                                                  inner_people=np.array(people), **{f"o_{k}": v for k, v in o.items()})
            for q, m in zip(people, inner):
                save.update({f"i_{q}_{k}": v for k, v in pack(m).items()})
            np.savez(f"{out}/{proto}-{name}.npz", **save)
            print(f"{proto} {name} n_test {len(rows)} acc {np.mean(P.argmax(1) == y[rows]):.4f} iters {o['n_iter']} "
                  f"inner {len(people)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
