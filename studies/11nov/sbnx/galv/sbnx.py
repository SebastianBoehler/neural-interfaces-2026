"""SBNX (11 Nov): SessionBN versions of member families never tried with session normalisation, 3-class, per fold of
all_loso + seen_session. Recipe = the FB slot's (SessionNet group 32, EA k5, 40 epochs, AdamW one-cycle, rot 0.05).
Saved proba (n, 3) at SessionBN k 2/5/20, rows sorted by index. A failing architecture logs FAIL and exits 0.
Usage: python sbnx.py TASK (seed = TASK // len(ARCHS), arch = ARCHS[TASK % len(ARCHS)]) | python sbnx.py smoke
"""
import sys, time, traceback
from pathlib import Path
WS = Path("/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026")
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402

OUT = WS / "runs/diag/sbnx/probs"
ARCHS = ["tidnet", "ctnet", "conformer", "atcnet", "simpleconv", "fbmsnet", "sccnet", "deep4"]
KS = (2.0, 5.0, 20.0)


def run(arch, d, proto, seed, out, epochs=0):
    t = f"sbnx_{arch}"
    if all((out / f"{t}_k{int(k)}-{proto}-s{seed}.npz").exists() for k in KS):
        return print("exists", t, proto, seed, flush=True)
    got = {k: ([], []) for k in KS}; t0 = time.time()
    for fold, tr, te in protocol.folds(d, proto):
        m = SB.SessionNet(d, arch=arch, seed=seed, group=32); m.rot = 0.05
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr))
        assert m.n_sbn > 0, "no BatchNorm layer to swap"
        for k in KS:
            m.set_k(k)
            got[k][0].append(np.flatnonzero(te)), got[k][1].append(protocol.predict_streams(m, d, te, proba=True))
        print(f"{t} s{seed} {proto} {fold} n_sbn={m.n_sbn} {time.time() - t0:.0f}s", flush=True)
        if epochs:
            break
    for k in KS:
        idx, P = np.concatenate(got[k][0]), np.concatenate(got[k][1]); o = np.argsort(idx); idx, P = idx[o], P[o]
        assert np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-4)
        np.savez(out / f"{t}_k{int(k)}-{proto}-s{seed}.npz", proba=P, index=idx)
        print(f"TOTAL {t}_k{int(k)} s{seed} {proto} n={len(idx)} acc={np.mean(P.argmax(1) == d['y'][idx]):.4f} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load()
    if sys.argv[1] == "smoke":
        out = WS / "runs/diag/sbnx/smoke"; out.mkdir(parents=True, exist_ok=True)
        for arch in ARCHS:
            try:
                run(arch, d, "seen_session", 0, out, epochs=1); print("SMOKE PASS", arch, flush=True)
            except Exception:
                print("SMOKE FAIL", arch, traceback.format_exc()[-600:], flush=True)
        return
    task = int(sys.argv[1]); arch, seed = ARCHS[task % len(ARCHS)], task // len(ARCHS)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        for proto in ("all_loso", "seen_session"):
            run(arch, d, proto, seed, OUT)
    except Exception:
        print("FAIL", arch, seed, traceback.format_exc()[-1500:], flush=True)


if __name__ == "__main__":
    main()
