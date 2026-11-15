"""s280 REVE collect: per-fold files runs/diag/reve280/<form>_s<seed>_<held>.npz -> runs/diag/reve280_probs18/ttt18<order>d<1|2>_<form>_base-all_loso-s<seed>.npz
(proba, index sorted; d1 and d2 are the same net/stream: the base posteriors do not depend on the draw, as for every s271 family)."""
import sys
import numpy as np
import reve_fm as F
import protocol


def main():
    d = protocol.load(); n = len(d["y"]); held = [h for h, _, _ in protocol.folds(d, "all_loso")]; D = F.WS / "runs/diag/ttt/probs18"; O = F.OUT.parent / "reve280_probs18"; O.mkdir(parents=True, exist_ok=True)
    ref = np.load(D / "ttt18within0d1_fb_base-all_loso-s0.npz")["index"]; made = 0
    for form in F.NAMES:
        for s in range(3):
            fs = [F.OUT / f"{form}_s{s}_{h}.npz" for h in held]
            if not all(f.exists() for f in fs):
                print("incomplete", form, s, sum(f.exists() for f in fs), "/10"); continue
            z = [np.load(f) for f in fs]; idx = np.concatenate([a["rows"] for a in z]); o = np.argsort(idx)
            assert (idx[o] == ref).all() and len(idx) == n
            for od in F.ORD:
                P = np.concatenate([a[f"proba_{od}"] for a in z])[o]
                for dr in ("d1", "d2"):
                    np.savez(O / f"ttt18{od}{dr}_{form}_base-all_loso-s{s}.npz", proba=P, index=idx[o])
            made += 1
    print("assembled", made, "of", len(F.NAMES) * 3)
