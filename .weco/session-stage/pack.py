"""Build the evaluation pack for the session-stage search and verify the one-file baseline against the chain replay."""
import importlib.util, os, sys
import numpy as np
TASK = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(TASK))
sys.path.insert(0, REPO + "/runs/archive_scratch_11nov/secop")
import env
A = env.A; SC = A.SC; L = A.L
spec = importlib.util.spec_from_file_location("opt", TASK + "/optimize.py"); opt = importlib.util.module_from_spec(spec); spec.loader.exec_module(opt)
out = {}
for p in A.PROTOS:
    out[f"sess|{p}"] = L.Z[f"sess|{p}"]; out[f"X|{p}"] = L.Z[f"pc128|{p}"]; out[f"F|{p}"] = L.Z[f"GE|{p}"]
    out[f"y|{p}"] = SC.B[f"y|{p}"]; out[f"cell|{p}"] = SC.B[f"cell|{p}"]; out[f"person|{p}"] = SC.B[f"person|{p}"]
    for bn in env.BASES:
        out[f"mix|{bn}|{p}"] = A.mix(bn, p, A.FAMS)[0]
for bn in ("B2_pool",):
    for p in A.PROTOS:
        ref = A.shadow(out[f"mix|{bn}|{p}"], p)[1]; ss = out[f"sess|{p}"]; got = np.empty_like(ref)
        for s in np.unique(ss):
            r = np.flatnonzero(ss == s); got[r] = opt.run_session(out[f"mix|{bn}|{p}"][r], out[f"X|{p}"][r], out[f"F|{p}"][r])
        print(f"verify {bn} {p}: max |one-file stage - chain replay| = {np.abs(got - ref).max():.2e}; argmax agreement {np.mean(got.argmax(1) == ref.argmax(1)):.5f}", flush=True)
os.makedirs(TASK + "/pack", exist_ok=True); np.savez(TASK + "/pack/data.npz", **out); print("packed", len(out), "arrays")
