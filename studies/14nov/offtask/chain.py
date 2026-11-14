"""OFFTASK chain (ledger s241 step 2a): the v6f chain (shufcombo, env exactly as runs/s215_<order>_off.log) with the weighted
template stages of hooks.py installed by monkeypatching (shufcombo is imported and run unedited, via runpy).
OFFW: W0 = hook with w = 1 (must reproduce runs/chainout/s215_<order>_off.npz), W1 = 1 - p_rest, W3 = (1 - p_rest)^3.
p_rest: seed mean of the rest net's 4th softmax output re-streamed in this order (runs/diag/restprest/<order>/, prest.py).
Outputs: per-window outputs runs/diag/offtask/<OFFW>_<order>.npz (shufcombo SAVE_OUT, recording layout); shufcombo's own
summary file is redirected to runs/diag/offtask/ so the shared runs/shufcombo_*.npz is never overwritten.
Usage: OFFW=W1 python studies/14nov/offtask/chain.py ORDER"""
import glob, os, runpy, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); O = sys.argv[1]; SC_ = O.rstrip("0123456789"); PS = O[len(SC_):]; ARM = os.environ["OFFW"]
GAMMA = {"W0": 0.0, "W1": 1.0, "W3": 3.0}[ARM]; OUTD = f"{REPO}/runs/diag/offtask"; os.makedirs(OUTD, exist_ok=True)
os.environ.update(SCHEME=SC_, PSEED=PS, STRICT="1", GATE_P0="1", OCU_DIR=f"runs/ocushuf/{O}", FT_V="gC5", REC_PFX="ttt7", T6_DIR=f"{REPO}/runs/ttt/probs7",
                  T17_DIR=f"{REPO}/runs/ttt/probs18", FT_TAG=f"ttt18{O}L", SAVE_OUT=f"runs/diag/offtask/{ARM}_{O}.npz", SCHEMES=SC_, PERMS=str(int(PS) + 1))
for k in ("RS_DIR", "CM_NPZ", "EXTRA_VIEW", "RUNOFF", "RUN_IDS", "NUIS", "TSEM_TGT", "ALLSEED", "EYEW", "REST_W", "EM_LAM", "TS_LAM", "BTS", "OCU_TS", "LABEL_OUT", "PROTO"):
    assert k not in os.environ, f"{k} is set: the run would not be the s215 off chain"
sys.path[:0] = [REPO + "/studies/13nov", os.path.dirname(os.path.abspath(__file__))]
import shuffle as S  # noqa: E402  (shufcombo's `import shuffle as S` gets this module, so the hooks reach its stages)
import hooks  # noqa: E402
N = len(S.A.L.Z["y|all_loso"])
if GAMMA:
    fs = sorted(glob.glob(f"{REPO}/runs/diag/restprest/{O}/rest4_sh_r1-all_loso-s*.npz")); assert len(fs) == 3, fs
    for f in fs:
        assert (np.load(f)["index"] == np.arange(N)).all(), f
    pr = np.mean([np.load(f)["p4"][:, 3] for f in fs], 0); w = (1 - pr) ** GAMMA
else:
    w = np.ones(N)
hooks.install(S.gm, w); print(f"OFFTASK {ARM} {O}: template weights w = (1 - p_rest)^{GAMMA:g}, mean {w.mean():.3f}, q10/50/90 {np.percentile(w, [10, 50, 90]).round(3)}", flush=True)
_savez = np.savez


def savez(path, *a, **k):  # shufcombo's summary file goes to our own directory
    p = str(path)
    if os.path.basename(p).startswith("shufcombo_"):
        p = f"{OUTD}/{ARM}_{os.path.basename(p)}"
    return _savez(p, *a, **k)


np.savez = savez
runpy.run_path(f"{REPO}/studies/13nov/shufcombo.py", run_name="__main__")
assert S.gm.em_g.__module__ == "hooks" and S.gm.L.G.tsem_lik.__module__ == "hooks", "the hooks were replaced during the run"
