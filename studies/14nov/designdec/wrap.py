"""DESIGNDEC package form (ledger s239, test ii): studies/13nov/shufcombo.py run unmodified through runpy, with stage.py's
joint stage installed in gm, once per variant in one process (the chainout env: STRICT, GATE_P0, gC5, ttt18<order>L).
shufcombo's own summary npz (runs/shufcombo_*.npz) is redirected into runs/designdec/ so the regular files are never
overwritten; SAVE_OUT (P0r / V6Fr / V6r per-window outputs) goes to runs/designdec/sc_<order>_<fb>_<bnd>.npz.
Variant off/J must reproduce runs/chainout/<order>.npz for the bases run (checked by score.py).
Usage: [BASES=B2_pool] python studies/14nov/designdec/wrap.py <order> off/J,rep/J,ext/J,rep/true,ext/true[,ext/J/ts (stage: both | em | ts)]"""
import os, runpy, sys, time
order, variants = sys.argv[1], sys.argv[2].split(","); sc = order.rstrip("0123456789"); REPO = os.path.abspath(__file__ + "/../../../..")
env = dict(SCHEME=sc, PSEED=order[len(sc):], STRICT="1", GATE_P0="1", OCU_DIR=f"runs/ocushuf/{order}", FT_V="gC5", REC_PFX="ttt7",
           T6_DIR="runs/ttt/probs7", T17_DIR="runs/ttt/probs18", FT_TAG=f"ttt18{order}L", PYTHONWARNINGS="ignore")
os.environ.update({k: os.environ.get(k, v) for k, v in env.items()}); os.environ.setdefault("BASES", "B2_pool"); os.chdir(REPO)  # input dirs overridable (Leonardo: runs/diag/...)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import stage as ST  # noqa: E402
OUT = f"{REPO}/runs/designdec"; os.makedirs(OUT, exist_ok=True); _savez = np.savez


def savez(path, *a, **k):
    p = str(path)
    if os.path.basename(p).startswith("shufcombo_"):
        p = f"{OUT}/{TAG}_{os.path.basename(p)}"
    assert p.startswith(OUT), f"wrap.py would write outside runs/designdec: {p}"
    return _savez(p, *a, **k)


np.savez = savez; ST.install()
for v in variants:
    fb, bnd, *stg = v.split("/"); ST.CFG.update(fb=fb, bnd=bnd, stg=stg[0] if stg else "both"); ST.check()
    TAG = f"sc_{order}_{fb}_{bnd}" + (f"_{stg[0]}" if stg else "")
    os.environ["SAVE_OUT"] = f"runs/designdec/{TAG}.npz"; ST.STATE.update(latch=[], calls=0); t0 = time.time()
    print(f"=== DESIGNDEC wrap {order} {v} (BASES {os.environ['BASES']})", flush=True)
    runpy.run_path(f"{REPO}/studies/13nov/shufcombo.py", run_name="__main__")
    lat = ST.STATE["latch"]; print(f"=== {v}: {time.time() - t0:.0f}s, joint calls {ST.STATE['calls']}, latched {sum(w is not None for w in lat)}/{len(lat)}", flush=True)
