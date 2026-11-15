"""TTT18 (13 Nov, ledger s171): gC5 (the fine-tuning form that fits the run time, s120 addendum 3) re-streamed in a
non-chronological delivery order (runs/shuffle_perms.npz), same nets as ttt7 seeds 0-5; scored with shufcombo.py
(FT_KIND=gC5). Saved to runs/diag/ttt/probs18 as ttt18<SCHEME><PSEED>_<arch>_{base,gC5}-<proto>-s<seed>.npz.
CHAIN_NPZ=runs/labels/chain_v51b_<order>.npz: package-faithful pseudo-labels (tag suffix L, s178).
Usage (Leonardo): SCHEME=within PSEED=0 python studies/13nov/ttt18.py TASK | smoke"""
import importlib.util, os
from pathlib import Path
import numpy as np
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); SCHEME = os.environ["SCHEME"]; PSEED = int(os.environ["PSEED"]); DRAW = int(os.environ.get("DRAW", 0))
LB = os.environ.get("CHAIN_NPZ")  # s178: v5.1b's labels as emitted in this order (shufcombo.py LABEL_OUT); must name the same order
LSFX = {f"chain_v51b_{SCHEME}{PSEED}.npz": "L", f"chain_v6r_{SCHEME}{PSEED}.npz": "R", f"chain_rt_{SCHEME}{PSEED}.npz": "T",
        f"chain_stk_{SCHEME}{PSEED}.npz": "K", f"chain_rts_{SCHEME}{PSEED}.npz": "Q",
        f"chain_blend_{SCHEME}{PSEED}.npz": "B"}  # B (s200): labels = mean of the v5.1b and v6-chain outputs  # Q (s199): returning rows routed for every order  # K (s198): new people from the v6 chain, returning people from routed v5.1b where it exists  # T (s190): returning rows labelled by routed v5.1b  # s187: R = labels from the v6 chain (eye + rest)
assert LB is None or os.path.basename(LB) in LSFX, f"label file {LB} is not this order's"
_spec = importlib.util.spec_from_file_location("ttt7", WS / "studies/13nov/ttt7.py"); T7 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T7)
P = np.load(WS / "runs/shuffle_perms.npz"); PERM = {}
for proto in ("all_loso", "seen_session") if SCHEME != "rec" else ():  # s191: SCHEME=rec streams in recording order (PERM None)
    order = P[f"{SCHEME}|{PSEED}|{proto}"]; rk = np.full(int(order.max()) + 10000, -1); rk[order] = np.arange(len(order)); PERM[proto] = rk
PERM = PERM or None
# s191: EXT=<tokens> pools external corpora as extra training people (protocol.load(external=...); folds keep them in training);
# tag X. CTRL=1 is the same run without them (tag C), a retrained control for a paired comparison.
EXT, CTRL = tuple(t for t in os.environ.get("EXT", "").split(",") if t), bool(os.environ.get("CTRL"))
assert not (EXT and CTRL), "EXT and CTRL are the two arms of one comparison"
T7.TAG, T7.OUT, T7.SMOKE_OUT, T7.PERM, T7.DRAW = f"ttt18{SCHEME}{PSEED}" + (f"d{DRAW}" if DRAW else "") + (LSFX[os.path.basename(LB)] if LB else ""), WS / "runs/diag/ttt/probs18", f"smoke18{SCHEME}", PERM, DRAW  # DRAW > 0: fresh augmentation draws
T7.VARS = {} if os.environ.get("BASE_ONLY") else {"gC5": T7.VARS["gC5"]}  # BASE_ONLY (s271): the base posteriors only, no fine-tuning arm; tag unchanged
# s194: SWEEP=1 runs run-time-neutral forms of gC5 side by side (5 steps per refit, every 10 windows, warm, 1 view, GPU draw);
# fields (steps, every, lr, scope, lam, form, warm, views, gpu, t0); gC5 itself is the in-run control. Tag suffix S.
SWEEP = {"gC5": (5, 10, 5e-4, "all", 1.0, "hard", True, 1, True, 30),
         "s01": (5, 10, 2.5e-4, "all", 1.0, "hard", True, 1, True, 30), "s02": (5, 10, 1e-3, "all", 1.0, "hard", True, 1, True, 30),
         "s03": (5, 10, 2e-3, "all", 1.0, "hard", True, 1, True, 30), "s04": (5, 10, 5e-4, "head", 1.0, "hard", True, 1, True, 30),
         "s05": (5, 10, 2e-3, "head", 1.0, "hard", True, 1, True, 30), "s06": (5, 10, 5e-4, "all", 0.0, "hard", True, 1, True, 30),
         "s07": (5, 10, 5e-4, "all", 0.5, "hard", True, 1, True, 30), "s08": (5, 10, 5e-4, "all", 2.0, "hard", True, 1, True, 30),
         "s09": (5, 10, 5e-4, "all", 4.0, "hard", True, 1, True, 30), "s10": (5, 10, 5e-4, "all", 1.0, "soft", True, 1, True, 30),
         "s11": (5, 10, 1e-3, "all", 1.0, "soft", True, 1, True, 30), "s12": (5, 10, 5e-4, "all", 1.0, "hard", True, 1, True, 20),
         "s13": (5, 10, 5e-4, "all", 1.0, "hard", True, 1, True, 40), "s14": (5, 10, 1e-3, "all", 2.0, "hard", True, 1, True, 30),
         "s15": (5, 10, 1e-3, "all", 1.0, "hard", True, 1, True, 20)}
# s197: SWEEP=2 continues along the axis that wave 1 found (higher learning rate), run-time neutral as before. Tag S2.
SWEEP2 = {"gC5": SWEEP["gC5"], "s03": SWEEP["s03"],
          "t01": (5, 10, 3e-3, "all", 1.0, "hard", True, 1, True, 30), "t02": (5, 10, 4e-3, "all", 1.0, "hard", True, 1, True, 30),
          "t03": (5, 10, 6e-3, "all", 1.0, "hard", True, 1, True, 30), "t04": (5, 10, 8e-3, "all", 1.0, "hard", True, 1, True, 30),
          "t05": (5, 10, 2e-3, "all", 2.0, "hard", True, 1, True, 30), "t06": (5, 10, 4e-3, "all", 2.0, "hard", True, 1, True, 30),
          "t07": (5, 10, 2e-3, "all", 1.0, "hard", True, 1, True, 20), "t08": (5, 10, 4e-3, "all", 1.0, "hard", True, 1, True, 20),
          "t09": (5, 10, 2e-3, "all", 1.0, "soft", True, 1, True, 30), "t10": (5, 10, 2e-3, "all", 0.0, "hard", True, 1, True, 30),
          "t11": (5, 10, 3e-3, "all", 2.0, "hard", True, 1, True, 20), "t12": (5, 10, 2e-3, "all", 4.0, "hard", True, 1, True, 30)}
# s198: the stack's combined form (the three axes that helped) and EMA forms; SWEEP=3 is a finer sweep around s03/s14/s15
EXTRA = {"c01": (5, 10, 2e-3, "all", 2.0, "hard", True, 1, True, 20), "e01": SWEEP["s03"], "e02": SWEEP["s03"], "e03": (5, 10, 2e-3, "all", 2.0, "hard", True, 1, True, 20)}
EXTRA["cum10"] = (10, 10, 5e-4, "all", 1.0, "hard", True, 1, True, 30)  # s224: ttt6's cum10 (warm start, 10 steps every 10), GPU-drawn augmentation
EMA_OF = {"e01": 0.5, "e02": 0.8, "e03": 0.8, "u07": 0.8, "u08": 0.9}
SWEEP3 = {"gC5": SWEEP["gC5"], "s03": SWEEP["s03"], "c01": EXTRA["c01"],
          "u01": (5, 10, 1.5e-3, "all", 1.0, "hard", True, 1, True, 30), "u02": (5, 10, 2.5e-3, "all", 1.0, "hard", True, 1, True, 30),
          "u03": (5, 10, 2e-3, "all", 1.5, "hard", True, 1, True, 30), "u04": (5, 10, 2e-3, "all", 3.0, "hard", True, 1, True, 30),
          "u05": (5, 10, 2e-3, "all", 1.0, "hard", True, 1, True, 15), "u06": (5, 10, 2e-3, "all", 1.0, "hard", True, 1, True, 25),
          "u07": SWEEP["s03"], "u08": SWEEP["s03"], "e03": EXTRA["e03"]}
ALL = {**SWEEP, **SWEEP2, **EXTRA, **SWEEP3}
if os.environ.get("SWEEP"):
    sw = os.environ["SWEEP"]; T7.VARS = {"1": SWEEP, "2": SWEEP2, "3": SWEEP3}[sw]; T7.TAG += {"1": "S", "2": "S2", "3": "S3"}[sw]
if os.environ.get("FORMS"):  # s198: a named subset (the stack run); tag suffix F
    T7.VARS = {k: ALL[k] for k in os.environ["FORMS"].split(",")}; T7.TAG += "F"
T7.EMA = {k: d for k, d in EMA_OF.items() if k in T7.VARS}
assert all(k in T7.VARS for k in T7.EMA), "EMA set for a form that does not run"
T7.TAG += "X" if EXT else ("C" if CTRL else "")
if EXT:
    _load = T7.protocol.load

    def _load_ext(external=()):  # ttt7.main's protocol.load() now appends EXT; fails loudly if nothing was appended
        d = _load(external=EXT); n = int((d["split"] == "external").sum()); assert n > 0, f"EXT {EXT} appended no windows"
        print(f"EXT {EXT}: {n} external training windows, {len(set(d['subject'][d['split'] == 'external']))} people", flush=True)
        # deep._fit switches to a class-balanced weighted loss whenever d["tag"] is set, and SessionBN training does not support
        # weighting. Both sets are class-balanced (PROTEUS 885/886/884, Scherer 710 each), so the weights would be about 1:
        # assert that, then clear the tag. (protocol's "~m" nolow flag is not read by deep/session_bn at all: a no-op there.)
        c = np.bincount(d["y"].astype(int), minlength=3); assert c.max() / c.min() < 1.02, f"classes not balanced: {c}"
        assert not any("~" in t for t in EXT), "EXT options (~m, ~w, ~n) are not honoured by SessionBN training"
        d["tag"] = ""; return d
    T7.protocol.load = _load_ext
assert T7.run.__globals__["PERM"] is PERM and T7.stream.__globals__["DRAW"] == DRAW
assert not EXT or T7.main.__globals__["protocol"].load is _load_ext, "ttt7.main would not see the external data"

if __name__ == "__main__":
    T7.main()
