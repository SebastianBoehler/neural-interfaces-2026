"""Stage-strength profile for the sealed portfolio (31 Oct): R19e view (gaze + events)
at lam 0 (= R15), 0.25, 0.5 (R19e), 0.8, and R19 (gaze only, lam 0.5); six-seed halves."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import *  # d, load, mixp, rebal, cm
import session_em as SE

X = d["X"].astype(np.float64); ch = list(d["ch_names"]); st = d["stream"]
G = SE.gaze_features(X, ch)
FE = np.c_[G, SE.event_features(X, ch, st, SE.global_mad(X, ch))]
del X


def spec(a, b, tfb):
    return ["v2", f"fbcnet_ea_tf:{a}-{tfb}*0.5", f"fbcnet_raw_tf:{a}-{tfb}*0.5"] + [f"{n}:{a}-{b}" for n in (
        "shallow_ea", "shallow_raw", "fbcnet_raw", "fbcnet_ea_rot05", "sincshallow_ea_ptm7r_rot05", "fblight_ea_rot05")]


ARMS = {"R15": None, "R19": (G, 0.5), "e.25": (FE, 0.25), "R19e": (FE, 0.5), "e.8": (FE, 0.8)}
for tag, sp in (("s0-2", spec(0, 2, 1)), ("s3-5", spec(3, 5, 4)), ("all6", None)):
    for proto in ("dev_loso_all", "dev_session", "all_loso"):
        _, Ps, w, idx, _ = load(proto, sp)
        base = rebal(mixp(Ps, w), idx)
        sc = {k: cm(idx, (base if a is None else SE.apply(a[0][idx], base, st[idx], lam=a[1])).argmax(1))
              for k, a in ARMS.items()}
        print(f"{tag} {proto:13s} " + " ".join(f"{k} {v:.2f}" for k, v in sc.items()), flush=True)
