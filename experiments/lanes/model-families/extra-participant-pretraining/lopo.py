"""Lane B6 stage B: laneH/lopo.py with the MI84 / NETBCI arms. python laneB6/lopo.py ARM SEED [FOLD ...]  ARM G/N/M/X/Z as pretrain_b6.py
(trunk fbcnet_raw-hb<arm>-sSEED.pt, same SessionBN recipe, seeds, folds and orders as lane H arms A/B; Z = 1-epoch smoke)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "laneH")]
import lopo as H  # noqa: E402

H.CFG.update({a: f"fbcnet_raw_pthb{a.lower()}_sbn" for a in "GNMXZ"})
assert H.CFG["X"] == "fbcnet_raw_pthbx_sbn"
if __name__ == "__main__":
    import os
    if sys.argv[1] == "Z":
        os.environ["LANEH_QUICK"] = "1"  # honoured in H.run: 1 epoch
    arm, seed = sys.argv[1], int(sys.argv[2])
    work = Path(os.environ["LANEH_WORK"])
    for held in sys.argv[3:] or [f"P{i:02}" for i in range(1, 11)] + ["all"]:
        H.run(arm, seed, held, work)
