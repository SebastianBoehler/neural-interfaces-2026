"""Lane V: laneH/lopo.py (arm B recipe: hp7 trunk, SessionBN raw FBCNet, same seeds/orders) with a virtual-subject augmentation.
python laneV/lopo_v.py ARM SEED [FOLD ...]   ARM in vaug.ARMS (V0 = arm B untouched). Output stem h_{ARM}_{seed}_{held} in $LANEH_WORK."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "laneH"), str(ROOT / "laneV")]
import numpy as np  # noqa: E402
import lopo as H  # noqa: E402
import session_bn  # noqa: E402
import vaug  # noqa: E402

H.CFG.update({a: H.CFG["B"] for a in vaug.ARMS})  # same trunk / net / optimiser as arm B
_make, STATE = session_bn.make, {}


def make(cfg, d, seed):
    net = _make(cfg, d, seed)
    assert net.init and "hp7" in net.init and net.rot == 0.0 and net.epochs == 40, "not the arm B recipe"
    STATE["h"] = vaug.install(net, STATE["arm"], seed, d, np.load(ROOT / "data/proteus_windows.npz")["ch_names"])
    assert (STATE["h"] is None) == (STATE["arm"] == "V0")
    return net


session_bn.make = make
if __name__ == "__main__":
    import os
    arm, seed = sys.argv[1], int(sys.argv[2]); STATE["arm"] = arm
    work = Path(os.environ["LANEH_WORK"])
    for held in sys.argv[3:] or [f"P{i:02}" for i in range(1, 11)]:
        H.run(arm, seed, held, work)
        h = STATE["h"]
        if h:
            print("VSTATS", arm, seed, held, {k: round(v, 4) for k, v in h["vx"].stats.items()}, flush=True)
