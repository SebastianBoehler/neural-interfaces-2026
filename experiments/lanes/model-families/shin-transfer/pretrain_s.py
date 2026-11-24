"""Lane S stage A: raw FBCNet trunk = Lane H's m7r pretraining (hp7) with the Shin cohorts in a correctly mapped task block.

python laneS/pretrain_s.py ARM SEED   ARM P: shin2017a (MI) + shin2017b (SUB, rest aux) in the shared 4-way block, rest as hp7
                                      ARM W: hp7 + shin2018 (WG -> 2, baseline aux 3) in the shared block
                                      ARM D: both;  ARM Z: D with 1 epoch (smoke).  Block: 0 MI, 1 SUB, 2 WORD, 3 rest/baseline.
Calls experiments/pretrain.py main() unchanged. Class names -> block indices come from experiments/class_map.py and are
asserted per corpus (unknown or unexpected names raise). Output runs/pretrain/fbcnet_raw-hs<arm>-sSEED.pt (+ .map.json)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding")]
import class_map  # noqa: E402
import pretrain as P  # noqa: E402
import protocol  # noqa: E402

M7 = "physionetmi,cho2017,lee2019_mi,dreyer2023,schirrmeister2017,shin2017a,shin2017b"
ARMS = {"P": (M7, "shin2017a,shin2017b"), "W": (M7 + ",shin2018", "shin2018"),
        "D": (M7 + ",shin2018", "shin2017a,shin2017b,shin2018"), "Z": (M7 + ",shin2018", "shin2017a,shin2017b,shin2018")}
BLOCK = {**class_map.TASK, **{c: 3 for c in class_map.AUX}}  # rest / baseline: the 4th (auxiliary) output, never a task class
P.UNIFIED = BLOCK
SEEN, MAPS, _orig = [], {}, P.load_corpus


def load_corpus(name, align):
    if name != "shin2018":
        x, y, classes, n = _orig(name, align)
    else:  # one file, windows.npz; first window of every trial (laneH/laneC convention)
        Xs, ys, ss = [], [], []
        for f in sorted(glob.glob(str(ROOT / "data" / name / "windows*.npz"))):
            with np.load(f) as z:
                classes = class_map.class_names(name, z)
                keep = z["win"] == 0
                s = np.char.add(np.char.add(z["subject"].astype(str), "|"), z["session"].astype(str))
                Xs.append(protocol.fill_null(z["X"][keep], s[keep])), ys.append(z["y"][keep]), ss.append(s[keep])
        X, y, s = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
        x, n = P.align_sessions(X.astype(np.float32), s, align), len(np.unique(np.char.partition(s, "|")[:, 0]))
    if name in EXPECT_UNI:
        assert set(classes) == class_map.EXPECT[name], (name, classes)
        MAPS[name] = {c: BLOCK[c] for c in classes}
        SEEN.append(name)
    return x, y, classes, n


EXPECT_UNI = set()
P.load_corpus = load_corpus
if __name__ == "__main__":
    arm, seed = sys.argv[1], sys.argv[2]
    corpora, uni = ARMS[arm]
    EXPECT_UNI.update(uni.split(","))
    argv = ["fbcnet", "raw", f"hs{arm.lower()}", corpora, "--unified", uni, "--seed", seed, "--rot", "0.05"]
    sys.argv = ["pretrain.py", *argv, *(["--epochs", "1"] if arm == "Z" else [])]
    P.main()
    assert sorted(SEEN) == sorted(EXPECT_UNI), f"unified corpora not read through the checked loader: {SEEN}"
    out = ROOT / f"runs/pretrain/fbcnet_raw-hs{arm.lower()}-s{seed}.map.json"
    out.write_text(json.dumps(MAPS, indent=1) + "\n")
    print("MAP", json.dumps(MAPS), flush=True)
