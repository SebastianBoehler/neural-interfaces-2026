"""Stage-B-only fit of CBraMod on PROTEUS (arm A, held P02, seed 0) through Lane C's trainer; prints held BA per order."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "laneC")]
import numpy as np
import train_cbramod as T

root, work, ck, held = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
class A: pass
args = A(); args.root, args.work, args.checkpoint = root, work, ck
T.lane_c.stage_b(args, "A", 0, held)
z = np.load(work / f"reve_A_0_{held}.npz")
y = np.load(root / "data/proteus_windows.npz")["y"][z["index"]]
p = z["full0"]
print("mean max-posterior", float(p.max(1).mean()), "pred counts", np.bincount(p.argmax(1), minlength=3))
print("held", held, "recall per class", [float((p.argmax(1)[y == c] == c).mean()) for c in range(3)], "BA", float(np.mean([(p.argmax(1)[y == c] == c).mean() for c in range(3)])))
