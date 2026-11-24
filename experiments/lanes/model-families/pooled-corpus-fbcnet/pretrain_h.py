"""Lane H stage A: pooled-corpus supervised trunk for the raw FBCNet member (cap250 slot 48 recipe).

python laneH/pretrain_h.py TAG SEED   TAG hp7 = the shipped m7r pretraining (7 corpora, one head each, rot 0.05, 30 epochs);
                                      TAG hp9 = those 7 + Scherer 2015 + Shin 2018 word generation, ALL in one task-matched
                                      4-way block (motor / subtraction / word generation / rest), head transferred.
Calls experiments/pretrain.py main() unchanged; this wrapper only (a) lets load_corpus read the two new corpora and
(b) asserts that the patch was used. Output runs/pretrain/fbcnet_raw-TAG-sSEED.pt under the work root."""
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding")]
import pretrain as P  # noqa: E402
import protocol  # noqa: E402

M7 = "physionetmi,cho2017,lee2019_mi,dreyer2023,schirrmeister2017,shin2017a,shin2017b"
NEW = {"scherer2015": ["hands", "subtraction", "S16"],  # export labels HAND 0, SUB 1, WORD 2 (laneC/data.py SCHERER_EXPORT)
       "shin2018": None}  # classes WG / baseline in the file
P.UNIFIED.update({"WG": 2, "baseline": 3})
USED = []
_orig = P.load_corpus


def load_corpus(name, align):
    if name not in NEW:
        return _orig(name, align)
    files = sorted(glob.glob(str(ROOT / "data" / name / "windows*.npz")))
    assert files, name
    Xs, ys, ss, classes = [], [], [], NEW[name]
    for f in files:
        with np.load(f) as z:
            classes = list(z["classes"]) if "classes" in z.files else classes
            keep = z["win"] == 0 if "win" in z.files else np.ones(len(z["y"]), bool)  # laneC: first window of a trial
            s = np.char.add(np.char.add(z["subject"].astype(str), "|"), z["session"].astype(str))
            Xs.append(protocol.fill_null(z["X"][keep], s[keep])), ys.append(z["y"][keep]), ss.append(s[keep])
    X, y, s = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
    USED.append(name)
    return P.align_sessions(X.astype(np.float32), s, align), y, classes, len(np.unique(np.char.partition(s, "|")[:, 0]))


P.load_corpus = load_corpus
if __name__ == "__main__":
    tag, seed = sys.argv[1], sys.argv[2]
    pooled = M7 + ",scherer2015,shin2018"
    argv = {"hp7": ["fbcnet", "raw", "hp7", M7],
            "hp9": ["fbcnet", "raw", "hp9", pooled, "--unified", pooled],
            "hs9": ["fbcnet", "raw", "hs9", pooled, "--unified", pooled, "--epochs", "1"]}[tag]  # hs9: smoke, 1 epoch
    sys.argv = ["pretrain.py", *argv, "--seed", seed, "--rot", "0.05"]
    P.main()
    assert sorted(USED) == ([] if tag == "hp7" else ["scherer2015", "shin2018"]), f"corpus patch state wrong for {tag}: {USED}"
