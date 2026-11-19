"""Lane B6 stage A: Lane H's hp7 raw-FBCNet trunk recipe (7 corpora, one head each, rot 0.05, 30 ep) plus the MI84 / NETBCI corpora, each with its OWN head.

python laneB6/pretrain_b6.py ARM SEED   ARM G: hp7 + ds007221g (Graz hand/feet/rest, 14 people)       trunk tag hbg
                                        ARM N: hp7 + netbci (19 people, right hand vs rest)             hbn
                                        ARM M: hp7 + ds007221g + netbci                                 hbm
                                        ARM X: hp7 + ds007221g + ds007221lr (Hybrid people, Graz condition) + netbci   hbx
                                        ARM Z: M with 1 epoch (smoke)                                   hbz
Calls experiments/pretrain.py main() unchanged. The wrapper asserts that every new corpus was read, that its class list equals
class_map.OWN_HEAD (explicit name -> local index map, unmapped names raised at export) and that every class has windows; it never
joins the shared 4-way block, so no label of these corpora can reach a PROTEUS task index. Output runs/pretrain/fbcnet_raw-hb<arm>-sSEED.pt (+ .map.json)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding")]
import class_map  # noqa: E402
import pretrain as P  # noqa: E402

M7 = "physionetmi,cho2017,lee2019_mi,dreyer2023,schirrmeister2017,shin2017a,shin2017b"
NEW = {"G": ["ds007221g"], "N": ["netbci"], "M": ["ds007221g", "netbci"], "X": ["ds007221g", "ds007221lr", "netbci"], "Z": ["ds007221g", "netbci"]}
SEEN, MAPS, _orig = {}, {}, P.load_corpus


def load_corpus(name, align):
    x, y, classes, n = _orig(name, align)
    if name in class_map.OWN_HEAD:
        assert list(classes) == class_map.OWN_HEAD[name], (name, classes)
        cnt = np.bincount(y, minlength=len(classes))
        assert (cnt > 0).all() and len(y) == len(x), (name, cnt.tolist())
        SEEN[name] = dict(windows=len(y), people=n, counts=cnt.tolist())
        MAPS[name] = {c: i for i, c in enumerate(classes)}
    return x, y, classes, n


P.load_corpus = load_corpus
if __name__ == "__main__":
    arm, seed = sys.argv[1], sys.argv[2]
    argv = ["fbcnet", "raw", f"hb{arm.lower()}", ",".join([M7, *NEW[arm]]), "--seed", seed, "--rot", "0.05"]
    sys.argv = ["pretrain.py", *argv, *(["--epochs", "1"] if arm == "Z" else [])]
    P.main()
    assert sorted(SEEN) == sorted(NEW[arm]), f"own-head corpora not read through the checked loader: {sorted(SEEN)}"
    (ROOT / f"runs/pretrain/fbcnet_raw-hb{arm.lower()}-s{seed}.map.json").write_text(json.dumps(dict(maps=MAPS, seen=SEEN), indent=1) + "\n")
    print("MAP", json.dumps(MAPS), json.dumps(SEEN), flush=True)
