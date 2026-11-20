"""Lane FT-B6s stage A: the shipped Sinc trunk recipe (sincshallow, EA, 7 corpora, one head each, rot 0.05, 30 ep; docs/AUDIT_NXIF.md row 4)
with and without the B6 extra people. python laneB6/pretrain_sinc.py ARM SEED
  ARM X: M7 + ds007221g + ds007221lr + netbci (own heads, as B6 arm X)   -> runs/pretrain/sincshallow_ea-m7rx-sSEED.pt
  ARM C: M7 only, the shipped recipe re-run (control for re-pretraining noise) -> sincshallow_ea-m7rc-sSEED.pt
  ARM Z: X with 1 epoch (smoke)                                                  -> sincshallow_ea-m7rz-sSEED.pt
Uses pretrain_b6's checked corpus loader (class lists equal class_map.OWN_HEAD, own heads only) and experiments/pretrain.py main() unchanged."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "laneB6")]
import pretrain_b6 as B  # noqa: E402  (patches P.load_corpus, fills B.SEEN / B.MAPS)

TAG = {"X": "m7rx", "C": "m7rc", "Z": "m7rz"}
NEW = {"X": B.NEW["X"], "C": [], "Z": B.NEW["X"]}
if __name__ == "__main__":
    arm, seed = sys.argv[1], sys.argv[2]
    corpora = ",".join([B.M7, *NEW[arm]])
    sys.argv = ["pretrain.py", "sincshallow", "ea", TAG[arm], corpora, "--seed", seed, "--rot", "0.05", *(["--epochs", "1"] if arm == "Z" else [])]
    B.P.main()
    assert sorted(B.SEEN) == sorted(NEW[arm]), f"own-head corpora not read through the checked loader: {sorted(B.SEEN)}"
    out = ROOT / f"runs/pretrain/sincshallow_ea-{TAG[arm]}-s{seed}.pt"
    assert out.is_file(), out
    out.with_suffix(".map.json").write_text(json.dumps(dict(maps=B.MAPS, seen=B.SEEN, corpora=corpora), indent=1) + "\n")
    print("DONE", out, json.dumps(B.SEEN), flush=True)
