"""Lane E: extra Eye / Rest seeds with the shipped recipe (scripts/train_v6_members.py, unchanged), all ten public people.

python lane_e/train_extra_seeds.py KIND SEED OUT_DIR [smoke]     KIND = eye | rest ; SEED in 3..5 (0..2 are shipped)
Imports the shipped trainer as a module and calls its eye()/rest() with the seed; nothing in the recipe is copied or
changed. Refuses seeds 0-2 so the shipped fits can not be overwritten. Env NI26_WS = the code root (scripts/, experiments/,
studies/, data/, 2026-competition/ present).
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["NI26_WS"])
sys.argv, (kind, seed, out), smoke = sys.argv[:1], sys.argv[1:4], len(sys.argv) > 4 and sys.argv[4] == "smoke"
seed = int(seed)
assert kind in ("eye", "rest") and 3 <= seed <= 5, "seeds 0-2 are the shipped fits"
spec = importlib.util.spec_from_file_location("train_v6_members", ROOT / "scripts/train_v6_members.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
T.OUT = Path(out)
T.OUT.mkdir(parents=True, exist_ok=True)
res = (T.eye if kind == "eye" else T.rest)(seed, smoke)
print(json.dumps(res), flush=True)
