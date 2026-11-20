"""Union of the members of several packages (same people) -> one package; no training. Eval-only ensembles.

python cleanslate/ttt_base/merge.py OUT PKG1 PKG2 ... [--pick NAME,NAME]   (--pick keeps members whose `name` is listed)
"""
import json
import os
import shutil
import sys
from pathlib import Path

import torch

out, *pk = sys.argv[1:]
pick = None
if "--pick" in pk:
    i = pk.index("--pick")
    pick, pk = set(pk[i + 1].split(",")), pk[:i]
blobs = [torch.load(Path(p) / "weights.pt", map_location="cpu", weights_only=False) for p in pk]
tj = [json.load(open(Path(p) / "train.json")) for p in pk]
assert all(t["people"] == tj[0]["people"] for t in tj), "packages differ in training people"
regen = bool(os.environ.get("REGEN"))  # REGEN=1: write the CURRENT generated submission.py (code only; weights are unchanged)
assert regen or len({(Path(p) / "submission.py").read_text() for p in pk}) == 1, "packages differ in submission.py"
mem = [m for b in blobs for m in b["members"] if pick is None or m["name"] in pick]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
torch.save(dict(members=mem, ttt=blobs[0]["ttt"], people=blobs[0]["people"], args=blobs[0]["args"]), out / "weights.pt")
if regen:
    import importlib.util
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("ttt_base_train", here / "train.py")
    TB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TB)  # ttt_base/train.py (generate_submission)
    TB.generate_submission(out)
else:
    shutil.copy(Path(pk[0]) / "submission.py", out / "submission.py")
tj[0]["members"] = [dict(member=f"{m['name']}:{m['seed']}") for m in mem]
json.dump(tj[0], open(out / "train.json", "w"), indent=1)
print("merged", len(mem), "members from", pk)
