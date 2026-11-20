"""Union of the members of several packages (same people) -> one package; no training. Eval-only ensembles.

python cleanslate/ttt_base/merge.py OUT PKG1 PKG2 ... [--pick NAME,NAME]   (--pick keeps members whose `name` is listed)
"""
import json
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
assert len({(Path(p) / "submission.py").read_text() for p in pk}) == 1, "packages differ in submission.py"
mem = [m for b in blobs for m in b["members"] if pick is None or m["name"] in pick]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
torch.save(dict(members=mem, ttt=blobs[0]["ttt"], people=blobs[0]["people"], args=blobs[0]["args"]), out / "weights.pt")
shutil.copy(Path(pk[0]) / "submission.py", out / "submission.py")
tj[0]["members"] = [dict(member=f"{m['name']}:{m['seed']}") for m in mem]
json.dump(tj[0], open(out / "train.json", "w"), indent=1)
print("merged", len(mem), "members from", pk)
