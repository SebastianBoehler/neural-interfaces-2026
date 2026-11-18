import json
from pathlib import Path

from returning_assets import ROOT, sha

origin = ROOT / "ni26_js321_20261117_inputreuse2"
complete = json.loads((origin / "complete.json").read_text())
target = Path("protected")
target.mkdir()
source = origin / "prepared_jsall"
if set(complete["files"]) != {p.name for p in source.iterdir() if p.is_file()}:
    raise ValueError("all-public JS package inventory changed")
for item in source.iterdir():
    if not item.is_file():
        continue
    if sha(item) != complete["files"][item.name]:
        raise ValueError("all-public JS member changed: " + item.name)
    if item.name != "submission.py":
        (target / item.name).symlink_to(item)
        continue
    body = item.read_text()
    before = "model = ReuseEnsemble("
    marker = "class Solver(CompetSolver):"
    if body.count(before) != 1 or body.count(marker) != 1:
        raise ValueError("private package constructor differs")
    module = body.replace(before, "model = ProtectedEnsemble(").replace(marker, Path("policy.py").read_text() + "\n\n" + marker)
    compile(module, str(target / item.name), "exec")
    (target / item.name).write_text(module)
