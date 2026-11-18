"""Assemble private source bundles. Source and model assets stay immutable."""
import json
from pathlib import Path

from returning_assets import ROOT, sha

origin = ROOT / "ni26_js321_20261117_returning_bundle2"
manifest = json.loads((origin / "bundle_complete.json").read_text())
for fold in ("P04_s23", "P10_s23"):
    target = Path("packages") / fold
    target.mkdir(parents=True)
    source = origin / fold / "js"
    entry = next(item for item in manifest["artifacts"] if item["path"] == str(source.relative_to(origin)))
    for name, digest in entry["files"].items():
        item = source / name
        if sha(item) != digest:
            raise ValueError(f"source changed: {name}")
        if name != "submission.py":
            (target / name).symlink_to(item)
    body = (source / "submission.py").read_text()
    if body.count("class Solver(CompetSolver):") != 1 or body.count("model = ReuseEnsemble(") != 1:
        raise ValueError("private source constructor differs")
    body = body.replace("class Solver(CompetSolver):", Path("policy.py").read_text() + "\n\nclass Solver(CompetSolver):")
    body = body.replace("model = ReuseEnsemble(", "model = ProtectedEnsemble(")
    compile(body, str(target / "submission.py"), "exec")
    (target / "submission.py").write_text(body)
